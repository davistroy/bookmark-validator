"""
Async HTTP Client for URL Validation

This module provides asynchronous HTTP client functionality for validating URLs
with rate limiting and concurrent request management.
"""

from __future__ import annotations

import asyncio
import logging
import time
from asyncio import Semaphore
from typing import TYPE_CHECKING, List, Optional

import aiohttp

from .batch_types import BatchResult, ValidationResult

if TYPE_CHECKING:
    from ..utils.intelligent_rate_limiter import IntelligentRateLimiter
    from ..utils.security_validator import SecurityValidator
    from ..utils.browser_simulator import BrowserSimulator


class AsyncHttpClient:
    """
    Async HTTP client for URL validation with rate limiting.

    Provides async methods for validating URLs individually or in batches
    with proper rate limiting and concurrent request management.
    """

    # Valid HTTP status codes for success
    SUCCESS_CODES = {
        200, 201, 202, 203, 204, 205, 206,
        300, 301, 302, 303, 304, 307, 308,
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_concurrent: int = 10,
        verify_ssl: bool = True,
        rate_limiter: Optional["IntelligentRateLimiter"] = None,
        security_validator: Optional["SecurityValidator"] = None,
        browser_simulator: Optional["BrowserSimulator"] = None,
    ):
        """
        Initialize async HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
            verify_ssl: Whether to verify SSL certificates
            rate_limiter: Optional rate limiter instance
            security_validator: Optional security validator
            browser_simulator: Optional browser simulator for headers
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.verify_ssl = verify_ssl
        self.rate_limiter = rate_limiter
        self.security_validator = security_validator
        self.browser_simulator = browser_simulator

        self._async_session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

    async def _initialize_session(self) -> None:
        """Initialize aiohttp session for async validation"""
        if self._async_session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout, connect=10.0)
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent,
                limit_per_host=5,  # Max connections per host
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=self.verify_ssl,
            )

            self._async_session = aiohttp.ClientSession(
                timeout=timeout, connector=connector
            )

    async def close(self) -> None:
        """Close the async session"""
        if self._async_session:
            await self._async_session.close()
            self._async_session = None

    async def _apply_rate_limiting(self, url: str) -> None:
        """Apply rate limiting for async requests"""
        if not self.rate_limiter:
            return

        # Use the existing rate limiter's logic but with async sleep
        domain = self.rate_limiter._extract_domain(url)

        with self.rate_limiter.lock:
            stats = self.rate_limiter.domain_stats[domain]
            delay = self.rate_limiter._get_domain_delay(domain)

            # Calculate required wait time
            current_time = time.time()
            time_since_last = current_time - stats.last_request_time
            wait_time = max(0, delay - time_since_last)

            # Update stats immediately to prevent race conditions
            stats.last_request_time = current_time
            stats.request_count += 1
            self.rate_limiter.active_domains.add(domain)

        # Apply wait asynchronously
        if wait_time > 0:
            self.logger.debug(f"Async rate limiting {domain}: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

            with self.rate_limiter.lock:
                stats.total_wait_time += wait_time

    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL has valid format"""
        from urllib.parse import urlparse

        if not url or not isinstance(url, str):
            return False

        url = url.strip()
        if not url:
            return False

        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped"""
        skip_patterns = [
            "javascript:",
            "mailto:",
            "tel:",
            "ftp:",
            "file:",
            "data:",
            "#",
            "about:blank",
        ]

        url_lower = url.lower().strip()
        for pattern in skip_patterns:
            if url_lower.startswith(pattern):
                return True
        return False

    def _classify_http_error(self, status_code: int) -> str:
        """Classify HTTP error by status code"""
        if 400 <= status_code < 500:
            error_types = {
                400: "bad_request",
                401: "unauthorized",
                403: "forbidden",
                404: "not_found",
                405: "method_not_allowed",
                408: "request_timeout",
                429: "rate_limited",
            }
            return error_types.get(status_code, "client_error")
        elif 500 <= status_code < 600:
            error_types = {
                500: "internal_server_error",
                501: "not_implemented",
                502: "bad_gateway",
                503: "service_unavailable",
                504: "gateway_timeout",
            }
            return error_types.get(status_code, "server_error")
        else:
            return "http_error"

    def _parse_content_length(
        self, content_length_header: Optional[str]
    ) -> Optional[int]:
        """Parse content-length header"""
        if not content_length_header:
            return None
        try:
            return int(content_length_header)
        except (ValueError, TypeError):
            return None

    async def validate_url(self, url: str) -> ValidationResult:
        """
        Asynchronously validate a single URL.

        Args:
            url: URL to validate

        Returns:
            ValidationResult object
        """
        import os

        # Test mode guard - return mock validation for testing
        if os.getenv("BOOKMARK_PROCESSOR_TEST_MODE") == "true":
            return self._get_mock_validation_result(url)

        start_time = time.time()

        # Security validation first (sync, but fast)
        security_result = None
        if self.security_validator:
            security_result = self.security_validator.validate_url_security(url)
            if not security_result.is_safe:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error_message=security_result.blocked_reason
                    or "Security validation failed",
                    error_type="security_error",
                    response_time=time.time() - start_time,
                    security_validation=security_result,
                )

        # Basic URL validation
        if not self._is_valid_url_format(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="Invalid URL format",
                error_type="format_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Check skip patterns
        if self._should_skip_url(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="URL type not supported for validation",
                error_type="unsupported_scheme",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        try:
            # Initialize session if needed
            if self._async_session is None:
                await self._initialize_session()

            # Get headers for this request
            headers = {}
            if self.browser_simulator:
                headers = self.browser_simulator.get_headers(url)

            # Apply async rate limiting
            await self._apply_rate_limiting(url)

            # Make async request
            async with self._async_session.head(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                allow_redirects=True,
                ssl=self.verify_ssl,
            ) as response:
                response_time = time.time() - start_time

                # Check if successful
                is_valid = response.status in self.SUCCESS_CODES

                result = ValidationResult(
                    url=url,
                    is_valid=is_valid,
                    status_code=response.status,
                    final_url=str(response.url) if str(response.url) != url else None,
                    response_time=response_time,
                    content_type=response.headers.get("content-type"),
                    content_length=self._parse_content_length(
                        response.headers.get("content-length")
                    ),
                    security_validation=security_result,
                )

                if not is_valid:
                    result.error_message = f"HTTP {response.status}"
                    result.error_type = self._classify_http_error(response.status)

                return result

        except asyncio.TimeoutError:
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Timeout after {self.timeout}s",
                error_type="timeout",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )
        except aiohttp.ClientError as e:
            error_msg = str(e)
            error_type = "connection_error"

            if "dns" in error_msg.lower() or "name resolution" in error_msg.lower():
                error_type = "dns_error"
            elif "refused" in error_msg.lower():
                error_type = "connection_refused"

            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=error_msg,
                error_type=error_type,
                response_time=time.time() - start_time,
                security_validation=security_result,
            )
        except Exception as e:
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type="unknown_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

    async def validate_batch(self, urls: List[str], batch_id: str) -> BatchResult:
        """
        Asynchronously validate a batch of URLs with rate limiting.

        Args:
            urls: List of URLs to validate
            batch_id: Unique identifier for this batch

        Returns:
            BatchResult with processing details
        """
        start_time = time.time()

        self.logger.info(f"Processing async batch {batch_id} with {len(urls)} URLs")

        # Create semaphore for concurrent validation within this batch
        validation_semaphore = Semaphore(min(self.max_concurrent, len(urls)))

        async def validate_with_semaphore(url: str) -> ValidationResult:
            async with validation_semaphore:
                return await self.validate_url(url)

        # Process all URLs concurrently with rate limiting
        tasks = [validate_with_semaphore(url) for url in urls]
        validation_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        final_results = []
        error_messages = []

        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                error_msg = f"{urls[i]}: {str(result)}"
                error_messages.append(error_msg)

                # Create error result
                error_result = ValidationResult(
                    url=urls[i],
                    is_valid=False,
                    error_message=str(result),
                    error_type="async_validation_error",
                    response_time=0.0,
                )
                final_results.append(error_result)
            else:
                final_results.append(result)

        # Calculate batch statistics
        successful_count = sum(1 for result in final_results if result.is_valid)
        failed_count = len(final_results) - successful_count
        processing_time = time.time() - start_time
        average_time_per_item = processing_time / len(urls) if urls else 0
        error_rate = failed_count / len(urls) if urls else 0

        batch_result = BatchResult(
            batch_id=batch_id,
            items_processed=len(urls),
            items_successful=successful_count,
            items_failed=failed_count,
            processing_time=processing_time,
            average_item_time=average_time_per_item,
            error_rate=error_rate,
            results=final_results,
            errors=error_messages[:10],  # Limit to first 10 errors
        )

        self.logger.info(
            f"Async batch {batch_id} completed: {successful_count}/{len(urls)} URLs "
            f"valid in {processing_time:.2f}s "
            f"(avg {average_time_per_item:.3f}s per URL)"
        )

        return batch_result

    def _get_mock_validation_result(self, url: str) -> ValidationResult:
        """
        Generate mock validation result for test mode.

        Args:
            url: URL being validated

        Returns:
            Mock ValidationResult
        """
        # Most URLs are valid in test mode, except a few specific test cases
        is_valid = not any(
            invalid in url.lower() for invalid in ["invalid", "404", "error", "timeout"]
        )

        return ValidationResult(
            url=url,
            is_valid=is_valid,
            status_code=200 if is_valid else 404,
            final_url=url,
            error_message=None if is_valid else "Mock test failure",
            error_type=None if is_valid else "test_error",
            response_time=0.1,  # Fast mock response
            content_type="text/html" if is_valid else None,
        )


__all__ = [
    "AsyncHttpClient",
]
