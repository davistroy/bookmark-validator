"""
Core URL Validator

Provides synchronous URL validation with retry logic, rate limiting,
and browser simulation.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import requests

from ...utils.browser_simulator import BrowserSimulator
from ...utils.intelligent_rate_limiter import IntelligentRateLimiter
from ...utils.progress_tracker import ProcessingStage, ProgressTracker
from ...utils.retry_handler import RetryHandler
from ...utils.secure_logging import secure_logger
from ...utils.security_validator import SecurityValidationResult, SecurityValidator
from ..batch_types import ValidationResult, ValidationStats
from .helpers import (
    SUCCESS_CODES,
    classify_http_error,
    is_valid_url_format,
    parse_content_length,
    should_skip_url,
)


class URLValidator:
    """Enhanced URL validation with retry logic and intelligent rate limiting"""

    def __init__(
        self,
        timeout: float = 30.0,
        max_redirects: int = 5,
        max_concurrent: int = 10,
        user_agent_rotation: bool = True,
        verify_ssl: bool = True,
        rate_limiter: Optional[IntelligentRateLimiter] = None,
    ):
        """
        Initialize URL validator.

        Args:
            timeout: Request timeout in seconds
            max_redirects: Maximum number of redirects to follow
            max_concurrent: Maximum concurrent requests
            user_agent_rotation: Whether to rotate user agents
            verify_ssl: Whether to verify SSL certificates
            rate_limiter: Optional rate limiter instance
        """
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_concurrent = max_concurrent
        self.verify_ssl = verify_ssl

        # Initialize components
        self.rate_limiter = rate_limiter or IntelligentRateLimiter(
            max_concurrent=max_concurrent
        )
        self.browser_simulator = BrowserSimulator(rotate_agents=user_agent_rotation)
        self.retry_handler = RetryHandler()
        self.security_validator = SecurityValidator()

        # Create session
        self.session = self._create_session()

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = ValidationStats()

        logging.info(
            f"Initialized URL validator (timeout={timeout}s, "
            f"max_concurrent={max_concurrent}, ssl_verify={verify_ssl})"
        )

    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate a single URL.

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

        # Security validation first
        security_result = self.security_validator.validate_url_security(url)
        if not security_result.is_safe:
            # Log security event
            if security_result.risk_level in ["high", "critical"]:
                secure_logger.log_url_validation_failure(
                    url,
                    security_result.blocked_reason or "Security validation failed",
                    security_result.issues,
                )

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
        if not is_valid_url_format(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="Invalid URL format",
                error_type="format_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Check skip patterns
        if should_skip_url(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="URL type not supported for validation",
                error_type="unsupported_scheme",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Apply rate limiting
        self.rate_limiter.wait_if_needed(url)

        try:
            # Get headers for this request
            headers = self.browser_simulator.get_headers(url)

            # Make request
            response = self.session.head(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=self.verify_ssl,
            )

            response_time = time.time() - start_time

            # Check if successful
            is_valid = response.status_code in SUCCESS_CODES

            result = ValidationResult(
                url=url,
                is_valid=is_valid,
                status_code=response.status_code,
                final_url=response.url if response.url != url else None,
                response_time=response_time,
                content_type=response.headers.get("content-type"),
                content_length=parse_content_length(
                    response.headers.get("content-length")
                ),
                security_validation=security_result,
            )

            if not is_valid:
                result.error_message = f"HTTP {response.status_code}"
                result.error_type = classify_http_error(response.status_code)

            # Record success for rate limiter
            self.rate_limiter.record_success(url)

            return result

        except requests.exceptions.SSLError as e:
            # Try without SSL verification if allowed
            if self.verify_ssl:
                logging.debug(f"SSL error for {url}, trying without verification: {e}")
                try:
                    response = self.session.head(
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=True,
                        verify=False,
                    )

                    response_time = time.time() - start_time
                    is_valid = response.status_code in SUCCESS_CODES

                    result = ValidationResult(
                        url=url,
                        is_valid=is_valid,
                        status_code=response.status_code,
                        final_url=response.url if response.url != url else None,
                        response_time=response_time,
                        content_type=response.headers.get("content-type"),
                        content_length=parse_content_length(
                            response.headers.get("content-length")
                        ),
                        error_message=(
                            "SSL certificate warning"
                            if is_valid
                            else f"HTTP {response.status_code}"
                        ),
                        error_type=(
                            "ssl_warning"
                            if is_valid
                            else classify_http_error(response.status_code)
                        ),
                        security_validation=security_result,
                    )

                    self.rate_limiter.record_success(url)
                    return result

                except Exception:
                    pass  # Fall through to error handling

            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"SSL Error: {str(e)}",
                error_type="ssl_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        except requests.exceptions.Timeout:
            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Timeout after {self.timeout}s",
                error_type="timeout",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        except requests.exceptions.ConnectionError as e:
            self.rate_limiter.record_error(url)
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
            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type="unknown_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

    def batch_validate(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
        enable_retries: bool = True,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> List[ValidationResult]:
        """
        Validate multiple URLs with concurrent processing and retry logic.

        Args:
            urls: List of URLs to validate
            progress_callback: Optional progress callback function
            enable_retries: Whether to enable retry logic for failed URLs
            progress_tracker: Optional progress tracker for detailed per-batch
                monitoring

        Returns:
            List of ValidationResult objects
        """
        logging.info(f"Starting batch validation of {len(urls)} URLs")
        start_time = time.time()

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        if len(unique_urls) != len(urls):
            logging.info(f"Removed {len(urls) - len(unique_urls)} duplicate URLs")

        # Initialize progress tracking for batch processing
        if progress_tracker:
            progress_tracker.start_stage(
                ProcessingStage.VALIDATING_URLS, len(unique_urls)
            )
            progress_tracker.set_active_tasks(self.max_concurrent)

        results = []

        # First pass: concurrent validation
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all validation tasks
            future_to_url = {
                executor.submit(self.validate_url, url): url for url in unique_urls
            }

            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update statistics
                    with self.lock:
                        self.stats.update_from_result(result)

                    # Add failed URLs to retry queue
                    if enable_retries and not result.is_valid:
                        if result.error_type not in [
                            "format_error",
                            "unsupported_scheme",
                        ]:
                            error = Exception(
                                result.error_message or "Validation failed"
                            )
                            self.retry_handler.add_failed_url(url, error)

                    completed += 1

                    # Update progress tracking
                    if progress_tracker:
                        progress_tracker.update_progress(
                            items_delta=1,
                            failed_delta=1 if not result.is_valid else 0,
                            stage_specific=True,
                        )

                    if progress_callback:
                        progress_callback(
                            f"Validated {completed}/{len(unique_urls)}: {url}"
                        )

                except Exception as e:
                    logging.error(f"Error validating {url}: {e}")
                    error_result = ValidationResult(
                        url=url,
                        is_valid=False,
                        error_message=str(e),
                        error_type="validation_error",
                    )
                    results.append(error_result)

                    if enable_retries:
                        self.retry_handler.add_failed_url(url, e)

                    # Update progress tracking for errors
                    if progress_tracker:
                        progress_tracker.update_progress(
                            items_delta=1, failed_delta=1, stage_specific=True
                        )

        # Second pass: retry failed URLs
        if enable_retries and self.retry_handler.retry_queue:
            retry_count = len(self.retry_handler.retry_queue)
            logging.info(f"Retrying {retry_count} failed URLs")

            # Track retry progress if progress tracker available
            if progress_tracker:
                # Start a sub-stage for retries
                progress_tracker.log_error(
                    f"Starting retry of {retry_count} failed URLs"
                )

            retry_results = self.retry_handler.retry_failed_items(
                self.validate_url, progress_callback
            )

            # Update results with successful retries
            retry_urls = {r.url for r in retry_results}
            results = [r for r in results if r.url not in retry_urls]
            results.extend(retry_results)

            # Track retry completion
            if progress_tracker:
                successful_retries = sum(1 for r in retry_results if r.is_valid)
                progress_tracker.log_error(
                    f"Retry completed: {successful_retries}/{retry_count} successful"
                )

        total_time = time.time() - start_time

        # Final statistics
        valid_count = sum(1 for r in results if r.is_valid)
        logging.info(
            f"Batch validation completed in {total_time:.2f}s: "
            f"{valid_count}/{len(results)} URLs valid "
            f"({valid_count/len(results)*100:.1f}%)"
        )

        # Complete progress tracking
        if progress_tracker:
            progress_tracker.set_active_tasks(0)  # Clear active tasks
            # Log final batch statistics
            progress_tracker.log_error(
                f"Batch validation summary: {valid_count}/{len(results)} valid "
                f"({valid_count/len(results)*100:.1f}%) in {total_time:.2f}s"
            )

        return results

    def batch_validate_optimized(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> List[ValidationResult]:
        """
        Optimized batch validation with memory-efficient processing and progress
        tracking

        Args:
            urls: List of URLs to validate
            progress_callback: Optional progress callback function
            batch_size: Number of URLs to process in each batch
            max_workers: Number of concurrent workers
            progress_tracker: Optional progress tracker for detailed tracking

        Returns:
            List of ValidationResult objects
        """
        logging.info(
            f"Starting optimized batch validation of {len(urls)} URLs "
            f"in batches of {batch_size}"
        )
        start_time = time.time()

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        total_urls = len(unique_urls)

        # Initialize progress tracker stage if provided
        if progress_tracker:
            progress_tracker.start_stage(ProcessingStage.VALIDATING_URLS, total_urls)
            logging.info(f"Progress tracker initialized for {total_urls} URLs")

        if max_workers is None:
            max_workers = min(self.max_concurrent, 20)  # Cap at 20 for optimization

        all_results = []
        processed_count = 0
        failed_count = 0

        # Process in batches to manage memory
        for batch_start in range(0, total_urls, batch_size):
            batch_end = min(batch_start + batch_size, total_urls)
            batch_urls = unique_urls[batch_start:batch_end]
            batch_number = batch_start // batch_size + 1

            logging.info(f"Processing batch {batch_number}: {len(batch_urls)} URLs")

            # Validate batch with threading
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all validations
                future_to_url = {
                    executor.submit(self.validate_url, url): url for url in batch_urls
                }

                batch_results = []
                batch_failed = 0
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        processed_count += 1

                        # Track failures for progress tracker
                        if not result.is_valid:
                            batch_failed += 1
                            failed_count += 1

                        # Update progress tracker if available
                        if progress_tracker:
                            progress_tracker.update_progress(
                                items_delta=1,
                                failed_delta=1 if not result.is_valid else 0,
                                stage_specific=True,
                            )

                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                f"Validated {processed_count}/{total_urls}: {url}"
                            )

                    except Exception as e:
                        logging.error(f"Error validating {url}: {e}")
                        # Create error result
                        error_result = ValidationResult(
                            url=url,
                            is_valid=False,
                            error_message=f"Validation error: {str(e)}",
                            error_type="validation_error",
                        )
                        batch_results.append(error_result)
                        processed_count += 1
                        batch_failed += 1
                        failed_count += 1

                        # Update progress tracker for errors
                        if progress_tracker:
                            progress_tracker.update_progress(
                                items_delta=1, failed_delta=1, stage_specific=True
                            )
                            progress_tracker.log_error(
                                f"Validation error for {url}: {e}"
                            )

            # Add batch results to total
            all_results.extend(batch_results)

            # Log batch completion with progress tracker
            if progress_tracker:
                batch_success_rate = (
                    (len(batch_results) - batch_failed) / len(batch_results)
                ) * 100
                logging.info(
                    f"Batch {batch_number} completed: "
                    f"{len(batch_results) - batch_failed}/{len(batch_results)} valid "
                    f"({batch_success_rate:.1f}% success rate)"
                )

            # Force garbage collection between batches
            import gc

            gc.collect()

            # Brief pause between batches to allow rate limiting
            if batch_end < total_urls:
                time.sleep(0.1)

        total_time = time.time() - start_time
        valid_count = sum(1 for r in all_results if r.is_valid)
        overall_success_rate = (valid_count / len(all_results)) * 100

        # Log comprehensive completion statistics
        logging.info(
            f"Optimized batch validation completed: "
            f"{valid_count}/{len(all_results)} valid URLs "
            f"({overall_success_rate:.1f}% success rate) in {total_time:.2f}s"
        )

        if progress_tracker:
            logging.info(
                f"Progress tracker final state: {processed_count} processed, "
                f"{failed_count} failed"
            )

        return all_results

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        with self.lock:
            stats_dict = {
                "total_urls": self.stats.total_urls,
                "valid_urls": self.stats.valid_urls,
                "invalid_urls": self.stats.invalid_urls,
                "success_rate": self.stats.valid_urls / max(self.stats.total_urls, 1),
                "average_response_time": self.stats.average_response_time,
                "total_processing_time": self.stats.total_time,
                "redirected_urls": self.stats.redirected_urls,
                "error_distribution": dict(self.stats.error_distribution),
                "rate_limiter_stats": self.rate_limiter.get_all_stats(),
            }

            if self.retry_handler:
                stats_dict["retry_stats"] = self.retry_handler.get_retry_statistics()

            return stats_dict

    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        with self.lock:
            self.stats = ValidationStats()
            self.rate_limiter.reset_domain_stats()
            if self.retry_handler:
                self.retry_handler.clear_completed()

        logging.info("Reset validation statistics")

    def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, "session"):
            self.session.close()

        logging.info("URL validator closed")

    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()

        # Configure adapters for retries at the HTTP level
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=0,  # We handle retries at application level
            connect=1,  # Allow one connection retry
            read=1,  # Allow one read retry
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )

        # Enhanced adapter with larger connection pool
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # Number of connection pools
            pool_maxsize=50,  # Max connections per pool
            pool_block=False,  # Don't block when pool is full
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default timeout
        session.timeout = self.timeout

        return session

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
            security_validation=(
                SecurityValidationResult(
                    is_safe=True,
                    risk_level="low",
                    issues=[],
                    blocked_reason=None,
                )
                if is_valid
                else SecurityValidationResult(
                    is_safe=False,
                    risk_level="medium",
                    issues=["Test failure"],
                    blocked_reason="Mock test error",
                )
            ),
        )


__all__ = ["URLValidator"]
