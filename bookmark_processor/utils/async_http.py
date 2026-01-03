"""
Modern Async HTTP Utilities

Provides a clean async HTTP client using httpx with structured concurrency
via Python 3.11+ TaskGroup. Falls back to asyncio.gather for older Python.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Check Python version for TaskGroup support
SUPPORTS_TASKGROUP = sys.version_info >= (3, 11)

T = TypeVar("T")


@dataclass
class HTTPResponse:
    """Standardized HTTP response container."""

    url: str
    status_code: int
    headers: Dict[str, str]
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    final_url: Optional[str] = None
    response_time: float = 0.0
    is_success: bool = False
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    burst_size: int = 20
    per_domain_limit: float = 2.0

    # Domain-specific overrides
    domain_limits: Dict[str, float] = field(default_factory=lambda: {
        "github.com": 1.5,
        "google.com": 2.0,
        "youtube.com": 2.0,
        "linkedin.com": 2.0,
        "twitter.com": 1.5,
        "facebook.com": 3.0,
    })


class AsyncRateLimiter:
    """Async-aware rate limiter with per-domain tracking."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._domain_last_request: Dict[str, float] = {}
        self._global_semaphore = asyncio.Semaphore(self.config.burst_size)
        self._lock = asyncio.Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if not domain:
                return "unknown"
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"

    def _get_domain_limit(self, domain: str) -> float:
        """Get rate limit for a specific domain."""
        for pattern, limit in self.config.domain_limits.items():
            if pattern in domain:
                return limit
        return self.config.per_domain_limit

    async def acquire(self, url: str) -> None:
        """Acquire rate limit permission for a URL."""
        domain = self._get_domain(url)

        async with self._global_semaphore:
            async with self._lock:
                # Get or create domain semaphore
                if domain not in self._domain_semaphores:
                    self._domain_semaphores[domain] = asyncio.Semaphore(1)

                # Check last request time for this domain
                last_request = self._domain_last_request.get(domain, 0)
                now = asyncio.get_event_loop().time()
                delay = self._get_domain_limit(domain)

                time_since_last = now - last_request
                if time_since_last < delay:
                    await asyncio.sleep(delay - time_since_last)

                self._domain_last_request[domain] = asyncio.get_event_loop().time()


class AsyncHTTPClient:
    """
    Modern async HTTP client using httpx.

    Features:
    - Connection pooling
    - Automatic retries with exponential backoff
    - Rate limiting (global and per-domain)
    - Browser-like headers
    - Structured concurrency with TaskGroup (Python 3.11+)
    """

    # Browser-like headers
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_connections: int = 100,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_connections = max_connections
        self.rate_limiter = rate_limiter or AsyncRateLimiter()
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AsyncHTTPClient":
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=20,
            ),
            follow_redirects=True,
            verify=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        retry: bool = True,
    ) -> HTTPResponse:
        """
        Perform a GET request with retries and rate limiting.

        Args:
            url: URL to fetch
            headers: Optional additional headers
            retry: Whether to retry on failure

        Returns:
            HTTPResponse with results
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        # Prepare headers
        request_headers = self.DEFAULT_HEADERS.copy()
        if headers:
            request_headers.update(headers)

        # Apply rate limiting
        await self.rate_limiter.acquire(url)

        attempt = 0
        last_error = None
        start_time = asyncio.get_event_loop().time()

        while attempt <= (self.max_retries if retry else 0):
            try:
                response = await self._client.get(url, headers=request_headers)
                elapsed = asyncio.get_event_loop().time() - start_time

                return HTTPResponse(
                    url=url,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content_type=response.headers.get("content-type"),
                    content_length=response.headers.get("content-length"),
                    final_url=str(response.url),
                    response_time=elapsed,
                    is_success=200 <= response.status_code < 400,
                )

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
            except httpx.HTTPError as e:
                last_error = f"HTTP error: {e}"
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            attempt += 1
            if attempt <= self.max_retries and retry:
                # Exponential backoff
                delay = min(2 ** attempt, 30)
                await asyncio.sleep(delay)

        elapsed = asyncio.get_event_loop().time() - start_time
        return HTTPResponse(
            url=url,
            status_code=0,
            headers={},
            response_time=elapsed,
            is_success=False,
            error=last_error,
        )

    async def validate_urls(
        self,
        urls: List[str],
        max_concurrent: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[HTTPResponse]:
        """
        Validate multiple URLs concurrently using structured concurrency.

        Args:
            urls: List of URLs to validate
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback(completed, total)

        Returns:
            List of HTTPResponse objects in same order as input
        """
        if not urls:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)
        results: Dict[int, HTTPResponse] = {}
        completed = 0

        async def validate_one(index: int, url: str) -> None:
            nonlocal completed
            async with semaphore:
                result = await self.get(url)
                results[index] = result
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(urls))

        # Use TaskGroup for Python 3.11+ or gather for older versions
        if SUPPORTS_TASKGROUP:
            async with asyncio.TaskGroup() as tg:
                for i, url in enumerate(urls):
                    tg.create_task(validate_one(i, url))
        else:
            tasks = [validate_one(i, url) for i, url in enumerate(urls)]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Return results in original order
        return [results.get(i, HTTPResponse(
            url=urls[i],
            status_code=0,
            headers={},
            is_success=False,
            error="Task failed",
        )) for i in range(len(urls))]

    async def head(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> HTTPResponse:
        """
        Perform a HEAD request (faster than GET for validation).

        Args:
            url: URL to check
            headers: Optional additional headers

        Returns:
            HTTPResponse with results
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        request_headers = self.DEFAULT_HEADERS.copy()
        if headers:
            request_headers.update(headers)

        await self.rate_limiter.acquire(url)

        start_time = asyncio.get_event_loop().time()
        try:
            response = await self._client.head(
                url,
                headers=request_headers,
                follow_redirects=True,
            )
            elapsed = asyncio.get_event_loop().time() - start_time

            return HTTPResponse(
                url=url,
                status_code=response.status_code,
                headers=dict(response.headers),
                content_type=response.headers.get("content-type"),
                content_length=response.headers.get("content-length"),
                final_url=str(response.url),
                response_time=elapsed,
                is_success=200 <= response.status_code < 400,
            )
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            return HTTPResponse(
                url=url,
                status_code=0,
                headers={},
                response_time=elapsed,
                is_success=False,
                error=str(e),
            )


async def validate_urls_batch(
    urls: List[str],
    timeout: float = 30.0,
    max_concurrent: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[HTTPResponse]:
    """
    Convenience function to validate a batch of URLs.

    Args:
        urls: List of URLs to validate
        timeout: Request timeout in seconds
        max_concurrent: Maximum concurrent requests
        progress_callback: Optional progress callback

    Returns:
        List of HTTPResponse objects
    """
    async with AsyncHTTPClient(timeout=timeout) as client:
        return await client.validate_urls(
            urls,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback,
        )


def run_validation(
    urls: List[str],
    timeout: float = 30.0,
    max_concurrent: int = 50,
) -> List[HTTPResponse]:
    """
    Synchronous wrapper for URL validation.

    Args:
        urls: List of URLs to validate
        timeout: Request timeout in seconds
        max_concurrent: Maximum concurrent requests

    Returns:
        List of HTTPResponse objects
    """
    return asyncio.run(validate_urls_batch(urls, timeout, max_concurrent))
