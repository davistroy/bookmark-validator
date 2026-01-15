"""
Enhanced Async Pipeline.

Provides fully asynchronous pipeline execution for improved performance
on network-bound operations like URL validation and content fetching.
"""

import asyncio
import logging
from asyncio import Semaphore
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None
    ClientSession = None
    ClientTimeout = None
    TCPConnector = None

from .data_models import Bookmark
from .pipeline.config import PipelineConfig


@dataclass
class ValidationResult:
    """Result of URL validation."""
    url: str
    is_valid: bool
    status_code: Optional[int] = None
    final_url: Optional[str] = None
    response_time: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class ContentData:
    """Content data from a URL."""
    url: str
    content: str = ""
    title: Optional[str] = None
    description: Optional[str] = None
    content_type: Optional[str] = None
    fetch_time: float = 0.0
    error: Optional[str] = None


@dataclass
class AIProcessingResult:
    """Result of AI processing."""
    url: str
    enhanced_description: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    method: str = "none"
    error: Optional[str] = None


@dataclass
class AsyncPipelineStats:
    """Statistics for async pipeline execution."""
    total_urls: int = 0
    validation_success: int = 0
    validation_failed: int = 0
    content_fetched: int = 0
    content_failed: int = 0
    ai_processed: int = 0
    ai_failed: int = 0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Timing breakdowns
    validation_time: float = 0.0
    content_time: float = 0.0
    ai_time: float = 0.0

    @property
    def total_time(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def throughput(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_urls / self.total_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_urls": self.total_urls,
            "validation_success": self.validation_success,
            "validation_failed": self.validation_failed,
            "content_fetched": self.content_fetched,
            "content_failed": self.content_failed,
            "ai_processed": self.ai_processed,
            "ai_failed": self.ai_failed,
            "total_time": self.total_time,
            "throughput": self.throughput,
            "validation_time": self.validation_time,
            "content_time": self.content_time,
            "ai_time": self.ai_time,
        }


class AsyncPipelineExecutor:
    """
    Fully async execution for network-bound operations.

    This executor provides parallel processing of URL validation,
    content fetching, and cloud AI API calls for improved throughput.

    Features:
    - Concurrent URL validation with configurable limits
    - Parallel content fetching with rate limiting
    - Async cloud AI API processing
    - Per-domain rate limiting to avoid blocks
    - Automatic retry with exponential backoff

    Example:
        >>> executor = AsyncPipelineExecutor(config, max_concurrent=20)
        >>> results = await executor.validate_urls_async(bookmarks)
        >>> content = await executor.fetch_content_async(valid_urls)
    """

    # Default rate limits per domain (requests per second)
    DEFAULT_DOMAIN_LIMITS = {
        "github.com": 0.5,      # 1 request per 2 seconds
        "google.com": 0.5,
        "youtube.com": 0.5,
        "linkedin.com": 0.5,
        "twitter.com": 0.5,
        "x.com": 0.5,
        "default": 2.0,        # 2 requests per second default
    }

    def __init__(
        self,
        config: PipelineConfig,
        max_concurrent: int = 20,
        timeout: float = 30.0,
        domain_limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the async pipeline executor.

        Args:
            config: Pipeline configuration
            max_concurrent: Maximum concurrent requests (default 20)
            timeout: Request timeout in seconds (default 30)
            domain_limits: Per-domain rate limits (requests per second)
        """
        if not HAS_AIOHTTP:
            raise ImportError(
                "aiohttp is required for AsyncPipelineExecutor. "
                "Install with: pip install aiohttp"
            )

        self.config = config
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.domain_limits = domain_limits or self.DEFAULT_DOMAIN_LIMITS
        self.logger = logging.getLogger(__name__)

        # Semaphore for global concurrency control
        self._semaphore: Optional[Semaphore] = None

        # Per-domain tracking for rate limiting
        self._domain_last_request: Dict[str, datetime] = {}
        self._domain_locks: Dict[str, asyncio.Lock] = {}

        # Statistics
        self.stats = AsyncPipelineStats()

        # Session management
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "AsyncPipelineExecutor":
        """Async context manager entry."""
        await self._init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self._close_session()

    async def _init_session(self) -> None:
        """Initialize the aiohttp session."""
        if self._session is None:
            connector = TCPConnector(
                limit=self.max_concurrent,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            timeout_config = ClientTimeout(total=self.timeout)
            self._session = ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                }
            )
            self._semaphore = Semaphore(self.max_concurrent)

    async def _close_session(self) -> None:
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return domain if domain else "default"
        except Exception:
            return "default"

    async def _wait_for_rate_limit(self, domain: str) -> None:
        """
        Wait if needed to respect rate limits for a domain.

        Args:
            domain: Domain to check rate limit for
        """
        # Get or create lock for domain
        if domain not in self._domain_locks:
            self._domain_locks[domain] = asyncio.Lock()

        async with self._domain_locks[domain]:
            rate_limit = self.domain_limits.get(
                domain,
                self.domain_limits.get("default", 2.0)
            )
            min_interval = 1.0 / rate_limit

            last_request = self._domain_last_request.get(domain)
            if last_request:
                elapsed = (datetime.now() - last_request).total_seconds()
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)

            self._domain_last_request[domain] = datetime.now()

    async def _validate_single_url(
        self,
        url: str,
        retry_count: int = 3
    ) -> ValidationResult:
        """
        Validate a single URL with retry logic.

        Args:
            url: URL to validate
            retry_count: Number of retries

        Returns:
            ValidationResult
        """
        domain = self._get_domain(url)
        start_time = datetime.now()

        for attempt in range(retry_count):
            try:
                await self._wait_for_rate_limit(domain)

                async with self._semaphore:
                    async with self._session.head(
                        url,
                        allow_redirects=True,
                        ssl=self.config.verify_ssl
                    ) as response:
                        response_time = (datetime.now() - start_time).total_seconds()

                        return ValidationResult(
                            url=url,
                            is_valid=response.status < 400,
                            status_code=response.status,
                            final_url=str(response.url),
                            response_time=response_time
                        )

            except asyncio.TimeoutError:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error_message="Request timed out",
                    error_type="timeout",
                    response_time=(datetime.now() - start_time).total_seconds()
                )

            except aiohttp.ClientError as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error_message=str(e),
                    error_type="client_error",
                    response_time=(datetime.now() - start_time).total_seconds()
                )

            except Exception as e:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error_message=str(e),
                    error_type="unknown",
                    response_time=(datetime.now() - start_time).total_seconds()
                )

        return ValidationResult(
            url=url,
            is_valid=False,
            error_message="All retries failed",
            error_type="retry_exhausted"
        )

    async def validate_urls_async(
        self,
        bookmarks: List[Bookmark],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Validate URLs concurrently.

        Args:
            bookmarks: List of bookmarks to validate
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping URL to ValidationResult
        """
        if not bookmarks:
            return {}

        await self._init_session()
        self.stats.total_urls = len(bookmarks)
        self.stats.start_time = datetime.now()

        urls = [b.url for b in bookmarks if b.url]

        # Create tasks for all URLs
        tasks = [
            self._validate_single_url(url)
            for url in urls
        ]

        # Execute with progress tracking
        results: Dict[str, ValidationResult] = {}
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results[result.url] = result

            if result.is_valid:
                self.stats.validation_success += 1
            else:
                self.stats.validation_failed += 1

            completed += 1
            if progress_callback and completed % 10 == 0:
                progress_callback(completed, len(urls))

        self.stats.validation_time = (
            datetime.now() - self.stats.start_time
        ).total_seconds()

        self.logger.info(
            f"Validated {len(urls)} URLs: "
            f"{self.stats.validation_success} valid, "
            f"{self.stats.validation_failed} invalid "
            f"in {self.stats.validation_time:.2f}s"
        )

        return results

    async def _fetch_single_content(
        self,
        url: str,
        max_length: int = 100000
    ) -> ContentData:
        """
        Fetch content from a single URL.

        Args:
            url: URL to fetch
            max_length: Maximum content length to fetch

        Returns:
            ContentData
        """
        domain = self._get_domain(url)
        start_time = datetime.now()

        try:
            await self._wait_for_rate_limit(domain)

            async with self._semaphore:
                async with self._session.get(
                    url,
                    allow_redirects=True,
                    ssl=self.config.verify_ssl
                ) as response:
                    if response.status >= 400:
                        return ContentData(
                            url=url,
                            error=f"HTTP {response.status}"
                        )

                    content_type = response.headers.get("Content-Type", "")

                    # Only fetch text content
                    if "text" not in content_type and "html" not in content_type:
                        return ContentData(
                            url=url,
                            content_type=content_type,
                            error="Non-text content"
                        )

                    # Read content with limit
                    content = await response.text()
                    if len(content) > max_length:
                        content = content[:max_length]

                    fetch_time = (datetime.now() - start_time).total_seconds()

                    # Extract title from HTML
                    title = self._extract_title(content)
                    description = self._extract_description(content)

                    return ContentData(
                        url=url,
                        content=content,
                        title=title,
                        description=description,
                        content_type=content_type,
                        fetch_time=fetch_time
                    )

        except asyncio.TimeoutError:
            return ContentData(url=url, error="Request timed out")
        except Exception as e:
            return ContentData(url=url, error=str(e))

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML content."""
        try:
            import re
            match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
        return None

    def _extract_description(self, html: str) -> Optional[str]:
        """Extract meta description from HTML content."""
        try:
            import re
            # Try meta description
            match = re.search(
                r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
                html,
                re.IGNORECASE
            )
            if match:
                return match.group(1).strip()

            # Try og:description
            match = re.search(
                r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
                html,
                re.IGNORECASE
            )
            if match:
                return match.group(1).strip()

        except Exception:
            pass
        return None

    async def fetch_content_async(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, ContentData]:
        """
        Fetch content from URLs concurrently.

        Args:
            urls: List of URLs to fetch
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping URL to ContentData
        """
        if not urls:
            return {}

        await self._init_session()
        start_time = datetime.now()

        # Create tasks
        tasks = [self._fetch_single_content(url) for url in urls]

        # Execute with progress tracking
        results: Dict[str, ContentData] = {}
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results[result.url] = result

            if result.error:
                self.stats.content_failed += 1
            else:
                self.stats.content_fetched += 1

            completed += 1
            if progress_callback and completed % 10 == 0:
                progress_callback(completed, len(urls))

        self.stats.content_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(
            f"Fetched {len(urls)} URLs: "
            f"{self.stats.content_fetched} success, "
            f"{self.stats.content_failed} failed "
            f"in {self.stats.content_time:.2f}s"
        )

        return results

    async def _process_ai_single(
        self,
        bookmark: Bookmark,
        content: Optional[ContentData],
        api_client: Any
    ) -> AIProcessingResult:
        """
        Process AI description for a single bookmark.

        Args:
            bookmark: Bookmark to process
            content: Optional content data
            api_client: AI API client

        Returns:
            AIProcessingResult
        """
        start_time = datetime.now()

        try:
            # Build prompt from available data
            prompt_content = ""
            if content and content.content:
                prompt_content = content.content[:2000]
            elif bookmark.excerpt:
                prompt_content = bookmark.excerpt
            elif bookmark.note:
                prompt_content = bookmark.note

            if not prompt_content:
                return AIProcessingResult(
                    url=bookmark.url,
                    error="No content available for AI processing"
                )

            # Call AI API (assuming api_client has async method)
            if hasattr(api_client, "generate_description_async"):
                result = await api_client.generate_description_async(
                    title=bookmark.title or "",
                    content=prompt_content
                )
            else:
                # Fall back to sync if no async method
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: api_client.generate_description(
                        title=bookmark.title or "",
                        content=prompt_content
                    )
                )

            processing_time = (datetime.now() - start_time).total_seconds()

            return AIProcessingResult(
                url=bookmark.url,
                enhanced_description=result.get("description", ""),
                confidence=result.get("confidence", 0.5),
                processing_time=processing_time,
                method="cloud"
            )

        except Exception as e:
            return AIProcessingResult(
                url=bookmark.url,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    async def process_ai_async(
        self,
        bookmarks: List[Bookmark],
        contents: Dict[str, ContentData],
        api_client: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, AIProcessingResult]:
        """
        Process AI descriptions concurrently (for cloud APIs).

        For local AI models that don't parallelize well, this falls
        back to sequential processing.

        Args:
            bookmarks: List of bookmarks to process
            contents: Dictionary of URL to ContentData
            api_client: AI API client (optional, uses config if not provided)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping URL to AIProcessingResult
        """
        if not bookmarks:
            return {}

        start_time = datetime.now()

        # If no API client or local engine, process sequentially
        if api_client is None or self.config.ai_engine == "local":
            return await self._process_ai_sequential(
                bookmarks, contents, progress_callback
            )

        # Create tasks for cloud API calls
        tasks = [
            self._process_ai_single(
                bookmark,
                contents.get(bookmark.url),
                api_client
            )
            for bookmark in bookmarks
        ]

        # Execute with concurrency limit
        ai_semaphore = Semaphore(5)  # Limit concurrent AI calls

        async def limited_task(task):
            async with ai_semaphore:
                return await task

        limited_tasks = [limited_task(task) for task in tasks]

        # Execute with progress tracking
        results: Dict[str, AIProcessingResult] = {}
        completed = 0

        for coro in asyncio.as_completed(limited_tasks):
            result = await coro
            results[result.url] = result

            if result.error:
                self.stats.ai_failed += 1
            else:
                self.stats.ai_processed += 1

            completed += 1
            if progress_callback and completed % 5 == 0:
                progress_callback(completed, len(bookmarks))

        self.stats.ai_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(
            f"AI processed {len(bookmarks)} bookmarks: "
            f"{self.stats.ai_processed} success, "
            f"{self.stats.ai_failed} failed "
            f"in {self.stats.ai_time:.2f}s"
        )

        return results

    async def _process_ai_sequential(
        self,
        bookmarks: List[Bookmark],
        contents: Dict[str, ContentData],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, AIProcessingResult]:
        """
        Process AI sequentially for local models.

        Args:
            bookmarks: List of bookmarks
            contents: Content data dictionary
            progress_callback: Progress callback

        Returns:
            Dictionary of results
        """
        from .ai_processor import EnhancedAIProcessor

        results: Dict[str, AIProcessingResult] = {}

        try:
            processor = EnhancedAIProcessor(
                max_description_length=self.config.max_description_length
            )

            for i, bookmark in enumerate(bookmarks):
                start_time = datetime.now()

                try:
                    content = contents.get(bookmark.url)
                    content_text = content.content if content else ""

                    # Use sync processor in executor
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda b=bookmark, c=content_text: processor.process_single(b, content=c)
                    )

                    if result:
                        results[bookmark.url] = AIProcessingResult(
                            url=bookmark.url,
                            enhanced_description=result.enhanced_description,
                            processing_time=(datetime.now() - start_time).total_seconds(),
                            method="local"
                        )
                        self.stats.ai_processed += 1
                    else:
                        results[bookmark.url] = AIProcessingResult(
                            url=bookmark.url,
                            error="Processing returned no result"
                        )
                        self.stats.ai_failed += 1

                except Exception as e:
                    results[bookmark.url] = AIProcessingResult(
                        url=bookmark.url,
                        error=str(e)
                    )
                    self.stats.ai_failed += 1

                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, len(bookmarks))

        except Exception as e:
            self.logger.error(f"AI processing error: {e}")

        return results

    async def execute_full_pipeline(
        self,
        bookmarks: List[Bookmark],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Tuple[Dict[str, ValidationResult], Dict[str, ContentData], Dict[str, AIProcessingResult]]:
        """
        Execute full async pipeline: validation -> content -> AI.

        Args:
            bookmarks: List of bookmarks to process
            progress_callback: Optional callback (stage, current, total)

        Returns:
            Tuple of (validation_results, content_data, ai_results)
        """
        async with self:
            self.stats = AsyncPipelineStats()
            self.stats.start_time = datetime.now()
            self.stats.total_urls = len(bookmarks)

            # Stage 1: Validate URLs
            if progress_callback:
                progress_callback("Validating URLs", 0, len(bookmarks))

            validation_results = await self.validate_urls_async(
                bookmarks,
                progress_callback=lambda c, t: progress_callback("Validating URLs", c, t)
                if progress_callback else None
            )

            # Get valid URLs for content fetching
            valid_urls = [
                url for url, result in validation_results.items()
                if result.is_valid
            ]

            # Stage 2: Fetch Content
            if progress_callback:
                progress_callback("Fetching Content", 0, len(valid_urls))

            content_data = await self.fetch_content_async(
                valid_urls,
                progress_callback=lambda c, t: progress_callback("Fetching Content", c, t)
                if progress_callback else None
            )

            # Stage 3: AI Processing (if enabled)
            ai_results: Dict[str, AIProcessingResult] = {}
            if self.config.ai_enabled:
                valid_bookmarks = [
                    b for b in bookmarks
                    if b.url in validation_results
                    and validation_results[b.url].is_valid
                ]

                if progress_callback:
                    progress_callback("AI Processing", 0, len(valid_bookmarks))

                ai_results = await self.process_ai_async(
                    valid_bookmarks,
                    content_data,
                    progress_callback=lambda c, t: progress_callback("AI Processing", c, t)
                    if progress_callback else None
                )

            self.stats.end_time = datetime.now()

            self.logger.info(
                f"Full pipeline complete in {self.stats.total_time:.2f}s: "
                f"{self.stats.validation_success} validated, "
                f"{self.stats.content_fetched} fetched, "
                f"{self.stats.ai_processed} AI processed"
            )

            return validation_results, content_data, ai_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return self.stats.to_dict()

    async def close(self) -> None:
        """Clean up resources."""
        await self._close_session()


def run_async_pipeline(
    bookmarks: List[Bookmark],
    config: PipelineConfig,
    max_concurrent: int = 20
) -> Tuple[Dict[str, ValidationResult], Dict[str, ContentData], Dict[str, AIProcessingResult]]:
    """
    Convenience function to run async pipeline synchronously.

    Args:
        bookmarks: List of bookmarks to process
        config: Pipeline configuration
        max_concurrent: Maximum concurrent requests

    Returns:
        Tuple of (validation_results, content_data, ai_results)
    """
    executor = AsyncPipelineExecutor(config, max_concurrent=max_concurrent)
    return asyncio.run(executor.execute_full_pipeline(bookmarks))
