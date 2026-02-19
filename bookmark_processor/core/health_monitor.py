"""
Bookmark Health Monitoring.

This module provides async health monitoring capabilities for bookmarks,
including URL validation, content change detection, and Wayback Machine
integration for archiving dead links.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .data_models import Bookmark
from .data_sources.state_tracker import ProcessingStateTracker


@dataclass
class HealthCheckResult:
    """
    Result of a single bookmark health check.

    Attributes:
        url: The checked URL
        status: Health status (healthy, redirected, dead, timeout, content_changed)
        http_status: HTTP status code if applicable
        redirect_url: New URL if redirected
        content_changed: Whether the content has changed since last check
        last_checked: When the check was performed
        wayback_url: Wayback Machine archive URL if available
        response_time: Time taken to check the URL in seconds
        error_message: Error message if check failed
    """

    url: str
    status: str  # healthy, redirected, dead, timeout, content_changed, error
    http_status: Optional[int] = None
    redirect_url: Optional[str] = None
    content_changed: bool = False
    last_checked: datetime = field(default_factory=datetime.now)
    wayback_url: Optional[str] = None
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    content_hash: Optional[str] = None

    def __str__(self) -> str:
        return f"HealthCheckResult(url={self.url[:50]}..., status={self.status})"


@dataclass
class HealthReport:
    """
    Comprehensive health report for a set of bookmarks.

    Attributes:
        total: Total number of bookmarks checked
        healthy: Number of healthy bookmarks
        redirected: Number of redirected bookmarks
        dead: Number of dead/broken links
        timeout: Number of timeouts
        content_changed: Number with content changes
        newly_dead: Number newly dead since last check
        recovered: Number recovered from previously dead
        archived: Number archived to Wayback Machine
        results: Individual check results
        checked_at: When the report was generated
    """

    total: int
    healthy: int
    redirected: int
    dead: int
    timeout: int
    content_changed: int
    newly_dead: int
    recovered: int
    archived: int
    results: List[HealthCheckResult]
    checked_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    @property
    def healthy_percentage(self) -> float:
        """Get percentage of healthy bookmarks."""
        if self.total == 0:
            return 0.0
        return (self.healthy / self.total) * 100

    @property
    def problematic(self) -> List[HealthCheckResult]:
        """Get results that need attention (not healthy)."""
        return [r for r in self.results if r.status != "healthy"]

    def __str__(self) -> str:
        return (
            f"HealthReport(total={self.total}, healthy={self.healthy}, "
            f"dead={self.dead}, redirected={self.redirected})"
        )


class HealthMonitorError(Exception):
    """Exception raised by health monitor operations."""
    pass


class WaybackMachineClient:
    """Client for interacting with the Wayback Machine API."""

    AVAILABILITY_API = "https://archive.org/wayback/available"
    SAVE_API = "https://web.archive.org/save/"

    def __init__(self, timeout: float = 30.0):
        """
        Initialize the Wayback Machine client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    async def check_availability(self, url: str) -> Optional[str]:
        """
        Check if a URL is available in the Wayback Machine.

        Args:
            url: URL to check

        Returns:
            Archived URL if available, None otherwise
        """
        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.AVAILABILITY_API,
                    params={"url": url}
                )

                if response.status_code == 200:
                    data = response.json()
                    snapshot = data.get("archived_snapshots", {}).get("closest", {})
                    if snapshot.get("available"):
                        return snapshot.get("url")

                return None

        except Exception as e:
            self.logger.warning(f"Wayback availability check failed for {url}: {e}")
            return None

    async def archive(self, url: str) -> Optional[str]:
        """
        Submit a URL to the Wayback Machine for archiving.

        Args:
            url: URL to archive

        Returns:
            Archived URL if successful, None otherwise
        """
        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.SAVE_API}{url}",
                    follow_redirects=True
                )

                if response.status_code == 200:
                    # The response URL should be the archived version
                    return str(response.url)

                return None

        except Exception as e:
            self.logger.warning(f"Wayback archive failed for {url}: {e}")
            return None


class BookmarkHealthMonitor:
    """
    Monitor bookmark health over time.

    This class provides async health checking for bookmarks, tracking
    status changes, content modifications, and optionally archiving
    dead links to the Wayback Machine.

    Example:
        >>> monitor = BookmarkHealthMonitor(archive_dead=True)
        >>> report = await monitor.check_health(bookmarks, stale_after=timedelta(days=30))
        >>> print(f"Found {report.dead} dead links")
    """

    # User agent for health checks
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        state_tracker: Optional[ProcessingStateTracker] = None,
        archive_dead: bool = False,
        max_concurrent: int = 20,
        timeout: float = 30.0,
        follow_redirects: bool = True,
        max_redirects: int = 5
    ):
        """
        Initialize the health monitor.

        Args:
            state_tracker: Optional state tracker for persistence
            archive_dead: Whether to archive dead links to Wayback Machine
            max_concurrent: Maximum concurrent checks
            timeout: Request timeout in seconds
            follow_redirects: Whether to follow redirects
            max_redirects: Maximum number of redirects to follow
        """
        if not HTTPX_AVAILABLE:
            raise HealthMonitorError(
                "httpx is required for health monitoring. "
                "Install with: pip install httpx"
            )

        self.state_tracker = state_tracker
        self.archive_dead = archive_dead
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self.logger = logging.getLogger(__name__)

        self.wayback = WaybackMachineClient(timeout=timeout) if archive_dead else None

        # Semaphore for concurrent request limiting
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Cache for previous check results
        self._previous_results: Dict[str, HealthCheckResult] = {}

    async def check_health(
        self,
        bookmarks: List[Bookmark],
        stale_after: Optional[timedelta] = None,
        progress_callback: Optional[callable] = None
    ) -> HealthReport:
        """
        Check the health of bookmarks.

        Args:
            bookmarks: List of bookmarks to check
            stale_after: Only check bookmarks not checked within this duration
            progress_callback: Optional callback for progress updates

        Returns:
            HealthReport with comprehensive results
        """
        start_time = datetime.now()

        if not bookmarks:
            return HealthReport(
                total=0, healthy=0, redirected=0, dead=0, timeout=0,
                content_changed=0, newly_dead=0, recovered=0, archived=0,
                results=[]
            )

        # Filter bookmarks if stale_after is specified
        if stale_after:
            bookmarks = self._filter_stale(bookmarks, stale_after)

        if not bookmarks:
            self.logger.info("No stale bookmarks to check")
            return HealthReport(
                total=0, healthy=0, redirected=0, dead=0, timeout=0,
                content_changed=0, newly_dead=0, recovered=0, archived=0,
                results=[]
            )

        self.logger.info(f"Checking health of {len(bookmarks)} bookmarks")

        # Initialize semaphore
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Load previous results for comparison
        self._load_previous_results()

        # Check all bookmarks
        results = await self._check_all(bookmarks, progress_callback)

        # Compile report
        report = self._compile_report(results, start_time)

        # Archive dead links if enabled
        if self.archive_dead:
            await self._archive_dead_links(report)

        return report

    def _filter_stale(
        self,
        bookmarks: List[Bookmark],
        stale_after: timedelta
    ) -> List[Bookmark]:
        """Filter to only bookmarks that need checking."""
        if not self.state_tracker:
            return bookmarks

        cutoff = datetime.now() - stale_after
        stale_bookmarks = []

        for bookmark in bookmarks:
            info = self.state_tracker.get_processed_info(bookmark.url)
            if info:
                processed_at = datetime.fromisoformat(info.get("processed_at", ""))
                if processed_at < cutoff:
                    stale_bookmarks.append(bookmark)
            else:
                stale_bookmarks.append(bookmark)

        self.logger.info(
            f"Found {len(stale_bookmarks)} stale bookmarks out of {len(bookmarks)}"
        )
        return stale_bookmarks

    def _load_previous_results(self) -> None:
        """Load previous check results from state tracker."""
        if not self.state_tracker:
            return

        # This would typically load from a health check history table
        # For now, we'll rely on the existing processed_bookmarks table
        pass

    async def _check_all(
        self,
        bookmarks: List[Bookmark],
        progress_callback: Optional[callable] = None
    ) -> List[HealthCheckResult]:
        """Check all bookmarks concurrently."""
        tasks = [
            self._check_with_semaphore(bookmark, i, len(bookmarks), progress_callback)
            for i, bookmark in enumerate(bookmarks)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Check failed: {result}")
                processed_results.append(HealthCheckResult(
                    url=bookmarks[i].url,
                    status="error",
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _check_with_semaphore(
        self,
        bookmark: Bookmark,
        index: int,
        total: int,
        progress_callback: Optional[callable] = None
    ) -> HealthCheckResult:
        """Check a single bookmark with semaphore limiting."""
        async with self._semaphore:
            result = await self._check_single(bookmark)

            if progress_callback:
                try:
                    progress_callback(index + 1, total, result)
                except Exception as e:
                    self.logger.warning(f"Progress callback error: {e}")

            return result

    async def _check_single(self, bookmark: Bookmark) -> HealthCheckResult:
        """
        Check the health of a single bookmark.

        Args:
            bookmark: The bookmark to check

        Returns:
            HealthCheckResult with check details
        """
        url = bookmark.url
        start_time = datetime.now()

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=self.follow_redirects,
                max_redirects=self.max_redirects,
                headers={"User-Agent": self.USER_AGENT}
            ) as client:
                # Use HEAD request first for efficiency
                try:
                    response = await client.head(url)
                except httpx.HTTPStatusError:
                    # Fall back to GET if HEAD fails
                    response = await client.get(url)

                response_time = (datetime.now() - start_time).total_seconds()

                # Check for redirects
                redirect_url = None
                if response.history:
                    redirect_url = str(response.url)
                    if redirect_url != url:
                        return HealthCheckResult(
                            url=url,
                            status="redirected",
                            http_status=response.status_code,
                            redirect_url=redirect_url,
                            last_checked=datetime.now(),
                            response_time=response_time
                        )

                # Check status
                if response.status_code >= 200 and response.status_code < 400:
                    # Check for content changes
                    content_changed = await self._check_content_changed(
                        bookmark, client
                    )

                    return HealthCheckResult(
                        url=url,
                        status="content_changed" if content_changed else "healthy",
                        http_status=response.status_code,
                        content_changed=content_changed,
                        last_checked=datetime.now(),
                        response_time=response_time
                    )
                else:
                    return HealthCheckResult(
                        url=url,
                        status="dead",
                        http_status=response.status_code,
                        last_checked=datetime.now(),
                        response_time=response_time,
                        error_message=f"HTTP {response.status_code}"
                    )

        except httpx.TimeoutException:
            return HealthCheckResult(
                url=url,
                status="timeout",
                last_checked=datetime.now(),
                error_message="Request timed out"
            )
        except httpx.TooManyRedirects:
            return HealthCheckResult(
                url=url,
                status="dead",
                last_checked=datetime.now(),
                error_message="Too many redirects"
            )
        except httpx.ConnectError as e:
            return HealthCheckResult(
                url=url,
                status="dead",
                last_checked=datetime.now(),
                error_message=f"Connection error: {e}"
            )
        except Exception as e:
            return HealthCheckResult(
                url=url,
                status="error",
                last_checked=datetime.now(),
                error_message=str(e)
            )

    async def _check_content_changed(
        self,
        bookmark: Bookmark,
        client: httpx.AsyncClient
    ) -> bool:
        """Check if the content of a page has changed."""
        if not self.state_tracker:
            return False

        # Get stored hash
        info = self.state_tracker.get_processed_info(bookmark.url)
        if not info or not info.get("content_hash"):
            return False

        stored_hash = info.get("content_hash")

        try:
            # Fetch content for hash comparison
            response = await client.get(bookmark.url)
            content = response.text[:10000]  # Only hash first 10KB

            current_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

            return current_hash != stored_hash

        except Exception as e:
            self.logger.warning(f"Content check failed for {bookmark.url}: {e}")
            return False

    def _compile_report(
        self,
        results: List[HealthCheckResult],
        start_time: datetime
    ) -> HealthReport:
        """Compile results into a health report."""
        # Count statuses
        status_counts = {
            "healthy": 0,
            "redirected": 0,
            "dead": 0,
            "timeout": 0,
            "content_changed": 0,
            "error": 0
        }

        for result in results:
            status = result.status
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts["error"] += 1

        # Determine newly dead and recovered
        newly_dead = 0
        recovered = 0

        for result in results:
            previous = self._previous_results.get(result.url)
            if previous:
                if result.status == "dead" and previous.status != "dead":
                    newly_dead += 1
                elif result.status == "healthy" and previous.status == "dead":
                    recovered += 1

        duration = (datetime.now() - start_time).total_seconds()

        return HealthReport(
            total=len(results),
            healthy=status_counts["healthy"],
            redirected=status_counts["redirected"],
            dead=status_counts["dead"] + status_counts["error"],
            timeout=status_counts["timeout"],
            content_changed=status_counts["content_changed"],
            newly_dead=newly_dead,
            recovered=recovered,
            archived=0,  # Updated after archiving
            results=results,
            duration_seconds=duration
        )

    async def _archive_dead_links(self, report: HealthReport) -> None:
        """Archive dead links to the Wayback Machine."""
        if not self.wayback:
            return

        dead_results = [r for r in report.results if r.status in ("dead", "error")]
        archived_count = 0

        for result in dead_results:
            # First check if already archived
            archived_url = await self.wayback.check_availability(result.url)

            if archived_url:
                result.wayback_url = archived_url
                archived_count += 1
            else:
                # Try to archive
                archived_url = await self.wayback.archive(result.url)
                if archived_url:
                    result.wayback_url = archived_url
                    archived_count += 1

            # Rate limit
            await asyncio.sleep(1)

        report.archived = archived_count
        self.logger.info(f"Archived {archived_count} dead links")

    async def check_single_url(self, url: str) -> HealthCheckResult:
        """
        Check the health of a single URL.

        Args:
            url: URL to check

        Returns:
            HealthCheckResult with check details
        """
        bookmark = Bookmark(url=url)
        return await self._check_single(bookmark)

    def generate_report_text(self, report: HealthReport) -> str:
        """
        Generate a human-readable text report.

        Args:
            report: The health report

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BOOKMARK HEALTH REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Checked at: {report.checked_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration: {report.duration_seconds:.1f} seconds")
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total checked:    {report.total}")
        lines.append(f"Healthy:          {report.healthy} ({report.healthy_percentage:.1f}%)")
        lines.append(f"Redirected:       {report.redirected}")
        lines.append(f"Dead/Broken:      {report.dead}")
        lines.append(f"Timeouts:         {report.timeout}")
        lines.append(f"Content changed:  {report.content_changed}")
        lines.append(f"Newly dead:       {report.newly_dead}")
        lines.append(f"Recovered:        {report.recovered}")
        if self.archive_dead:
            lines.append(f"Archived:         {report.archived}")
        lines.append("")

        # List problematic URLs
        problematic = report.problematic
        if problematic:
            lines.append("PROBLEMATIC URLS")
            lines.append("-" * 40)
            for result in problematic[:20]:  # Limit to 20
                status_emoji = {
                    "dead": "[DEAD]",
                    "timeout": "[TIMEOUT]",
                    "redirected": "[REDIRECT]",
                    "content_changed": "[CHANGED]",
                    "error": "[ERROR]"
                }.get(result.status, "[?]")

                lines.append(f"{status_emoji} {result.url[:60]}")
                if result.redirect_url:
                    lines.append(f"  -> {result.redirect_url[:60]}")
                if result.wayback_url:
                    lines.append(f"  [archived] {result.wayback_url[:60]}")

            if len(problematic) > 20:
                lines.append(f"  ... and {len(problematic) - 20} more")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_report(
        self,
        report: HealthReport,
        output_path: Path,
        format: str = "text"
    ) -> None:
        """
        Save the health report to a file.

        Args:
            report: The health report
            output_path: Path to save the report
            format: Output format (text, json, csv)
        """
        import json
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "text":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self.generate_report_text(report))

        elif format == "json":
            data = {
                "summary": {
                    "total": report.total,
                    "healthy": report.healthy,
                    "redirected": report.redirected,
                    "dead": report.dead,
                    "timeout": report.timeout,
                    "content_changed": report.content_changed,
                    "checked_at": report.checked_at.isoformat(),
                    "duration_seconds": report.duration_seconds
                },
                "results": [
                    {
                        "url": r.url,
                        "status": r.status,
                        "http_status": r.http_status,
                        "redirect_url": r.redirect_url,
                        "wayback_url": r.wayback_url,
                        "response_time": r.response_time,
                        "error_message": r.error_message
                    }
                    for r in report.results
                ]
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "URL", "Status", "HTTP Status", "Redirect URL",
                    "Wayback URL", "Response Time", "Error"
                ])
                for r in report.results:
                    writer.writerow([
                        r.url, r.status, r.http_status or "",
                        r.redirect_url or "", r.wayback_url or "",
                        r.response_time or "", r.error_message or ""
                    ])

        self.logger.info(f"Saved health report to {output_path}")
