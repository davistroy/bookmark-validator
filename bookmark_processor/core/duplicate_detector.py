"""
Duplicate URL Detection and Removal Module

Detects and handles duplicate URLs in bookmark collections with intelligent
resolution strategies and comprehensive logging.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .data_models import Bookmark


@dataclass
class DuplicateGroup:
    """Group of duplicate URLs with metadata"""

    normalized_url: str
    bookmarks: List[Bookmark] = field(default_factory=list)
    keep_index: Optional[int] = None
    resolution_reason: Optional[str] = None

    def add_bookmark(self, bookmark: Bookmark):
        """Add a bookmark to the duplicate group"""
        self.bookmarks.append(bookmark)

    def get_kept_bookmark(self) -> Optional[Bookmark]:
        """Get the bookmark that will be kept"""
        if self.keep_index is not None and 0 <= self.keep_index < len(self.bookmarks):
            return self.bookmarks[self.keep_index]
        return None

    def get_removed_bookmarks(self) -> List[Bookmark]:
        """Get list of bookmarks that will be removed"""
        if self.keep_index is None:
            return []
        return [b for i, b in enumerate(self.bookmarks) if i != self.keep_index]


@dataclass
class DuplicateDetectionResult:
    """Result of duplicate detection process"""

    total_bookmarks: int
    unique_urls: int
    duplicate_groups: List[DuplicateGroup]
    duplicates_count: int
    removed_count: int
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_bookmarks": self.total_bookmarks,
            "unique_urls": self.unique_urls,
            "duplicate_groups_count": len(self.duplicate_groups),
            "duplicates_count": self.duplicates_count,
            "removed_count": self.removed_count,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        return (
            f"Duplicate Detection Summary:\n"
            f"  Total bookmarks: {self.total_bookmarks}\n"
            f"  Unique URLs: {self.unique_urls}\n"
            f"  Duplicate groups: {len(self.duplicate_groups)}\n"
            f"  Total duplicates: {self.duplicates_count}\n"
            f"  Removed: {self.removed_count}\n"
            f"  Processing time: {self.processing_time:.2f}s"
        )


class DuplicateDetector:
    """Detects and handles duplicate URLs in bookmark collections"""

    def __init__(
        self,
        normalize_www: bool = True,
        normalize_protocol: bool = True,
        normalize_trailing_slash: bool = True,
        normalize_query_params: bool = True,
        case_sensitive: bool = False,
    ):
        """
        Initialize duplicate detector.

        Args:
            normalize_www: Whether to normalize www prefix
            normalize_protocol: Whether to treat http/https as same
            normalize_trailing_slash: Whether to normalize trailing slashes
            normalize_query_params: Whether to sort query parameters
            case_sensitive: Whether URL comparison is case-sensitive
        """
        self.normalize_www = normalize_www
        self.normalize_protocol = normalize_protocol
        self.normalize_trailing_slash = normalize_trailing_slash
        self.normalize_query_params = normalize_query_params
        self.case_sensitive = case_sensitive

        self.logger = logging.getLogger(__name__)

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL string
        """
        if not url:
            return ""

        try:
            # Parse URL
            parsed = urlparse(url.strip())

            # Protocol normalization
            scheme = parsed.scheme.lower()
            if self.normalize_protocol and scheme in ["http", "https"]:
                scheme = "https"  # Standardize to https

            # Domain normalization
            netloc = parsed.netloc
            if not self.case_sensitive:
                netloc = netloc.lower()

            # WWW normalization
            if self.normalize_www:
                if netloc.startswith("www."):
                    netloc = netloc[4:]
                elif "://www." in url and not netloc.startswith("www."):
                    # Handle edge cases
                    pass

            # Path normalization
            path = parsed.path
            if not self.case_sensitive:
                path = path.lower()

            # Trailing slash normalization
            if self.normalize_trailing_slash and path != "/":
                path = path.rstrip("/")

            # Query parameter normalization
            query = parsed.query
            if self.normalize_query_params and query:
                # Parse, sort, and reconstruct query string
                params = parse_qs(query, keep_blank_values=True)
                sorted_params = sorted(params.items())
                query = urlencode(sorted_params, doseq=True)

            # Fragment is ignored for duplicate detection
            fragment = ""

            # Reconstruct normalized URL
            normalized = urlunparse(
                (scheme, netloc, path, parsed.params, query, fragment)
            )

            return normalized

        except Exception as e:
            self.logger.warning(f"Error normalizing URL {url}: {e}")
            return url

    def detect_duplicates(self, bookmarks: List[Bookmark]) -> Dict[str, DuplicateGroup]:
        """
        Detect duplicate URLs in bookmark list.

        Args:
            bookmarks: List of bookmarks to check

        Returns:
            Dictionary mapping normalized URLs to duplicate groups
        """
        url_groups = defaultdict(DuplicateGroup)

        for i, bookmark in enumerate(bookmarks):
            if not bookmark.url:
                continue

            normalized = self.normalize_url(bookmark.url)

            if normalized not in url_groups:
                url_groups[normalized] = DuplicateGroup(normalized_url=normalized)

            url_groups[normalized].add_bookmark(bookmark)

        # Filter out non-duplicates
        duplicate_groups = {
            url: group for url, group in url_groups.items() if len(group.bookmarks) > 1
        }

        return duplicate_groups

    def resolve_duplicates(
        self, duplicate_groups: Dict[str, DuplicateGroup], strategy: str = "newest"
    ) -> Dict[str, DuplicateGroup]:
        """
        Resolve which bookmark to keep from each duplicate group.

        Args:
            duplicate_groups: Dictionary of duplicate groups
            strategy: Resolution strategy ('newest', 'oldest', 'most_complete', 'highest_quality')

        Returns:
            Updated duplicate groups with keep_index set
        """
        for normalized_url, group in duplicate_groups.items():
            if strategy == "newest":
                # Keep the most recently created bookmark
                keep_index = self._resolve_by_newest(group.bookmarks)
                group.resolution_reason = "Kept newest bookmark"

            elif strategy == "oldest":
                # Keep the oldest bookmark
                keep_index = self._resolve_by_oldest(group.bookmarks)
                group.resolution_reason = "Kept oldest bookmark"

            elif strategy == "most_complete":
                # Keep bookmark with most metadata
                keep_index = self._resolve_by_completeness(group.bookmarks)
                group.resolution_reason = "Kept most complete bookmark"

            elif strategy == "highest_quality":
                # Keep bookmark with best quality (combination of factors)
                keep_index = self._resolve_by_quality(group.bookmarks)
                group.resolution_reason = "Kept highest quality bookmark"

            else:
                # Default to newest
                keep_index = self._resolve_by_newest(group.bookmarks)
                group.resolution_reason = "Kept newest bookmark (default)"

            group.keep_index = keep_index

        return duplicate_groups

    def _resolve_by_newest(self, bookmarks: List[Bookmark]) -> int:
        """Keep the most recently created bookmark"""
        newest_index = 0
        newest_date = bookmarks[0].created

        for i, bookmark in enumerate(bookmarks[1:], 1):
            if bookmark.created and bookmark.created > newest_date:
                newest_date = bookmark.created
                newest_index = i

        return newest_index

    def _resolve_by_oldest(self, bookmarks: List[Bookmark]) -> int:
        """Keep the oldest bookmark"""
        oldest_index = 0
        oldest_date = bookmarks[0].created

        for i, bookmark in enumerate(bookmarks[1:], 1):
            if bookmark.created and bookmark.created < oldest_date:
                oldest_date = bookmark.created
                oldest_index = i

        return oldest_index

    def _resolve_by_completeness(self, bookmarks: List[Bookmark]) -> int:
        """Keep bookmark with most complete metadata"""
        scores = []

        for bookmark in bookmarks:
            score = 0
            # Score based on field completeness
            if bookmark.title and len(bookmark.title.strip()) > 0:
                score += 2
            if bookmark.note and len(bookmark.note.strip()) > 0:
                score += 3  # User notes are valuable
            if bookmark.excerpt and len(bookmark.excerpt.strip()) > 0:
                score += 1
            if bookmark.tags and len(bookmark.tags) > 0:
                score += 2
            if bookmark.folder and bookmark.folder != "/":
                score += 1
            if bookmark.created:
                score += 1

            scores.append(score)

        # Return index of highest scoring bookmark
        return scores.index(max(scores))

    def _resolve_by_quality(self, bookmarks: List[Bookmark]) -> int:
        """Keep bookmark with highest overall quality"""
        scores = []

        for bookmark in bookmarks:
            score = 0

            # Completeness factors
            if bookmark.title and len(bookmark.title.strip()) > 0:
                score += 2
            if bookmark.note and len(bookmark.note.strip()) > 0:
                score += 4  # User notes are most valuable
            if bookmark.tags and len(bookmark.tags) > 0:
                score += 2

            # Quality factors
            if bookmark.note and len(bookmark.note) > 50:
                score += 2  # Longer notes indicate more effort
            if bookmark.tags and len(bookmark.tags) > 3:
                score += 1  # Well-tagged
            if bookmark.folder and "/" in bookmark.folder and bookmark.folder != "/":
                score += 1  # Organized in subfolder

            # Recency factor (slight preference for newer)
            if bookmark.created:
                # More recent bookmarks get a small bonus
                try:
                    # Handle timezone-aware vs naive datetime comparison
                    now = datetime.now()
                    if bookmark.created.tzinfo is not None:
                        # If bookmark.created has timezone info, make now timezone-aware
                        from datetime import timezone

                        now = datetime.now(timezone.utc)
                        if bookmark.created.tzinfo != timezone.utc:
                            # Convert to UTC for comparison
                            bookmark_created = bookmark.created.astimezone(timezone.utc)
                        else:
                            bookmark_created = bookmark.created
                    else:
                        # If bookmark.created is naive, use naive now
                        bookmark_created = bookmark.created

                    days_old = (now - bookmark_created).days
                    if days_old < 30:
                        score += 2
                    elif days_old < 90:
                        score += 1
                except (TypeError, AttributeError):
                    # If datetime comparison fails, skip recency factor
                    pass

            scores.append(score)

        # Return index of highest scoring bookmark
        max_score = max(scores)
        # If there's a tie, prefer the newer one
        for i in reversed(range(len(scores))):
            if scores[i] == max_score:
                return i

        return 0

    def process_bookmarks(
        self,
        bookmarks: List[Bookmark],
        strategy: str = "highest_quality",
        dry_run: bool = False,
    ) -> Tuple[List[Bookmark], DuplicateDetectionResult]:
        """
        Process bookmarks to detect and remove duplicates.

        Args:
            bookmarks: List of bookmarks to process
            strategy: Duplicate resolution strategy
            dry_run: If True, don't actually remove duplicates

        Returns:
            Tuple of (deduplicated bookmarks, detection result)
        """
        start_time = datetime.now()

        # Detect duplicates
        duplicate_groups = self.detect_duplicates(bookmarks)

        # Resolve duplicates
        if duplicate_groups:
            duplicate_groups = self.resolve_duplicates(duplicate_groups, strategy)

        # Calculate statistics
        total_bookmarks = len(bookmarks)
        duplicates_count = sum(
            len(group.bookmarks) - 1 for group in duplicate_groups.values()
        )

        # Remove duplicates if not dry run
        if not dry_run and duplicate_groups:
            # Create list of bookmarks to remove (using id() for comparison)
            bookmarks_to_remove_ids = set()
            for group in duplicate_groups.values():
                for removed_bookmark in group.get_removed_bookmarks():
                    bookmarks_to_remove_ids.add(id(removed_bookmark))

            # Filter bookmarks
            deduplicated = [
                b for b in bookmarks if id(b) not in bookmarks_to_remove_ids
            ]
        else:
            deduplicated = bookmarks

        # Create result
        processing_time = (datetime.now() - start_time).total_seconds()

        result = DuplicateDetectionResult(
            total_bookmarks=total_bookmarks,
            unique_urls=total_bookmarks - duplicates_count,
            duplicate_groups=list(duplicate_groups.values()),
            duplicates_count=duplicates_count,
            removed_count=duplicates_count if not dry_run else 0,
            processing_time=processing_time,
        )

        # Log summary
        self.logger.info(result.get_summary())

        return deduplicated, result

    def generate_report(self, result: DuplicateDetectionResult) -> str:
        """
        Generate detailed duplicate detection report.

        Args:
            result: Detection result

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "DUPLICATE URL DETECTION REPORT",
            "=" * 60,
            f"Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Processing time: {result.processing_time:.2f} seconds",
            "",
            "SUMMARY",
            "-" * 20,
            f"Total bookmarks analyzed: {result.total_bookmarks}",
            f"Unique URLs found: {result.unique_urls}",
            f"Duplicate groups found: {len(result.duplicate_groups)}",
            f"Total duplicates: {result.duplicates_count}",
            f"Bookmarks removed: {result.removed_count}",
            "",
        ]

        if result.duplicate_groups:
            report_lines.extend(["DUPLICATE GROUPS", "-" * 20, ""])

            for i, group in enumerate(result.duplicate_groups, 1):
                kept = group.get_kept_bookmark()
                removed = group.get_removed_bookmarks()

                report_lines.extend(
                    [
                        f"Group {i}: {group.normalized_url}",
                        f"  Resolution: {group.resolution_reason}",
                        f"  Duplicates: {len(group.bookmarks)}",
                        "",
                    ]
                )

                if kept:
                    report_lines.extend(
                        [
                            "  KEPT:",
                            f"    Title: {kept.title}",
                            f"    Created: {kept.created}",
                            (
                                f"    Note: {kept.note[:50]}..."
                                if kept.note and len(kept.note) > 50
                                else f"    Note: {kept.note}"
                            ),
                            "",
                        ]
                    )

                if removed:
                    report_lines.append("  REMOVED:")
                    for bookmark in removed:
                        report_lines.extend(
                            [
                                f"    - Title: {bookmark.title}",
                                f"      Created: {bookmark.created}",
                            ]
                        )
                    report_lines.append("")

        report_lines.extend(["=" * 60, "END OF REPORT"])

        return "\n".join(report_lines)
