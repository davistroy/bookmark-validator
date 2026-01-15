"""
Filter Infrastructure for Bookmark Processing.

This module provides a composable filtering system for selecting
subsets of bookmarks based on various criteria including folder,
tags, date range, domain, and processing status.
"""

import fnmatch
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from urllib.parse import urlparse

from .data_models import Bookmark


class BookmarkFilter(ABC):
    """
    Abstract base class for bookmark filters.

    Filters can be combined using & (AND) and | (OR) operators
    to create complex filter chains.
    """

    @abstractmethod
    def matches(self, bookmark: Bookmark) -> bool:
        """
        Check if a bookmark matches this filter.

        Args:
            bookmark: The bookmark to check

        Returns:
            True if the bookmark matches the filter criteria
        """
        pass

    def __and__(self, other: "BookmarkFilter") -> "CompositeFilter":
        """
        Combine filters with AND logic.

        Args:
            other: Another filter to combine with

        Returns:
            CompositeFilter with AND logic
        """
        if isinstance(other, CompositeFilter) and other.operator == "and":
            # Flatten nested AND filters
            return CompositeFilter([self] + other.filters, operator="and")
        return CompositeFilter([self, other], operator="and")

    def __or__(self, other: "BookmarkFilter") -> "CompositeFilter":
        """
        Combine filters with OR logic.

        Args:
            other: Another filter to combine with

        Returns:
            CompositeFilter with OR logic
        """
        if isinstance(other, CompositeFilter) and other.operator == "or":
            # Flatten nested OR filters
            return CompositeFilter([self] + other.filters, operator="or")
        return CompositeFilter([self, other], operator="or")

    def __invert__(self) -> "NotFilter":
        """
        Negate this filter.

        Returns:
            NotFilter that inverts this filter's logic
        """
        return NotFilter(self)

    def filter(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """
        Filter a list of bookmarks.

        Args:
            bookmarks: List of bookmarks to filter

        Returns:
            List of bookmarks that match the filter
        """
        return [b for b in bookmarks if self.matches(b)]


class CompositeFilter(BookmarkFilter):
    """
    Composite filter that combines multiple filters.

    Supports AND and OR operations between child filters.
    """

    def __init__(
        self,
        filters: List[BookmarkFilter],
        operator: str = "and",
    ):
        """
        Initialize composite filter.

        Args:
            filters: List of filters to combine
            operator: "and" or "or"
        """
        self.filters = filters
        self.operator = operator.lower()

        if self.operator not in ("and", "or"):
            raise ValueError(f"Invalid operator: {operator}. Must be 'and' or 'or'.")

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark matches the composite filter."""
        if not self.filters:
            return True

        if self.operator == "and":
            return all(f.matches(bookmark) for f in self.filters)
        else:  # or
            return any(f.matches(bookmark) for f in self.filters)


class NotFilter(BookmarkFilter):
    """Filter that negates another filter."""

    def __init__(self, filter_to_negate: BookmarkFilter):
        """
        Initialize NOT filter.

        Args:
            filter_to_negate: The filter to negate
        """
        self.inner_filter = filter_to_negate

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark does NOT match the inner filter."""
        return not self.inner_filter.matches(bookmark)


class FolderFilter(BookmarkFilter):
    """
    Filter bookmarks by folder pattern.

    Supports glob-style patterns for matching folder paths.
    """

    def __init__(self, pattern: str, case_sensitive: bool = False):
        """
        Initialize folder filter.

        Args:
            pattern: Glob pattern to match folders (e.g., "Tech/*", "*/Python/*")
            case_sensitive: Whether matching should be case-sensitive
        """
        self.pattern = pattern
        self.case_sensitive = case_sensitive

        # Pre-compile regex for better performance
        regex_pattern = self._glob_to_regex(pattern)
        flags = 0 if case_sensitive else re.IGNORECASE
        self._regex = re.compile(regex_pattern, flags)

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob pattern to regex."""
        # Escape special regex characters except * and ?
        escaped = ""
        for char in pattern:
            if char == "*":
                escaped += ".*"
            elif char == "?":
                escaped += "."
            elif char in r"\.[]{}()+^$|":
                escaped += "\\" + char
            else:
                escaped += char
        return f"^{escaped}$"

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark's folder matches the pattern."""
        folder = bookmark.folder or ""
        return bool(self._regex.match(folder))


class TagFilter(BookmarkFilter):
    """
    Filter bookmarks by tag presence.

    Supports matching any or all specified tags.
    """

    def __init__(
        self,
        tags: Union[str, List[str]],
        mode: str = "any",
        case_sensitive: bool = False,
    ):
        """
        Initialize tag filter.

        Args:
            tags: Tag(s) to filter by
            mode: "any" (match any tag) or "all" (match all tags)
            case_sensitive: Whether matching should be case-sensitive
        """
        if isinstance(tags, str):
            tags = [tags]
        self.tags = tags
        self.mode = mode.lower()
        self.case_sensitive = case_sensitive

        if self.mode not in ("any", "all"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'any' or 'all'.")

        # Normalize tags for comparison
        if not case_sensitive:
            self._normalized_tags = {t.lower() for t in self.tags}
        else:
            self._normalized_tags = set(self.tags)

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark has the specified tags."""
        bookmark_tags = bookmark.tags or []

        if not self.case_sensitive:
            bookmark_tags_set = {t.lower() for t in bookmark_tags}
        else:
            bookmark_tags_set = set(bookmark_tags)

        if self.mode == "any":
            return bool(bookmark_tags_set & self._normalized_tags)
        else:  # all
            return self._normalized_tags.issubset(bookmark_tags_set)


class DateRangeFilter(BookmarkFilter):
    """
    Filter bookmarks by creation date range.

    Supports filtering by start date, end date, or both.
    """

    def __init__(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ):
        """
        Initialize date range filter.

        Args:
            start: Start date (inclusive). None for no lower bound.
            end: End date (inclusive). None for no upper bound.
        """
        self.start = start
        self.end = end

        if start is None and end is None:
            raise ValueError("At least one of start or end must be specified.")

        if start and end and start > end:
            raise ValueError("Start date must be before or equal to end date.")

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark's creation date is within the range."""
        created = bookmark.created

        if created is None:
            return False

        if self.start and created < self.start:
            return False

        if self.end and created > self.end:
            return False

        return True

    @classmethod
    def from_string(cls, date_range: str) -> "DateRangeFilter":
        """
        Create a DateRangeFilter from a string specification.

        Args:
            date_range: String in format "start:end" where either can be empty.
                       Dates should be ISO format (YYYY-MM-DD).

        Returns:
            DateRangeFilter instance

        Examples:
            "2024-01-01:2024-12-31" - Full year 2024
            "2024-01-01:" - From 2024-01-01 onwards
            ":2024-12-31" - Up to 2024-12-31
        """
        parts = date_range.split(":", 1)

        if len(parts) != 2:
            raise ValueError(
                f"Invalid date range format: {date_range}. "
                "Expected format: 'start:end'"
            )

        start_str, end_str = parts
        start = None
        end = None

        if start_str.strip():
            try:
                start = datetime.fromisoformat(start_str.strip())
            except ValueError:
                raise ValueError(f"Invalid start date: {start_str}")

        if end_str.strip():
            try:
                end = datetime.fromisoformat(end_str.strip())
                # Set end to end of day
                end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
            except ValueError:
                raise ValueError(f"Invalid end date: {end_str}")

        return cls(start=start, end=end)


class DomainFilter(BookmarkFilter):
    """
    Filter bookmarks by URL domain.

    Supports matching against multiple domains.
    """

    def __init__(
        self,
        domains: Union[str, List[str]],
        include_subdomains: bool = True,
    ):
        """
        Initialize domain filter.

        Args:
            domains: Domain(s) to filter by (e.g., "github.com", "google.com")
            include_subdomains: Whether to include subdomains (e.g., "api.github.com")
        """
        if isinstance(domains, str):
            domains = [d.strip() for d in domains.split(",")]
        self.domains = [d.lower().strip() for d in domains if d.strip()]
        self.include_subdomains = include_subdomains

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark's URL is from one of the specified domains."""
        url = bookmark.url or ""

        if not url:
            return False

        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()

            # Remove port if present
            if ":" in hostname:
                hostname = hostname.split(":")[0]

            for domain in self.domains:
                if self.include_subdomains:
                    # Match exact domain or any subdomain
                    if hostname == domain or hostname.endswith(f".{domain}"):
                        return True
                else:
                    # Match exact domain only
                    if hostname == domain:
                        return True

            return False

        except Exception:
            return False


class StatusFilter(BookmarkFilter):
    """
    Filter bookmarks by processing status.

    Supports filtering by validation status, AI processing status, etc.
    """

    def __init__(self, statuses: Union[str, List[str]]):
        """
        Initialize status filter.

        Args:
            statuses: Status(es) to filter by. Supported values:
                     - "validated": URL has been validated
                     - "invalid": URL validation failed
                     - "processed": AI processing completed
                     - "unprocessed": AI processing not done
                     - "tags_optimized": Tags have been optimized
                     - "error": Any error occurred during processing
        """
        if isinstance(statuses, str):
            statuses = [statuses]
        self.statuses = [s.lower().strip() for s in statuses]

        # Validate status values
        valid_statuses = {
            "validated", "invalid", "processed", "unprocessed",
            "tags_optimized", "error", "content_extracted", "pending"
        }
        for status in self.statuses:
            if status not in valid_statuses:
                raise ValueError(
                    f"Invalid status: {status}. Valid values: {valid_statuses}"
                )

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark has any of the specified statuses."""
        status = bookmark.processing_status

        for s in self.statuses:
            if s == "validated" and status.url_validated:
                return True
            elif s == "invalid" and status.url_validation_error:
                return True
            elif s == "processed" and status.ai_processed:
                return True
            elif s == "unprocessed" and not status.ai_processed:
                return True
            elif s == "tags_optimized" and status.tags_optimized:
                return True
            elif s == "content_extracted" and status.content_extracted:
                return True
            elif s == "pending" and not status.url_validated:
                return True
            elif s == "error" and (
                status.url_validation_error
                or status.content_extraction_error
                or status.ai_processing_error
            ):
                return True

        return False


class CustomFilter(BookmarkFilter):
    """
    Filter using a custom predicate function.

    Allows for arbitrary filtering logic.
    """

    def __init__(self, predicate: Callable[[Bookmark], bool], name: str = "custom"):
        """
        Initialize custom filter.

        Args:
            predicate: Function that takes a Bookmark and returns bool
            name: Name for this filter (for debugging)
        """
        self.predicate = predicate
        self.name = name

    def matches(self, bookmark: Bookmark) -> bool:
        """Apply the custom predicate."""
        return self.predicate(bookmark)


class URLPatternFilter(BookmarkFilter):
    """
    Filter bookmarks by URL pattern matching.

    Supports regex patterns for flexible URL matching.
    """

    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        """
        Initialize URL pattern filter.

        Args:
            pattern: Regex pattern to match URLs
            flags: Regex flags (default: case insensitive)
        """
        self.pattern = pattern
        self._regex = re.compile(pattern, flags)

    def matches(self, bookmark: Bookmark) -> bool:
        """Check if bookmark URL matches the pattern."""
        url = bookmark.url or ""
        return bool(self._regex.search(url))


@dataclass
class FilterChain:
    """
    Apply multiple filters with configurable logic.

    Provides a convenient way to build and apply filter chains
    from configuration or CLI arguments.
    """

    filters: List[BookmarkFilter] = field(default_factory=list)
    operator: str = "and"  # "and" or "or"

    def add(self, filter_obj: BookmarkFilter) -> "FilterChain":
        """
        Add a filter to the chain.

        Args:
            filter_obj: Filter to add

        Returns:
            Self for method chaining
        """
        self.filters.append(filter_obj)
        return self

    def apply(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """
        Apply all filters to a list of bookmarks.

        Args:
            bookmarks: List of bookmarks to filter

        Returns:
            List of bookmarks that match the filter chain
        """
        if not self.filters:
            return bookmarks

        # Create a composite filter from all filters
        composite = CompositeFilter(self.filters, operator=self.operator)
        return composite.filter(bookmarks)

    def matches(self, bookmark: Bookmark) -> bool:
        """
        Check if a bookmark matches the filter chain.

        Args:
            bookmark: Bookmark to check

        Returns:
            True if bookmark matches
        """
        if not self.filters:
            return True

        composite = CompositeFilter(self.filters, operator=self.operator)
        return composite.matches(bookmark)

    def count_matching(self, bookmarks: List[Bookmark]) -> int:
        """
        Count how many bookmarks match the filter chain.

        Args:
            bookmarks: List of bookmarks to check

        Returns:
            Count of matching bookmarks
        """
        return len(self.apply(bookmarks))

    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> "FilterChain":
        """
        Create a FilterChain from CLI arguments.

        Args:
            args: Dictionary of CLI arguments with keys like:
                 - filter_folder: Folder pattern
                 - filter_tag: Tag(s) to filter
                 - filter_date: Date range string
                 - filter_domain: Domain(s) to filter
                 - filter_status: Processing status
                 - retry_invalid: Re-process invalid URLs

        Returns:
            FilterChain configured from the arguments
        """
        chain = cls()

        # Folder filter
        if args.get("filter_folder"):
            chain.add(FolderFilter(args["filter_folder"]))

        # Tag filter
        if args.get("filter_tag"):
            tags = args["filter_tag"]
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]
            chain.add(TagFilter(tags, mode=args.get("tag_mode", "any")))

        # Date range filter
        if args.get("filter_date"):
            chain.add(DateRangeFilter.from_string(args["filter_date"]))

        # Domain filter
        if args.get("filter_domain"):
            chain.add(DomainFilter(args["filter_domain"]))

        # Status filter
        if args.get("filter_status"):
            chain.add(StatusFilter(args["filter_status"]))

        # Retry invalid (convenience shortcut)
        if args.get("retry_invalid"):
            chain.add(StatusFilter(["invalid"]))

        return chain

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FilterChain":
        """
        Create a FilterChain from a configuration dictionary.

        Args:
            config: Dictionary with filter specifications

        Returns:
            FilterChain configured from the dictionary
        """
        return cls.from_cli_args(config)

    def __len__(self) -> int:
        """Return the number of filters in the chain."""
        return len(self.filters)

    def __bool__(self) -> bool:
        """Return True if there are any filters."""
        return bool(self.filters)


# Convenience factory functions

def folder_filter(pattern: str) -> FolderFilter:
    """Create a folder filter with the given pattern."""
    return FolderFilter(pattern)


def tag_filter(
    tags: Union[str, List[str]],
    mode: str = "any",
) -> TagFilter:
    """Create a tag filter for the given tags."""
    return TagFilter(tags, mode=mode)


def date_filter(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> DateRangeFilter:
    """Create a date range filter."""
    return DateRangeFilter(start=start, end=end)


def domain_filter(domains: Union[str, List[str]]) -> DomainFilter:
    """Create a domain filter for the given domains."""
    return DomainFilter(domains)


def status_filter(statuses: Union[str, List[str]]) -> StatusFilter:
    """Create a status filter."""
    return StatusFilter(statuses)


def url_pattern_filter(pattern: str) -> URLPatternFilter:
    """Create a URL pattern filter."""
    return URLPatternFilter(pattern)
