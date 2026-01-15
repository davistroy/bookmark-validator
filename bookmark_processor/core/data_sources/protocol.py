"""
Data Source Protocol for Bookmark Processing.

This module defines the abstract protocol for bookmark data sources,
enabling multiple data source implementations (CSV, MCP, future sources).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..data_models import Bookmark


@dataclass
class BulkUpdateResult:
    """
    Result of a bulk update operation.

    Attributes:
        total: Total number of bookmarks in the operation
        succeeded: Number of successfully updated bookmarks
        failed: Number of failed updates
        errors: List of error details for failed updates
    """

    total: int
    succeeded: int
    failed: int
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total == 0:
            return 0.0
        return (self.succeeded / self.total) * 100

    @property
    def has_errors(self) -> bool:
        """Check if there were any errors."""
        return self.failed > 0

    def __str__(self) -> str:
        return (
            f"BulkUpdateResult(total={self.total}, "
            f"succeeded={self.succeeded}, "
            f"failed={self.failed}, "
            f"success_rate={self.success_rate:.1f}%)"
        )


@runtime_checkable
class BookmarkDataSource(Protocol):
    """
    Protocol for bookmark data sources.

    This protocol defines the interface that all bookmark data sources
    must implement, enabling a consistent API for different storage backends
    (CSV files, MCP/API connections, databases, etc.).

    Example Usage:
        >>> source = CSVDataSource(Path("bookmarks.csv"), Path("output.csv"))
        >>> bookmarks = source.fetch_bookmarks()
        >>> for bookmark in bookmarks:
        ...     bookmark.enhanced_description = "New description"
        >>> result = source.bulk_update(bookmarks)
        >>> source.save()  # For sources that support it
    """

    @abstractmethod
    def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        """
        Fetch bookmarks from the data source.

        Args:
            filters: Optional dictionary of filter criteria. Supported keys
                    depend on the data source, but common ones include:
                    - folder: Folder pattern (glob supported)
                    - tags: List of tags to filter by
                    - domain: Domain(s) to filter by
                    - since_last_run: Boolean to filter unprocessed bookmarks

        Returns:
            List of Bookmark objects matching the criteria

        Raises:
            DataSourceError: If fetching fails
        """
        ...

    @abstractmethod
    def update_bookmark(self, bookmark: Bookmark) -> bool:
        """
        Update a single bookmark in the data source.

        Args:
            bookmark: The bookmark to update

        Returns:
            True if update succeeded, False otherwise

        Raises:
            DataSourceError: If update fails due to connection issues
        """
        ...

    @abstractmethod
    def bulk_update(
        self,
        bookmarks: List[Bookmark]
    ) -> BulkUpdateResult:
        """
        Bulk update multiple bookmarks.

        This method is more efficient than calling update_bookmark()
        repeatedly for large numbers of bookmarks.

        Args:
            bookmarks: List of bookmarks to update

        Returns:
            BulkUpdateResult with statistics and any errors

        Raises:
            DataSourceError: If bulk update fails completely
        """
        ...

    @property
    @abstractmethod
    def supports_incremental(self) -> bool:
        """
        Whether this data source supports incremental updates.

        Sources that support incremental updates can track which bookmarks
        have been processed and only return unprocessed ones.

        Returns:
            True if incremental updates are supported
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Human-readable name for this data source.

        Returns:
            Name string for display purposes
        """
        ...


class AbstractBookmarkDataSource(ABC):
    """
    Abstract base class for bookmark data sources.

    This class provides a base implementation with common functionality
    that concrete data sources can extend.
    """

    @abstractmethod
    def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        """Fetch bookmarks from the data source."""
        pass

    @abstractmethod
    def update_bookmark(self, bookmark: Bookmark) -> bool:
        """Update a single bookmark."""
        pass

    def bulk_update(
        self,
        bookmarks: List[Bookmark]
    ) -> BulkUpdateResult:
        """
        Default bulk update implementation using individual updates.

        Concrete classes should override this for better performance
        if the data source supports batch operations.
        """
        succeeded = 0
        failed = 0
        errors = []

        for bookmark in bookmarks:
            try:
                if self.update_bookmark(bookmark):
                    succeeded += 1
                else:
                    failed += 1
                    errors.append({
                        "url": bookmark.url,
                        "error": "Update returned False"
                    })
            except Exception as e:
                failed += 1
                errors.append({
                    "url": bookmark.url,
                    "error": str(e)
                })

        return BulkUpdateResult(
            total=len(bookmarks),
            succeeded=succeeded,
            failed=failed,
            errors=errors
        )

    @property
    @abstractmethod
    def supports_incremental(self) -> bool:
        """Whether this source supports incremental updates."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name for this source."""
        pass


class DataSourceError(Exception):
    """
    Exception raised for data source errors.

    Attributes:
        message: Error description
        source_name: Name of the data source that raised the error
        original_error: The underlying exception if any
    """

    def __init__(
        self,
        message: str,
        source_name: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.source_name = source_name
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = []
        if self.source_name:
            parts.append(f"[{self.source_name}]")
        parts.append(self.message)
        if self.original_error:
            parts.append(f"(Caused by: {type(self.original_error).__name__}: {self.original_error})")
        return " ".join(parts)


class DataSourceConnectionError(DataSourceError):
    """Exception raised when connection to data source fails."""
    pass


class DataSourceReadError(DataSourceError):
    """Exception raised when reading from data source fails."""
    pass


class DataSourceWriteError(DataSourceError):
    """Exception raised when writing to data source fails."""
    pass


class DataSourceValidationError(DataSourceError):
    """Exception raised when data validation fails."""
    pass
