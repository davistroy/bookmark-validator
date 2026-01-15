"""
CSV Data Source Implementation.

This module provides a CSV-based data source that wraps the existing
RaindropCSVHandler to implement the BookmarkDataSource protocol.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..csv_handler import RaindropCSVHandler
from ..data_models import Bookmark
from ..filters import FilterChain
from .protocol import (
    AbstractBookmarkDataSource,
    BulkUpdateResult,
    DataSourceError,
    DataSourceReadError,
    DataSourceValidationError,
    DataSourceWriteError,
)


class CSVDataSource(AbstractBookmarkDataSource):
    """
    CSV-based data source for bookmark processing.

    This class wraps the existing RaindropCSVHandler to implement the
    BookmarkDataSource protocol, enabling CSV files to be used as a
    data source for bookmark processing pipelines.

    The CSV source operates on an in-memory collection of bookmarks,
    loading from input file and writing to output file on save.

    Attributes:
        input_path: Path to the input CSV file
        output_path: Path to the output CSV file
        handler: The RaindropCSVHandler instance

    Example:
        >>> source = CSVDataSource(Path("export.csv"), Path("import.csv"))
        >>> bookmarks = source.fetch_bookmarks()
        >>> for bookmark in bookmarks:
        ...     bookmark.enhanced_description = "New description"
        >>> result = source.bulk_update(bookmarks)
        >>> source.save()
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        csv_handler: Optional[RaindropCSVHandler] = None
    ):
        """
        Initialize the CSV data source.

        Args:
            input_path: Path to the input CSV file (raindrop.io export format)
            output_path: Path to the output CSV file (raindrop.io import format)
            csv_handler: Optional RaindropCSVHandler instance (for testing)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.handler = csv_handler or RaindropCSVHandler()
        self._bookmarks: Optional[List[Bookmark]] = None
        self._url_index: Optional[Dict[str, int]] = None
        self._loaded = False
        self._modified = False
        self.logger = logging.getLogger(__name__)

    def _load_bookmarks(self) -> None:
        """
        Load bookmarks from the input CSV file.

        This method is called lazily on first access to bookmarks.

        Raises:
            DataSourceReadError: If loading fails
        """
        if self._loaded:
            return

        try:
            self.logger.info(f"Loading bookmarks from {self.input_path}")
            self._bookmarks = self.handler.load_and_transform_csv(self.input_path)
            self._build_url_index()
            self._loaded = True
            self.logger.info(f"Loaded {len(self._bookmarks)} bookmarks")

        except FileNotFoundError as e:
            raise DataSourceReadError(
                f"CSV file not found: {self.input_path}",
                source_name=self.source_name,
                original_error=e
            )
        except Exception as e:
            raise DataSourceReadError(
                f"Failed to load CSV file: {self.input_path}",
                source_name=self.source_name,
                original_error=e
            )

    def _build_url_index(self) -> None:
        """Build an index of URL to bookmark list position for fast lookups."""
        if self._bookmarks is None:
            self._url_index = {}
            return

        self._url_index = {}
        for i, bookmark in enumerate(self._bookmarks):
            if bookmark.url:
                # Use normalized URL as key
                key = bookmark.normalized_url or bookmark.url
                self._url_index[key] = i

    def _get_bookmark_index(self, bookmark: Bookmark) -> Optional[int]:
        """
        Get the index of a bookmark in the internal list.

        Args:
            bookmark: The bookmark to find

        Returns:
            Index in the list, or None if not found
        """
        if self._url_index is None:
            self._build_url_index()

        # Try normalized URL first, then raw URL
        key = bookmark.normalized_url or bookmark.url
        if key in self._url_index:
            return self._url_index[key]

        # Fallback to raw URL lookup
        if bookmark.url in self._url_index:
            return self._url_index[bookmark.url]

        # Final fallback: linear search
        if self._bookmarks:
            for i, b in enumerate(self._bookmarks):
                if b.url == bookmark.url:
                    return i

        return None

    def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        """
        Fetch bookmarks from the CSV file.

        Args:
            filters: Optional dictionary of filter criteria. Supported keys:
                    - filter_folder: Folder pattern (glob supported)
                    - filter_tag: Tag(s) to filter by
                    - filter_date: Date range string ("start:end")
                    - filter_domain: Domain(s) to filter by
                    - filter_status: Processing status filter
                    - retry_invalid: Only return previously invalid URLs

        Returns:
            List of Bookmark objects matching the criteria

        Raises:
            DataSourceReadError: If loading fails
        """
        self._load_bookmarks()

        if self._bookmarks is None:
            return []

        # Return all bookmarks if no filters specified
        if not filters:
            return list(self._bookmarks)

        # Build filter chain from the filter dictionary
        filter_chain = FilterChain.from_dict(filters)

        if not filter_chain:
            return list(self._bookmarks)

        # Apply filters
        filtered = filter_chain.apply(self._bookmarks)
        self.logger.info(
            f"Filtered {len(filtered)} of {len(self._bookmarks)} bookmarks "
            f"({len(filter_chain)} filters applied)"
        )

        return filtered

    def update_bookmark(self, bookmark: Bookmark) -> bool:
        """
        Update a single bookmark in the data source.

        The bookmark is identified by its URL. If found, all fields
        are updated from the provided bookmark object.

        Args:
            bookmark: The bookmark to update

        Returns:
            True if update succeeded, False if bookmark not found

        Raises:
            DataSourceError: If update fails
        """
        self._load_bookmarks()

        if self._bookmarks is None:
            return False

        index = self._get_bookmark_index(bookmark)
        if index is None:
            self.logger.warning(f"Bookmark not found for update: {bookmark.url}")
            return False

        # Update the bookmark in the list
        self._bookmarks[index] = bookmark
        self._modified = True
        return True

    def bulk_update(
        self,
        bookmarks: List[Bookmark]
    ) -> BulkUpdateResult:
        """
        Bulk update multiple bookmarks.

        For CSV data source, this updates the in-memory collection.
        Call save() to persist changes to the output file.

        Args:
            bookmarks: List of bookmarks to update

        Returns:
            BulkUpdateResult with statistics and any errors
        """
        self._load_bookmarks()

        succeeded = 0
        failed = 0
        errors = []

        for bookmark in bookmarks:
            if self.update_bookmark(bookmark):
                succeeded += 1
            else:
                failed += 1
                errors.append({
                    "url": bookmark.url,
                    "error": "Bookmark not found in source"
                })

        return BulkUpdateResult(
            total=len(bookmarks),
            succeeded=succeeded,
            failed=failed,
            errors=errors
        )

    def add_bookmark(self, bookmark: Bookmark) -> bool:
        """
        Add a new bookmark to the data source.

        Args:
            bookmark: The bookmark to add

        Returns:
            True if add succeeded, False if bookmark already exists
        """
        self._load_bookmarks()

        if self._bookmarks is None:
            self._bookmarks = []
            self._url_index = {}

        # Check if bookmark already exists
        if self._get_bookmark_index(bookmark) is not None:
            self.logger.warning(f"Bookmark already exists: {bookmark.url}")
            return False

        # Add to list and update index
        index = len(self._bookmarks)
        self._bookmarks.append(bookmark)
        key = bookmark.normalized_url or bookmark.url
        self._url_index[key] = index
        self._modified = True

        return True

    def remove_bookmark(self, bookmark: Bookmark) -> bool:
        """
        Remove a bookmark from the data source.

        Args:
            bookmark: The bookmark to remove

        Returns:
            True if removal succeeded, False if not found
        """
        self._load_bookmarks()

        if self._bookmarks is None:
            return False

        index = self._get_bookmark_index(bookmark)
        if index is None:
            return False

        # Remove from list and rebuild index
        del self._bookmarks[index]
        self._build_url_index()
        self._modified = True

        return True

    def save(self) -> None:
        """
        Save all bookmarks to the output CSV file.

        This writes the in-memory bookmark collection to the output file
        in raindrop.io import format.

        Raises:
            DataSourceWriteError: If saving fails
        """
        if self._bookmarks is None:
            raise DataSourceValidationError(
                "No bookmarks to save - load bookmarks first",
                source_name=self.source_name
            )

        try:
            self.logger.info(f"Saving {len(self._bookmarks)} bookmarks to {self.output_path}")
            self.handler.save_import_csv(self._bookmarks, self.output_path)
            self._modified = False
            self.logger.info(f"Successfully saved to {self.output_path}")

        except Exception as e:
            raise DataSourceWriteError(
                f"Failed to save CSV file: {self.output_path}",
                source_name=self.source_name,
                original_error=e
            )

    def get_bookmark_count(self) -> int:
        """
        Get the total number of bookmarks.

        Returns:
            Number of bookmarks in the data source
        """
        self._load_bookmarks()
        return len(self._bookmarks) if self._bookmarks else 0

    def get_bookmark_by_url(self, url: str) -> Optional[Bookmark]:
        """
        Get a bookmark by its URL.

        Args:
            url: The URL to search for

        Returns:
            Bookmark if found, None otherwise
        """
        self._load_bookmarks()

        if self._bookmarks is None or self._url_index is None:
            return None

        # Try direct lookup first
        index = self._url_index.get(url)
        if index is not None:
            return self._bookmarks[index]

        # Try linear search as fallback (for raw URL lookup)
        for bookmark in self._bookmarks:
            if bookmark.url == url:
                return bookmark

        return None

    @property
    def is_modified(self) -> bool:
        """Check if the data source has unsaved modifications."""
        return self._modified

    @property
    def is_loaded(self) -> bool:
        """Check if bookmarks have been loaded."""
        return self._loaded

    @property
    def supports_incremental(self) -> bool:
        """
        Whether this data source supports incremental updates.

        CSV files don't inherently support incremental updates,
        but when combined with ProcessingStateTracker, incremental
        processing can be achieved.

        Returns:
            False (CSV doesn't support incremental natively)
        """
        return False

    @property
    def source_name(self) -> str:
        """
        Human-readable name for this data source.

        Returns:
            "CSV File"
        """
        return "CSV File"

    def __len__(self) -> int:
        """Return the number of bookmarks."""
        return self.get_bookmark_count()

    def __repr__(self) -> str:
        return (
            f"CSVDataSource(input={self.input_path}, "
            f"output={self.output_path}, "
            f"loaded={self._loaded}, "
            f"count={len(self._bookmarks) if self._bookmarks else 0})"
        )
