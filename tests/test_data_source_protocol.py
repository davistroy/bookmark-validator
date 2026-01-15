"""
Unit tests for the data source protocol and base classes.

Tests the BookmarkDataSource protocol, BulkUpdateResult, and
exception classes.
"""

from typing import Any, Dict, List, Optional

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.data_sources import (
    AbstractBookmarkDataSource,
    BookmarkDataSource,
    BulkUpdateResult,
    DataSourceConnectionError,
    DataSourceError,
    DataSourceReadError,
    DataSourceValidationError,
    DataSourceWriteError,
)


class TestBulkUpdateResult:
    """Test BulkUpdateResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic BulkUpdateResult."""
        result = BulkUpdateResult(
            total=10,
            succeeded=8,
            failed=2,
            errors=[{"url": "http://test.com", "error": "Failed"}]
        )

        assert result.total == 10
        assert result.succeeded == 8
        assert result.failed == 2
        assert len(result.errors) == 1

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = BulkUpdateResult(total=10, succeeded=8, failed=2)
        assert result.success_rate == 80.0

        # All succeeded
        result = BulkUpdateResult(total=10, succeeded=10, failed=0)
        assert result.success_rate == 100.0

        # None succeeded
        result = BulkUpdateResult(total=10, succeeded=0, failed=10)
        assert result.success_rate == 0.0

    def test_success_rate_zero_total(self):
        """Test success rate with zero total."""
        result = BulkUpdateResult(total=0, succeeded=0, failed=0)
        assert result.success_rate == 0.0

    def test_has_errors_property(self):
        """Test has_errors property."""
        # No errors
        result = BulkUpdateResult(total=10, succeeded=10, failed=0)
        assert not result.has_errors

        # Has errors
        result = BulkUpdateResult(total=10, succeeded=8, failed=2)
        assert result.has_errors

    def test_default_errors_list(self):
        """Test that errors defaults to empty list."""
        result = BulkUpdateResult(total=10, succeeded=10, failed=0)
        assert result.errors == []

    def test_str_representation(self):
        """Test string representation."""
        result = BulkUpdateResult(total=10, succeeded=8, failed=2)
        result_str = str(result)

        assert "10" in result_str
        assert "8" in result_str
        assert "2" in result_str
        assert "80.0%" in result_str


class TestDataSourceExceptions:
    """Test data source exception classes."""

    def test_data_source_error_basic(self):
        """Test basic DataSourceError."""
        error = DataSourceError("Test error")
        assert "Test error" in str(error)
        assert error.message == "Test error"
        assert error.source_name is None
        assert error.original_error is None

    def test_data_source_error_with_source_name(self):
        """Test DataSourceError with source name."""
        error = DataSourceError("Test error", source_name="CSV File")
        assert "[CSV File]" in str(error)
        assert "Test error" in str(error)

    def test_data_source_error_with_original_error(self):
        """Test DataSourceError with original error."""
        original = ValueError("Original error")
        error = DataSourceError(
            "Test error",
            source_name="CSV File",
            original_error=original
        )

        assert "Test error" in str(error)
        assert "ValueError" in str(error)
        assert "Original error" in str(error)
        assert error.original_error is original

    def test_data_source_connection_error(self):
        """Test DataSourceConnectionError."""
        error = DataSourceConnectionError("Connection failed")
        assert isinstance(error, DataSourceError)
        assert "Connection failed" in str(error)

    def test_data_source_read_error(self):
        """Test DataSourceReadError."""
        error = DataSourceReadError("Read failed", source_name="API")
        assert isinstance(error, DataSourceError)
        assert "Read failed" in str(error)
        assert "[API]" in str(error)

    def test_data_source_write_error(self):
        """Test DataSourceWriteError."""
        error = DataSourceWriteError("Write failed")
        assert isinstance(error, DataSourceError)
        assert "Write failed" in str(error)

    def test_data_source_validation_error(self):
        """Test DataSourceValidationError."""
        error = DataSourceValidationError("Invalid data")
        assert isinstance(error, DataSourceError)
        assert "Invalid data" in str(error)


class ConcreteDataSource(AbstractBookmarkDataSource):
    """Concrete implementation for testing abstract base class."""

    def __init__(self):
        self.bookmarks = []
        self._name = "Test Source"

    def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        return self.bookmarks

    def update_bookmark(self, bookmark: Bookmark) -> bool:
        for i, b in enumerate(self.bookmarks):
            if b.url == bookmark.url:
                self.bookmarks[i] = bookmark
                return True
        return False

    @property
    def supports_incremental(self) -> bool:
        return False

    @property
    def source_name(self) -> str:
        return self._name


class TestAbstractBookmarkDataSource:
    """Test AbstractBookmarkDataSource base class."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""
        source = ConcreteDataSource()
        assert source.source_name == "Test Source"
        assert source.supports_incremental is False
        assert source.fetch_bookmarks() == []

    def test_default_bulk_update(self):
        """Test default bulk_update implementation."""
        source = ConcreteDataSource()

        # Add some bookmarks
        b1 = Bookmark(url="http://test1.com", title="Test 1")
        b2 = Bookmark(url="http://test2.com", title="Test 2")
        source.bookmarks = [b1, b2]

        # Update bookmarks
        b1_updated = Bookmark(url="http://test1.com", title="Test 1 Updated")
        b2_updated = Bookmark(url="http://test2.com", title="Test 2 Updated")
        b3 = Bookmark(url="http://test3.com", title="Test 3")  # Not in source

        result = source.bulk_update([b1_updated, b2_updated, b3])

        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1
        assert len(result.errors) == 1
        assert result.errors[0]["url"] == "http://test3.com"

    def test_update_bookmark_not_found(self):
        """Test update_bookmark returns False when not found."""
        source = ConcreteDataSource()
        bookmark = Bookmark(url="http://notfound.com", title="Not Found")

        result = source.update_bookmark(bookmark)
        assert result is False

    def test_update_bookmark_found(self):
        """Test update_bookmark returns True when found."""
        source = ConcreteDataSource()
        original = Bookmark(url="http://test.com", title="Original")
        source.bookmarks = [original]

        updated = Bookmark(url="http://test.com", title="Updated")
        result = source.update_bookmark(updated)

        assert result is True
        assert source.bookmarks[0].title == "Updated"


class TestBookmarkDataSourceProtocol:
    """Test BookmarkDataSource protocol compliance."""

    def test_protocol_compliance(self):
        """Test that ConcreteDataSource complies with protocol."""
        source = ConcreteDataSource()

        # Runtime checkable protocol
        assert isinstance(source, BookmarkDataSource)

    def test_protocol_methods_exist(self):
        """Test that protocol methods are callable."""
        source = ConcreteDataSource()

        # Check methods exist and are callable
        assert callable(source.fetch_bookmarks)
        assert callable(source.update_bookmark)
        assert callable(source.bulk_update)

        # Check properties exist
        assert hasattr(source, "supports_incremental")
        assert hasattr(source, "source_name")


class BrokenDataSource(AbstractBookmarkDataSource):
    """Data source that raises exceptions for testing."""

    def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        raise DataSourceReadError("Read failed")

    def update_bookmark(self, bookmark: Bookmark) -> bool:
        raise Exception("Update failed")

    @property
    def supports_incremental(self) -> bool:
        return True

    @property
    def source_name(self) -> str:
        return "Broken Source"


class TestBulkUpdateWithExceptions:
    """Test bulk_update behavior when individual updates raise exceptions."""

    def test_bulk_update_handles_exceptions(self):
        """Test that bulk_update handles exceptions gracefully."""
        source = BrokenDataSource()

        b1 = Bookmark(url="http://test1.com", title="Test 1")
        b2 = Bookmark(url="http://test2.com", title="Test 2")

        result = source.bulk_update([b1, b2])

        # All should fail
        assert result.total == 2
        assert result.succeeded == 0
        assert result.failed == 2
        assert len(result.errors) == 2

        # Check errors contain exception message
        for error in result.errors:
            assert "Update failed" in error["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
