"""
Unit tests for the CSV data source implementation.

Tests CSVDataSource class that wraps RaindropCSVHandler.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.data_sources import (
    BulkUpdateResult,
    CSVDataSource,
    DataSourceReadError,
    DataSourceValidationError,
    DataSourceWriteError,
)


class TestCSVDataSourceBasics:
    """Test basic CSVDataSource functionality."""

    def test_initialization(self, temp_dir):
        """Test CSVDataSource initialization."""
        input_path = temp_dir / "input.csv"
        output_path = temp_dir / "output.csv"

        source = CSVDataSource(input_path, output_path)

        assert source.input_path == input_path
        assert source.output_path == output_path
        assert source.source_name == "CSV File"
        assert source.supports_incremental is False
        assert source.is_loaded is False
        assert source.is_modified is False

    def test_initialization_with_string_paths(self, temp_dir):
        """Test initialization with string paths."""
        input_path = str(temp_dir / "input.csv")
        output_path = str(temp_dir / "output.csv")

        source = CSVDataSource(input_path, output_path)

        assert isinstance(source.input_path, Path)
        assert isinstance(source.output_path, Path)

    def test_repr(self, temp_dir):
        """Test string representation."""
        source = CSVDataSource(temp_dir / "input.csv", temp_dir / "output.csv")

        repr_str = repr(source)
        assert "CSVDataSource" in repr_str
        assert "input=" in repr_str
        assert "output=" in repr_str


class TestCSVDataSourceLoading:
    """Test loading bookmarks from CSV files."""

    def test_load_bookmarks(self, sample_csv_file, temp_dir):
        """Test loading bookmarks from CSV file."""
        output_path = temp_dir / "output.csv"
        source = CSVDataSource(sample_csv_file, output_path)

        bookmarks = source.fetch_bookmarks()

        assert len(bookmarks) > 0
        assert all(isinstance(b, Bookmark) for b in bookmarks)
        assert source.is_loaded is True

    def test_load_from_nonexistent_file(self, temp_dir):
        """Test loading from non-existent file raises error."""
        source = CSVDataSource(
            temp_dir / "nonexistent.csv",
            temp_dir / "output.csv"
        )

        with pytest.raises(DataSourceReadError) as exc_info:
            source.fetch_bookmarks()

        assert "not found" in str(exc_info.value).lower()

    def test_lazy_loading(self, sample_csv_file, temp_dir):
        """Test that loading is lazy."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        # Not loaded until fetch_bookmarks called
        assert source.is_loaded is False

        source.fetch_bookmarks()

        assert source.is_loaded is True

    def test_bookmark_count(self, sample_csv_file, temp_dir):
        """Test getting bookmark count."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        count = source.get_bookmark_count()

        assert count > 0
        assert len(source) == count


class TestCSVDataSourceFiltering:
    """Test filtering bookmarks."""

    def test_filter_by_folder(self, sample_csv_file, temp_dir):
        """Test filtering by folder pattern."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        # Load all first to check we have programming bookmarks
        all_bookmarks = source.fetch_bookmarks()
        programming_count = sum(
            1 for b in all_bookmarks
            if b.folder and "Programming" in b.folder
        )

        # Filter by folder
        filters = {"filter_folder": "Programming/*"}
        filtered = source.fetch_bookmarks(filters=filters)

        assert len(filtered) == programming_count

    def test_filter_by_tag(self, sample_csv_file, temp_dir):
        """Test filtering by tag."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        filters = {"filter_tag": "python"}
        filtered = source.fetch_bookmarks(filters=filters)

        # All filtered bookmarks should have python tag
        for bookmark in filtered:
            assert any("python" in t.lower() for t in bookmark.tags)

    def test_no_filters_returns_all(self, sample_csv_file, temp_dir):
        """Test that no filters returns all bookmarks."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        all_bookmarks = source.fetch_bookmarks()
        filtered = source.fetch_bookmarks(filters=None)

        assert len(all_bookmarks) == len(filtered)


class TestCSVDataSourceUpdates:
    """Test updating bookmarks."""

    def test_update_existing_bookmark(self, sample_csv_file, temp_dir):
        """Test updating an existing bookmark."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        bookmarks = source.fetch_bookmarks()
        original = bookmarks[0]
        original_url = original.url

        # Create updated version
        updated = Bookmark(
            url=original_url,
            title="Updated Title",
            folder="Updated/Folder"
        )

        result = source.update_bookmark(updated)

        assert result is True
        assert source.is_modified is True

        # Verify update
        refreshed = source.get_bookmark_by_url(original_url)
        assert refreshed.title == "Updated Title"

    def test_update_nonexistent_bookmark(self, sample_csv_file, temp_dir):
        """Test updating non-existent bookmark returns False."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")
        source.fetch_bookmarks()

        bookmark = Bookmark(url="http://nonexistent.com", title="Test")

        result = source.update_bookmark(bookmark)

        assert result is False

    def test_bulk_update(self, sample_csv_file, temp_dir):
        """Test bulk update of bookmarks."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        bookmarks = source.fetch_bookmarks()

        # Update all bookmarks
        for bookmark in bookmarks:
            bookmark.enhanced_description = "Bulk updated"

        # Add one that doesn't exist
        nonexistent = Bookmark(url="http://nonexistent.com", title="Missing")
        bookmarks_to_update = bookmarks + [nonexistent]

        result = source.bulk_update(bookmarks_to_update)

        assert isinstance(result, BulkUpdateResult)
        assert result.succeeded == len(bookmarks)
        assert result.failed == 1

    def test_add_bookmark(self, sample_csv_file, temp_dir):
        """Test adding a new bookmark."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        initial_count = source.get_bookmark_count()

        new_bookmark = Bookmark(
            url="http://newbookmark.com",
            title="New Bookmark",
            folder="Test"
        )

        result = source.add_bookmark(new_bookmark)

        assert result is True
        assert source.get_bookmark_count() == initial_count + 1
        assert source.is_modified is True

    def test_add_duplicate_bookmark_fails(self, sample_csv_file, temp_dir):
        """Test that adding duplicate bookmark fails."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        bookmarks = source.fetch_bookmarks()
        existing = bookmarks[0]

        # Try to add duplicate
        duplicate = Bookmark(url=existing.url, title="Duplicate")

        result = source.add_bookmark(duplicate)

        assert result is False

    def test_remove_bookmark(self, sample_csv_file, temp_dir):
        """Test removing a bookmark."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        bookmarks = source.fetch_bookmarks()
        initial_count = len(bookmarks)
        to_remove = bookmarks[0]

        result = source.remove_bookmark(to_remove)

        assert result is True
        assert source.get_bookmark_count() == initial_count - 1
        assert source.get_bookmark_by_url(to_remove.url) is None

    def test_remove_nonexistent_bookmark(self, sample_csv_file, temp_dir):
        """Test removing non-existent bookmark returns False."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")
        source.fetch_bookmarks()

        bookmark = Bookmark(url="http://nonexistent.com", title="Test")

        result = source.remove_bookmark(bookmark)

        assert result is False


class TestCSVDataSourceSaving:
    """Test saving bookmarks to CSV."""

    def test_save_bookmarks(self, sample_csv_file, temp_dir):
        """Test saving bookmarks to output file."""
        output_path = temp_dir / "output.csv"
        source = CSVDataSource(sample_csv_file, output_path)

        bookmarks = source.fetch_bookmarks()

        # Modify a bookmark
        bookmarks[0].enhanced_description = "Modified"
        source.update_bookmark(bookmarks[0])

        source.save()

        # Verify file was created
        assert output_path.exists()
        assert source.is_modified is False

    def test_save_without_loading_raises_error(self, temp_dir):
        """Test saving without loading raises error."""
        source = CSVDataSource(
            temp_dir / "input.csv",
            temp_dir / "output.csv"
        )

        with pytest.raises(DataSourceValidationError):
            source.save()

    def test_save_to_readonly_location_raises_error(self, sample_csv_file, temp_dir):
        """Test that save to invalid location raises error."""
        # Use a path that's likely to fail on write
        source = CSVDataSource(
            sample_csv_file,
            Path("/nonexistent/directory/output.csv")
        )

        source.fetch_bookmarks()

        with pytest.raises(DataSourceWriteError):
            source.save()


class TestCSVDataSourceLookup:
    """Test bookmark lookup functionality."""

    def test_get_bookmark_by_url(self, sample_csv_file, temp_dir):
        """Test getting bookmark by URL."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        bookmarks = source.fetch_bookmarks()
        expected_url = bookmarks[0].url

        result = source.get_bookmark_by_url(expected_url)

        assert result is not None
        assert result.url == expected_url

    def test_get_bookmark_by_nonexistent_url(self, sample_csv_file, temp_dir):
        """Test getting non-existent bookmark returns None."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")
        source.fetch_bookmarks()

        result = source.get_bookmark_by_url("http://nonexistent.com")

        assert result is None


class TestCSVDataSourceWithMockedHandler:
    """Test CSVDataSource with mocked RaindropCSVHandler."""

    def test_custom_handler_injection(self, temp_dir):
        """Test that custom handler can be injected."""
        mock_handler = MagicMock()
        mock_handler.load_and_transform_csv.return_value = [
            Bookmark(url="http://test.com", title="Test")
        ]

        source = CSVDataSource(
            temp_dir / "input.csv",
            temp_dir / "output.csv",
            csv_handler=mock_handler
        )

        bookmarks = source.fetch_bookmarks()

        mock_handler.load_and_transform_csv.assert_called_once()
        assert len(bookmarks) == 1

    def test_handler_error_wrapped(self, temp_dir):
        """Test that handler errors are wrapped properly."""
        mock_handler = MagicMock()
        mock_handler.load_and_transform_csv.side_effect = Exception("Handler error")

        source = CSVDataSource(
            temp_dir / "input.csv",
            temp_dir / "output.csv",
            csv_handler=mock_handler
        )

        with pytest.raises(DataSourceReadError) as exc_info:
            source.fetch_bookmarks()

        assert "Handler error" in str(exc_info.value)


class TestCSVDataSourceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_csv_file(self, temp_dir):
        """Test loading empty CSV file raises DataSourceReadError."""
        # Create empty CSV with headers (but no data)
        # The underlying RaindropCSVHandler raises an error for empty CSVs
        empty_csv = temp_dir / "empty.csv"
        empty_csv.write_text(
            "id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n"
        )

        source = CSVDataSource(empty_csv, temp_dir / "output.csv")

        # Empty CSV should raise an error since handler doesn't allow empty files
        with pytest.raises(DataSourceReadError):
            source.fetch_bookmarks()

    def test_special_characters_in_bookmarks(self, temp_dir):
        """Test handling bookmarks with special characters."""
        mock_handler = MagicMock()
        mock_handler.load_and_transform_csv.return_value = [
            Bookmark(
                url="http://test.com/path?q=search&x=1",
                title='Title with "quotes" and <brackets>',
                folder="Path/With/Slashes",
                tags=["tag-with-dash", "tag_with_underscore"]
            )
        ]

        source = CSVDataSource(
            temp_dir / "input.csv",
            temp_dir / "output.csv",
            csv_handler=mock_handler
        )

        bookmarks = source.fetch_bookmarks()

        assert len(bookmarks) == 1
        assert "quotes" in bookmarks[0].title
        assert "?" in bookmarks[0].url

    def test_multiple_fetches_return_same_data(self, sample_csv_file, temp_dir):
        """Test that multiple fetches return consistent data."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        first_fetch = source.fetch_bookmarks()
        second_fetch = source.fetch_bookmarks()

        assert len(first_fetch) == len(second_fetch)
        for b1, b2 in zip(first_fetch, second_fetch):
            assert b1.url == b2.url


class TestCSVDataSourceIntegration:
    """Integration tests with real RaindropCSVHandler."""

    def test_full_workflow(self, sample_csv_file, temp_dir):
        """Test complete read-modify-write workflow."""
        output_path = temp_dir / "output.csv"
        source = CSVDataSource(sample_csv_file, output_path)

        # Load bookmarks
        bookmarks = source.fetch_bookmarks()
        initial_count = len(bookmarks)

        # Add a new bookmark
        new_bookmark = Bookmark(
            url="http://newsite.example.com",
            title="New Site",
            folder="Test/New",
            tags=["new", "test"]
        )
        source.add_bookmark(new_bookmark)

        # Update existing bookmark
        bookmarks[0].enhanced_description = "Updated description"
        source.update_bookmark(bookmarks[0])

        # Save
        source.save()

        # Verify the output file was created
        assert output_path.exists()

        # Verify file content (check CSV has rows)
        import csv
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Header + data rows (initial_count + 1 new bookmark)
            assert len(rows) == initial_count + 2  # header + data rows

        # Verify the source data is correct
        assert source.get_bookmark_count() == initial_count + 1

    def test_filter_then_update_workflow(self, sample_csv_file, temp_dir):
        """Test filtering then updating bookmarks."""
        source = CSVDataSource(sample_csv_file, temp_dir / "output.csv")

        # Filter to get subset
        filters = {"filter_tag": "python"}
        filtered = source.fetch_bookmarks(filters=filters)

        if filtered:
            # Update filtered bookmarks
            for bookmark in filtered:
                bookmark.note = "Python-related content"
                source.update_bookmark(bookmark)

            assert source.is_modified is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
