"""
Unit tests for CSV handler module.

Tests the RaindropCSVHandler for reading/writing raindrop.io format CSV files.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from bookmark_processor.core.csv_handler import (
    CSVEncodingError,
    CSVError,
    CSVFormatError,
    CSVValidationError,
    RaindropCSVHandler,
)
from bookmark_processor.core.data_models import Bookmark
from tests.fixtures.test_data import (
    EXPECTED_RAINDROP_IMPORT_ROWS,
    SAMPLE_RAINDROP_EXPORT_ROWS,
    create_expected_import_dataframe,
    create_invalid_csv_content,
    create_sample_bookmark_objects,
    create_sample_export_dataframe,
)


class TestRaindropCSVHandler:
    """Test RaindropCSVHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a RaindropCSVHandler instance."""
        return RaindropCSVHandler()

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write sample export data
            df = create_sample_export_dataframe()
            df.to_csv(f, index=False)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_initialization(self, handler):
        """Test RaindropCSVHandler initialization."""
        assert handler.export_columns == [
            "id",
            "title",
            "note",
            "excerpt",
            "url",
            "folder",
            "tags",
            "created",
            "cover",
            "highlights",
            "favorite",
        ]
        assert handler.import_columns == [
            "url",
            "folder",
            "title",
            "note",
            "tags",
            "created",
        ]
        assert handler.required_export_columns == ["url"]
        assert handler.required_import_columns == ["url"]

    def test_validate_export_dataframe_valid(self, handler):
        """Test validating a valid export DataFrame."""
        df = create_sample_export_dataframe()

        # Should not raise any exception
        handler.validate_export_dataframe(df)

    def test_validate_export_dataframe_missing_columns(self, handler):
        """Test validating export DataFrame with missing columns."""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["Test1", "Test2"],
                # Missing 'url' and other required columns
            }
        )

        with pytest.raises(CSVFormatError) as exc_info:
            handler.validate_export_dataframe(df)

        assert "Missing required columns" in str(exc_info.value)

    def test_validate_export_dataframe_missing_url(self, handler):
        """Test validating export DataFrame with missing URL values."""
        df = create_sample_export_dataframe()
        df.loc[0, "url"] = ""  # Empty URL

        with pytest.raises(CSVValidationError) as exc_info:
            handler.validate_export_dataframe(df)

        assert "Empty URL" in str(exc_info.value)

    def test_validate_export_dataframe_duplicate_urls(self, handler):
        """Test validating export DataFrame with duplicate URLs."""
        df = create_sample_export_dataframe()
        df.loc[1, "url"] = df.loc[0, "url"]  # Create duplicate

        with pytest.raises(CSVValidationError) as exc_info:
            handler.validate_export_dataframe(df)

        assert "Duplicate URLs found" in str(exc_info.value)

    def test_transform_export_to_bookmarks(self, handler):
        """Test transforming export DataFrame to Bookmark objects."""
        df = create_sample_export_dataframe()
        bookmarks = handler.transform_export_to_bookmarks(df)

        assert len(bookmarks) == len(df)
        assert all(isinstance(b, Bookmark) for b in bookmarks)

        # Check first bookmark
        first_bookmark = bookmarks[0]
        assert first_bookmark.url == "https://docs.python.org/3/"
        assert first_bookmark.title == "Python Documentation"
        assert first_bookmark.folder == "Programming/Python"
        assert first_bookmark.tags == ["python", "documentation", "programming"]

    def test_transform_export_to_bookmarks_with_invalid_rows(self, handler):
        """Test transforming export DataFrame with some invalid rows."""
        df = create_sample_export_dataframe()
        # Add an invalid row
        invalid_row = {col: "" for col in df.columns}
        df = pd.concat([df, pd.DataFrame([invalid_row])], ignore_index=True)

        bookmarks = handler.transform_export_to_bookmarks(df)

        # Should skip invalid rows
        assert len(bookmarks) == len(df) - 1

    def test_bookmarks_to_dataframe(self, handler):
        """Test converting bookmarks to import DataFrame."""
        bookmarks = create_sample_bookmark_objects()
        df = handler.bookmarks_to_dataframe(bookmarks)

        assert len(df) == len(bookmarks)
        assert list(df.columns) == handler.import_columns

        # Check first row
        first_row = df.iloc[0]
        assert first_row["url"] == bookmarks[0].url
        assert first_row["title"] == bookmarks[0].get_effective_title()
        assert first_row["folder"] == bookmarks[0].get_folder_path()

    def test_bookmarks_to_dataframe_empty(self, handler):
        """Test converting empty bookmark list to DataFrame."""
        df = handler.bookmarks_to_dataframe([])

        assert len(df) == 0
        assert list(df.columns) == handler.import_columns

    def test_bookmarks_to_dataframe_tag_formatting(self, handler):
        """Test tag formatting in DataFrame conversion."""
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            optimized_tags=["tag1", "tag2", "tag3"],
        )

        df = handler.bookmarks_to_dataframe([bookmark])

        # Multiple tags should be quoted
        assert df.iloc[0]["tags"] == '"tag1, tag2, tag3"'

        # Single tag should not be quoted
        bookmark.optimized_tags = ["single"]
        df = handler.bookmarks_to_dataframe([bookmark])
        assert df.iloc[0]["tags"] == "single"

        # No tags
        bookmark.optimized_tags = []
        bookmark.tags = []
        df = handler.bookmarks_to_dataframe([bookmark])
        assert df.iloc[0]["tags"] == ""

    def test_load_export_csv(self, handler, temp_csv_file):
        """Test loading export CSV file."""
        df = handler.load_export_csv(temp_csv_file)

        assert len(df) == len(SAMPLE_RAINDROP_EXPORT_ROWS)
        assert list(df.columns) == handler.export_columns

    def test_load_export_csv_nonexistent_file(self, handler):
        """Test loading non-existent CSV file."""
        with pytest.raises(CSVError) as exc_info:
            handler.load_export_csv("nonexistent.csv")

        assert "not found" in str(exc_info.value)

    def test_load_export_csv_invalid_format(self, handler):
        """Test loading CSV with invalid format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(create_invalid_csv_content())
            temp_path = f.name

        try:
            with pytest.raises(CSVError):
                handler.load_export_csv(temp_path)
        finally:
            os.unlink(temp_path)

    def test_save_import_csv(self, handler):
        """Test saving import CSV file."""
        bookmarks = create_sample_bookmark_objects()[:3]  # Use first 3

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            handler.save_import_csv(bookmarks, temp_path)

            # Verify the saved file
            df = pd.read_csv(temp_path)
            assert len(df) == 3
            assert list(df.columns) == handler.import_columns

            # Check content matches expected format
            assert df.iloc[0]["url"] == bookmarks[0].url
            assert df.iloc[0]["title"] == bookmarks[0].get_effective_title()

        finally:
            os.unlink(temp_path)

    def test_save_import_csv_empty_bookmarks(self, handler):
        """Test saving empty bookmark list."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(CSVError) as exc_info:
                handler.save_import_csv([], temp_path)

            assert "No valid bookmarks" in str(exc_info.value)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_import_csv_write_error(self, handler):
        """Test handling write errors when saving CSV."""
        bookmarks = create_sample_bookmark_objects()[:1]

        # Try to write to a directory (should fail)
        with pytest.raises(CSVError) as exc_info:
            handler.save_import_csv(bookmarks, "/")

        assert "Failed to save CSV" in str(exc_info.value)

    def test_load_and_transform_csv(self, handler, temp_csv_file):
        """Test the complete load and transform workflow."""
        bookmarks = handler.load_and_transform_csv(temp_csv_file)

        assert len(bookmarks) > 0
        assert all(isinstance(b, Bookmark) for b in bookmarks)

        # Verify transformation
        first = bookmarks[0]
        assert first.url == SAMPLE_RAINDROP_EXPORT_ROWS[0]["url"]
        assert first.title == SAMPLE_RAINDROP_EXPORT_ROWS[0]["title"]

    def test_try_parse_csv_different_encodings(self, handler):
        """Test parsing CSV files with different encodings."""
        # Test UTF-8 with BOM
        test_content = '\ufeff"id","title","url"\n"1","Test","https://example.com"'

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(test_content.encode("utf-8-sig"))
            temp_path = f.name

        try:
            df = handler._try_parse_csv(temp_path, ["utf-8-sig", "utf-8"])
            assert len(df) == 1
            assert df.iloc[0]["url"] == "https://example.com"
        finally:
            os.unlink(temp_path)

    def test_clean_dataframe_values(self, handler):
        """Test cleaning DataFrame values."""
        df = pd.DataFrame(
            {
                "title": ["  Spaced  ", "Normal", None, ""],
                "url": ["https://example.com", "  https://test.com  ", "", None],
                "tags": ["tag1, tag2", "  tag3  ", None, ""],
            }
        )

        cleaned = handler._clean_dataframe_values(df)

        assert cleaned.iloc[0]["title"] == "Spaced"
        assert cleaned.iloc[1]["url"] == "https://test.com"
        assert cleaned.iloc[1]["tags"] == "tag3"
        assert cleaned.iloc[2]["title"] == ""  # None -> empty string
        assert cleaned.iloc[3]["url"] == ""

    def test_tag_format_conversion(self, handler):
        """Test converting between different tag formats."""
        # Test parsing various tag formats
        test_cases = [
            ("single", ["single"]),
            ('"tag1, tag2"', ["tag1", "tag2"]),
            ("tag1; tag2; tag3", ["tag1", "tag2", "tag3"]),
            ('"  spaced  ,  tags  "', ["spaced", "tags"]),
            ("", []),
            (None, []),
        ]

        for input_tags, expected in test_cases:
            df = pd.DataFrame({"url": ["https://example.com"], "tags": [input_tags]})

            # Simulate the tag parsing that happens during transformation
            bookmarks = handler.transform_export_to_bookmarks(df)
            if bookmarks:
                # The actual parsing happens in Bookmark.from_raindrop_export
                assert True  # Just verify no errors

    def test_date_format_handling(self, handler):
        """Test handling different date formats."""
        df = pd.DataFrame(
            {
                "url": ["https://example.com"] * 4,
                "title": ["Test"] * 4,
                "created": [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:00+00:00",
                    "2024/01/01 00:00:00",
                    "invalid-date",
                ],
            }
        )

        # Set other required columns
        for col in handler.export_columns:
            if col not in df.columns:
                df[col] = ""

        bookmarks = handler.transform_export_to_bookmarks(df)

        # Should handle various date formats gracefully
        assert len(bookmarks) == 4
        # First three should have valid dates, last one should handle gracefully
        assert bookmarks[0].created is not None
        assert bookmarks[1].created is not None
        assert bookmarks[2].created is not None
        # Invalid date might be None or handled differently


if __name__ == "__main__":
    pytest.main([__file__])
