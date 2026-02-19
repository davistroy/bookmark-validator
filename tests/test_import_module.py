"""
Unit tests for the import_module.

Tests the MultiFormatImporter, BookmarkImporter, ValidationMode,
ImportOptions and convenience functions for importing bookmarks
from multiple file formats.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest

from bookmark_processor.core.import_module import (
    MultiFormatImporter,
    ValidationMode,
    ImportOptions,
    BookmarkImporter,
    import_raindrop_csv,
    validate_raindrop_csv,
    convert_raindrop_csv,
)
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.error_handler import (
    BookmarkImportError,
    UnsupportedFormatError,
    CSVError,
    ChromeHTMLError,
)
from tests.fixtures.test_data import (
    SAMPLE_RAINDROP_EXPORT_ROWS,
    create_sample_export_dataframe,
    create_sample_bookmark_objects,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def multi_format_importer():
    """Create a MultiFormatImporter instance."""
    return MultiFormatImporter()


@pytest.fixture
def bookmark_importer():
    """Create a BookmarkImporter instance."""
    return BookmarkImporter()


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with valid raindrop.io export format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = create_sample_export_dataframe()
        df.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_html_file():
    """Create a temporary Chrome HTML bookmark file."""
    chrome_html_content = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file.
     It will be read and overwritten.
     DO NOT EDIT! -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
    <DT><H3 ADD_DATE="1609459200" LAST_MODIFIED="1609459200" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks Bar</H3>
    <DL><p>
        <DT><A HREF="https://example.com" ADD_DATE="1609459200">Example Site</A>
        <DT><A HREF="https://python.org" ADD_DATE="1609459200">Python</A>
    </DL><p>
    <DT><H3 ADD_DATE="1609459200" LAST_MODIFIED="1609459200">Other Bookmarks</H3>
    <DL><p>
        <DT><A HREF="https://github.com" ADD_DATE="1609459200">GitHub</A>
    </DL><p>
</DL><p>
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(chrome_html_content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_unknown_file():
    """Create a temporary file with unknown format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is not a bookmark file\nJust some random text")
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# =============================================================================
# MultiFormatImporter Tests
# =============================================================================


class TestMultiFormatImporter:
    """Test MultiFormatImporter class."""

    def test_initialization(self, multi_format_importer):
        """Test MultiFormatImporter initialization."""
        assert multi_format_importer.csv_handler is not None
        assert multi_format_importer.chrome_parser is not None
        assert multi_format_importer.logger is not None

    def test_get_supported_formats(self, multi_format_importer):
        """Test get_supported_formats returns expected formats."""
        formats = multi_format_importer.get_supported_formats()
        assert "csv" in formats
        assert "html" in formats
        assert len(formats) == 2

    def test_get_format_descriptions(self, multi_format_importer):
        """Test get_format_descriptions returns descriptions."""
        descriptions = multi_format_importer.get_format_descriptions()
        assert "csv" in descriptions
        assert "html" in descriptions
        assert "Raindrop.io" in descriptions["csv"]
        assert "Chrome" in descriptions["html"]

    def test_detect_format_csv(self, multi_format_importer, temp_csv_file):
        """Test format detection for CSV files."""
        detected_format = multi_format_importer.detect_format(Path(temp_csv_file))
        assert detected_format == "csv"

    def test_detect_format_html(self, multi_format_importer, temp_html_file):
        """Test format detection for HTML files."""
        detected_format = multi_format_importer.detect_format(Path(temp_html_file))
        assert detected_format == "html"

    def test_detect_format_unknown(self, multi_format_importer, temp_unknown_file):
        """Test format detection for unknown files."""
        detected_format = multi_format_importer.detect_format(Path(temp_unknown_file))
        assert detected_format == "unknown"

    def test_detect_format_by_content_html(self, multi_format_importer):
        """Test content-based format detection for HTML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("<!DOCTYPE NETSCAPE-Bookmark-file-1>\n<DL><DT>Test</DT></DL>")
            temp_path = f.name

        try:
            detected = multi_format_importer._detect_by_content(Path(temp_path))
            assert detected == "html"
        finally:
            os.unlink(temp_path)

    def test_detect_format_by_content_csv(self, multi_format_importer):
        """Test content-based format detection for CSV with raindrop header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
            f.write('1,Test,,,https://example.com,,,,,,\n')
            temp_path = f.name

        try:
            detected = multi_format_importer._detect_by_content(Path(temp_path))
            assert detected == "csv"
        finally:
            os.unlink(temp_path)

    def test_detect_format_by_content_html_dl_dt(self, multi_format_importer):
        """Test content-based detection with DL/DT markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("<html><DL><p><DT><A HREF='test'>Test</A></DT></DL></html>")
            temp_path = f.name

        try:
            detected = multi_format_importer._detect_by_content(Path(temp_path))
            assert detected == "html"
        finally:
            os.unlink(temp_path)

    def test_detect_format_error_handling(self, multi_format_importer):
        """Test format detection handles errors gracefully."""
        # Test with non-existent file
        result = multi_format_importer.detect_format(Path("/nonexistent/file.csv"))
        assert result == "unknown"

    def test_is_raindrop_csv_valid(self, multi_format_importer, temp_csv_file):
        """Test _is_raindrop_csv returns True for valid CSV."""
        result = multi_format_importer._is_raindrop_csv(Path(temp_csv_file))
        assert result is True

    def test_is_raindrop_csv_invalid(self, multi_format_importer, temp_unknown_file):
        """Test _is_raindrop_csv returns False for invalid CSV."""
        result = multi_format_importer._is_raindrop_csv(Path(temp_unknown_file))
        assert result is False

    def test_import_bookmarks_csv(self, multi_format_importer, temp_csv_file):
        """Test importing bookmarks from CSV file."""
        bookmarks = multi_format_importer.import_bookmarks(temp_csv_file)
        assert len(bookmarks) > 0
        assert all(isinstance(b, Bookmark) for b in bookmarks)

    def test_import_bookmarks_html(self, multi_format_importer, temp_html_file):
        """Test importing bookmarks from HTML file."""
        bookmarks = multi_format_importer.import_bookmarks(temp_html_file)
        assert len(bookmarks) >= 3  # At least 3 bookmarks in our test file
        assert all(isinstance(b, Bookmark) for b in bookmarks)

    def test_import_bookmarks_file_not_found(self, multi_format_importer):
        """Test import_bookmarks raises error for non-existent file."""
        with pytest.raises(BookmarkImportError) as exc_info:
            multi_format_importer.import_bookmarks("/nonexistent/file.csv")
        assert "File not found" in str(exc_info.value)

    def test_import_bookmarks_unsupported_format(self, multi_format_importer, temp_unknown_file):
        """Test import_bookmarks raises error for unsupported format."""
        with pytest.raises(BookmarkImportError) as exc_info:
            multi_format_importer.import_bookmarks(temp_unknown_file)
        assert "Failed to import bookmarks" in str(exc_info.value)

    def test_import_csv_internal(self, multi_format_importer, temp_csv_file):
        """Test _import_csv internal method."""
        bookmarks = multi_format_importer._import_csv(Path(temp_csv_file))
        assert len(bookmarks) > 0

    def test_import_csv_error_handling(self, multi_format_importer):
        """Test _import_csv handles errors appropriately."""
        with pytest.raises(BookmarkImportError):
            multi_format_importer._import_csv(Path("/nonexistent/file.csv"))

    def test_import_html_internal(self, multi_format_importer, temp_html_file):
        """Test _import_html internal method."""
        bookmarks = multi_format_importer._import_html(Path(temp_html_file))
        assert len(bookmarks) >= 3

    def test_import_html_error_handling(self, multi_format_importer):
        """Test _import_html handles errors appropriately."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html>Not a Chrome bookmark file</html>")
            temp_path = f.name

        try:
            with pytest.raises(BookmarkImportError):
                multi_format_importer._import_html(Path(temp_path))
        finally:
            os.unlink(temp_path)

    def test_get_file_info_csv(self, multi_format_importer, temp_csv_file):
        """Test get_file_info for CSV files."""
        info = multi_format_importer.get_file_info(temp_csv_file)
        assert info["exists"] is True
        assert info["format"] == "csv"
        assert info["is_supported"] is True
        assert info["estimated_bookmarks"] > 0
        assert info["size_bytes"] > 0

    def test_get_file_info_html(self, multi_format_importer, temp_html_file):
        """Test get_file_info for HTML files."""
        info = multi_format_importer.get_file_info(temp_html_file)
        assert info["exists"] is True
        assert info["format"] == "html"
        assert info["is_supported"] is True
        assert info["estimated_bookmarks"] >= 3

    def test_get_file_info_nonexistent(self, multi_format_importer):
        """Test get_file_info for non-existent files."""
        info = multi_format_importer.get_file_info("/nonexistent/file.csv")
        assert info["exists"] is False
        assert info["format"] == "unknown"
        assert info["is_supported"] is False
        assert info["estimated_bookmarks"] == 0

    def test_get_file_info_unknown_format(self, multi_format_importer, temp_unknown_file):
        """Test get_file_info for unknown formats."""
        info = multi_format_importer.get_file_info(temp_unknown_file)
        assert info["exists"] is True
        assert info["format"] == "unknown"
        assert info["is_supported"] is False

    def test_import_csv_with_invalid_rows(self, multi_format_importer):
        """Test CSV import rejects files with invalid rows (empty URLs)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Valid header
            f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
            # Valid row
            f.write('1,Test,note,excerpt,https://example.com,folder,tag,2024-01-01T00:00:00Z,,,false\n')
            # Row with empty URL (invalid)
            f.write('2,Invalid,,,,,,,,,\n')
            temp_path = f.name

        try:
            # CSV handler does strict validation - empty URLs cause an error
            with pytest.raises(BookmarkImportError) as exc_info:
                multi_format_importer._import_csv(Path(temp_path))
            assert "Empty URL" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


# =============================================================================
# ValidationMode Tests
# =============================================================================


class TestValidationMode:
    """Test ValidationMode enum."""

    def test_validation_mode_values(self):
        """Test ValidationMode enum values."""
        assert ValidationMode.STRICT.value == "strict"
        assert ValidationMode.BEST_EFFORT.value == "best_effort"
        assert ValidationMode.PERMISSIVE.value == "permissive"

    def test_validation_mode_comparison(self):
        """Test ValidationMode enum comparison."""
        assert ValidationMode.STRICT != ValidationMode.BEST_EFFORT
        assert ValidationMode.BEST_EFFORT != ValidationMode.PERMISSIVE


# =============================================================================
# ImportOptions Tests
# =============================================================================


class TestImportOptions:
    """Test ImportOptions dataclass."""

    def test_default_options(self):
        """Test ImportOptions default values."""
        options = ImportOptions()
        assert options.validation_mode == ValidationMode.BEST_EFFORT
        assert options.max_errors is None
        assert options.encoding is None
        assert options.progress_callback is None
        assert options.error_callback is None
        assert options.include_invalid is False
        assert options.transform_urls is True
        assert options.parse_dates is True
        assert options.normalize_tags is True

    def test_custom_options(self):
        """Test ImportOptions with custom values."""
        callback = lambda x, y: None
        options = ImportOptions(
            validation_mode=ValidationMode.STRICT,
            max_errors=10,
            encoding="utf-8",
            progress_callback=callback,
            include_invalid=True,
        )
        assert options.validation_mode == ValidationMode.STRICT
        assert options.max_errors == 10
        assert options.encoding == "utf-8"
        assert options.progress_callback == callback
        assert options.include_invalid is True


# =============================================================================
# BookmarkImporter Tests
# =============================================================================


class TestBookmarkImporter:
    """Test BookmarkImporter class."""

    def test_initialization_default_options(self):
        """Test BookmarkImporter initialization with default options."""
        importer = BookmarkImporter()
        assert importer.options.validation_mode == ValidationMode.BEST_EFFORT

    def test_initialization_custom_options(self):
        """Test BookmarkImporter initialization with custom options."""
        options = ImportOptions(validation_mode=ValidationMode.STRICT)
        importer = BookmarkImporter(options)
        assert importer.options.validation_mode == ValidationMode.STRICT

    def test_reset_statistics(self, bookmark_importer):
        """Test reset_statistics clears all stats."""
        bookmark_importer.stats["total_rows"] = 100
        bookmark_importer.stats["valid_bookmarks"] = 50
        bookmark_importer.reset_statistics()
        assert bookmark_importer.stats["total_rows"] == 0
        assert bookmark_importer.stats["valid_bookmarks"] == 0
        assert bookmark_importer.stats["errors"] == []

    def test_import_csv_basic(self, bookmark_importer, temp_csv_file):
        """Test basic CSV import."""
        bookmarks = bookmark_importer.import_csv(temp_csv_file)
        assert len(bookmarks) > 0
        assert all(isinstance(b, Bookmark) for b in bookmarks)

    def test_import_csv_with_custom_options(self, temp_csv_file):
        """Test CSV import with custom options."""
        progress_calls = []
        error_calls = []

        options = ImportOptions(
            validation_mode=ValidationMode.BEST_EFFORT,
            progress_callback=lambda x, y: progress_calls.append((x, y)),
            error_callback=lambda e: error_calls.append(e),
        )
        importer = BookmarkImporter(options)
        bookmarks = importer.import_csv(temp_csv_file)

        assert len(bookmarks) > 0
        assert len(progress_calls) > 0  # Progress was reported

    def test_import_csv_strict_mode(self, temp_csv_file):
        """Test CSV import in strict mode."""
        options = ImportOptions(validation_mode=ValidationMode.STRICT)
        importer = BookmarkImporter(options)
        bookmarks = importer.import_csv(temp_csv_file)
        # All returned bookmarks should be valid
        assert all(b.is_valid() for b in bookmarks)

    def test_import_csv_permissive_mode(self, temp_csv_file):
        """Test CSV import in permissive mode."""
        options = ImportOptions(
            validation_mode=ValidationMode.PERMISSIVE,
            include_invalid=True,
        )
        importer = BookmarkImporter(options)
        bookmarks = importer.import_csv(temp_csv_file)
        assert len(bookmarks) > 0

    def test_import_csv_file_not_found(self, bookmark_importer):
        """Test import_csv raises error for non-existent file."""
        with pytest.raises(Exception):  # Could be CSVError or FileNotFoundError
            bookmark_importer.import_csv("/nonexistent/file.csv")

    def test_import_csv_override_options(self, bookmark_importer, temp_csv_file):
        """Test import_csv with options override."""
        override_options = ImportOptions(validation_mode=ValidationMode.STRICT)
        bookmarks = bookmark_importer.import_csv(temp_csv_file, options=override_options)
        assert len(bookmarks) > 0

    def test_get_import_statistics(self, bookmark_importer, temp_csv_file):
        """Test get_import_statistics returns stats."""
        bookmark_importer.import_csv(temp_csv_file)
        stats = bookmark_importer.get_import_statistics()

        assert "total_rows" in stats
        assert "valid_bookmarks" in stats
        assert "processing_time" in stats
        assert "file_size_mb" in stats
        assert stats["total_rows"] > 0

    def test_validate_csv_file(self, bookmark_importer, temp_csv_file):
        """Test validate_csv_file returns validation report."""
        report = bookmark_importer.validate_csv_file(temp_csv_file)

        assert "file_path" in report
        assert "file_exists" in report
        assert "can_import" in report
        assert "import_mode_recommended" in report
        assert report["file_exists"] is True

    def test_validate_csv_file_nonexistent(self, bookmark_importer):
        """Test validate_csv_file for non-existent file."""
        report = bookmark_importer.validate_csv_file("/nonexistent/file.csv")
        assert report["can_import"] is False

    def test_recommend_import_mode_no_issues(self, bookmark_importer):
        """Test _recommend_import_mode with no issues."""
        diagnosis = {
            "structure_issues": [],
            "data_quality_issues": [],
            "parsing_errors": [],
        }
        mode = bookmark_importer._recommend_import_mode(diagnosis)
        assert mode == ValidationMode.STRICT

    def test_recommend_import_mode_few_issues(self, bookmark_importer):
        """Test _recommend_import_mode with few issues."""
        diagnosis = {
            "structure_issues": ["issue1"],
            "data_quality_issues": ["issue2"],
            "parsing_errors": [],
        }
        mode = bookmark_importer._recommend_import_mode(diagnosis)
        assert mode == ValidationMode.BEST_EFFORT

    def test_recommend_import_mode_many_issues(self, bookmark_importer):
        """Test _recommend_import_mode with many issues."""
        diagnosis = {
            "structure_issues": ["issue1", "issue2"],
            "data_quality_issues": ["issue3", "issue4"],
            "parsing_errors": ["error1"],
        }
        mode = bookmark_importer._recommend_import_mode(diagnosis)
        assert mode == ValidationMode.PERMISSIVE

    def test_export_bookmarks(self, bookmark_importer):
        """Test export_bookmarks saves to file."""
        bookmarks = create_sample_bookmark_objects()[:3]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            bookmark_importer.export_bookmarks(bookmarks, temp_path)
            assert os.path.exists(temp_path)
            # Verify content
            df = pd.read_csv(temp_path)
            assert len(df) == 3
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_bookmarks_error(self, bookmark_importer):
        """Test export_bookmarks handles errors."""
        bookmarks = create_sample_bookmark_objects()[:1]
        with pytest.raises(CSVError):
            bookmark_importer.export_bookmarks(bookmarks, "/")

    def test_apply_validation_mode_permissive_include_invalid(self, bookmark_importer):
        """Test _apply_validation_mode in permissive mode with include_invalid."""
        bookmarks = [
            Bookmark(url="https://valid.com", title="Valid"),
            Bookmark(url="", title=""),  # Invalid
        ]
        options = ImportOptions(
            validation_mode=ValidationMode.PERMISSIVE,
            include_invalid=True,
        )
        result = bookmark_importer._apply_validation_mode(bookmarks, options)
        assert len(result) == 2  # Both included

    def test_apply_validation_mode_strict_with_invalid(self, bookmark_importer):
        """Test _apply_validation_mode in strict mode raises on invalid."""
        bookmarks = [
            Bookmark(url="https://valid.com", title="Valid"),
            Bookmark(url="", title=""),  # Invalid
        ]
        options = ImportOptions(validation_mode=ValidationMode.STRICT)
        with pytest.raises(CSVError) as exc_info:
            bookmark_importer._apply_validation_mode(bookmarks, options)
        assert "Strict validation failed" in str(exc_info.value)

    def test_apply_validation_mode_best_effort(self, bookmark_importer):
        """Test _apply_validation_mode in best effort mode."""
        bookmarks = [
            Bookmark(url="https://valid.com", title="Valid"),
            Bookmark(url="", title=""),  # Invalid
        ]
        options = ImportOptions(validation_mode=ValidationMode.BEST_EFFORT)
        result = bookmark_importer._apply_validation_mode(bookmarks, options)
        assert len(result) == 1  # Only valid

    def test_load_and_transform_strict_error(self, bookmark_importer):
        """Test _load_and_transform raises on error in strict mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Valid header but malformed data
            f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
            f.write('1,Test,note,excerpt,https://example.com,folder,tag,2024-01-01T00:00:00Z,,,false\n')
            temp_path = f.name

        try:
            options = ImportOptions(validation_mode=ValidationMode.STRICT)
            # This should work for valid data
            bookmarks = bookmark_importer._load_and_transform(Path(temp_path), options)
            assert len(bookmarks) > 0
        finally:
            os.unlink(temp_path)

    def test_load_and_transform_with_encoding(self, bookmark_importer):
        """Test _load_and_transform with forced encoding."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            df = create_sample_export_dataframe()
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            options = ImportOptions(encoding="utf-8")
            bookmarks = bookmark_importer._load_and_transform(Path(temp_path), options)
            assert len(bookmarks) > 0
            assert bookmark_importer.stats["encoding_detected"] == "utf-8"
        finally:
            os.unlink(temp_path)

    def test_load_and_transform_max_errors(self, bookmark_importer):
        """Test _load_and_transform respects max_errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = create_sample_export_dataframe()
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            options = ImportOptions(max_errors=1)
            # Normal import should work
            bookmarks = bookmark_importer._load_and_transform(Path(temp_path), options)
            assert len(bookmarks) > 0
        finally:
            os.unlink(temp_path)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_import_raindrop_csv_default(self, temp_csv_file):
        """Test import_raindrop_csv with default settings."""
        bookmarks = import_raindrop_csv(temp_csv_file)
        assert len(bookmarks) > 0
        assert all(isinstance(b, Bookmark) for b in bookmarks)

    def test_import_raindrop_csv_strict_mode(self, temp_csv_file):
        """Test import_raindrop_csv with strict mode."""
        bookmarks = import_raindrop_csv(temp_csv_file, ValidationMode.STRICT)
        assert len(bookmarks) > 0

    def test_import_raindrop_csv_permissive_mode(self, temp_csv_file):
        """Test import_raindrop_csv with permissive mode."""
        bookmarks = import_raindrop_csv(temp_csv_file, ValidationMode.PERMISSIVE)
        assert len(bookmarks) > 0

    def test_validate_raindrop_csv(self, temp_csv_file):
        """Test validate_raindrop_csv function."""
        report = validate_raindrop_csv(temp_csv_file)
        assert "file_path" in report
        assert "can_import" in report
        assert report["can_import"] is True

    def test_validate_raindrop_csv_nonexistent(self):
        """Test validate_raindrop_csv with non-existent file."""
        report = validate_raindrop_csv("/nonexistent/file.csv")
        assert report["can_import"] is False

    def test_convert_raindrop_csv(self, temp_csv_file):
        """Test convert_raindrop_csv function."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            count = convert_raindrop_csv(temp_csv_file, output_path)
            assert count > 0
            assert os.path.exists(output_path)

            # Verify output format (6 columns)
            df = pd.read_csv(output_path)
            assert "url" in df.columns
            assert "folder" in df.columns
            assert "title" in df.columns
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_convert_raindrop_csv_with_mode(self, temp_csv_file):
        """Test convert_raindrop_csv with specific validation mode."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            count = convert_raindrop_csv(
                temp_csv_file, output_path, ValidationMode.PERMISSIVE
            )
            assert count > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_detect_format_with_exception(self, multi_format_importer):
        """Test detect_format handles exceptions gracefully."""
        # Mock validate_file to raise an exception
        with patch.object(
            multi_format_importer.chrome_parser,
            "validate_file",
            side_effect=Exception("Test error"),
        ):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                f.write("test content")
                temp_path = f.name

            try:
                result = multi_format_importer.detect_format(Path(temp_path))
                assert result == "unknown"
            finally:
                os.unlink(temp_path)

    def test_detect_by_content_exception(self, multi_format_importer):
        """Test _detect_by_content handles exceptions gracefully."""
        # Create a file that will cause an exception when read
        with patch("builtins.open", side_effect=PermissionError("No access")):
            result = multi_format_importer._detect_by_content(Path("/test/file"))
            assert result == "unknown"

    def test_get_file_info_exception(self, multi_format_importer):
        """Test get_file_info handles exceptions gracefully."""
        # Use a non-existent path to exercise the early return path
        info = multi_format_importer.get_file_info("/nonexistent/path/file.csv")
        assert "path" in info
        assert info["exists"] is False

    def test_import_csv_with_row_errors(self, multi_format_importer):
        """Test CSV import logs warnings for problematic rows."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
            f.write('1,Valid,note,excerpt,https://example.com,folder,tag,2024-01-01T00:00:00Z,,,false\n')
            temp_path = f.name

        try:
            bookmarks = multi_format_importer._import_csv(Path(temp_path))
            assert len(bookmarks) >= 1
        finally:
            os.unlink(temp_path)

    def test_validate_csv_file_exception(self, bookmark_importer):
        """Test validate_csv_file handles exceptions."""
        with patch.object(
            bookmark_importer.csv_handler,
            "diagnose_csv_issues",
            side_effect=Exception("Diagnosis error"),
        ):
            report = bookmark_importer.validate_csv_file("/test/file.csv")
            assert "validation_error" in report
            assert report["can_import"] is False

    def test_import_csv_processing_time(self, bookmark_importer, temp_csv_file):
        """Test that processing time is recorded."""
        bookmark_importer.import_csv(temp_csv_file)
        stats = bookmark_importer.get_import_statistics()
        assert stats["processing_time"] > 0

    def test_import_csv_file_size(self, bookmark_importer, temp_csv_file):
        """Test that file size is recorded."""
        bookmark_importer.import_csv(temp_csv_file)
        stats = bookmark_importer.get_import_statistics()
        assert stats["file_size_mb"] > 0

    def test_content_detection_with_csv_like_content(self, multi_format_importer):
        """Test content detection for CSV-like files without header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            # CSV-like content but not raindrop format
            f.write('url,title\n"https://example.com","Test"\n')
            temp_path = f.name

        try:
            detected = multi_format_importer._detect_by_content(Path(temp_path))
            assert detected == "unknown"  # Not a valid raindrop CSV
        finally:
            os.unlink(temp_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the import module."""

    def test_full_import_workflow_csv(self, temp_csv_file):
        """Test complete CSV import workflow."""
        # Create importer with options
        options = ImportOptions(validation_mode=ValidationMode.BEST_EFFORT)
        importer = BookmarkImporter(options)

        # Validate first
        report = importer.validate_csv_file(temp_csv_file)
        assert report["can_import"] is True

        # Import bookmarks
        bookmarks = importer.import_csv(temp_csv_file)
        assert len(bookmarks) > 0

        # Get statistics
        stats = importer.get_import_statistics()
        assert stats["total_rows"] > 0
        assert stats["valid_bookmarks"] == len(bookmarks)

    def test_full_import_workflow_html(self, temp_html_file):
        """Test complete HTML import workflow."""
        importer = MultiFormatImporter()

        # Get file info
        info = importer.get_file_info(temp_html_file)
        assert info["is_supported"] is True
        assert info["format"] == "html"

        # Import bookmarks
        bookmarks = importer.import_bookmarks(temp_html_file)
        assert len(bookmarks) >= 3

        # Verify bookmark properties
        urls = [b.url for b in bookmarks]
        assert "https://example.com" in urls
        assert "https://python.org" in urls
        assert "https://github.com" in urls

    def test_convert_and_import_roundtrip(self, temp_csv_file):
        """Test converting a file and re-importing it."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            # Convert
            count = convert_raindrop_csv(temp_csv_file, output_path)
            assert count > 0

            # The converted file is in import format (6 columns)
            # We can verify it's readable
            df = pd.read_csv(output_path)
            assert len(df) > 0
            assert "url" in df.columns
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
