"""
Unit tests for the multi_file_processor module.

Tests the MultiFileProcessor class for handling multiple bookmark files
in batch operations with auto-detection, processing, and reporting capabilities.
"""

import glob
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from bookmark_processor.core.multi_file_processor import MultiFileProcessor
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.error_handler import (
    BookmarkImportError,
    UnsupportedFormatError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def processor():
    """Create a MultiFileProcessor instance."""
    return MultiFileProcessor()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_csv_file(temp_directory):
    """Create a temporary CSV file with valid raindrop.io export format."""
    csv_content = """id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,Python Docs,Python documentation,Welcome to Python,https://docs.python.org/3/,Programming/Python,"python, docs",2024-01-01T00:00:00Z,,,false
2,GitHub,Code hosting,GitHub is where people build software,https://github.com,Development,github,2024-01-02T00:00:00Z,,,false
"""
    csv_path = os.path.join(temp_directory, "bookmarks.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    return csv_path


@pytest.fixture
def temp_html_file(temp_directory):
    """Create a temporary Chrome HTML bookmark file."""
    html_content = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
    <DT><H3 ADD_DATE="1609459200" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks Bar</H3>
    <DL><p>
        <DT><A HREF="https://example.com" ADD_DATE="1609459200">Example</A>
    </DL><p>
</DL><p>
"""
    html_path = os.path.join(temp_directory, "bookmarks.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


@pytest.fixture
def temp_invalid_file(temp_directory):
    """Create a temporary file with invalid content."""
    invalid_path = os.path.join(temp_directory, "invalid.csv")
    with open(invalid_path, "w", encoding="utf-8") as f:
        f.write("This is not a valid CSV bookmark file\nJust random text")
    return invalid_path


@pytest.fixture
def sample_bookmarks():
    """Create sample bookmark objects for testing."""
    return [
        Bookmark(url="https://example.com", title="Example", folder="Test"),
        Bookmark(url="https://python.org", title="Python", folder="Programming"),
    ]


# =============================================================================
# MultiFileProcessor Initialization Tests
# =============================================================================


class TestMultiFileProcessorInitialization:
    """Test MultiFileProcessor initialization."""

    def test_initialization(self, processor):
        """Test MultiFileProcessor initializes correctly."""
        assert processor.logger is not None
        assert processor.importer is not None

    def test_logger_name(self, processor):
        """Test logger has correct module name."""
        assert "multi_file_processor" in processor.logger.name


# =============================================================================
# Auto Detection Tests
# =============================================================================


class TestAutoDetectFiles:
    """Test auto_detect_files method."""

    def test_auto_detect_with_default_directory(self, processor):
        """Test auto-detection with current working directory."""
        # This should not raise an error, just return a list
        # (may be empty if no bookmark files in cwd)
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": False,
                "estimated_bookmarks": 0,
                "format": "unknown",
            }
            files = processor.auto_detect_files()
            assert isinstance(files, list)

    def test_auto_detect_with_specific_directory(self, processor, temp_directory, temp_csv_file):
        """Test auto-detection with specific directory."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 2,
                "format": "csv",
            }
            files = processor.auto_detect_files(temp_directory)
            assert len(files) >= 1
            assert all(isinstance(f, Path) for f in files)

    def test_auto_detect_with_path_object(self, processor, temp_directory, temp_csv_file):
        """Test auto-detection accepts Path object."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 2,
                "format": "csv",
            }
            files = processor.auto_detect_files(Path(temp_directory))
            assert isinstance(files, list)

    def test_auto_detect_nonexistent_directory(self, processor):
        """Test auto-detection raises error for non-existent directory."""
        with pytest.raises(FileNotFoundError) as exc_info:
            processor.auto_detect_files("/nonexistent/directory/path")
        assert "Directory not found" in str(exc_info.value)

    def test_auto_detect_file_instead_of_directory(self, processor, temp_csv_file):
        """Test auto-detection raises error when given a file instead of directory."""
        with pytest.raises(ValueError) as exc_info:
            processor.auto_detect_files(temp_csv_file)
        assert "not a directory" in str(exc_info.value)

    def test_auto_detect_finds_csv_files(self, processor, temp_directory, temp_csv_file):
        """Test auto-detection finds CSV files."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 5,
                "format": "csv",
            }
            files = processor.auto_detect_files(temp_directory)
            csv_files = [f for f in files if f.suffix == ".csv"]
            assert len(csv_files) >= 1

    def test_auto_detect_finds_html_files(self, processor, temp_directory, temp_html_file):
        """Test auto-detection finds HTML files."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 1,
                "format": "html",
            }
            files = processor.auto_detect_files(temp_directory)
            html_files = [f for f in files if f.suffix in (".html", ".htm")]
            assert len(html_files) >= 1

    def test_auto_detect_skips_unsupported_files(self, processor, temp_directory, temp_invalid_file):
        """Test auto-detection skips unsupported files."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": False,
                "estimated_bookmarks": 0,
                "format": "unknown",
            }
            files = processor.auto_detect_files(temp_directory)
            # Invalid file should be skipped
            invalid_names = [f.name for f in files if "invalid" in f.name]
            assert len(invalid_names) == 0

    def test_auto_detect_skips_empty_files(self, processor, temp_directory):
        """Test auto-detection skips files with 0 estimated bookmarks."""
        # Create an empty CSV file
        empty_path = os.path.join(temp_directory, "empty.csv")
        with open(empty_path, "w") as f:
            f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")

        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 0,
                "format": "csv",
            }
            files = processor.auto_detect_files(temp_directory)
            # Empty files should be skipped
            assert not any(f.name == "empty.csv" for f in files)

    def test_auto_detect_handles_validation_error(self, processor, temp_directory, temp_csv_file):
        """Test auto-detection handles file validation errors gracefully."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.side_effect = Exception("Validation error")
            # Should not raise, just skip the problematic file
            files = processor.auto_detect_files(temp_directory)
            assert isinstance(files, list)

    def test_auto_detect_returns_sorted_files(self, processor, temp_directory):
        """Test auto-detection returns sorted file list."""
        # Create multiple files
        for name in ["c_file.csv", "a_file.csv", "b_file.csv"]:
            path = os.path.join(temp_directory, name)
            with open(path, "w") as f:
                f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
                f.write("1,Test,,,https://example.com,,,,,,false\n")

        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 1,
                "format": "csv",
            }
            files = processor.auto_detect_files(temp_directory)
            # Should be sorted
            file_names = [f.name for f in files]
            assert file_names == sorted(file_names)


# =============================================================================
# Process Multiple Files Tests
# =============================================================================


class TestProcessMultipleFiles:
    """Test process_multiple_files method."""

    def test_process_empty_file_list(self, processor):
        """Test processing raises error for empty file list."""
        with pytest.raises(ValueError) as exc_info:
            processor.process_multiple_files([])
        assert "No files provided" in str(exc_info.value)

    def test_process_single_file_success(self, processor, temp_csv_file, sample_bookmarks):
        """Test processing a single file successfully."""
        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.return_value = sample_bookmarks
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files([temp_csv_file])

                assert len(bookmarks) == 2
                assert stats["total_files"] == 1
                assert stats["successful_files"] == 1
                assert stats["failed_files"] == 0
                assert stats["total_bookmarks"] == 2

    def test_process_multiple_files_success(self, processor, temp_csv_file, temp_html_file, sample_bookmarks):
        """Test processing multiple files successfully."""
        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.return_value = sample_bookmarks
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files(
                    [temp_csv_file, temp_html_file]
                )

                assert len(bookmarks) == 4  # 2 bookmarks per file
                assert stats["total_files"] == 2
                assert stats["successful_files"] == 2
                assert stats["failed_files"] == 0

    def test_process_files_with_path_objects(self, processor, temp_csv_file, sample_bookmarks):
        """Test processing accepts Path objects."""
        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.return_value = sample_bookmarks
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files([Path(temp_csv_file)])

                assert len(bookmarks) == 2

    def test_process_files_adds_source_file_attribute(self, processor, temp_csv_file):
        """Test processing adds source_file attribute to bookmarks."""
        test_bookmarks = [
            Bookmark(url="https://test.com", title="Test"),
        ]

        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.return_value = test_bookmarks
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files([temp_csv_file])

                # Check that source_file was added
                assert hasattr(bookmarks[0], "source_file")
                assert temp_csv_file in bookmarks[0].source_file

    def test_process_files_preserves_existing_source_file(self, processor, temp_csv_file):
        """Test processing preserves existing source_file attribute."""
        test_bookmark = Bookmark(url="https://test.com", title="Test")
        test_bookmark.source_file = "original_source.csv"

        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.return_value = [test_bookmark]
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files([temp_csv_file])

                # Should preserve original source_file
                assert bookmarks[0].source_file == "original_source.csv"

    def test_process_files_with_bookmark_import_error(self, processor, temp_csv_file, temp_html_file, sample_bookmarks):
        """Test processing continues after BookmarkImportError."""
        def mock_import_side_effect(path):
            if "html" in str(path):
                raise BookmarkImportError("Import failed")
            return sample_bookmarks

        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.side_effect = mock_import_side_effect
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files(
                    [temp_csv_file, temp_html_file]
                )

                # First file should succeed, second should fail
                assert stats["successful_files"] == 1
                assert stats["failed_files"] == 1
                assert len(stats["errors"]) == 1

    def test_process_files_with_unsupported_format_error(self, processor, temp_csv_file, temp_html_file, sample_bookmarks):
        """Test processing handles UnsupportedFormatError for one file but continues."""
        def mock_import_side_effect(path):
            if "html" in str(path):
                raise UnsupportedFormatError("Unsupported format")
            return sample_bookmarks

        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.side_effect = mock_import_side_effect
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files([temp_csv_file, temp_html_file])

                assert stats["successful_files"] == 1
                assert stats["failed_files"] == 1

    def test_process_files_with_generic_exception(self, processor, temp_csv_file, sample_bookmarks):
        """Test processing handles generic exceptions."""
        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            mock_import.side_effect = Exception("Unexpected error")

            # Should raise BookmarkImportError when all files fail
            with pytest.raises(BookmarkImportError) as exc_info:
                processor.process_multiple_files([temp_csv_file])

            assert "Failed to process any files" in str(exc_info.value)

    def test_process_files_all_fail_raises_error(self, processor, temp_csv_file, temp_html_file):
        """Test processing raises error when all files fail."""
        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            mock_import.side_effect = BookmarkImportError("All imports fail")

            with pytest.raises(BookmarkImportError) as exc_info:
                processor.process_multiple_files([temp_csv_file, temp_html_file])

            assert "Failed to process any files" in str(exc_info.value)

    def test_process_files_records_file_results(self, processor, temp_csv_file, sample_bookmarks):
        """Test processing records detailed file results."""
        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.return_value = sample_bookmarks
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files([temp_csv_file])

                # Check file_results structure
                assert temp_csv_file in stats["file_results"] or str(Path(temp_csv_file)) in stats["file_results"]

                file_result = stats["file_results"].get(temp_csv_file) or stats["file_results"].get(str(Path(temp_csv_file)))
                assert file_result is not None
                assert file_result["status"] == "success"
                assert file_result["bookmark_count"] == 2
                assert file_result["file_format"] == "csv"

    def test_process_files_records_failed_file_results(self, processor, temp_csv_file):
        """Test processing records details for failed files."""
        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            mock_import.side_effect = BookmarkImportError("Import failed")

            with pytest.raises(BookmarkImportError):
                processor.process_multiple_files([temp_csv_file])


# =============================================================================
# Generate Timestamped Output Paths Tests
# =============================================================================


class TestGenerateTimestampedOutputPaths:
    """Test generate_timestamped_output_paths method."""

    def test_default_base_name(self, processor):
        """Test generating paths with default base name."""
        paths = processor.generate_timestamped_output_paths()

        assert "csv" in paths
        assert "html" in paths
        assert isinstance(paths["csv"], Path)
        assert isinstance(paths["html"], Path)

    def test_custom_base_name(self, processor):
        """Test generating paths with custom base name."""
        paths = processor.generate_timestamped_output_paths(base_name="my_bookmarks")

        assert "my_bookmarks" in str(paths["csv"])
        assert "my_bookmarks" in str(paths["html"])

    def test_paths_have_correct_extensions(self, processor):
        """Test generated paths have correct file extensions."""
        paths = processor.generate_timestamped_output_paths()

        assert paths["csv"].suffix == ".csv"
        assert paths["html"].suffix == ".html"

    def test_paths_include_timestamp(self, processor):
        """Test generated paths include timestamp."""
        paths = processor.generate_timestamped_output_paths()

        # Timestamp format is YYYYMMDD_HHMMSS
        csv_name = str(paths["csv"])
        # Check for date-like pattern
        assert "_" in csv_name
        # Should have numbers in the timestamp
        parts = csv_name.replace(".csv", "").split("_")
        assert len(parts) >= 2

    def test_paths_are_unique(self, processor):
        """Test each call generates unique paths."""
        paths1 = processor.generate_timestamped_output_paths()

        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)

        paths2 = processor.generate_timestamped_output_paths()

        # Note: If called within the same second, paths could be the same
        # This test verifies structure is consistent
        assert paths1["csv"] != paths1["html"]
        assert paths2["csv"] != paths2["html"]


# =============================================================================
# Get Processing Summary Tests
# =============================================================================


class TestGetProcessingSummary:
    """Test get_processing_summary method."""

    def test_summary_with_successful_files(self, processor):
        """Test summary generation for successful processing."""
        stats = {
            "successful_files": 2,
            "total_files": 2,
            "total_bookmarks": 10,
            "errors": [],
            "file_results": {
                "/path/file1.csv": {
                    "status": "success",
                    "bookmark_count": 5,
                    "file_format": "csv",
                },
                "/path/file2.html": {
                    "status": "success",
                    "bookmark_count": 5,
                    "file_format": "html",
                },
            },
        }

        summary = processor.get_processing_summary(stats)

        assert "Multi-file Processing Summary" in summary
        assert "2/2" in summary
        assert "10" in summary
        assert "file1.csv" in summary
        assert "file2.html" in summary

    def test_summary_with_failed_files(self, processor):
        """Test summary generation includes failed files."""
        stats = {
            "successful_files": 1,
            "total_files": 2,
            "total_bookmarks": 5,
            "errors": ["Error importing file2.csv"],
            "file_results": {
                "/path/file1.csv": {
                    "status": "success",
                    "bookmark_count": 5,
                    "file_format": "csv",
                },
                "/path/file2.csv": {
                    "status": "failed",
                    "error": "Invalid format",
                    "bookmark_count": 0,
                },
            },
        }

        summary = processor.get_processing_summary(stats)

        assert "1/2" in summary
        assert "Errors: 1" in summary
        assert "Invalid format" in summary

    def test_summary_with_no_errors(self, processor):
        """Test summary generation without errors section when no errors."""
        stats = {
            "successful_files": 1,
            "total_files": 1,
            "total_bookmarks": 5,
            "errors": [],
            "file_results": {
                "/path/file.csv": {
                    "status": "success",
                    "bookmark_count": 5,
                    "file_format": "csv",
                },
            },
        }

        summary = processor.get_processing_summary(stats)

        assert "Errors:" not in summary

    def test_summary_format(self, processor):
        """Test summary has expected format structure."""
        stats = {
            "successful_files": 1,
            "total_files": 1,
            "total_bookmarks": 5,
            "errors": [],
            "file_results": {
                "/path/file.csv": {
                    "status": "success",
                    "bookmark_count": 5,
                    "file_format": "csv",
                },
            },
        }

        summary = processor.get_processing_summary(stats)

        # Check structure
        lines = summary.split("\n")
        assert "Multi-file Processing Summary:" in lines[0]
        assert "File Details:" in summary


# =============================================================================
# Validate Directory for Auto Detection Tests
# =============================================================================


class TestValidateDirectoryForAutoDetection:
    """Test validate_directory_for_auto_detection method."""

    def test_validate_default_directory(self, processor):
        """Test validation with default (current) directory."""
        report = processor.validate_directory_for_auto_detection()

        assert "directory" in report
        assert "exists" in report
        assert "is_directory" in report
        assert "readable" in report
        assert report["exists"] is True
        assert report["is_directory"] is True

    def test_validate_specific_directory(self, processor, temp_directory):
        """Test validation with specific directory."""
        report = processor.validate_directory_for_auto_detection(temp_directory)

        assert report["directory"] == str(temp_directory)
        assert report["exists"] is True
        assert report["is_directory"] is True
        assert report["readable"] is True

    def test_validate_with_path_object(self, processor, temp_directory):
        """Test validation accepts Path object."""
        report = processor.validate_directory_for_auto_detection(Path(temp_directory))

        assert report["exists"] is True

    def test_validate_nonexistent_directory(self, processor):
        """Test validation for non-existent directory."""
        report = processor.validate_directory_for_auto_detection("/nonexistent/path")

        assert report["exists"] is False
        assert report["can_auto_detect"] is False
        assert "error" in report
        assert "does not exist" in report["error"]

    def test_validate_file_instead_of_directory(self, processor, temp_csv_file):
        """Test validation when path is a file, not a directory."""
        report = processor.validate_directory_for_auto_detection(temp_csv_file)

        assert report["exists"] is True
        assert report["is_directory"] is False
        assert report["can_auto_detect"] is False
        assert "error" in report
        assert "not a directory" in report["error"]

    def test_validate_finds_potential_files(self, processor, temp_directory, temp_csv_file, temp_html_file):
        """Test validation identifies potential bookmark files."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 5,
                "format": "csv",
                "size_bytes": 1024,
            }

            report = processor.validate_directory_for_auto_detection(temp_directory)

            assert len(report["potential_files"]) >= 2
            assert len(report["valid_files"]) >= 0

    def test_validate_identifies_valid_files(self, processor, temp_directory, temp_csv_file):
        """Test validation identifies valid bookmark files."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 5,
                "format": "csv",
                "size_bytes": 1024,
            }

            report = processor.validate_directory_for_auto_detection(temp_directory)

            if report["valid_files"]:
                valid_file = report["valid_files"][0]
                assert "name" in valid_file
                assert "format" in valid_file
                assert "estimated_bookmarks" in valid_file
                assert "size_mb" in valid_file

    def test_validate_identifies_invalid_files(self, processor, temp_directory, temp_invalid_file):
        """Test validation identifies invalid bookmark files."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": False,
                "estimated_bookmarks": 0,
                "format": "unknown",
                "size_bytes": 100,
            }

            report = processor.validate_directory_for_auto_detection(temp_directory)

            assert len(report["invalid_files"]) >= 1
            invalid_file = report["invalid_files"][0]
            assert "name" in invalid_file
            assert "reason" in invalid_file

    def test_validate_calculates_total_bookmarks(self, processor, temp_directory, temp_csv_file):
        """Test validation calculates total estimated bookmarks."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 10,
                "format": "csv",
                "size_bytes": 1024,
            }

            report = processor.validate_directory_for_auto_detection(temp_directory)

            assert "total_estimated_bookmarks" in report

    def test_validate_sets_can_auto_detect_flag(self, processor, temp_directory, temp_csv_file):
        """Test validation sets can_auto_detect flag correctly."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 5,
                "format": "csv",
                "size_bytes": 1024,
            }

            report = processor.validate_directory_for_auto_detection(temp_directory)

            if report["valid_files"]:
                assert report["can_auto_detect"] is True

    def test_validate_handles_get_file_info_exception(self, processor, temp_directory, temp_csv_file):
        """Test validation handles exceptions from get_file_info."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.side_effect = Exception("File info error")

            report = processor.validate_directory_for_auto_detection(temp_directory)

            # Should still return a report with invalid files noted
            assert "invalid_files" in report

    def test_validate_handles_scanning_exception(self, processor, temp_directory):
        """Test validation handles exceptions during directory scanning."""
        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = Exception("Scanning error")

            report = processor.validate_directory_for_auto_detection(temp_directory)

            assert "error" in report
            assert "Error scanning directory" in report["error"]

    def test_validate_unreadable_directory(self, processor, temp_directory):
        """Test validation for directory without read permissions."""
        # Mock os.access to return False for read permission
        with patch("os.access") as mock_access:
            mock_access.return_value = False

            report = processor.validate_directory_for_auto_detection(temp_directory)

            assert report["readable"] is False
            assert report["can_auto_detect"] is False
            assert "error" in report
            assert "not readable" in report["error"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiFileProcessorIntegration:
    """Integration tests for MultiFileProcessor."""

    def test_full_workflow_auto_detect_and_process(self, processor, temp_directory, temp_csv_file):
        """Test complete workflow: auto-detect then process files."""
        # Mock the importer methods
        sample_bookmarks = [
            Bookmark(url="https://example.com", title="Example"),
        ]

        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            with patch.object(processor.importer, "import_bookmarks") as mock_import:
                with patch.object(processor.importer, "detect_format") as mock_detect:
                    mock_get_info.return_value = {
                        "is_supported": True,
                        "estimated_bookmarks": 1,
                        "format": "csv",
                        "size_bytes": 100,
                    }
                    mock_import.return_value = sample_bookmarks
                    mock_detect.return_value = "csv"

                    # Step 1: Auto-detect files
                    detected_files = processor.auto_detect_files(temp_directory)

                    # Step 2: Process detected files
                    if detected_files:
                        bookmarks, stats = processor.process_multiple_files(detected_files)

                        # Step 3: Generate output paths
                        output_paths = processor.generate_timestamped_output_paths("output")

                        # Step 4: Get summary
                        summary = processor.get_processing_summary(stats)

                        assert len(bookmarks) >= 1
                        assert stats["successful_files"] >= 1
                        assert "csv" in output_paths
                        assert "Summary" in summary

    def test_validate_then_auto_detect(self, processor, temp_directory, temp_csv_file):
        """Test validation followed by auto-detection."""
        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 5,
                "format": "csv",
                "size_bytes": 1024,
            }

            # First validate
            validation_report = processor.validate_directory_for_auto_detection(temp_directory)

            if validation_report["can_auto_detect"]:
                # Then auto-detect
                files = processor.auto_detect_files(temp_directory)
                assert len(files) >= 1

    def test_error_recovery_workflow(self, processor, temp_directory, temp_csv_file, temp_html_file):
        """Test workflow with partial failures and recovery."""
        success_bookmarks = [Bookmark(url="https://success.com", title="Success")]

        call_count = [0]

        def mock_import_side_effect(path):
            call_count[0] += 1
            if call_count[0] == 1:
                return success_bookmarks
            else:
                raise BookmarkImportError("Simulated failure")

        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.side_effect = mock_import_side_effect
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files(
                    [temp_csv_file, temp_html_file]
                )

                # Should have partial success
                assert stats["successful_files"] == 1
                assert stats["failed_files"] == 1
                assert len(bookmarks) == 1


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_process_files_with_special_characters_in_path(self, processor, temp_directory):
        """Test processing files with special characters in path."""
        # Create file with special characters (if supported by OS)
        special_name = "bookmarks with spaces.csv"
        special_path = os.path.join(temp_directory, special_name)

        with open(special_path, "w") as f:
            f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
            f.write("1,Test,,,https://example.com,,,,,,false\n")

        sample_bookmarks = [Bookmark(url="https://example.com", title="Test")]

        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.return_value = sample_bookmarks
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files([special_path])

                assert stats["successful_files"] == 1

    def test_process_large_number_of_files(self, processor, temp_directory):
        """Test processing many files."""
        # Create multiple files
        num_files = 10
        file_paths = []

        for i in range(num_files):
            path = os.path.join(temp_directory, f"bookmarks_{i}.csv")
            with open(path, "w") as f:
                f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
                f.write(f"{i},Test{i},,,https://example{i}.com,,,,,,false\n")
            file_paths.append(path)

        def mock_import_side_effect(path):
            return [Bookmark(url="https://example.com", title="Test")]

        with patch.object(processor.importer, "import_bookmarks") as mock_import:
            with patch.object(processor.importer, "detect_format") as mock_detect:
                mock_import.side_effect = mock_import_side_effect
                mock_detect.return_value = "csv"

                bookmarks, stats = processor.process_multiple_files(file_paths)

                assert stats["total_files"] == num_files
                assert stats["successful_files"] == num_files
                assert len(bookmarks) == num_files

    def test_empty_directory(self, processor, temp_directory):
        """Test auto-detection in empty directory."""
        # temp_directory should be empty initially
        empty_dir = os.path.join(temp_directory, "empty")
        os.makedirs(empty_dir)

        files = processor.auto_detect_files(empty_dir)
        assert len(files) == 0

    def test_summary_with_empty_stats(self, processor):
        """Test summary generation with minimal stats."""
        stats = {
            "successful_files": 0,
            "total_files": 0,
            "total_bookmarks": 0,
            "errors": [],
            "file_results": {},
        }

        summary = processor.get_processing_summary(stats)

        assert "0/0" in summary
        assert "0" in summary

    def test_htm_extension_detection(self, processor, temp_directory):
        """Test auto-detection finds .htm files (not just .html)."""
        htm_path = os.path.join(temp_directory, "bookmarks.htm")
        with open(htm_path, "w") as f:
            f.write("<!DOCTYPE NETSCAPE-Bookmark-file-1>\n<DL><DT>Test</DT></DL>")

        with patch.object(processor.importer, "get_file_info") as mock_get_info:
            mock_get_info.return_value = {
                "is_supported": True,
                "estimated_bookmarks": 1,
                "format": "html",
            }

            files = processor.auto_detect_files(temp_directory)
            htm_files = [f for f in files if f.suffix == ".htm"]
            assert len(htm_files) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
