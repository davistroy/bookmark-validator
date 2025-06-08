"""
Unit tests for data models module.

Tests the Bookmark class and related data structures.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from bookmark_processor.core.data_models import (
    Bookmark,
    BookmarkMetadata,
    ProcessingResults,
    ProcessingStatus,
)
from tests.fixtures.test_data import (
    SAMPLE_RAINDROP_EXPORT_ROWS,
    create_sample_bookmark_objects,
    create_sample_processed_bookmark,
)


class TestBookmarkMetadata:
    """Test BookmarkMetadata class."""

    def test_default_creation(self):
        """Test creating BookmarkMetadata with default values."""
        metadata = BookmarkMetadata()

        assert metadata.title is None
        assert metadata.description is None
        assert metadata.keywords == []
        assert metadata.author is None
        assert metadata.publication_date is None
        assert metadata.canonical_url is None

    def test_creation_with_values(self):
        """Test creating BookmarkMetadata with specific values."""
        keywords = ["python", "programming"]
        pub_date = datetime.now()

        metadata = BookmarkMetadata(
            title="Test Title",
            description="Test Description",
            keywords=keywords,
            author="Test Author",
            publication_date=pub_date,
            canonical_url="https://example.com",
        )

        assert metadata.title == "Test Title"
        assert metadata.description == "Test Description"
        assert metadata.keywords == keywords
        assert metadata.author == "Test Author"
        assert metadata.publication_date == pub_date
        assert metadata.canonical_url == "https://example.com"


class TestProcessingStatus:
    """Test ProcessingStatus class."""

    def test_default_creation(self):
        """Test creating ProcessingStatus with default values."""
        status = ProcessingStatus()

        assert status.url_validated is False
        assert status.url_validation_error is None
        assert status.content_extracted is False
        assert status.content_extraction_error is None
        assert status.ai_processed is False
        assert status.ai_processing_error is None
        assert status.tags_optimized is False
        assert status.processing_attempts == 0
        assert status.last_attempt is None

    def test_creation_with_values(self):
        """Test creating ProcessingStatus with specific values."""
        attempt_time = datetime.now()

        status = ProcessingStatus(
            url_validated=True,
            url_validation_error="Connection timeout",
            content_extracted=True,
            ai_processed=False,
            ai_processing_error="Model not available",
            processing_attempts=3,
            last_attempt=attempt_time,
        )

        assert status.url_validated is True
        assert status.url_validation_error == "Connection timeout"
        assert status.content_extracted is True
        assert status.ai_processed is False
        assert status.ai_processing_error == "Model not available"
        assert status.processing_attempts == 3
        assert status.last_attempt == attempt_time


class TestBookmark:
    """Test Bookmark class."""

    def test_default_creation(self):
        """Test creating Bookmark with default values."""
        bookmark = Bookmark()

        assert bookmark.id is None
        assert bookmark.title == ""
        assert bookmark.note == ""
        assert bookmark.excerpt == ""
        assert bookmark.url == ""
        assert bookmark.folder == ""
        assert bookmark.tags == []
        assert bookmark.created is None
        assert bookmark.enhanced_description == ""
        assert bookmark.optimized_tags == []
        assert bookmark.normalized_url == ""
        assert bookmark.folder_hierarchy == []

    def test_creation_with_url_normalization(self):
        """Test that URL normalization works during creation."""
        bookmark = Bookmark(url="https://example.com/path/")

        assert bookmark.url == "https://example.com/path/"
        assert bookmark.normalized_url == "https://example.com/path"

    def test_creation_with_folder_parsing(self):
        """Test that folder hierarchy parsing works during creation."""
        bookmark = Bookmark(folder="Tech/Programming/Python")

        assert bookmark.folder == "Tech/Programming/Python"
        assert bookmark.folder_hierarchy == ["Tech", "Programming", "Python"]

    def test_url_normalization_edge_cases(self):
        """Test URL normalization with various edge cases."""
        test_cases = [
            ("", ""),
            ("not-a-url", "not-a-url"),
            ("javascript:void(0)", "javascript:void(0)"),
            ("mailto:test@example.com", "mailto:test@example.com"),
            ("example.com", "https://example.com"),
            ("www.example.com", "https://www.example.com"),
            ("https://Example.COM/Path?b=2&a=1", "https://example.com/Path?a=1&b=2"),
            ("https://example.com/path/", "https://example.com/path"),
            ("https://example.com///", "https://example.com"),
        ]

        for input_url, expected in test_cases:
            bookmark = Bookmark(url=input_url)
            assert bookmark.normalized_url == expected, f"Failed for {input_url}"

    def test_folder_parsing_edge_cases(self):
        """Test folder hierarchy parsing with various edge cases."""
        test_cases = [
            ("", []),
            ("SingleFolder", ["SingleFolder"]),
            ("Folder1/Folder2", ["Folder1", "Folder2"]),
            ("  Spaced  /  Folders  ", ["Spaced", "Folders"]),
            ("Folder/", ["Folder"]),
            ("/Folder", ["Folder"]),
            ("//Double//Slash//", ["Double", "Slash"]),
        ]

        for input_folder, expected in test_cases:
            bookmark = Bookmark(folder=input_folder)
            assert bookmark.folder_hierarchy == expected, f"Failed for '{input_folder}'"

    def test_from_raindrop_export(self):
        """Test creating Bookmark from raindrop.io export data."""
        row_data = SAMPLE_RAINDROP_EXPORT_ROWS[0]
        bookmark = Bookmark.from_raindrop_export(row_data)

        assert bookmark.id == "1"
        assert bookmark.title == "Python Documentation"
        assert bookmark.note == "Official Python documentation"
        assert (
            bookmark.excerpt
            == "Welcome to Python.org, the official documentation for Python programming language."
        )
        assert bookmark.url == "https://docs.python.org/3/"
        assert bookmark.folder == "Programming/Python"
        assert bookmark.tags == ["python", "documentation", "programming"]
        assert bookmark.created is not None
        assert bookmark.favorite is False

    def test_from_raindrop_export_with_edge_cases(self):
        """Test from_raindrop_export with edge cases."""
        # Test with empty/missing fields
        row_data = {
            "id": "",
            "title": "",
            "note": "",
            "excerpt": "",
            "url": "",
            "folder": "",
            "tags": "",
            "created": "",
            "cover": "",
            "highlights": "",
            "favorite": "",
        }

        bookmark = Bookmark.from_raindrop_export(row_data)

        assert bookmark.id == ""
        assert bookmark.title == ""
        assert bookmark.tags == []
        assert bookmark.created is None
        assert bookmark.favorite is False

    def test_from_raindrop_export_tag_parsing(self):
        """Test tag parsing from different formats."""
        test_cases = [
            ("", []),
            ("single", ["single"]),
            ("tag1, tag2, tag3", ["tag1", "tag2", "tag3"]),
            ('"quoted, tags, here"', ["quoted", "tags", "here"]),
            ("  spaced  ,  tags  ", ["spaced", "tags"]),
        ]

        for tags_input, expected in test_cases:
            row_data = {**SAMPLE_RAINDROP_EXPORT_ROWS[0], "tags": tags_input}
            bookmark = Bookmark.from_raindrop_export(row_data)
            assert bookmark.tags == expected, f"Failed for '{tags_input}'"

    def test_from_raindrop_export_date_parsing(self):
        """Test date parsing from different formats."""
        test_cases = [
            ("", None),
            ("2024-01-01T00:00:00Z", datetime(2024, 1, 1, tzinfo=None)),
            ("invalid-date", None),
        ]

        for date_input, expected in test_cases:
            row_data = {**SAMPLE_RAINDROP_EXPORT_ROWS[0], "created": date_input}
            bookmark = Bookmark.from_raindrop_export(row_data)
            if expected is None:
                assert bookmark.created is None
            else:
                assert bookmark.created is not None

    def test_from_raindrop_export_favorite_parsing(self):
        """Test favorite parsing from different formats."""
        test_cases = [
            ("", False),
            ("false", False),
            ("true", True),
            ("1", True),
            ("yes", True),
            ("no", False),
            ("invalid", False),
        ]

        for favorite_input, expected in test_cases:
            row_data = {**SAMPLE_RAINDROP_EXPORT_ROWS[0], "favorite": favorite_input}
            bookmark = Bookmark.from_raindrop_export(row_data)
            assert bookmark.favorite == expected, f"Failed for '{favorite_input}'"

    def test_get_folder_path(self):
        """Test getting folder path as string."""
        bookmark = Bookmark(folder="Tech/Programming/Python")
        assert bookmark.get_folder_path() == "Tech/Programming/Python"

        bookmark = Bookmark()
        assert bookmark.get_folder_path() == ""

    def test_get_effective_title(self):
        """Test getting the most appropriate title."""
        # Test with title
        bookmark = Bookmark(title="Test Title", url="https://example.com")
        assert bookmark.get_effective_title() == "Test Title"

        # Test with metadata title
        bookmark = Bookmark(url="https://example.com")
        bookmark.extracted_metadata = BookmarkMetadata(title="Metadata Title")
        assert bookmark.get_effective_title() == "Metadata Title"

        # Test with URL fallback
        bookmark = Bookmark(url="https://example.com/path")
        assert bookmark.get_effective_title() == "example.com"

        # Test with no URL
        bookmark = Bookmark()
        assert bookmark.get_effective_title() == "Untitled Bookmark"

    def test_get_effective_description(self):
        """Test getting the most appropriate description."""
        bookmark = Bookmark()

        # Test with enhanced description
        bookmark.enhanced_description = "Enhanced"
        assert bookmark.get_effective_description() == "Enhanced"

        # Test with note fallback
        bookmark.enhanced_description = ""
        bookmark.note = "Note"
        assert bookmark.get_effective_description() == "Note"

        # Test with excerpt fallback
        bookmark.note = ""
        bookmark.excerpt = "Excerpt"
        assert bookmark.get_effective_description() == "Excerpt"

        # Test with metadata fallback
        bookmark.excerpt = ""
        bookmark.extracted_metadata = BookmarkMetadata(description="Metadata")
        assert bookmark.get_effective_description() == "Metadata"

        # Test with no description
        bookmark.extracted_metadata = None
        assert bookmark.get_effective_description() == ""

    def test_get_all_tags(self):
        """Test getting all available tags."""
        bookmark = Bookmark(tags=["original1", "original2"])
        bookmark.optimized_tags = [
            "optimized1",
            "optimized2",
            "original1",
        ]  # Some overlap
        bookmark.extracted_metadata = BookmarkMetadata(
            keywords=["keyword1", "original1"]
        )

        all_tags = bookmark.get_all_tags()

        # Should be unique, cleaned, and sorted
        expected = ["keyword1", "optimized1", "optimized2", "original1", "original2"]
        assert all_tags == expected

    def test_get_final_tags(self):
        """Test getting final optimized tags for export."""
        bookmark = Bookmark(tags=["original1", "original2", "original3"])

        # Test with optimized tags available
        bookmark.optimized_tags = ["opt1", "opt2", "opt3", "opt4", "opt5"]
        final_tags = bookmark.get_final_tags(max_tags=3)
        assert final_tags == ["opt1", "opt2", "opt3"]

        # Test with original tags fallback
        bookmark.optimized_tags = []
        final_tags = bookmark.get_final_tags(max_tags=2)
        assert final_tags == ["original1", "original2"]

        # Test with tag cleaning
        bookmark.tags = ["  spaced  ", "", "normal", None]
        final_tags = bookmark.get_final_tags()
        assert final_tags == ["spaced", "normal"]

    def test_is_valid(self):
        """Test bookmark validation."""
        # Valid bookmark
        bookmark = Bookmark(url="https://example.com", title="Test")
        assert bookmark.is_valid() is True

        # No URL
        bookmark = Bookmark(title="Test")
        assert bookmark.is_valid() is False

        # No title (but should use fallback)
        bookmark = Bookmark(url="https://example.com")
        assert bookmark.is_valid() is True  # URL provides domain as fallback title

        # Empty URL
        bookmark = Bookmark(url="", title="Test")
        assert bookmark.is_valid() is False

        # Whitespace only
        bookmark = Bookmark(url="   ", title="   ")
        assert bookmark.is_valid() is False

    def test_to_export_dict(self):
        """Test converting bookmark to export dictionary."""
        bookmark = create_sample_processed_bookmark()
        export_dict = bookmark.to_export_dict()

        expected_keys = ["url", "folder", "title", "note", "tags", "created"]
        assert list(export_dict.keys()) == expected_keys

        assert export_dict["url"] == bookmark.url
        assert export_dict["folder"] == bookmark.get_folder_path()
        assert export_dict["title"] == bookmark.get_effective_title()
        assert export_dict["note"] == bookmark.get_effective_description()

        # Test tag formatting (multiple tags should be quoted per CLAUDE.md)
        bookmark.optimized_tags = ["tag1", "tag2", "tag3"]
        export_dict = bookmark.to_export_dict()
        assert export_dict["tags"] == '"tag1, tag2, tag3"'

        # Test single tag (no quotes)
        bookmark.optimized_tags = ["single"]
        export_dict = bookmark.to_export_dict()
        assert export_dict["tags"] == "single"

        # Test no tags
        bookmark.optimized_tags = []
        bookmark.tags = []
        export_dict = bookmark.to_export_dict()
        assert export_dict["tags"] == ""

    def test_to_dict(self):
        """Test converting bookmark to dictionary for serialization."""
        bookmark = create_sample_processed_bookmark()
        bookmark_dict = bookmark.to_dict()

        # Check that all important fields are present
        assert "url" in bookmark_dict
        assert "title" in bookmark_dict
        assert "tags" in bookmark_dict
        assert "enhanced_description" in bookmark_dict
        assert "optimized_tags" in bookmark_dict
        assert "normalized_url" in bookmark_dict
        assert "folder_hierarchy" in bookmark_dict

        # Check date serialization
        if bookmark.created:
            assert isinstance(bookmark_dict["created"], str)

    def test_copy(self):
        """Test copying a bookmark."""
        original = create_sample_processed_bookmark()
        copy = original.copy()

        # Should be equal but different objects
        assert copy.url == original.url
        assert copy.title == original.title
        assert copy.tags == original.tags
        assert copy is not original
        assert copy.tags is not original.tags  # Deep copy of lists


class TestProcessingResults:
    """Test ProcessingResults class."""

    def test_default_creation(self):
        """Test creating ProcessingResults with default values."""
        results = ProcessingResults()

        assert results.total_bookmarks == 0
        assert results.processed_bookmarks == 0
        assert results.valid_bookmarks == 0
        assert results.invalid_bookmarks == 0
        assert results.errors == []
        assert results.processing_time == 0.0

    def test_creation_with_values(self):
        """Test creating ProcessingResults with specific values."""
        errors = ["Error 1", "Error 2"]

        results = ProcessingResults(
            total_bookmarks=100,
            processed_bookmarks=95,
            valid_bookmarks=90,
            invalid_bookmarks=10,
            errors=errors,
            processing_time=120.5,
        )

        assert results.total_bookmarks == 100
        assert results.processed_bookmarks == 95
        assert results.valid_bookmarks == 90
        assert results.invalid_bookmarks == 10
        assert results.errors == errors
        assert results.processing_time == 120.5

    def test_string_representation(self):
        """Test string representation of ProcessingResults."""
        results = ProcessingResults(
            total_bookmarks=100,
            processed_bookmarks=95,
            valid_bookmarks=90,
            invalid_bookmarks=10,
            errors=["Error 1", "Error 2"],
            processing_time=120.5,
        )

        str_repr = str(results)
        assert "total=100" in str_repr
        assert "processed=95" in str_repr
        assert "valid=90" in str_repr
        assert "invalid=10" in str_repr
        assert "errors=2" in str_repr
        assert "time=120.50s" in str_repr

    def test_success_rate(self):
        """Test success rate calculation."""
        results = ProcessingResults(total_bookmarks=100, valid_bookmarks=85)
        assert results.get_success_rate() == 85.0

        # Test with zero total
        results = ProcessingResults(total_bookmarks=0, valid_bookmarks=0)
        assert results.get_success_rate() == 0.0

    def test_url_validation_rate(self):
        """Test URL validation rate calculation."""
        results = ProcessingResults(url_validation_success=80, url_validation_failed=20)
        assert results.get_url_validation_rate() == 80.0

        # Test with zero attempts
        results = ProcessingResults()
        assert results.get_url_validation_rate() == 0.0

    def test_ai_processing_rate(self):
        """Test AI processing rate calculation."""
        results = ProcessingResults(ai_processing_success=70, ai_processing_failed=30)
        assert results.get_ai_processing_rate() == 70.0

        # Test with zero attempts
        results = ProcessingResults()
        assert results.get_ai_processing_rate() == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
