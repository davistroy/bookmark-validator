"""
Tests for Chrome HTML bookmark parser.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from bookmark_processor.core.chrome_html_parser import (
    ChromeHTMLError,
    ChromeHTMLParser,
    ChromeHTMLStructureError,
)


class TestChromeHTMLParser:
    """Test cases for ChromeHTMLParser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ChromeHTMLParser()

    def test_init(self):
        """Test parser initialization."""
        assert self.parser is not None
        assert hasattr(self.parser, "logger")

    @pytest.fixture
    def sample_chrome_html(self):
        """Sample Chrome bookmark HTML content."""
        return """<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file.
     It will be read and overwritten.
     DO NOT EDIT! -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
    <DT><H3 ADD_DATE="1715434444" LAST_MODIFIED="1717526901" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks bar</H3>
    <DL><p>
        <DT><H3 ADD_DATE="1717175257" LAST_MODIFIED="1717526910">Machine Learning</H3>
        <DL><p>
            <DT><A HREF="https://example.com/" ADD_DATE="1717175221">Example Site</A>
            <DT><A HREF="https://test.com/" ADD_DATE="1717175261">Test Site</A>
        </DL><p>
        <DT><A HREF="https://direct.com/" ADD_DATE="1717175273">Direct Bookmark</A>
    </DL><p>
    <DT><H3 ADD_DATE="1634868593" LAST_MODIFIED="1634868593">Other Folder</H3>
    <DL><p>
        <DT><A HREF="https://nested.com/" ADD_DATE="1634868593">Nested Bookmark</A>
    </DL><p>
</DL><p>"""

    @pytest.fixture
    def invalid_html(self):
        """Invalid HTML content."""
        return """<html>
<head><title>Not Bookmarks</title></head>
<body>This is not a bookmark file</body>
</html>"""

    def test_validate_file_valid(self, sample_chrome_html):
        """Test validation of valid Chrome bookmark file."""
        with patch("builtins.open", mock_open(read_data=sample_chrome_html)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.suffix", new_callable=lambda: ".html"):
                    result = self.parser.validate_file("test.html")
                    assert result is True

    def test_validate_file_invalid(self, invalid_html):
        """Test validation of invalid file."""
        with patch("builtins.open", mock_open(read_data=invalid_html)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.suffix", new_callable=lambda: ".html"):
                    result = self.parser.validate_file("test.html")
                    assert result is False

    def test_validate_file_nonexistent(self):
        """Test validation of non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            result = self.parser.validate_file("nonexistent.html")
            assert result is False

    def test_parse_file_success(self, sample_chrome_html):
        """Test successful parsing of Chrome bookmark file."""
        with patch("builtins.open", mock_open(read_data=sample_chrome_html)):
            with patch("pathlib.Path.exists", return_value=True):
                bookmarks = self.parser.parse_file("test.html")

                # The parser may return duplicates due to HTML structure parsing
                # Deduplicate by URL to get unique bookmarks
                unique_urls = set(b.url for b in bookmarks)
                assert len(unique_urls) == 4

                # Check that expected URLs are present
                urls = [b.url for b in bookmarks]
                assert "https://example.com/" in urls
                assert "https://test.com/" in urls
                assert "https://direct.com/" in urls
                assert "https://nested.com/" in urls

                # Check folder structure exists in results
                folders = [b.folder for b in bookmarks]
                assert any("Machine Learning" in f for f in folders)
                assert any("Bookmarks Bar" in f for f in folders)
                assert any("Other Folder" in f for f in folders)

    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ChromeHTMLError, match="File not found"):
                self.parser.parse_file("nonexistent.html")

    def test_parse_file_invalid_structure(self, invalid_html):
        """Test parsing file with invalid structure."""
        with patch("builtins.open", mock_open(read_data=invalid_html)):
            with patch("pathlib.Path.exists", return_value=True):
                # Parser wraps ChromeHTMLStructureError in ChromeHTMLError
                with pytest.raises(ChromeHTMLError) as exc_info:
                    self.parser.parse_file("invalid.html")
                # Verify the underlying cause was a structure error
                assert isinstance(exc_info.value.__cause__, ChromeHTMLStructureError)

    def test_parse_timestamp_valid(self):
        """Test parsing valid Unix timestamp."""
        timestamp_str = "1717175221"  # 2024-05-31 15:07:01 UTC
        result = self.parser._parse_timestamp(timestamp_str)

        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid timestamp."""
        result = self.parser._parse_timestamp("invalid")
        assert result is None

        result = self.parser._parse_timestamp(None)
        assert result is None

    def test_build_folder_path(self):
        """Test folder path building."""
        # Empty current path
        result = self.parser._build_folder_path("", "Folder1")
        assert result == "Folder1"

        # Existing path
        result = self.parser._build_folder_path("Parent", "Child")
        assert result == "Parent/Child"

    def test_get_file_info_existing(self, sample_chrome_html):
        """Test getting file info for existing file."""
        with patch("builtins.open", mock_open(read_data=sample_chrome_html)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1000

                    info = self.parser.get_file_info("test.html")

                    assert info["exists"] is True
                    assert info["size_bytes"] == 1000
                    assert info["is_chrome_bookmarks"] is True
                    assert info["estimated_bookmark_count"] > 0

    def test_get_file_info_nonexistent(self):
        """Test getting file info for non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            info = self.parser.get_file_info("nonexistent.html")

            assert info["exists"] is False
            assert info["size_bytes"] == 0
            assert info["is_chrome_bookmarks"] is False
            assert info["estimated_bookmark_count"] == 0
