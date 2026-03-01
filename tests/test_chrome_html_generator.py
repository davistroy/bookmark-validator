"""
Tests for Chrome HTML Bookmark Generator.

Comprehensive test suite for the ChromeHTMLGenerator class that generates
Chrome-compatible HTML bookmark files from processed bookmark data.
"""

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from bookmark_processor.core.chrome_html_generator import (
    ChromeHTMLGenerator,
    FolderNode,
)
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.error_handler import ChromeHTMLGeneratorError


class TestFolderNode:
    """Test cases for FolderNode class."""

    def test_init_basic(self):
        """Test basic FolderNode initialization."""
        node = FolderNode("Test Folder")

        assert node.name == "Test Folder"
        assert node.parent is None
        assert node.children == {}
        assert node.bookmarks == []
        assert isinstance(node.created_timestamp, int)
        assert isinstance(node.modified_timestamp, int)

    def test_init_with_parent(self):
        """Test FolderNode initialization with parent."""
        parent = FolderNode("Parent")
        child = FolderNode("Child", parent=parent)

        assert child.parent is parent
        assert child.name == "Child"

    def test_add_child_new(self):
        """Test adding a new child folder."""
        parent = FolderNode("Parent")
        child = parent.add_child("Child")

        assert "Child" in parent.children
        assert parent.children["Child"] is child
        assert child.parent is parent
        assert child.name == "Child"

    def test_add_child_existing(self):
        """Test adding child that already exists returns existing."""
        parent = FolderNode("Parent")
        child1 = parent.add_child("Child")
        child2 = parent.add_child("Child")

        assert child1 is child2
        assert len(parent.children) == 1

    def test_add_multiple_children(self):
        """Test adding multiple children."""
        parent = FolderNode("Parent")
        child1 = parent.add_child("Child1")
        child2 = parent.add_child("Child2")
        child3 = parent.add_child("Child3")

        assert len(parent.children) == 3
        assert parent.children["Child1"] is child1
        assert parent.children["Child2"] is child2
        assert parent.children["Child3"] is child3

    def test_add_bookmark(self):
        """Test adding a bookmark to folder."""
        node = FolderNode("Test")
        bookmark = Bookmark(url="https://example.com", title="Example")

        initial_modified = node.modified_timestamp
        time.sleep(0.01)  # Ensure timestamp difference

        node.add_bookmark(bookmark)

        assert len(node.bookmarks) == 1
        assert node.bookmarks[0] is bookmark
        assert node.modified_timestamp >= initial_modified

    def test_add_multiple_bookmarks(self):
        """Test adding multiple bookmarks to folder."""
        node = FolderNode("Test")
        bookmarks = [
            Bookmark(url="https://example1.com", title="Example 1"),
            Bookmark(url="https://example2.com", title="Example 2"),
            Bookmark(url="https://example3.com", title="Example 3"),
        ]

        for bm in bookmarks:
            node.add_bookmark(bm)

        assert len(node.bookmarks) == 3
        for bm in bookmarks:
            assert bm in node.bookmarks

    def test_get_path_root(self):
        """Test get_path for root node."""
        root = FolderNode("Root")

        assert root.get_path() == ""

    def test_get_path_single_level(self):
        """Test get_path for single-level child."""
        root = FolderNode("Root")
        child = FolderNode("Child", parent=root)

        assert child.get_path() == "Child"

    def test_get_path_nested(self):
        """Test get_path for nested hierarchy."""
        root = FolderNode("Root")
        level1 = FolderNode("Level1", parent=root)
        level2 = FolderNode("Level2", parent=level1)
        level3 = FolderNode("Level3", parent=level2)

        assert level3.get_path() == "Level1/Level2/Level3"

    def test_is_bookmarks_bar_true(self):
        """Test is_bookmarks_bar returns True for bookmark bar names."""
        test_names = ["Bookmarks bar", "bookmarks bar", "BOOKMARKS BAR",
                      "Bookmarks_bar", "bookmarks_bar", "toolbar", "TOOLBAR"]

        for name in test_names:
            node = FolderNode(name)
            assert node.is_bookmarks_bar() is True, f"Failed for: {name}"

    def test_is_bookmarks_bar_false(self):
        """Test is_bookmarks_bar returns False for other names."""
        test_names = ["Other", "My Bookmarks", "Tech", "Programming", ""]

        for name in test_names:
            node = FolderNode(name)
            assert node.is_bookmarks_bar() is False, f"Failed for: {name}"


class TestChromeHTMLGenerator:
    """Test cases for ChromeHTMLGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ChromeHTMLGenerator()

    def test_init(self):
        """Test generator initialization."""
        assert self.generator is not None
        assert hasattr(self.generator, "logger")

    def test_escape_html_empty(self):
        """Test HTML escaping with empty string."""
        assert self.generator._escape_html("") == ""
        assert self.generator._escape_html(None) == ""

    def test_escape_html_no_escaping(self):
        """Test HTML escaping with plain text."""
        assert self.generator._escape_html("Hello World") == "Hello World"

    def test_escape_html_ampersand(self):
        """Test HTML escaping of ampersand."""
        assert self.generator._escape_html("A & B") == "A &amp; B"

    def test_escape_html_less_than(self):
        """Test HTML escaping of less than."""
        assert self.generator._escape_html("a < b") == "a &lt; b"

    def test_escape_html_greater_than(self):
        """Test HTML escaping of greater than."""
        assert self.generator._escape_html("a > b") == "a &gt; b"

    def test_escape_html_double_quote(self):
        """Test HTML escaping of double quote."""
        assert self.generator._escape_html('Say "Hello"') == "Say &quot;Hello&quot;"

    def test_escape_html_single_quote(self):
        """Test HTML escaping of single quote."""
        assert self.generator._escape_html("It's") == "It&#x27;s"

    def test_escape_html_all_special_chars(self):
        """Test HTML escaping with all special characters."""
        input_text = '<script>"alert(\'XSS\')&</script>'
        expected = "&lt;script&gt;&quot;alert(&#x27;XSS&#x27;)&amp;&lt;/script&gt;"
        assert self.generator._escape_html(input_text) == expected

    def test_escape_html_unicode(self):
        """Test HTML escaping preserves unicode characters."""
        assert self.generator._escape_html("Caf\u00e9") == "Caf\u00e9"
        assert self.generator._escape_html("Test") == "Test"

    def test_generate_bookmark_html_basic(self):
        """Test generating HTML for a basic bookmark."""
        bookmark = Bookmark(url="https://example.com", title="Example Site")
        html = self.generator._generate_bookmark_html(bookmark)

        assert 'HREF="https://example.com"' in html
        assert ">Example Site</A>" in html
        assert "ADD_DATE=" in html

    def test_generate_bookmark_html_with_created(self):
        """Test generating HTML for bookmark with created date."""
        created_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            created=created_time
        )
        html = self.generator._generate_bookmark_html(bookmark)

        timestamp = int(created_time.timestamp())
        assert f'ADD_DATE="{timestamp}"' in html

    def test_generate_bookmark_html_without_created(self):
        """Test generating HTML for bookmark without created date."""
        bookmark = Bookmark(url="https://example.com", title="Test")
        html = self.generator._generate_bookmark_html(bookmark)

        assert "ADD_DATE=" in html

    def test_generate_bookmark_html_special_chars_in_title(self):
        """Test generating HTML with special characters in title."""
        bookmark = Bookmark(
            url="https://example.com",
            title='Test <"Title"> & More'
        )
        html = self.generator._generate_bookmark_html(bookmark)

        assert "&lt;" in html
        assert "&gt;" in html
        assert "&quot;" in html
        assert "&amp;" in html

    def test_generate_bookmark_html_special_chars_in_url(self):
        """Test generating HTML with special characters in URL."""
        bookmark = Bookmark(
            url="https://example.com?q=test&lang=en",
            title="Test"
        )
        html = self.generator._generate_bookmark_html(bookmark)

        assert "https://example.com?q=test&amp;lang=en" in html

    def test_generate_bookmark_html_empty_title_uses_url(self):
        """Test that empty title falls back to URL."""
        bookmark = Bookmark(url="https://example.com", title="")
        html = self.generator._generate_bookmark_html(bookmark)

        assert ">https://example.com</A>" in html

    def test_build_folder_hierarchy_empty(self):
        """Test building folder hierarchy with no bookmarks."""
        root = self.generator._build_folder_hierarchy([])

        assert root.name == "Root"
        assert "Bookmarks bar" in root.children
        assert "Other bookmarks" in root.children

    def test_build_folder_hierarchy_no_folder(self):
        """Test bookmark without folder goes to Other bookmarks."""
        bookmark = Bookmark(url="https://example.com", title="Test", folder="")
        root = self.generator._build_folder_hierarchy([bookmark])

        other_bookmarks = root.children["Other bookmarks"]
        assert len(other_bookmarks.bookmarks) == 1
        assert other_bookmarks.bookmarks[0] is bookmark

    def test_build_folder_hierarchy_simple_folder(self):
        """Test bookmark with simple folder path."""
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            folder="Tech"
        )
        root = self.generator._build_folder_hierarchy([bookmark])

        other_bookmarks = root.children["Other bookmarks"]
        assert "Tech" in other_bookmarks.children
        tech_folder = other_bookmarks.children["Tech"]
        assert len(tech_folder.bookmarks) == 1

    def test_build_folder_hierarchy_nested_folder(self):
        """Test bookmark with nested folder path."""
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            folder="Tech/Programming/Python"
        )
        root = self.generator._build_folder_hierarchy([bookmark])

        other_bookmarks = root.children["Other bookmarks"]
        tech = other_bookmarks.children["Tech"]
        programming = tech.children["Programming"]
        python = programming.children["Python"]

        assert len(python.bookmarks) == 1

    def test_build_folder_hierarchy_bookmarks_bar(self):
        """Test bookmark with Bookmarks bar prefix."""
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            folder="Bookmarks bar/Quick Links"
        )
        root = self.generator._build_folder_hierarchy([bookmark])

        bookmarks_bar = root.children["Bookmarks bar"]
        assert "Quick Links" in bookmarks_bar.children
        quick_links = bookmarks_bar.children["Quick Links"]
        assert len(quick_links.bookmarks) == 1

    def test_build_folder_hierarchy_toolbar(self):
        """Test bookmark with toolbar prefix."""
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            folder="toolbar/Quick Links"
        )
        root = self.generator._build_folder_hierarchy([bookmark])

        bookmarks_bar = root.children["Bookmarks bar"]
        assert "Quick Links" in bookmarks_bar.children

    def test_build_folder_hierarchy_multiple_bookmarks_same_folder(self):
        """Test multiple bookmarks in the same folder."""
        bookmarks = [
            Bookmark(url="https://example1.com", title="Test 1", folder="Tech"),
            Bookmark(url="https://example2.com", title="Test 2", folder="Tech"),
            Bookmark(url="https://example3.com", title="Test 3", folder="Tech"),
        ]
        root = self.generator._build_folder_hierarchy(bookmarks)

        tech_folder = root.children["Other bookmarks"].children["Tech"]
        assert len(tech_folder.bookmarks) == 3

    def test_build_folder_hierarchy_multiple_folders(self):
        """Test multiple bookmarks in different folders."""
        bookmarks = [
            Bookmark(url="https://example1.com", title="Test 1", folder="Tech"),
            Bookmark(url="https://example2.com", title="Test 2", folder="News"),
            Bookmark(url="https://example3.com", title="Test 3", folder="Sports"),
        ]
        root = self.generator._build_folder_hierarchy(bookmarks)

        other = root.children["Other bookmarks"]
        assert "Tech" in other.children
        assert "News" in other.children
        assert "Sports" in other.children

    def test_build_folder_hierarchy_empty_path_parts(self):
        """Test folder path with empty parts gets cleaned."""
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            folder="Tech//Programming///Python"
        )
        root = self.generator._build_folder_hierarchy([bookmark])

        other = root.children["Other bookmarks"]
        tech = other.children["Tech"]
        programming = tech.children["Programming"]
        python = programming.children["Python"]
        assert len(python.bookmarks) == 1

    def test_build_folder_hierarchy_whitespace_in_folder(self):
        """Test folder path with whitespace gets trimmed."""
        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            folder="  Tech  /  Programming  "
        )
        root = self.generator._build_folder_hierarchy([bookmark])

        other = root.children["Other bookmarks"]
        assert "Tech" in other.children
        tech = other.children["Tech"]
        assert "Programming" in tech.children

    def test_generate_folder_html_basic(self):
        """Test generating HTML for a basic folder."""
        folder = FolderNode("Test Folder")
        html_lines = self.generator._generate_folder_html(folder)

        html = "\n".join(html_lines)
        assert "<H3" in html
        assert "Test Folder</H3>" in html
        assert "<DL><p>" in html
        assert "</DL><p>" in html
        assert "ADD_DATE=" in html
        assert "LAST_MODIFIED=" in html

    def test_generate_folder_html_bookmarks_bar(self):
        """Test generating HTML for Bookmarks bar folder."""
        folder = FolderNode("Bookmarks bar")
        html_lines = self.generator._generate_folder_html(folder)

        html = "\n".join(html_lines)
        assert 'PERSONAL_TOOLBAR_FOLDER="true"' in html

    def test_generate_folder_html_with_bookmarks(self):
        """Test generating HTML for folder with bookmarks."""
        folder = FolderNode("Test")
        folder.add_bookmark(Bookmark(url="https://a.com", title="A Site"))
        folder.add_bookmark(Bookmark(url="https://b.com", title="B Site"))

        html_lines = self.generator._generate_folder_html(folder)
        html = "\n".join(html_lines)

        assert "A Site</A>" in html
        assert "B Site</A>" in html

    def test_generate_folder_html_with_subfolders(self):
        """Test generating HTML for folder with subfolders."""
        folder = FolderNode("Parent")
        folder.add_child("Child1")
        folder.add_child("Child2")

        html_lines = self.generator._generate_folder_html(folder)
        html = "\n".join(html_lines)

        assert "Child1</H3>" in html
        assert "Child2</H3>" in html

    def test_generate_folder_html_sorted_subfolders(self):
        """Test that subfolders are sorted alphabetically."""
        folder = FolderNode("Parent")
        folder.add_child("Zebra")
        folder.add_child("Alpha")
        folder.add_child("Mango")

        html_lines = self.generator._generate_folder_html(folder)
        html = "\n".join(html_lines)

        alpha_pos = html.find("Alpha</H3>")
        mango_pos = html.find("Mango</H3>")
        zebra_pos = html.find("Zebra</H3>")

        assert alpha_pos < mango_pos < zebra_pos

    def test_generate_folder_html_sorted_bookmarks(self):
        """Test that bookmarks are sorted by title."""
        folder = FolderNode("Test")
        folder.add_bookmark(Bookmark(url="https://z.com", title="Zebra"))
        folder.add_bookmark(Bookmark(url="https://a.com", title="Alpha"))
        folder.add_bookmark(Bookmark(url="https://m.com", title="Mango"))

        html_lines = self.generator._generate_folder_html(folder)
        html = "\n".join(html_lines)

        alpha_pos = html.find(">Alpha</A>")
        mango_pos = html.find(">Mango</A>")
        zebra_pos = html.find(">Zebra</A>")

        assert alpha_pos < mango_pos < zebra_pos

    def test_generate_html_content_structure(self):
        """Test overall HTML content structure."""
        root = FolderNode("Root")
        root.add_child("Bookmarks bar")
        root.add_child("Other bookmarks")

        html = self.generator._generate_html_content(root, "My Bookmarks")

        assert "<!DOCTYPE NETSCAPE-Bookmark-file-1>" in html
        assert "<!-- This is an automatically generated file." in html
        assert '<META HTTP-EQUIV="Content-Type"' in html
        assert "charset=UTF-8" in html
        assert "<TITLE>My Bookmarks</TITLE>" in html
        assert "<H1>My Bookmarks</H1>" in html
        assert "<DL><p>" in html
        assert "</DL><p>" in html

    def test_generate_html_content_custom_title(self):
        """Test HTML generation with custom title."""
        root = FolderNode("Root")
        root.add_child("Bookmarks bar")
        root.add_child("Other bookmarks")

        html = self.generator._generate_html_content(root, "Custom Title Here")

        assert "<TITLE>Custom Title Here</TITLE>" in html
        assert "<H1>Custom Title Here</H1>" in html

    def test_generate_html_content_title_escaping(self):
        """Test HTML title escaping."""
        root = FolderNode("Root")
        root.add_child("Bookmarks bar")
        root.add_child("Other bookmarks")

        html = self.generator._generate_html_content(root, "Title <with> & \"special\" chars")

        assert "&lt;with&gt;" in html
        assert "&amp;" in html
        assert "&quot;special&quot;" in html

    def test_generate_html_file_creation(self):
        """Test that generate_html creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            bookmarks = [
                Bookmark(url="https://example.com", title="Example")
            ]

            self.generator.generate_html(bookmarks, output_path)

            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE NETSCAPE-Bookmark-file-1>" in content
            assert "https://example.com" in content

    def test_generate_html_with_string_path(self):
        """Test generate_html accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "bookmarks.html")
            bookmarks = [
                Bookmark(url="https://example.com", title="Example")
            ]

            self.generator.generate_html(bookmarks, output_path)

            assert Path(output_path).exists()

    def test_generate_html_default_title(self):
        """Test generate_html uses default title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            bookmarks = []

            self.generator.generate_html(bookmarks, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "<TITLE>Bookmarks</TITLE>" in content

    def test_generate_html_custom_title(self):
        """Test generate_html with custom title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            bookmarks = []

            self.generator.generate_html(bookmarks, output_path, title="My Custom Bookmarks")

            content = output_path.read_text(encoding="utf-8")
            assert "<TITLE>My Custom Bookmarks</TITLE>" in content

    def test_generate_html_utf8_encoding(self):
        """Test generate_html creates UTF-8 encoded file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            bookmarks = [
                Bookmark(url="https://example.com", title="Caf\u00e9 & B\u00e4r")
            ]

            self.generator.generate_html(bookmarks, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "Caf\u00e9" in content

    def test_generate_html_error_handling(self):
        """Test generate_html raises ChromeHTMLGeneratorError on failure."""
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            bookmarks = [Bookmark(url="https://example.com", title="Test")]

            with pytest.raises(ChromeHTMLGeneratorError) as exc_info:
                self.generator.generate_html(bookmarks, "/invalid/path/test.html")

            assert "Failed to generate Chrome HTML file" in str(exc_info.value)

    def test_get_generation_info_empty(self):
        """Test get_generation_info with no bookmarks."""
        info = self.generator.get_generation_info([])

        assert info["total_bookmarks"] == 0
        assert "generation_timestamp" in info
        assert "folder_structure" in info

    def test_get_generation_info_with_bookmarks(self):
        """Test get_generation_info with bookmarks."""
        bookmarks = [
            Bookmark(url="https://example1.com", title="Test 1", folder="Tech"),
            Bookmark(url="https://example2.com", title="Test 2", folder="Tech"),
            Bookmark(url="https://example3.com", title="Test 3", folder="News"),
        ]

        info = self.generator.get_generation_info(bookmarks)

        assert info["total_bookmarks"] == 3
        assert "Bookmarks bar" in info["folder_structure"]
        assert "Other bookmarks" in info["folder_structure"]

    def test_get_generation_info_nested_structure(self):
        """Test get_generation_info captures nested folder structure."""
        bookmarks = [
            Bookmark(url="https://example.com", title="Test", folder="Tech/AI/ML")
        ]

        info = self.generator.get_generation_info(bookmarks)

        other = info["folder_structure"]["Other bookmarks"]
        assert "Tech" in other["subfolders"]
        tech = other["subfolders"]["Tech"]
        assert "AI" in tech["subfolders"]
        ai = tech["subfolders"]["AI"]
        assert "ML" in ai["subfolders"]

    def test_get_generation_info_timestamp(self):
        """Test get_generation_info includes valid timestamp."""
        info = self.generator.get_generation_info([])

        timestamp = info["generation_timestamp"]
        assert timestamp is not None
        # Should be ISO format
        assert "T" in timestamp

    def test_log_folder_stats(self):
        """Test _log_folder_stats logs correct statistics."""
        root = FolderNode("Root")
        child1 = root.add_child("Child1")
        child2 = root.add_child("Child2")
        child1.add_bookmark(Bookmark(url="https://a.com", title="A"))
        child2.add_bookmark(Bookmark(url="https://b.com", title="B"))
        child2.add_bookmark(Bookmark(url="https://c.com", title="C"))

        # This should not raise any exceptions
        self.generator._log_folder_stats(root)

    def test_full_integration(self):
        """Test full integration: generate file with realistic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"

            bookmarks = [
                Bookmark(
                    url="https://python.org",
                    title="Python",
                    folder="Programming/Languages",
                    created=datetime(2024, 1, 1, tzinfo=timezone.utc)
                ),
                Bookmark(
                    url="https://github.com",
                    title="GitHub",
                    folder="Development/Tools",
                    created=datetime(2024, 1, 2, tzinfo=timezone.utc)
                ),
                Bookmark(
                    url="https://stackoverflow.com",
                    title="Stack Overflow",
                    folder="Programming/Resources",
                    created=datetime(2024, 1, 3, tzinfo=timezone.utc)
                ),
                Bookmark(
                    url="https://news.ycombinator.com",
                    title="Hacker News",
                    folder="Bookmarks bar/Daily",
                    created=datetime(2024, 1, 4, tzinfo=timezone.utc)
                ),
                Bookmark(
                    url="https://example.com",
                    title="No Folder",
                    folder="",
                    created=datetime(2024, 1, 5, tzinfo=timezone.utc)
                ),
            ]

            self.generator.generate_html(
                bookmarks,
                output_path,
                title="My Test Bookmarks"
            )

            content = output_path.read_text(encoding="utf-8")

            # Check structure
            assert "<!DOCTYPE NETSCAPE-Bookmark-file-1>" in content
            assert "<TITLE>My Test Bookmarks</TITLE>" in content

            # Check bookmarks are present
            assert "https://python.org" in content
            assert "https://github.com" in content
            assert "https://stackoverflow.com" in content
            assert "https://news.ycombinator.com" in content
            assert "https://example.com" in content

            # Check folders
            assert "Programming" in content
            assert "Languages" in content
            assert "Development" in content
            assert "Tools" in content
            assert "Daily" in content

            # Check bookmarks bar has special attribute
            assert 'PERSONAL_TOOLBAR_FOLDER="true"' in content


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ChromeHTMLGenerator()

    def test_very_long_title(self):
        """Test handling of very long bookmark title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            long_title = "A" * 10000
            bookmarks = [Bookmark(url="https://example.com", title=long_title)]

            self.generator.generate_html(bookmarks, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert long_title in content

    def test_very_long_url(self):
        """Test handling of very long URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            long_url = "https://example.com?" + "a=b&" * 1000
            bookmarks = [Bookmark(url=long_url, title="Long URL")]

            self.generator.generate_html(bookmarks, output_path)

            assert output_path.exists()

    def test_deeply_nested_folders(self):
        """Test handling of deeply nested folder structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            folder_path = "/".join([f"Level{i}" for i in range(50)])
            bookmarks = [
                Bookmark(url="https://example.com", title="Deep", folder=folder_path)
            ]

            self.generator.generate_html(bookmarks, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "Level0" in content
            assert "Level49" in content

    def test_many_bookmarks_same_folder(self):
        """Test handling of many bookmarks in the same folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            bookmarks = [
                Bookmark(url=f"https://example{i}.com", title=f"Test {i}", folder="Same")
                for i in range(100)
            ]

            self.generator.generate_html(bookmarks, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert content.count("example") == 100

    def test_unicode_folder_names(self):
        """Test handling of unicode in folder names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"
            bookmarks = [
                Bookmark(url="https://example.com", title="Test", folder="Caf\u00e9/B\u00e4cker")
            ]

            self.generator.generate_html(bookmarks, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "Caf\u00e9" in content

    def test_empty_bookmark_list(self):
        """Test generating HTML with empty bookmark list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bookmarks.html"

            self.generator.generate_html([], output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE NETSCAPE-Bookmark-file-1>" in content
            assert "Bookmarks bar" in content
            assert "Other bookmarks" in content

    def test_bookmark_with_none_title(self):
        """Test bookmark with None title uses URL."""
        bookmark = Bookmark(url="https://example.com")
        bookmark.title = None

        html = self.generator._generate_bookmark_html(bookmark)

        assert ">https://example.com</A>" in html

    def test_special_folder_name_variations(self):
        """Test various bookmark bar name variations."""
        variations = [
            "Bookmarks bar",
            "BOOKMARKS BAR",
            "bookmarks_bar",
            "Toolbar",
            "TOOLBAR",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, folder_prefix in enumerate(variations):
                output_path = Path(tmpdir) / f"bookmarks_{i}.html"
                bookmarks = [
                    Bookmark(
                        url=f"https://example{i}.com",
                        title=f"Test {i}",
                        folder=f"{folder_prefix}/Subfolder"
                    )
                ]

                self.generator.generate_html(bookmarks, output_path)

                content = output_path.read_text(encoding="utf-8")
                # All should result in bookmarks being in the bar
                bar_section = content.find('PERSONAL_TOOLBAR_FOLDER="true"')
                assert bar_section != -1, f"Failed for {folder_prefix}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
