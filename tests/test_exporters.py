"""
Tests for the multi-format bookmark exporters.

This module contains tests for all exporter implementations:
- JSONExporter
- MarkdownExporter
- ObsidianExporter
- NotionExporter
- OPMLExporter
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from bookmark_processor.core.data_models import Bookmark, ProcessingStatus
from bookmark_processor.core.exporters import (
    BookmarkExporter,
    ExportResult,
    ExportError,
    get_exporter,
    EXPORTERS,
)
from bookmark_processor.core.exporters.json_exporter import JSONExporter
from bookmark_processor.core.exporters.markdown_exporter import MarkdownExporter
from bookmark_processor.core.exporters.obsidian_exporter import ObsidianExporter
from bookmark_processor.core.exporters.notion_exporter import NotionExporter
from bookmark_processor.core.exporters.opml_exporter import OPMLExporter


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_bookmarks() -> List[Bookmark]:
    """Create sample bookmarks for testing."""
    return [
        Bookmark(
            id="1",
            title="Example Site",
            url="https://example.com",
            folder="Technology",
            tags=["tech", "example"],
            note="A sample bookmark",
            excerpt="This is an example site for testing purposes.",
            created=datetime(2024, 1, 15, 10, 30, 0),
            favorite=True,
        ),
        Bookmark(
            id="2",
            title="Python Documentation",
            url="https://docs.python.org",
            folder="Technology/Programming",
            tags=["python", "documentation", "programming"],
            note="Official Python docs",
            excerpt="Welcome to Python documentation.",
            created=datetime(2024, 2, 20, 14, 45, 0),
            favorite=False,
        ),
        Bookmark(
            id="3",
            title="News Article",
            url="https://news.example.org/article",
            folder="News",
            tags=["news", "current-events"],
            note="",
            excerpt="Breaking news about technology.",
            created=datetime(2024, 3, 10, 8, 0, 0),
            favorite=False,
        ),
        Bookmark(
            id="4",
            title="Recipe Site",
            url="https://recipes.example.com",
            folder="Recipes",
            tags=["food", "cooking"],
            note="Great recipes here",
            excerpt="",
            created=None,
            favorite=True,
        ),
    ]


@pytest.fixture
def empty_bookmarks() -> List[Bookmark]:
    """Create an empty bookmark list."""
    return []


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "export_output"
    output_dir.mkdir()
    return output_dir


# =========================================================================
# Common Tests
# =========================================================================

class TestExporterRegistry:
    """Tests for the exporter registry."""

    def test_get_exporter_json(self):
        """Test getting JSON exporter."""
        exporter_class = get_exporter("json")
        assert exporter_class == JSONExporter

    def test_get_exporter_markdown(self):
        """Test getting Markdown exporter."""
        exporter_class = get_exporter("markdown")
        assert exporter_class == MarkdownExporter

    def test_get_exporter_md_alias(self):
        """Test getting Markdown exporter via 'md' alias."""
        exporter_class = get_exporter("md")
        assert exporter_class == MarkdownExporter

    def test_get_exporter_obsidian(self):
        """Test getting Obsidian exporter."""
        exporter_class = get_exporter("obsidian")
        assert exporter_class == ObsidianExporter

    def test_get_exporter_notion(self):
        """Test getting Notion exporter."""
        exporter_class = get_exporter("notion")
        assert exporter_class == NotionExporter

    def test_get_exporter_opml(self):
        """Test getting OPML exporter."""
        exporter_class = get_exporter("opml")
        assert exporter_class == OPMLExporter

    def test_get_exporter_case_insensitive(self):
        """Test that format names are case insensitive."""
        assert get_exporter("JSON") == JSONExporter
        assert get_exporter("MARKDOWN") == MarkdownExporter
        assert get_exporter("Obsidian") == ObsidianExporter

    def test_get_exporter_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            get_exporter("invalid_format")


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_export_result_creation(self, temp_output_dir):
        """Test creating an ExportResult."""
        result = ExportResult(
            path=temp_output_dir / "test.json",
            count=10,
            format_name="JSON"
        )

        assert result.count == 10
        assert result.format_name == "JSON"
        assert isinstance(result.exported_at, datetime)

    def test_export_result_str(self, temp_output_dir):
        """Test ExportResult string representation."""
        result = ExportResult(
            path=temp_output_dir / "test.json",
            count=10,
            format_name="JSON"
        )

        str_repr = str(result)
        assert "JSON" in str_repr
        assert "10" in str_repr


# =========================================================================
# JSON Exporter Tests
# =========================================================================

class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_format_name(self):
        """Test format name property."""
        exporter = JSONExporter()
        assert exporter.format_name == "JSON"

    def test_file_extension(self):
        """Test file extension property."""
        exporter = JSONExporter()
        assert exporter.file_extension == "json"

    def test_export_creates_file(self, sample_bookmarks, temp_output_dir):
        """Test that export creates a JSON file."""
        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        result = exporter.export(sample_bookmarks, output_path)

        assert output_path.exists()
        assert result.count == len(sample_bookmarks)
        assert result.path == output_path

    def test_export_valid_json(self, sample_bookmarks, temp_output_dir):
        """Test that exported file is valid JSON."""
        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "bookmarks" in data
        assert "export_info" in data
        assert len(data["bookmarks"]) == len(sample_bookmarks)

    def test_export_includes_metadata(self, sample_bookmarks, temp_output_dir):
        """Test that export includes metadata when enabled."""
        exporter = JSONExporter(include_metadata=True)
        output_path = temp_output_dir / "bookmarks.json"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        first_bookmark = data["bookmarks"][0]
        assert "id" in first_bookmark
        assert "note" in first_bookmark

    def test_export_without_metadata(self, sample_bookmarks, temp_output_dir):
        """Test that export excludes metadata when disabled."""
        exporter = JSONExporter(include_metadata=False)
        output_path = temp_output_dir / "bookmarks.json"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        first_bookmark = data["bookmarks"][0]
        assert "id" not in first_bookmark
        assert "processing_status" not in first_bookmark

    def test_export_empty_raises_error(self, empty_bookmarks, temp_output_dir):
        """Test that exporting empty list raises error."""
        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        with pytest.raises(ExportError, match="No bookmarks"):
            exporter.export(empty_bookmarks, output_path)

    def test_export_by_folder(self, sample_bookmarks, temp_output_dir):
        """Test that export includes by_folder grouping."""
        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "by_folder" in data
        assert "Technology" in data["by_folder"]

    def test_export_statistics(self, sample_bookmarks, temp_output_dir):
        """Test that export includes statistics."""
        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stats = data["statistics"]
        assert stats["total_count"] == len(sample_bookmarks)
        assert stats["favorites"] == 2  # Two bookmarks are favorites

    def test_export_minimal(self, sample_bookmarks, temp_output_dir):
        """Test minimal export mode."""
        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks_minimal.json"

        result = exporter.export_minimal(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == len(sample_bookmarks)
        assert "url" in data[0]
        assert "title" in data[0]
        assert "tags" in data[0]


# =========================================================================
# Markdown Exporter Tests
# =========================================================================

class TestMarkdownExporter:
    """Tests for MarkdownExporter."""

    def test_format_name(self):
        """Test format name property."""
        exporter = MarkdownExporter()
        assert exporter.format_name == "Markdown"

    def test_file_extension(self):
        """Test file extension property."""
        exporter = MarkdownExporter()
        assert exporter.file_extension == "md"

    def test_export_single_file(self, sample_bookmarks, temp_output_dir):
        """Test single file export mode."""
        exporter = MarkdownExporter(mode="single")
        output_path = temp_output_dir / "bookmarks.md"

        result = exporter.export(sample_bookmarks, output_path)

        assert output_path.exists()
        assert result.count == len(sample_bookmarks)

    def test_export_single_file_content(self, sample_bookmarks, temp_output_dir):
        """Test single file export content."""
        exporter = MarkdownExporter(mode="single")
        output_path = temp_output_dir / "bookmarks.md"

        exporter.export(sample_bookmarks, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "# Bookmarks" in content
        assert "Example Site" in content
        assert "https://example.com" in content

    def test_export_directory_mode(self, sample_bookmarks, temp_output_dir):
        """Test directory export mode."""
        exporter = MarkdownExporter(mode="directory")
        output_path = temp_output_dir / "bookmarks_dir"

        result = exporter.export(sample_bookmarks, output_path)

        assert output_path.is_dir()
        # Should create README.md index
        assert (output_path / "README.md").exists()
        # Should create folder files
        assert (output_path / "Technology.md").exists()

    def test_export_includes_tags(self, sample_bookmarks, temp_output_dir):
        """Test that export includes tags."""
        exporter = MarkdownExporter(mode="single", include_tags=True)
        output_path = temp_output_dir / "bookmarks.md"

        exporter.export(sample_bookmarks, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "`tech`" in content or "tech" in content

    def test_export_includes_descriptions(self, sample_bookmarks, temp_output_dir):
        """Test that export includes descriptions."""
        exporter = MarkdownExporter(mode="single", include_descriptions=True)
        output_path = temp_output_dir / "bookmarks.md"

        exporter.export(sample_bookmarks, output_path)

        content = output_path.read_text(encoding="utf-8")
        # Should have description from excerpt
        assert "example site" in content.lower()

    def test_export_with_checkboxes(self, sample_bookmarks, temp_output_dir):
        """Test export with checkbox format."""
        exporter = MarkdownExporter(mode="single", use_checkboxes=True)
        output_path = temp_output_dir / "bookmarks.md"

        exporter.export(sample_bookmarks, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "- [ ]" in content

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            MarkdownExporter(mode="invalid")


# =========================================================================
# Obsidian Exporter Tests
# =========================================================================

class TestObsidianExporter:
    """Tests for ObsidianExporter."""

    def test_format_name(self):
        """Test format name property."""
        exporter = ObsidianExporter()
        assert exporter.format_name == "Obsidian"

    def test_file_extension(self):
        """Test file extension property."""
        exporter = ObsidianExporter()
        assert exporter.file_extension == "md"

    def test_export_creates_vault(self, sample_bookmarks, temp_output_dir):
        """Test that export creates vault structure."""
        exporter = ObsidianExporter()
        vault_path = temp_output_dir / "vault"

        result = exporter.export(sample_bookmarks, vault_path)

        assert vault_path.is_dir()
        assert result.count == len(sample_bookmarks)

    def test_export_creates_folder_structure(self, sample_bookmarks, temp_output_dir):
        """Test that export creates folder structure."""
        exporter = ObsidianExporter()
        vault_path = temp_output_dir / "vault"

        exporter.export(sample_bookmarks, vault_path)

        # Should create Technology folder
        assert (vault_path / "Technology").is_dir()
        # Should create nested folder
        assert (vault_path / "Technology" / "Programming").is_dir()

    def test_export_creates_notes(self, sample_bookmarks, temp_output_dir):
        """Test that export creates individual notes."""
        exporter = ObsidianExporter()
        vault_path = temp_output_dir / "vault"

        exporter.export(sample_bookmarks, vault_path)

        # Check for bookmark note in Technology folder
        tech_folder = vault_path / "Technology"
        md_files = list(tech_folder.glob("*.md"))
        assert len(md_files) > 0

    def test_export_note_has_frontmatter(self, sample_bookmarks, temp_output_dir):
        """Test that notes have YAML frontmatter."""
        exporter = ObsidianExporter()
        vault_path = temp_output_dir / "vault"

        exporter.export(sample_bookmarks, vault_path)

        # Find a note file
        note_files = list(vault_path.rglob("*.md"))
        # Exclude index files and MOC
        note_files = [f for f in note_files if "Index" not in f.name and "MOC" not in f.name]

        if note_files:
            content = note_files[0].read_text(encoding="utf-8")
            assert content.startswith("---")
            assert "url:" in content

    def test_export_creates_moc(self, sample_bookmarks, temp_output_dir):
        """Test that export creates Map of Content."""
        exporter = ObsidianExporter(create_moc=True)
        vault_path = temp_output_dir / "vault"

        exporter.export(sample_bookmarks, vault_path)

        moc_path = vault_path / "Bookmarks MOC.md"
        assert moc_path.exists()

    def test_export_creates_folder_notes(self, sample_bookmarks, temp_output_dir):
        """Test that export creates folder index notes."""
        exporter = ObsidianExporter(create_folder_notes=True)
        vault_path = temp_output_dir / "vault"

        exporter.export(sample_bookmarks, vault_path)

        # Should create index in Technology folder
        tech_folder = vault_path / "Technology"
        index_files = list(tech_folder.glob("*(Index).md"))
        assert len(index_files) > 0


# =========================================================================
# Notion Exporter Tests
# =========================================================================

class TestNotionExporter:
    """Tests for NotionExporter."""

    def test_format_name(self):
        """Test format name property."""
        exporter = NotionExporter()
        assert exporter.format_name == "Notion CSV"

    def test_file_extension(self):
        """Test file extension property."""
        exporter = NotionExporter()
        assert exporter.file_extension == "csv"

    def test_export_creates_csv(self, sample_bookmarks, temp_output_dir):
        """Test that export creates a CSV file."""
        exporter = NotionExporter()
        output_path = temp_output_dir / "notion.csv"

        result = exporter.export(sample_bookmarks, output_path)

        assert output_path.exists()
        assert result.count == len(sample_bookmarks)

    def test_export_valid_csv(self, sample_bookmarks, temp_output_dir):
        """Test that exported file is valid CSV."""
        exporter = NotionExporter()
        output_path = temp_output_dir / "notion.csv"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(sample_bookmarks)

    def test_export_has_required_columns(self, sample_bookmarks, temp_output_dir):
        """Test that export has required Notion columns."""
        exporter = NotionExporter()
        output_path = temp_output_dir / "notion.csv"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "Name" in rows[0]
        assert "URL" in rows[0]
        assert "Tags" in rows[0]
        assert "Folder" in rows[0]

    def test_export_with_status_column(self, sample_bookmarks, temp_output_dir):
        """Test export with status column."""
        exporter = NotionExporter(include_status=True)
        output_path = temp_output_dir / "notion.csv"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "Status" in rows[0]

    def test_export_tags_formatting(self, sample_bookmarks, temp_output_dir):
        """Test that tags are properly formatted."""
        exporter = NotionExporter(tag_separator=", ")
        output_path = temp_output_dir / "notion.csv"

        exporter.export(sample_bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # First bookmark has tags ["tech", "example"]
        assert "tech, example" in rows[0]["Tags"] or "tech" in rows[0]["Tags"]


# =========================================================================
# OPML Exporter Tests
# =========================================================================

class TestOPMLExporter:
    """Tests for OPMLExporter."""

    def test_format_name(self):
        """Test format name property."""
        exporter = OPMLExporter()
        assert exporter.format_name == "OPML"

    def test_file_extension(self):
        """Test file extension property."""
        exporter = OPMLExporter()
        assert exporter.file_extension == "opml"

    def test_export_creates_file(self, sample_bookmarks, temp_output_dir):
        """Test that export creates an OPML file."""
        exporter = OPMLExporter()
        output_path = temp_output_dir / "bookmarks.opml"

        result = exporter.export(sample_bookmarks, output_path)

        assert output_path.exists()
        assert result.count == len(sample_bookmarks)

    def test_export_valid_xml(self, sample_bookmarks, temp_output_dir):
        """Test that exported file is valid XML."""
        exporter = OPMLExporter()
        output_path = temp_output_dir / "bookmarks.opml"

        exporter.export(sample_bookmarks, output_path)

        # Should parse without error
        tree = ET.parse(output_path)
        root = tree.getroot()

        assert root.tag == "opml"
        assert root.attrib.get("version") == "2.0"

    def test_export_has_head_and_body(self, sample_bookmarks, temp_output_dir):
        """Test that OPML has head and body sections."""
        exporter = OPMLExporter()
        output_path = temp_output_dir / "bookmarks.opml"

        exporter.export(sample_bookmarks, output_path)

        tree = ET.parse(output_path)
        root = tree.getroot()

        head = root.find("head")
        body = root.find("body")

        assert head is not None
        assert body is not None

    def test_export_has_title(self, sample_bookmarks, temp_output_dir):
        """Test that OPML has title in head."""
        exporter = OPMLExporter(title="Test Bookmarks")
        output_path = temp_output_dir / "bookmarks.opml"

        exporter.export(sample_bookmarks, output_path)

        tree = ET.parse(output_path)
        root = tree.getroot()
        title = root.find("head/title")

        assert title is not None
        assert title.text == "Test Bookmarks"

    def test_export_folder_structure(self, sample_bookmarks, temp_output_dir):
        """Test that OPML has folder structure."""
        exporter = OPMLExporter()
        output_path = temp_output_dir / "bookmarks.opml"

        exporter.export(sample_bookmarks, output_path)

        tree = ET.parse(output_path)
        root = tree.getroot()
        body = root.find("body")

        # Should have outline elements for folders
        outlines = body.findall("outline")
        assert len(outlines) > 0

    def test_export_bookmark_attributes(self, sample_bookmarks, temp_output_dir):
        """Test that bookmarks have correct attributes."""
        exporter = OPMLExporter(use_html_url=True)
        output_path = temp_output_dir / "bookmarks.opml"

        exporter.export(sample_bookmarks, output_path)

        tree = ET.parse(output_path)
        root = tree.getroot()

        # Find a bookmark outline (type="link")
        for outline in root.iter("outline"):
            if outline.get("type") == "link":
                assert "htmlUrl" in outline.attrib
                assert "text" in outline.attrib
                break

    def test_export_flat(self, sample_bookmarks, temp_output_dir):
        """Test flat export mode."""
        exporter = OPMLExporter()
        output_path = temp_output_dir / "bookmarks_flat.opml"

        result = exporter.export_flat(sample_bookmarks, output_path)

        tree = ET.parse(output_path)
        root = tree.getroot()
        body = root.find("body")

        # All bookmarks should be direct children
        link_outlines = [o for o in body.findall("outline") if o.get("type") == "link"]
        assert len(link_outlines) == len(sample_bookmarks)


# =========================================================================
# Edge Cases and Error Handling
# =========================================================================

class TestExporterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_export_bookmark_without_title(self, temp_output_dir):
        """Test exporting bookmark without title."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title="",
                folder="Test"
            )
        ]

        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        result = exporter.export(bookmarks, output_path)

        # Should succeed, using URL as fallback
        assert result.count == 1

    def test_export_bookmark_without_folder(self, temp_output_dir):
        """Test exporting bookmark without folder."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title="Test",
                folder=""
            )
        ]

        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        result = exporter.export(bookmarks, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Should be in "Unsorted" folder
        assert "Unsorted" in data["by_folder"]

    def test_export_special_characters_in_title(self, temp_output_dir):
        """Test exporting bookmark with special characters."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title='Test "Special" <Characters> & More',
                folder="Test"
            )
        ]

        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        result = exporter.export(bookmarks, output_path)

        # Should succeed without errors
        assert result.count == 1

    def test_export_unicode_content(self, temp_output_dir):
        """Test exporting bookmark with unicode content."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title="Test Unicode Content",
                folder="Test"
            )
        ]

        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks.json"

        result = exporter.export(bookmarks, output_path)

        # Should handle unicode properly
        assert result.count == 1

    def test_export_adds_extension_if_missing(self, sample_bookmarks, temp_output_dir):
        """Test that export adds correct extension if missing."""
        exporter = JSONExporter()
        output_path = temp_output_dir / "bookmarks"  # No extension

        result = exporter.export(sample_bookmarks, output_path)

        # Should have added .json extension
        assert result.path.suffix == ".json"

    def test_export_creates_parent_directories(self, sample_bookmarks, tmp_path):
        """Test that export creates parent directories."""
        exporter = JSONExporter()
        output_path = tmp_path / "deep" / "nested" / "path" / "bookmarks.json"

        result = exporter.export(sample_bookmarks, output_path)

        assert output_path.exists()


# =========================================================================
# Integration Tests
# =========================================================================

@pytest.mark.integration
class TestExporterIntegration:
    """Integration tests for exporters."""

    def test_export_all_formats(self, sample_bookmarks, temp_output_dir):
        """Test exporting to all formats."""
        formats = ["json", "markdown", "obsidian", "notion", "opml"]

        for fmt in formats:
            ExporterClass = get_exporter(fmt)

            if fmt == "markdown":
                exporter = ExporterClass(mode="single")
            else:
                exporter = ExporterClass()

            if fmt == "obsidian":
                output_path = temp_output_dir / f"test_{fmt}"
            else:
                ext = exporter.file_extension
                output_path = temp_output_dir / f"test_{fmt}.{ext}"

            result = exporter.export(sample_bookmarks, output_path)

            assert result.count == len(sample_bookmarks)
            assert output_path.exists() or output_path.is_dir()
