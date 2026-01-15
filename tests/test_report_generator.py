"""
Unit tests for report generation infrastructure.

Tests the ReportGenerator, ReportSection, and related classes
for generating reports in multiple formats.
"""

import json
import tempfile
from pathlib import Path

import pytest

from bookmark_processor.utils.report_generator import (
    ReportGenerator,
    ReportSection,
    TableData,
)
from bookmark_processor.utils.report_styles import (
    ICONS,
    RICH_COLORS,
    ReportStyle,
    StyleConfig,
    get_icon,
    get_percentage_color,
    get_style_config,
)


class TestReportStyles:
    """Test ReportStyle enum and style utilities."""

    def test_report_style_enum_values(self):
        """Test that ReportStyle enum has expected values."""
        assert ReportStyle.RICH.value == "rich"
        assert ReportStyle.PLAIN.value == "plain"
        assert ReportStyle.MARKDOWN.value == "markdown"
        assert ReportStyle.JSON.value == "json"

    def test_get_style_config_rich(self):
        """Test getting Rich style configuration."""
        config = get_style_config(ReportStyle.RICH)

        assert isinstance(config, StyleConfig)
        assert config.use_icons is True
        assert config.use_colors is True
        assert config.use_borders is True

    def test_get_style_config_plain(self):
        """Test getting Plain style configuration."""
        config = get_style_config(ReportStyle.PLAIN)

        assert isinstance(config, StyleConfig)
        assert config.use_icons is False
        assert config.use_colors is False
        assert config.use_borders is False

    def test_get_style_config_markdown(self):
        """Test getting Markdown style configuration."""
        config = get_style_config(ReportStyle.MARKDOWN)

        assert isinstance(config, StyleConfig)
        assert config.use_icons is False
        assert config.bullet == "-"

    def test_get_style_config_json(self):
        """Test getting JSON style configuration."""
        config = get_style_config(ReportStyle.JSON)

        assert isinstance(config, StyleConfig)
        assert config.use_icons is False

    def test_get_icon_with_rich_style(self):
        """Test getting icons with Rich style."""
        icon = get_icon("success", ReportStyle.RICH)
        assert icon == ICONS["success"]
        assert icon != ""

    def test_get_icon_with_plain_style(self):
        """Test that icons are empty with Plain style."""
        icon = get_icon("success", ReportStyle.PLAIN)
        assert icon == ""

    def test_get_icon_unknown_name(self):
        """Test getting an unknown icon name returns empty string."""
        icon = get_icon("nonexistent_icon", ReportStyle.RICH)
        assert icon == ""

    def test_get_percentage_color_high(self):
        """Test percentage color for high values."""
        color = get_percentage_color(85.0)
        assert color == RICH_COLORS["percentage_high"]

    def test_get_percentage_color_medium(self):
        """Test percentage color for medium values."""
        color = get_percentage_color(55.0)
        assert color == RICH_COLORS["percentage_medium"]

    def test_get_percentage_color_low(self):
        """Test percentage color for low values."""
        color = get_percentage_color(25.0)
        assert color == RICH_COLORS["percentage_low"]

    def test_get_percentage_color_custom_thresholds(self):
        """Test percentage color with custom thresholds."""
        color = get_percentage_color(60.0, thresholds=(80.0, 50.0))
        assert color == RICH_COLORS["percentage_medium"]


class TestReportSection:
    """Test ReportSection class."""

    def test_default_creation(self):
        """Test creating ReportSection with defaults."""
        section = ReportSection(title="Test Section")

        assert section.title == "Test Section"
        assert section.content is None
        assert section.icon is None
        assert section.subsections == []
        assert section.section_type == "text"

    def test_creation_with_content(self):
        """Test creating ReportSection with content."""
        section = ReportSection(
            title="Metrics",
            content={"count": 100, "rate": "95%"},
            icon="metrics",
            section_type="metrics",
        )

        assert section.title == "Metrics"
        assert section.content == {"count": 100, "rate": "95%"}
        assert section.icon == "metrics"
        assert section.section_type == "metrics"

    def test_add_subsection(self):
        """Test adding subsections."""
        parent = ReportSection(title="Parent")
        child1 = ReportSection(title="Child 1")
        child2 = ReportSection(title="Child 2")

        parent.add_subsection(child1)
        parent.add_subsection(child2)

        assert len(parent.subsections) == 2
        assert parent.subsections[0].title == "Child 1"
        assert parent.subsections[1].title == "Child 2"

    def test_to_dict(self):
        """Test converting section to dictionary."""
        section = ReportSection(
            title="Test",
            content="Content",
            icon="info",
            section_type="text",
        )
        child = ReportSection(title="Child", content="Child content")
        section.add_subsection(child)

        result = section.to_dict()

        assert result["title"] == "Test"
        assert result["content"] == "Content"
        assert result["icon"] == "info"
        assert result["type"] == "text"
        assert len(result["subsections"]) == 1
        assert result["subsections"][0]["title"] == "Child"

    def test_to_dict_without_optional_fields(self):
        """Test to_dict without optional fields."""
        section = ReportSection(title="Simple")

        result = section.to_dict()

        assert result["title"] == "Simple"
        assert result["content"] is None
        assert "icon" not in result  # Optional field not included when None
        assert "subsections" not in result  # Empty list not included


class TestTableData:
    """Test TableData class."""

    def test_creation(self):
        """Test creating TableData."""
        table = TableData(
            headers=["Name", "Value"],
            rows=[["Item 1", 100], ["Item 2", 200]],
            title="Test Table",
        )

        assert table.headers == ["Name", "Value"]
        assert len(table.rows) == 2
        assert table.title == "Test Table"
        assert table.alignments is None

    def test_creation_with_alignments(self):
        """Test creating TableData with alignments."""
        table = TableData(
            headers=["Name", "Value"],
            rows=[],
            alignments=["left", "right"],
        )

        assert table.alignments == ["left", "right"]


class TestReportGenerator:
    """Test ReportGenerator class."""

    def test_default_initialization(self):
        """Test default initialization."""
        generator = ReportGenerator()

        assert generator.style == ReportStyle.RICH
        assert generator.sections == []
        assert generator.title is None
        assert generator.subtitle is None

    def test_initialization_with_style(self):
        """Test initialization with specific style."""
        generator = ReportGenerator(style=ReportStyle.MARKDOWN)

        assert generator.style == ReportStyle.MARKDOWN

    def test_set_title(self):
        """Test setting report title."""
        generator = ReportGenerator()
        result = generator.set_title("Test Report", "Subtitle")

        assert generator.title == "Test Report"
        assert generator.subtitle == "Subtitle"
        assert result is generator  # Method chaining

    def test_add_section(self):
        """Test adding a section."""
        generator = ReportGenerator()
        section = ReportSection(title="Test Section", content="Content")

        result = generator.add_section(section)

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Test Section"
        assert result is generator  # Method chaining

    def test_add_text_section(self):
        """Test adding a text section."""
        generator = ReportGenerator()
        generator.add_text_section("Header", "Some text content", icon="info")

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Header"
        assert generator.sections[0].content == "Some text content"
        assert generator.sections[0].icon == "info"
        assert generator.sections[0].section_type == "text"

    def test_add_table(self):
        """Test adding a table section."""
        generator = ReportGenerator()
        generator.add_table(
            "Results",
            headers=["Name", "Score"],
            rows=[["Test A", 95], ["Test B", 88]],
            icon="chart",
        )

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Results"
        assert generator.sections[0].section_type == "table"
        assert generator.sections[0].content["headers"] == ["Name", "Score"]
        assert len(generator.sections[0].content["rows"]) == 2

    def test_add_metrics(self):
        """Test adding a metrics section."""
        generator = ReportGenerator()
        generator.add_metrics(
            "Performance",
            metrics={"Total": 100, "Success Rate": "95%", "Errors": 5},
            icon="metrics",
        )

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Performance"
        assert generator.sections[0].section_type == "metrics"
        assert generator.sections[0].content["Total"] == 100

    def test_add_warning(self):
        """Test adding a warning section."""
        generator = ReportGenerator()
        generator.add_warning("This is a warning message")

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Warning"
        assert generator.sections[0].section_type == "warning"
        assert generator.sections[0].content == "This is a warning message"

    def test_add_tree(self):
        """Test adding a tree section."""
        generator = ReportGenerator()
        tree_data = {
            "Root": {
                "Branch1": {"Leaf1": "Value1"},
                "Branch2": "Value2",
            }
        }
        generator.add_tree("Hierarchy", tree_data, icon="folder")

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Hierarchy"
        assert generator.sections[0].section_type == "tree"
        assert "Root" in generator.sections[0].content

    def test_clear(self):
        """Test clearing the report."""
        generator = ReportGenerator()
        generator.set_title("Test Report")
        generator.add_text_section("Section", "Content")

        result = generator.clear()

        assert generator.title is None
        assert generator.subtitle is None
        assert generator.sections == []
        assert result is generator  # Method chaining


class TestReportGeneratorRendering:
    """Test ReportGenerator rendering methods."""

    @pytest.fixture
    def sample_generator(self):
        """Create a sample report generator with content."""
        generator = ReportGenerator()
        generator.set_title("Test Report", "A sample report")
        generator.add_text_section("Summary", "This is a summary.")
        generator.add_metrics(
            "Statistics",
            {"Total": 100, "Success": "85%", "Failed": 15},
        )
        generator.add_table(
            "Results",
            headers=["Item", "Status"],
            rows=[["Item 1", "OK"], ["Item 2", "Failed"]],
        )
        generator.add_warning("Some items need attention")
        return generator

    def test_render_plain(self, sample_generator):
        """Test rendering as plain text."""
        sample_generator.style = ReportStyle.PLAIN
        output = sample_generator.render_plain()

        assert "Test Report" in output
        assert "A sample report" in output
        assert "SUMMARY" in output
        assert "This is a summary" in output
        assert "STATISTICS" in output
        assert "Total" in output
        assert "85%" in output
        assert "RESULTS" in output
        assert "Item 1" in output
        assert "WARNING" in output

    def test_render_markdown(self, sample_generator):
        """Test rendering as Markdown."""
        output = sample_generator.render_markdown()

        assert "# Test Report" in output
        assert "*A sample report*" in output
        assert "## Summary" in output
        assert "This is a summary" in output
        assert "## Statistics" in output
        assert "**Total:**" in output
        assert "## Results" in output
        assert "| Item | Status |" in output
        assert "| Item 1 | OK |" in output
        assert "> **Warning:**" in output

    def test_render_json(self, sample_generator):
        """Test rendering as JSON."""
        output = sample_generator.render_json()

        assert output["title"] == "Test Report"
        assert output["subtitle"] == "A sample report"
        assert len(output["sections"]) == 4
        assert output["sections"][0]["title"] == "Summary"
        assert output["sections"][0]["content"] == "This is a summary."

    def test_render_json_string(self, sample_generator):
        """Test rendering as JSON string."""
        sample_generator.style = ReportStyle.JSON
        output = sample_generator.render()

        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["title"] == "Test Report"

    def test_render_terminal(self, sample_generator):
        """Test rendering for terminal."""
        output = sample_generator.render_terminal()

        # Should contain content (even if Rich is not available, falls back to plain)
        assert "Test Report" in output or "SUMMARY" in output

    def test_render_auto_detect_style(self, sample_generator):
        """Test that render() uses the configured style."""
        sample_generator.style = ReportStyle.MARKDOWN
        output = sample_generator.render()
        assert "# Test Report" in output

        sample_generator.style = ReportStyle.PLAIN
        output = sample_generator.render()
        assert "Test Report" in output  # Plain text contains title


class TestReportGeneratorSave:
    """Test ReportGenerator save functionality."""

    def test_save_markdown(self, tmp_path):
        """Test saving as Markdown file."""
        generator = ReportGenerator()
        generator.set_title("Test Report")
        generator.add_text_section("Section", "Content")

        output_path = tmp_path / "report.md"
        generator.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Test Report" in content
        assert "## Section" in content

    def test_save_json(self, tmp_path):
        """Test saving as JSON file."""
        generator = ReportGenerator()
        generator.set_title("Test Report")
        generator.add_metrics("Stats", {"count": 100})

        output_path = tmp_path / "report.json"
        generator.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        data = json.loads(content)
        assert data["title"] == "Test Report"

    def test_save_plain_text(self, tmp_path):
        """Test saving as plain text file."""
        generator = ReportGenerator()
        generator.set_title("Test Report")
        generator.add_text_section("Section", "Content")

        output_path = tmp_path / "report.txt"
        generator.save(output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Test Report" in content

    def test_save_auto_detect_format_md(self, tmp_path):
        """Test auto-detecting Markdown format from extension."""
        generator = ReportGenerator()
        generator.set_title("Test")

        output_path = tmp_path / "report.md"
        generator.save(output_path)

        content = output_path.read_text()
        assert "# Test" in content

    def test_save_auto_detect_format_json(self, tmp_path):
        """Test auto-detecting JSON format from extension."""
        generator = ReportGenerator()
        generator.set_title("Test")

        output_path = tmp_path / "report.json"
        generator.save(output_path)

        content = output_path.read_text()
        data = json.loads(content)
        assert data["title"] == "Test"

    def test_save_explicit_format_override(self, tmp_path):
        """Test explicit format overrides extension."""
        generator = ReportGenerator()
        generator.set_title("Test")

        # Save as JSON even though extension is .txt
        output_path = tmp_path / "report.txt"
        generator.save(output_path, format="json")

        content = output_path.read_text()
        data = json.loads(content)
        assert data["title"] == "Test"


class TestReportGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_report(self):
        """Test rendering an empty report."""
        generator = ReportGenerator(style=ReportStyle.PLAIN)
        output = generator.render()

        # Should not raise an error
        assert output is not None

    def test_report_without_title(self):
        """Test report without title."""
        generator = ReportGenerator(style=ReportStyle.MARKDOWN)
        generator.add_text_section("Section", "Content")

        output = generator.render()

        assert "## Section" in output
        # First line should be the section header, not a title
        first_line = output.strip().split("\n")[0]
        assert first_line == "## Section"

    def test_empty_table(self):
        """Test adding an empty table."""
        generator = ReportGenerator(style=ReportStyle.PLAIN)
        generator.add_table("Empty Table", headers=[], rows=[])

        output = generator.render()
        assert "EMPTY TABLE" in output

    def test_empty_metrics(self):
        """Test adding empty metrics."""
        generator = ReportGenerator(style=ReportStyle.MARKDOWN)
        generator.add_metrics("Empty Metrics", metrics={})

        output = generator.render()
        assert "## Empty Metrics" in output

    def test_nested_subsections(self):
        """Test deeply nested subsections."""
        parent = ReportSection(title="Parent", content="Parent content")
        child = ReportSection(title="Child", content="Child content")
        grandchild = ReportSection(title="Grandchild", content="Grandchild content")

        child.add_subsection(grandchild)
        parent.add_subsection(child)

        generator = ReportGenerator(style=ReportStyle.PLAIN)
        generator.add_section(parent)

        output = generator.render()

        assert "PARENT" in output
        assert "Child" in output
        assert "Grandchild" in output

    def test_special_characters_in_content(self):
        """Test handling special characters."""
        generator = ReportGenerator(style=ReportStyle.MARKDOWN)
        generator.add_text_section("Test", "Content with | pipe and * asterisk")
        generator.add_table(
            "Special",
            headers=["Name"],
            rows=[["Value with | pipe"]],
        )

        output = generator.render()

        # Should not crash
        assert "Content with | pipe" in output

    def test_unicode_content(self):
        """Test handling unicode content."""
        generator = ReportGenerator(style=ReportStyle.PLAIN)
        generator.set_title("Report")
        generator.add_text_section("Unicode", "Content with unicode chars")
        generator.add_metrics("Stats", {"Count": "100"})

        output = generator.render()
        assert "unicode" in output.lower()

    def test_large_table(self):
        """Test rendering a large table."""
        generator = ReportGenerator(style=ReportStyle.PLAIN)
        rows = [[f"Item {i}", f"Value {i}"] for i in range(100)]
        generator.add_table("Large Table", headers=["Item", "Value"], rows=rows)

        output = generator.render()

        assert "Item 0" in output
        assert "Item 99" in output

    def test_method_chaining(self):
        """Test that method chaining works correctly."""
        generator = (
            ReportGenerator()
            .set_title("Chained Report")
            .add_text_section("Section 1", "Content 1")
            .add_metrics("Metrics", {"value": 100})
            .add_warning("A warning")
        )

        assert generator.title == "Chained Report"
        assert len(generator.sections) == 3


class TestReportGeneratorIntegration:
    """Integration tests for complete report generation workflows."""

    def test_complete_report_workflow(self, tmp_path):
        """Test a complete report generation workflow."""
        # Create generator
        generator = ReportGenerator()

        # Set title
        generator.set_title("Processing Report", "Generated 2024-01-01")

        # Add summary
        generator.add_text_section(
            "Summary",
            "This report summarizes the bookmark processing results.",
            icon="info",
        )

        # Add statistics
        generator.add_metrics(
            "Statistics",
            {
                "Total Processed": "3,500",
                "Success Rate": "95.5%",
                "Failed": "157",
                "Duration": "2h 15m",
            },
            icon="chart",
        )

        # Add results table
        generator.add_table(
            "Top Errors",
            headers=["Error Type", "Count", "Percentage"],
            rows=[
                ["Timeout", 89, "56.7%"],
                ["404 Not Found", 45, "28.7%"],
                ["Connection Refused", 23, "14.6%"],
            ],
            icon="error",
        )

        # Add warning
        generator.add_warning("157 bookmarks require manual review")

        # Save in multiple formats
        md_path = tmp_path / "report.md"
        json_path = tmp_path / "report.json"
        txt_path = tmp_path / "report.txt"

        generator.save(md_path)
        generator.save(json_path)
        generator.save(txt_path)

        # Verify all files were created
        assert md_path.exists()
        assert json_path.exists()
        assert txt_path.exists()

        # Verify Markdown content
        md_content = md_path.read_text()
        assert "# Processing Report" in md_content
        assert "## Statistics" in md_content
        assert "| Timeout |" in md_content

        # Verify JSON content
        json_content = json_path.read_text()
        data = json.loads(json_content)
        assert data["title"] == "Processing Report"
        assert len(data["sections"]) == 4

        # Verify plain text content
        txt_content = txt_path.read_text(encoding="utf-8")
        assert "Processing Report" in txt_content
        assert "STATISTICS" in txt_content

    def test_report_with_tree_structure(self):
        """Test report with hierarchical tree data."""
        generator = ReportGenerator(style=ReportStyle.PLAIN)

        generator.set_title("Folder Structure")
        generator.add_tree(
            "Hierarchy",
            {
                "Tech": {
                    "Programming": {
                        "Python": 45,
                        "JavaScript": 32,
                    },
                    "AI": {
                        "Machine Learning": 28,
                        "NLP": 15,
                    },
                },
                "Personal": {
                    "Finance": 20,
                    "Health": 12,
                },
            },
        )

        output = generator.render()

        assert "Folder Structure" in output
        assert "HIERARCHY" in output
        assert "Tech" in output
        assert "Python" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
