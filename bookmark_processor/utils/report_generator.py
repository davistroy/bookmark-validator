"""
Report Generation Infrastructure.

This module provides a flexible report generation system that supports
multiple output formats: terminal (Rich), markdown, JSON, and plain text.
"""

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .report_styles import (
    ICONS,
    RICH_COLORS,
    ReportStyle,
    StyleConfig,
    get_icon,
    get_percentage_color,
    get_style_config,
)

# Rich library imports with graceful fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Panel = None
    Table = None
    Text = None
    Tree = None


@dataclass
class ReportSection:
    """
    Single section of a report with title and content.

    A section can contain various types of content including
    text, tables, metrics, warnings, and nested subsections.
    """

    title: str
    content: Union[str, Dict[str, Any], List[Any], None] = None
    icon: Optional[str] = None
    subsections: List["ReportSection"] = field(default_factory=list)
    section_type: str = "text"  # text, table, metrics, warning, tree

    def add_subsection(self, subsection: "ReportSection") -> None:
        """Add a subsection to this section."""
        self.subsections.append(subsection)

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary representation."""
        result = {
            "title": self.title,
            "content": self.content,
            "type": self.section_type,
        }

        if self.icon:
            result["icon"] = self.icon

        if self.subsections:
            result["subsections"] = [s.to_dict() for s in self.subsections]

        return result


@dataclass
class TableData:
    """Data structure for table content."""

    headers: List[str]
    rows: List[List[Any]]
    title: Optional[str] = None
    alignments: Optional[List[str]] = None  # left, center, right


class ReportGenerator:
    """
    Generate formatted reports in multiple styles.

    Supports Rich terminal output, markdown, JSON, and plain text formats.
    Provides a fluent interface for building reports section by section.
    """

    def __init__(self, style: ReportStyle = ReportStyle.RICH):
        """
        Initialize the report generator.

        Args:
            style: The output style to use (default: RICH)
        """
        self.style = style
        self.config = get_style_config(style)
        self.sections: List[ReportSection] = []
        self.title: Optional[str] = None
        self.subtitle: Optional[str] = None
        self._console: Optional["Console"] = None

    @property
    def console(self) -> Optional["Console"]:
        """Get or create Rich console instance."""
        if self._console is None and RICH_AVAILABLE:
            self._console = Console()
        return self._console

    def set_title(self, title: str, subtitle: Optional[str] = None) -> "ReportGenerator":
        """
        Set the report title and optional subtitle.

        Args:
            title: The main report title
            subtitle: Optional subtitle

        Returns:
            Self for method chaining
        """
        self.title = title
        self.subtitle = subtitle
        return self

    def add_section(self, section: ReportSection) -> "ReportGenerator":
        """
        Add a section to the report.

        Args:
            section: The section to add

        Returns:
            Self for method chaining
        """
        self.sections.append(section)
        return self

    def add_text_section(
        self,
        title: str,
        content: str,
        icon: Optional[str] = None,
    ) -> "ReportGenerator":
        """
        Add a simple text section.

        Args:
            title: Section title
            content: Text content
            icon: Optional icon name

        Returns:
            Self for method chaining
        """
        section = ReportSection(
            title=title,
            content=content,
            icon=icon,
            section_type="text",
        )
        return self.add_section(section)

    def add_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[Any]],
        icon: Optional[str] = None,
    ) -> "ReportGenerator":
        """
        Add a table section.

        Args:
            title: Table title
            headers: List of column headers
            rows: List of row data
            icon: Optional icon name

        Returns:
            Self for method chaining
        """
        table_data = TableData(headers=headers, rows=rows, title=title)
        section = ReportSection(
            title=title,
            content={"headers": headers, "rows": rows},
            icon=icon,
            section_type="table",
        )
        return self.add_section(section)

    def add_metrics(
        self,
        title: str,
        metrics: Dict[str, Any],
        icon: Optional[str] = None,
    ) -> "ReportGenerator":
        """
        Add a metrics section with key-value pairs.

        Args:
            title: Section title
            metrics: Dictionary of metric names to values
            icon: Optional icon name

        Returns:
            Self for method chaining
        """
        section = ReportSection(
            title=title,
            content=metrics,
            icon=icon,
            section_type="metrics",
        )
        return self.add_section(section)

    def add_warning(self, message: str, icon: Optional[str] = "warning") -> "ReportGenerator":
        """
        Add a warning message.

        Args:
            message: Warning message text
            icon: Icon name (default: "warning")

        Returns:
            Self for method chaining
        """
        section = ReportSection(
            title="Warning",
            content=message,
            icon=icon,
            section_type="warning",
        )
        return self.add_section(section)

    def add_tree(
        self,
        title: str,
        tree_data: Dict[str, Any],
        icon: Optional[str] = None,
    ) -> "ReportGenerator":
        """
        Add a tree structure section.

        Args:
            title: Section title
            tree_data: Nested dictionary representing tree structure
            icon: Optional icon name

        Returns:
            Self for method chaining
        """
        section = ReportSection(
            title=title,
            content=tree_data,
            icon=icon,
            section_type="tree",
        )
        return self.add_section(section)

    def render(self) -> str:
        """
        Render the report in the configured style.

        Returns:
            Rendered report as string
        """
        if self.style == ReportStyle.RICH:
            return self.render_terminal()
        elif self.style == ReportStyle.MARKDOWN:
            return self.render_markdown()
        elif self.style == ReportStyle.JSON:
            return json.dumps(self.render_json(), indent=2)
        else:
            return self.render_plain()

    def render_terminal(self) -> str:
        """
        Render the report for terminal output using Rich library.

        Returns:
            Terminal-formatted report string
        """
        if not RICH_AVAILABLE:
            return self.render_plain()

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)

        # Render title
        if self.title:
            title_text = self.title
            if self.subtitle:
                title_text = f"{self.title}\n{self.subtitle}"
            console.print(Panel(title_text, style=RICH_COLORS["header"], expand=False))
            console.print()

        # Render each section
        for section in self.sections:
            self._render_terminal_section(console, section)
            console.print()

        return output.getvalue()

    def _render_terminal_section(
        self,
        console: "Console",
        section: ReportSection,
        level: int = 0,
    ) -> None:
        """Render a single section for terminal output."""
        indent = self.config.indent * level
        icon_str = get_icon(section.icon, self.style) if section.icon else ""

        # Render section title
        title_parts = []
        if icon_str:
            title_parts.append(icon_str)
        title_parts.append(section.title.upper() if level == 0 else section.title)
        title = " ".join(title_parts)

        if level == 0:
            console.print(f"[{RICH_COLORS['header']}]{title}[/]")
            console.print(f"[dim]{self.config.header_format * len(section.title)}[/]")
        else:
            console.print(f"{indent}[{RICH_COLORS['subheader']}]{title}[/]")

        # Render content based on type
        if section.section_type == "text" and section.content:
            console.print(f"{indent}{self.config.indent}{section.content}")

        elif section.section_type == "table" and section.content:
            self._render_terminal_table(console, section.content, indent)

        elif section.section_type == "metrics" and section.content:
            self._render_terminal_metrics(console, section.content, indent)

        elif section.section_type == "warning" and section.content:
            console.print(
                f"{indent}[{RICH_COLORS['warning']}]{icon_str} {section.content}[/]"
            )

        elif section.section_type == "tree" and section.content:
            self._render_terminal_tree(console, section.content, indent)

        # Render subsections
        for subsection in section.subsections:
            self._render_terminal_section(console, subsection, level + 1)

    def _render_terminal_table(
        self,
        console: "Console",
        table_data: Dict[str, Any],
        indent: str,
    ) -> None:
        """Render a table for terminal output."""
        if not RICH_AVAILABLE:
            return

        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        table = Table(show_header=True, header_style=RICH_COLORS["header"])

        for header in headers:
            table.add_column(str(header))

        for row in rows:
            table.add_row(*[str(cell) for cell in row])

        console.print(table)

    def _render_terminal_metrics(
        self,
        console: "Console",
        metrics: Dict[str, Any],
        indent: str,
    ) -> None:
        """Render metrics for terminal output."""
        tree_chars = self.config

        items = list(metrics.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = tree_chars.tree_last_connector if is_last else tree_chars.tree_connector

            # Format value with color if it's a percentage
            if isinstance(value, str) and "%" in value:
                try:
                    pct_value = float(value.replace("%", "").strip().split()[0])
                    color = get_percentage_color(pct_value)
                    formatted_value = f"[{color}]{value}[/]"
                except (ValueError, IndexError):
                    formatted_value = f"[{RICH_COLORS['metric_value']}]{value}[/]"
            else:
                formatted_value = f"[{RICH_COLORS['metric_value']}]{value}[/]"

            console.print(
                f"{indent}{connector} [{RICH_COLORS['metric_label']}]{key}:[/] {formatted_value}"
            )

    def _render_terminal_tree(
        self,
        console: "Console",
        tree_data: Dict[str, Any],
        indent: str,
    ) -> None:
        """Render a tree structure for terminal output."""
        if not RICH_AVAILABLE:
            return

        def build_tree(data: Dict[str, Any], tree: "Tree") -> None:
            for key, value in data.items():
                if isinstance(value, dict):
                    branch = tree.add(f"[bold]{key}[/bold]")
                    build_tree(value, branch)
                else:
                    tree.add(f"{key}: {value}")

        root_tree = Tree("[bold]Root[/bold]")
        build_tree(tree_data, root_tree)
        console.print(root_tree)

    def render_markdown(self) -> str:
        """
        Render the report as Markdown.

        Returns:
            Markdown-formatted report string
        """
        lines = []

        # Render title
        if self.title:
            lines.append(f"# {self.title}")
            if self.subtitle:
                lines.append(f"\n*{self.subtitle}*")
            lines.append("")

        # Render each section
        for section in self.sections:
            self._render_markdown_section(lines, section, level=2)

        return "\n".join(lines)

    def _render_markdown_section(
        self,
        lines: List[str],
        section: ReportSection,
        level: int = 2,
    ) -> None:
        """Render a single section as Markdown."""
        # Section header
        header_prefix = "#" * level
        lines.append(f"{header_prefix} {section.title}")
        lines.append("")

        # Render content based on type
        if section.section_type == "text" and section.content:
            lines.append(str(section.content))
            lines.append("")

        elif section.section_type == "table" and section.content:
            self._render_markdown_table(lines, section.content)

        elif section.section_type == "metrics" and section.content:
            self._render_markdown_metrics(lines, section.content)

        elif section.section_type == "warning" and section.content:
            lines.append(f"> **Warning:** {section.content}")
            lines.append("")

        elif section.section_type == "tree" and section.content:
            self._render_markdown_tree(lines, section.content)

        # Render subsections
        for subsection in section.subsections:
            self._render_markdown_section(lines, subsection, level + 1)

    def _render_markdown_table(
        self,
        lines: List[str],
        table_data: Dict[str, Any],
    ) -> None:
        """Render a table as Markdown."""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not headers:
            return

        # Header row
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")

        # Separator row
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        # Data rows
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        lines.append("")

    def _render_markdown_metrics(
        self,
        lines: List[str],
        metrics: Dict[str, Any],
    ) -> None:
        """Render metrics as Markdown."""
        for key, value in metrics.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    def _render_markdown_tree(
        self,
        lines: List[str],
        tree_data: Dict[str, Any],
        indent: str = "",
    ) -> None:
        """Render a tree structure as Markdown."""
        for key, value in tree_data.items():
            if isinstance(value, dict):
                lines.append(f"{indent}- **{key}**")
                self._render_markdown_tree(lines, value, indent + "  ")
            else:
                lines.append(f"{indent}- {key}: {value}")

        if not indent:
            lines.append("")

    def render_plain(self) -> str:
        """
        Render the report as plain text.

        Returns:
            Plain text report string
        """
        lines = []

        # Render title
        if self.title:
            lines.append(self.config.header_format * 60)
            lines.append(self.title.center(60))
            if self.subtitle:
                lines.append(self.subtitle.center(60))
            lines.append(self.config.header_format * 60)
            lines.append("")

        # Render each section
        for section in self.sections:
            self._render_plain_section(lines, section)

        return "\n".join(lines)

    def _render_plain_section(
        self,
        lines: List[str],
        section: ReportSection,
        level: int = 0,
    ) -> None:
        """Render a single section as plain text."""
        indent = self.config.indent * level

        # Section header
        title = section.title.upper() if level == 0 else section.title
        lines.append(f"{indent}{title}")
        lines.append(f"{indent}{self.config.subheader_format * len(section.title)}")

        # Render content based on type
        if section.section_type == "text" and section.content:
            lines.append(f"{indent}{self.config.indent}{section.content}")

        elif section.section_type == "table" and section.content:
            self._render_plain_table(lines, section.content, indent)

        elif section.section_type == "metrics" and section.content:
            self._render_plain_metrics(lines, section.content, indent)

        elif section.section_type == "warning" and section.content:
            lines.append(f"{indent}WARNING: {section.content}")

        elif section.section_type == "tree" and section.content:
            self._render_plain_tree(lines, section.content, indent)

        lines.append("")

        # Render subsections
        for subsection in section.subsections:
            self._render_plain_section(lines, subsection, level + 1)

    def _render_plain_table(
        self,
        lines: List[str],
        table_data: Dict[str, Any],
        indent: str,
    ) -> None:
        """Render a table as plain text."""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not headers:
            return

        # Calculate column widths
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Header row
        header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
        lines.append(f"{indent}{header_line}")

        # Separator
        sep_line = "-+-".join("-" * w for w in widths)
        lines.append(f"{indent}{sep_line}")

        # Data rows
        for row in rows:
            data_line = " | ".join(
                str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                for i, cell in enumerate(row)
            )
            lines.append(f"{indent}{data_line}")

    def _render_plain_metrics(
        self,
        lines: List[str],
        metrics: Dict[str, Any],
        indent: str,
    ) -> None:
        """Render metrics as plain text."""
        items = list(metrics.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = (
                self.config.tree_last_connector
                if is_last
                else self.config.tree_connector
            )
            lines.append(f"{indent}{connector} {key}: {value}")

    def _render_plain_tree(
        self,
        lines: List[str],
        tree_data: Dict[str, Any],
        indent: str,
        prefix: str = "",
    ) -> None:
        """Render a tree structure as plain text."""
        items = list(tree_data.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = (
                self.config.tree_last_connector
                if is_last
                else self.config.tree_connector
            )
            continuation = "  " if is_last else self.config.tree_vertical + " "

            if isinstance(value, dict):
                lines.append(f"{indent}{prefix}{connector} {key}")
                self._render_plain_tree(
                    lines, value, indent, prefix + continuation
                )
            else:
                lines.append(f"{indent}{prefix}{connector} {key}: {value}")

    def render_json(self) -> Dict[str, Any]:
        """
        Render the report as a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the report
        """
        result: Dict[str, Any] = {}

        if self.title:
            result["title"] = self.title
            if self.subtitle:
                result["subtitle"] = self.subtitle

        result["sections"] = [section.to_dict() for section in self.sections]

        return result

    def save(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
    ) -> None:
        """
        Save the report to a file.

        Args:
            path: File path to save to
            format: Output format (md, json, txt). Auto-detected from extension if not provided.
        """
        path = Path(path)

        # Auto-detect format from extension
        if format is None:
            ext = path.suffix.lower()
            format_map = {
                ".md": "md",
                ".markdown": "md",
                ".json": "json",
                ".txt": "txt",
                ".text": "txt",
            }
            format = format_map.get(ext, "txt")

        # Render in appropriate format
        original_style = self.style

        if format == "md":
            self.style = ReportStyle.MARKDOWN
            content = self.render_markdown()
        elif format == "json":
            self.style = ReportStyle.JSON
            content = json.dumps(self.render_json(), indent=2)
        else:
            self.style = ReportStyle.PLAIN
            content = self.render_plain()

        # Restore original style
        self.style = original_style

        # Write to file
        path.write_text(content, encoding="utf-8")

    def print_to_console(self) -> None:
        """Print the report directly to the console."""
        if RICH_AVAILABLE and self.style == ReportStyle.RICH:
            # Use Rich console for direct printing
            console = Console()

            # Print title
            if self.title:
                title_text = self.title
                if self.subtitle:
                    title_text = f"{self.title}\n{self.subtitle}"
                console.print(Panel(title_text, style=RICH_COLORS["header"], expand=False))
                console.print()

            # Print each section
            for section in self.sections:
                self._render_terminal_section(console, section)
                console.print()
        else:
            # Fall back to print
            print(self.render())

    def clear(self) -> "ReportGenerator":
        """
        Clear all sections from the report.

        Returns:
            Self for method chaining
        """
        self.sections = []
        self.title = None
        self.subtitle = None
        return self
