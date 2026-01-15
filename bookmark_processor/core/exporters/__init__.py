"""
Multi-format bookmark exporters.

This module provides exporters for various bookmark formats including
JSON, Markdown, Obsidian, Notion, and OPML.
"""

from .base import BookmarkExporter, ExportResult, ExportError
from .json_exporter import JSONExporter
from .markdown_exporter import MarkdownExporter
from .obsidian_exporter import ObsidianExporter
from .notion_exporter import NotionExporter
from .opml_exporter import OPMLExporter

__all__ = [
    "BookmarkExporter",
    "ExportResult",
    "ExportError",
    "JSONExporter",
    "MarkdownExporter",
    "ObsidianExporter",
    "NotionExporter",
    "OPMLExporter",
]


# Format registry for easy access
EXPORTERS = {
    "json": JSONExporter,
    "markdown": MarkdownExporter,
    "md": MarkdownExporter,
    "obsidian": ObsidianExporter,
    "notion": NotionExporter,
    "opml": OPMLExporter,
}


def get_exporter(format_name: str) -> type:
    """
    Get an exporter class by format name.

    Args:
        format_name: Name of the format (json, markdown, obsidian, notion, opml)

    Returns:
        Exporter class for the specified format

    Raises:
        ValueError: If format is not supported
    """
    format_lower = format_name.lower()
    if format_lower not in EXPORTERS:
        supported = ", ".join(sorted(set(EXPORTERS.keys()) - {"md"}))
        raise ValueError(
            f"Unsupported export format: {format_name}. "
            f"Supported formats: {supported}"
        )
    return EXPORTERS[format_lower]
