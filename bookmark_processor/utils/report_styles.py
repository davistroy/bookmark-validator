"""
Report Style Definitions and Templates.

This module defines the styles and templates used for report generation,
supporting terminal (Rich), markdown, JSON, and plain text output formats.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReportStyle(Enum):
    """Enumeration of supported report output styles."""

    RICH = "rich"           # Rich console with colors/icons
    PLAIN = "plain"         # Plain text for piping
    MARKDOWN = "markdown"   # Markdown format
    JSON = "json"           # JSON format for programmatic access


@dataclass
class StyleConfig:
    """Configuration for a specific report style."""

    # Whether to use icons/emojis
    use_icons: bool = True

    # Whether to use colors (for terminal output)
    use_colors: bool = True

    # Whether to use borders/boxes
    use_borders: bool = True

    # Indentation string
    indent: str = "  "

    # Section header format
    header_format: str = "═"

    # Sub-section header format
    subheader_format: str = "─"

    # Bullet point character
    bullet: str = "•"

    # Tree connector characters
    tree_connector: str = "├─"
    tree_last_connector: str = "└─"
    tree_vertical: str = "│"


# Predefined style configurations
STYLE_CONFIGS: Dict[ReportStyle, StyleConfig] = {
    ReportStyle.RICH: StyleConfig(
        use_icons=True,
        use_colors=True,
        use_borders=True,
        indent="  ",
        header_format="═",
        subheader_format="─",
        bullet="•",
        tree_connector="├─",
        tree_last_connector="└─",
        tree_vertical="│",
    ),
    ReportStyle.PLAIN: StyleConfig(
        use_icons=False,
        use_colors=False,
        use_borders=False,
        indent="  ",
        header_format="=",
        subheader_format="-",
        bullet="-",
        tree_connector="|-",
        tree_last_connector="`-",
        tree_vertical="|",
    ),
    ReportStyle.MARKDOWN: StyleConfig(
        use_icons=False,
        use_colors=False,
        use_borders=False,
        indent="  ",
        header_format="#",
        subheader_format="##",
        bullet="-",
        tree_connector="  -",
        tree_last_connector="  -",
        tree_vertical="",
    ),
    ReportStyle.JSON: StyleConfig(
        use_icons=False,
        use_colors=False,
        use_borders=False,
        indent="  ",
        header_format="",
        subheader_format="",
        bullet="",
        tree_connector="",
        tree_last_connector="",
        tree_vertical="",
    ),
}


# Icon mappings for different content types
ICONS: Dict[str, str] = {
    # Status icons
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "pending": "⏳",
    "complete": "✓",
    "failed": "✗",

    # Category icons
    "description": "📝",
    "tags": "🏷️",
    "folder": "📁",
    "url": "🔗",
    "time": "⏱️",
    "memory": "💾",
    "rate": "🚀",
    "chart": "📊",
    "metrics": "📈",
    "quality": "⭐",

    # Action icons
    "processing": "🔄",
    "validation": "🔍",
    "ai": "🤖",
    "checkpoint": "💾",

    # Alert icons
    "attention": "🔔",
    "critical": "🚨",
    "review": "👀",
}


def get_icon(icon_name: str, style: ReportStyle = ReportStyle.RICH) -> str:
    """
    Get an icon for the given name, respecting the style settings.

    Args:
        icon_name: The name of the icon to retrieve
        style: The report style to use

    Returns:
        The icon string or empty string if icons are disabled
    """
    config = STYLE_CONFIGS.get(style, STYLE_CONFIGS[ReportStyle.PLAIN])

    if not config.use_icons:
        return ""

    return ICONS.get(icon_name, "")


def get_style_config(style: ReportStyle) -> StyleConfig:
    """
    Get the style configuration for a given style.

    Args:
        style: The report style

    Returns:
        StyleConfig for the given style
    """
    return STYLE_CONFIGS.get(style, STYLE_CONFIGS[ReportStyle.PLAIN])


# Color definitions for Rich console
RICH_COLORS: Dict[str, str] = {
    "header": "bold cyan",
    "subheader": "bold blue",
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "highlight": "bold white",
    "muted": "dim",
    "metric_value": "bold green",
    "metric_label": "white",
    "percentage_high": "green",
    "percentage_medium": "yellow",
    "percentage_low": "red",
}


def get_percentage_color(value: float, thresholds: tuple = (70.0, 40.0)) -> str:
    """
    Get the appropriate color for a percentage value.

    Args:
        value: The percentage value (0-100)
        thresholds: Tuple of (high_threshold, low_threshold)

    Returns:
        Color name for the value
    """
    high, low = thresholds

    if value >= high:
        return RICH_COLORS["percentage_high"]
    elif value >= low:
        return RICH_COLORS["percentage_medium"]
    else:
        return RICH_COLORS["percentage_low"]
