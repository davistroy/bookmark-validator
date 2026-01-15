"""
Base classes for bookmark exporters.

This module provides the abstract base class and common utilities
for all bookmark export formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from ..data_models import Bookmark


@dataclass
class ExportResult:
    """
    Result of an export operation.

    Attributes:
        path: Path to the exported file or directory
        count: Number of bookmarks exported
        format_name: Name of the export format used
        exported_at: Timestamp of the export
        additional_info: Any format-specific additional information
        warnings: List of non-fatal warnings during export
    """

    path: Path
    count: int
    format_name: str
    exported_at: datetime = field(default_factory=datetime.now)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"ExportResult(format={self.format_name}, "
            f"count={self.count}, path={self.path})"
        )


class ExportError(Exception):
    """
    Exception raised when export fails.

    Attributes:
        message: Error description
        format_name: Name of the export format
        path: Target path if available
        original_error: Underlying exception if any
    """

    def __init__(
        self,
        message: str,
        format_name: Optional[str] = None,
        path: Optional[Path] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.format_name = format_name
        self.path = path
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = []
        if self.format_name:
            parts.append(f"[{self.format_name}]")
        parts.append(self.message)
        if self.path:
            parts.append(f"(path: {self.path})")
        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")
        return " ".join(parts)


class BookmarkExporter(ABC):
    """
    Abstract base class for bookmark exporters.

    All exporters must implement the export() method and define
    format_name and file_extension properties.

    Example:
        >>> exporter = JSONExporter()
        >>> result = exporter.export(bookmarks, Path("output.json"))
        >>> print(f"Exported {result.count} bookmarks to {result.path}")
    """

    def __init__(self):
        """Initialize the exporter."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks to the specified path.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Target path for the export

        Returns:
            ExportResult with details about the export

        Raises:
            ExportError: If export fails
        """
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """
        Human-readable name of the export format.

        Returns:
            Format name string (e.g., "JSON", "Markdown")
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """
        Default file extension for this format.

        Returns:
            Extension string without leading dot (e.g., "json", "md")
        """
        pass

    def validate_bookmarks(self, bookmarks: List[Bookmark]) -> List[str]:
        """
        Validate bookmarks before export.

        Args:
            bookmarks: List of bookmarks to validate

        Returns:
            List of warning messages for any issues found
        """
        warnings = []

        if not bookmarks:
            warnings.append("No bookmarks provided for export")
            return warnings

        # Check for bookmarks without URLs
        no_url_count = sum(1 for b in bookmarks if not b.url)
        if no_url_count > 0:
            warnings.append(f"{no_url_count} bookmark(s) have no URL")

        # Check for bookmarks without titles
        no_title_count = sum(1 for b in bookmarks if not b.get_effective_title())
        if no_title_count > 0:
            warnings.append(f"{no_title_count} bookmark(s) have no title")

        return warnings

    def prepare_output_path(
        self,
        output_path: Union[str, Path],
        is_directory: bool = False
    ) -> Path:
        """
        Prepare and validate the output path.

        Args:
            output_path: Target path for export
            is_directory: Whether the path should be a directory

        Returns:
            Validated Path object

        Raises:
            ExportError: If path is invalid or cannot be created
        """
        path = Path(output_path)

        try:
            if is_directory:
                path.mkdir(parents=True, exist_ok=True)
            else:
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise ExportError(
                f"Permission denied creating path: {path}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )
        except Exception as e:
            raise ExportError(
                f"Failed to prepare output path: {path}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )

        return path

    def bookmark_to_dict(
        self,
        bookmark: Bookmark,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert a bookmark to a dictionary for serialization.

        Args:
            bookmark: Bookmark to convert
            include_metadata: Whether to include processing metadata

        Returns:
            Dictionary representation of the bookmark
        """
        data = {
            "url": bookmark.url,
            "title": bookmark.get_effective_title(),
            "description": bookmark.get_effective_description(),
            "folder": bookmark.folder,
            "tags": bookmark.get_final_tags(),
            "created": (
                bookmark.created.isoformat()
                if bookmark.created
                else None
            ),
        }

        if include_metadata:
            data["id"] = bookmark.id
            data["note"] = bookmark.note
            data["excerpt"] = bookmark.excerpt
            data["cover"] = bookmark.cover
            data["favorite"] = bookmark.favorite

            # Include processing status if available
            if bookmark.processing_status:
                data["processing_status"] = {
                    "url_validated": bookmark.processing_status.url_validated,
                    "content_extracted": bookmark.processing_status.content_extracted,
                    "ai_processed": bookmark.processing_status.ai_processed,
                    "tags_optimized": bookmark.processing_status.tags_optimized,
                }

            # Include enhanced data
            if bookmark.enhanced_description:
                data["enhanced_description"] = bookmark.enhanced_description
            if bookmark.optimized_tags:
                data["optimized_tags"] = bookmark.optimized_tags

        return data

    def sanitize_filename(self, name: str, max_length: int = 100) -> str:
        """
        Sanitize a string for use as a filename.

        Args:
            name: String to sanitize
            max_length: Maximum length of the result

        Returns:
            Sanitized filename-safe string
        """
        if not name:
            return "untitled"

        # Replace problematic characters
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = re.sub(r'\s+', " ", sanitized)
        sanitized = sanitized.strip(". ")

        # Truncate if necessary
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rsplit(" ", 1)[0]

        return sanitized or "untitled"

    def format_tags(self, tags: List[str], separator: str = ", ") -> str:
        """
        Format a list of tags as a string.

        Args:
            tags: List of tag strings
            separator: Separator between tags

        Returns:
            Formatted tag string
        """
        if not tags:
            return ""
        return separator.join(tags)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(format={self.format_name})"
