"""
JSON bookmark exporter.

Exports bookmarks to JSON format with full metadata preservation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BookmarkExporter, ExportResult, ExportError
from ..data_models import Bookmark


class JSONExporter(BookmarkExporter):
    """
    Export bookmarks to JSON format.

    Supports both full metadata export and compact mode for simpler output.
    The output includes all bookmark fields and can be used for backup
    or data interchange purposes.

    Example:
        >>> exporter = JSONExporter(indent=2, include_metadata=True)
        >>> result = exporter.export(bookmarks, Path("bookmarks.json"))
    """

    def __init__(
        self,
        indent: int = 2,
        include_metadata: bool = True,
        ensure_ascii: bool = False,
        sort_keys: bool = False,
        compact: bool = False
    ):
        """
        Initialize the JSON exporter.

        Args:
            indent: Number of spaces for indentation (None for no formatting)
            include_metadata: Whether to include processing metadata
            ensure_ascii: Whether to escape non-ASCII characters
            sort_keys: Whether to sort dictionary keys
            compact: If True, use minimal formatting (overrides indent)
        """
        super().__init__()
        self.indent = None if compact else indent
        self.include_metadata = include_metadata
        self.ensure_ascii = ensure_ascii
        self.sort_keys = sort_keys
        self.compact = compact

    @property
    def format_name(self) -> str:
        return "JSON"

    @property
    def file_extension(self) -> str:
        return "json"

    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks to a JSON file.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for the JSON file

        Returns:
            ExportResult with export details

        Raises:
            ExportError: If export fails
        """
        warnings = self.validate_bookmarks(bookmarks)

        if not bookmarks:
            raise ExportError(
                "No bookmarks to export",
                format_name=self.format_name
            )

        # Prepare output path
        path = self.prepare_output_path(output_path)

        # Ensure correct extension
        if not str(path).lower().endswith(".json"):
            path = path.with_suffix(".json")

        try:
            # Build export data
            export_data = self._build_export_data(bookmarks)

            # Write JSON file
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    export_data,
                    f,
                    indent=self.indent,
                    ensure_ascii=self.ensure_ascii,
                    sort_keys=self.sort_keys,
                    default=str  # Handle datetime and other types
                )

            self.logger.info(f"Exported {len(bookmarks)} bookmarks to {path}")

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=self.format_name,
                additional_info={
                    "include_metadata": self.include_metadata,
                    "compact": self.compact,
                    "file_size": path.stat().st_size
                },
                warnings=warnings
            )

        except PermissionError as e:
            raise ExportError(
                f"Permission denied writing to {path}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )
        except Exception as e:
            raise ExportError(
                f"Failed to export JSON: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )

    def _build_export_data(self, bookmarks: List[Bookmark]) -> Dict[str, Any]:
        """
        Build the export data structure.

        Args:
            bookmarks: List of bookmarks

        Returns:
            Dictionary with export data
        """
        bookmark_list = [
            self.bookmark_to_dict(b, include_metadata=self.include_metadata)
            for b in bookmarks
        ]

        # Group by folder for easier navigation
        folders: Dict[str, List[Dict]] = {}
        for bookmark in bookmark_list:
            folder = bookmark.get("folder") or "Unsorted"
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(bookmark)

        return {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "total_bookmarks": len(bookmarks),
                "format_version": "1.0",
                "generator": "bookmark-processor",
            },
            "bookmarks": bookmark_list,
            "by_folder": folders,
            "statistics": self._compute_statistics(bookmarks),
        }

    def _compute_statistics(self, bookmarks: List[Bookmark]) -> Dict[str, Any]:
        """
        Compute statistics about the bookmarks.

        Args:
            bookmarks: List of bookmarks

        Returns:
            Dictionary with statistics
        """
        all_tags = set()
        all_folders = set()

        for bookmark in bookmarks:
            all_tags.update(bookmark.get_final_tags())
            if bookmark.folder:
                all_folders.add(bookmark.folder)

        return {
            "total_count": len(bookmarks),
            "unique_tags": len(all_tags),
            "unique_folders": len(all_folders),
            "with_descriptions": sum(
                1 for b in bookmarks if b.get_effective_description()
            ),
            "with_tags": sum(1 for b in bookmarks if b.get_final_tags()),
            "favorites": sum(1 for b in bookmarks if b.favorite),
        }

    def export_minimal(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks with minimal data (URL, title, tags only).

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for the JSON file

        Returns:
            ExportResult with export details
        """
        path = self.prepare_output_path(output_path)

        if not str(path).lower().endswith(".json"):
            path = path.with_suffix(".json")

        minimal_data = [
            {
                "url": b.url,
                "title": b.get_effective_title(),
                "tags": b.get_final_tags()
            }
            for b in bookmarks
        ]

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(minimal_data, f, indent=self.indent, ensure_ascii=self.ensure_ascii)

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=f"{self.format_name} (minimal)",
                additional_info={"minimal": True}
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export minimal JSON: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )
