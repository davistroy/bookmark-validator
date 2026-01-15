"""
Notion-compatible CSV bookmark exporter.

Exports bookmarks to CSV format optimized for Notion database import.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import BookmarkExporter, ExportResult, ExportError
from ..data_models import Bookmark


class NotionExporter(BookmarkExporter):
    """
    Export bookmarks to Notion-compatible CSV format.

    Creates a CSV file that can be imported directly into a Notion database.
    Supports Notion's specific field types and formatting requirements.

    Example:
        >>> exporter = NotionExporter(include_status=True)
        >>> result = exporter.export(bookmarks, Path("notion_import.csv"))
    """

    # Notion database column headers
    DEFAULT_COLUMNS = [
        "Name",
        "URL",
        "Tags",
        "Description",
        "Folder",
        "Created",
        "Favorite",
    ]

    def __init__(
        self,
        include_status: bool = False,
        include_processing_info: bool = False,
        tag_separator: str = ", ",
        date_format: str = "%Y-%m-%d",
        use_notion_date_format: bool = True,
        custom_columns: Optional[List[str]] = None
    ):
        """
        Initialize the Notion exporter.

        Args:
            include_status: Include processing status column
            include_processing_info: Include detailed processing info columns
            tag_separator: Separator for multiple tags
            date_format: Format for date fields
            use_notion_date_format: Use Notion's preferred date format (YYYY-MM-DD)
            custom_columns: Optional list of custom column names to include
        """
        super().__init__()
        self.include_status = include_status
        self.include_processing_info = include_processing_info
        self.tag_separator = tag_separator
        self.date_format = "%Y-%m-%d" if use_notion_date_format else date_format
        self.custom_columns = custom_columns or []

    @property
    def format_name(self) -> str:
        return "Notion CSV"

    @property
    def file_extension(self) -> str:
        return "csv"

    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks to a Notion-compatible CSV file.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for the CSV file

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

        if not str(path).lower().endswith(".csv"):
            path = path.with_suffix(".csv")

        try:
            # Determine columns
            columns = self._get_columns()

            # Write CSV file
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
                writer.writeheader()

                for bookmark in bookmarks:
                    row = self._bookmark_to_row(bookmark)
                    writer.writerow(row)

            self.logger.info(f"Exported {len(bookmarks)} bookmarks to {path}")

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=self.format_name,
                additional_info={
                    "columns": columns,
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
                f"Failed to export Notion CSV: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )

    def _get_columns(self) -> List[str]:
        """Get the list of columns for the CSV."""
        columns = self.DEFAULT_COLUMNS.copy()

        if self.include_status:
            columns.append("Status")

        if self.include_processing_info:
            columns.extend([
                "URL Validated",
                "Content Extracted",
                "AI Processed",
                "Tags Optimized"
            ])

        # Add custom columns
        for col in self.custom_columns:
            if col not in columns:
                columns.append(col)

        return columns

    def _bookmark_to_row(self, bookmark: Bookmark) -> Dict[str, str]:
        """Convert a bookmark to a CSV row dictionary."""
        row = {
            "Name": bookmark.get_effective_title(),
            "URL": bookmark.url,
            "Tags": self._format_tags(bookmark.get_final_tags()),
            "Description": self._truncate_description(
                bookmark.get_effective_description()
            ),
            "Folder": bookmark.folder or "",
            "Created": self._format_date(bookmark.created),
            "Favorite": "Yes" if bookmark.favorite else "No",
        }

        if self.include_status:
            row["Status"] = self._determine_status(bookmark)

        if self.include_processing_info:
            status = bookmark.processing_status
            row["URL Validated"] = "Yes" if status and status.url_validated else "No"
            row["Content Extracted"] = "Yes" if status and status.content_extracted else "No"
            row["AI Processed"] = "Yes" if status and status.ai_processed else "No"
            row["Tags Optimized"] = "Yes" if status and status.tags_optimized else "No"

        return row

    def _format_tags(self, tags: List[str]) -> str:
        """Format tags for Notion (comma-separated)."""
        if not tags:
            return ""
        return self.tag_separator.join(tags)

    def _format_date(self, dt: Optional[datetime]) -> str:
        """Format a date for Notion."""
        if not dt:
            return ""
        return dt.strftime(self.date_format)

    def _truncate_description(
        self,
        description: str,
        max_length: int = 2000
    ) -> str:
        """Truncate description to fit Notion's limits."""
        if not description:
            return ""

        # Notion has a limit on text fields
        if len(description) <= max_length:
            return description

        return description[:max_length - 3] + "..."

    def _determine_status(self, bookmark: Bookmark) -> str:
        """Determine the status for a bookmark."""
        status = bookmark.processing_status

        if not status:
            return "Not Processed"

        if status.url_validation_error:
            return "Error - Invalid URL"

        if status.ai_processed:
            return "Processed"

        if status.content_extracted:
            return "Content Extracted"

        if status.url_validated:
            return "URL Validated"

        return "Pending"

    def export_with_relations(
        self,
        bookmarks: List[Bookmark],
        output_path: Path,
        relation_field: str = "Related"
    ) -> ExportResult:
        """
        Export bookmarks with relation suggestions based on shared tags.

        This creates a CSV that suggests relations between bookmarks
        that share common tags, useful for Notion's relation property.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for the CSV file
            relation_field: Name of the relation column

        Returns:
            ExportResult with export details
        """
        path = self.prepare_output_path(output_path)

        if not str(path).lower().endswith(".csv"):
            path = path.with_suffix(".csv")

        # Build tag-to-bookmarks index
        tag_index: Dict[str, List[str]] = {}
        for bookmark in bookmarks:
            for tag in bookmark.get_final_tags():
                if tag not in tag_index:
                    tag_index[tag] = []
                tag_index[tag].append(bookmark.get_effective_title())

        try:
            columns = self._get_columns() + [relation_field]

            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
                writer.writeheader()

                for bookmark in bookmarks:
                    row = self._bookmark_to_row(bookmark)

                    # Find related bookmarks through shared tags
                    related = set()
                    for tag in bookmark.get_final_tags():
                        for title in tag_index.get(tag, []):
                            if title != bookmark.get_effective_title():
                                related.add(title)

                    row[relation_field] = self.tag_separator.join(list(related)[:5])
                    writer.writerow(row)

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=f"{self.format_name} (with relations)",
                additional_info={
                    "columns": columns,
                    "relation_field": relation_field
                }
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export Notion CSV with relations: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )
