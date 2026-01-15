"""
Markdown bookmark exporter.

Exports bookmarks to Markdown format in single file or directory modes.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import BookmarkExporter, ExportResult, ExportError
from ..data_models import Bookmark


class MarkdownExporter(BookmarkExporter):
    """
    Export bookmarks to Markdown format.

    Supports two modes:
    - Single file: All bookmarks in one Markdown file with folder sections
    - Directory: One Markdown file per folder in a directory structure

    Example:
        >>> exporter = MarkdownExporter(mode="single", include_descriptions=True)
        >>> result = exporter.export(bookmarks, Path("bookmarks.md"))
    """

    def __init__(
        self,
        mode: str = "single",
        include_descriptions: bool = True,
        include_tags: bool = True,
        include_dates: bool = False,
        use_checkboxes: bool = False,
        link_style: str = "inline",  # inline, reference
        heading_level: int = 2
    ):
        """
        Initialize the Markdown exporter.

        Args:
            mode: Export mode - "single" for one file, "directory" for folder structure
            include_descriptions: Whether to include bookmark descriptions
            include_tags: Whether to include tags
            include_dates: Whether to include creation dates
            use_checkboxes: Whether to use checkbox format (for task lists)
            link_style: "inline" for [title](url) or "reference" for reference links
            heading_level: Starting heading level for folders (1-4)
        """
        super().__init__()
        self.mode = mode.lower()
        self.include_descriptions = include_descriptions
        self.include_tags = include_tags
        self.include_dates = include_dates
        self.use_checkboxes = use_checkboxes
        self.link_style = link_style.lower()
        self.heading_level = max(1, min(4, heading_level))

        if self.mode not in ("single", "directory"):
            raise ValueError(f"Invalid mode: {mode}. Use 'single' or 'directory'.")

    @property
    def format_name(self) -> str:
        return "Markdown"

    @property
    def file_extension(self) -> str:
        return "md"

    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks to Markdown.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for output (file for single mode, directory for directory mode)

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

        if self.mode == "single":
            return self._export_single_file(bookmarks, output_path, warnings)
        else:
            return self._export_directory(bookmarks, output_path, warnings)

    def _export_single_file(
        self,
        bookmarks: List[Bookmark],
        output_path: Path,
        warnings: List[str]
    ) -> ExportResult:
        """Export all bookmarks to a single Markdown file."""
        path = self.prepare_output_path(output_path)

        if not str(path).lower().endswith(".md"):
            path = path.with_suffix(".md")

        try:
            # Group bookmarks by folder
            by_folder = self._group_by_folder(bookmarks)

            # Build markdown content
            content = self._build_single_file_content(by_folder, bookmarks)

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"Exported {len(bookmarks)} bookmarks to {path}")

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=self.format_name,
                additional_info={
                    "mode": "single",
                    "folders": len(by_folder),
                    "file_size": path.stat().st_size
                },
                warnings=warnings
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export Markdown: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )

    def _export_directory(
        self,
        bookmarks: List[Bookmark],
        output_path: Path,
        warnings: List[str]
    ) -> ExportResult:
        """Export bookmarks to a directory with one file per folder."""
        path = self.prepare_output_path(output_path, is_directory=True)

        try:
            by_folder = self._group_by_folder(bookmarks)
            files_created = 0

            for folder_name, folder_bookmarks in by_folder.items():
                # Create safe filename from folder name
                filename = self.sanitize_filename(folder_name) + ".md"

                # Handle nested folders - create subdirectory structure
                folder_parts = folder_name.split("/")
                if len(folder_parts) > 1:
                    subdir = path / "/".join(
                        self.sanitize_filename(p) for p in folder_parts[:-1]
                    )
                    subdir.mkdir(parents=True, exist_ok=True)
                    file_path = subdir / (self.sanitize_filename(folder_parts[-1]) + ".md")
                else:
                    file_path = path / filename

                # Build content for this folder
                content = self._build_folder_content(folder_name, folder_bookmarks)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                files_created += 1

            # Create index file
            index_content = self._build_index_content(by_folder)
            index_path = path / "README.md"
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(index_content)

            self.logger.info(
                f"Exported {len(bookmarks)} bookmarks to {files_created} files in {path}"
            )

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=self.format_name,
                additional_info={
                    "mode": "directory",
                    "files_created": files_created,
                    "folders": len(by_folder)
                },
                warnings=warnings
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export Markdown directory: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )

    def _group_by_folder(self, bookmarks: List[Bookmark]) -> Dict[str, List[Bookmark]]:
        """Group bookmarks by folder."""
        by_folder: Dict[str, List[Bookmark]] = {}

        for bookmark in bookmarks:
            folder = bookmark.folder or "Unsorted"
            if folder not in by_folder:
                by_folder[folder] = []
            by_folder[folder].append(bookmark)

        # Sort folders alphabetically
        return dict(sorted(by_folder.items()))

    def _build_single_file_content(
        self,
        by_folder: Dict[str, List[Bookmark]],
        all_bookmarks: List[Bookmark]
    ) -> str:
        """Build content for a single Markdown file."""
        lines = []

        # Header
        lines.append("# Bookmarks")
        lines.append("")
        lines.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")
        lines.append(f"**Total:** {len(all_bookmarks)} bookmarks in {len(by_folder)} folders")
        lines.append("")

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for folder_name, folder_bookmarks in by_folder.items():
            anchor = self._folder_to_anchor(folder_name)
            lines.append(f"- [{folder_name}](#{anchor}) ({len(folder_bookmarks)})")
        lines.append("")

        # Reference links collection (if using reference style)
        references = []

        # Content by folder
        for folder_name, folder_bookmarks in by_folder.items():
            heading = "#" * self.heading_level
            lines.append(f"{heading} {folder_name}")
            lines.append("")

            for i, bookmark in enumerate(folder_bookmarks):
                bookmark_lines, ref = self._format_bookmark(bookmark, f"ref-{len(references)}")
                lines.extend(bookmark_lines)
                if ref:
                    references.append(ref)

            lines.append("")

        # Add reference links at the end if using reference style
        if self.link_style == "reference" and references:
            lines.append("---")
            lines.append("")
            lines.extend(references)

        return "\n".join(lines)

    def _build_folder_content(
        self,
        folder_name: str,
        bookmarks: List[Bookmark]
    ) -> str:
        """Build content for a folder-specific Markdown file."""
        lines = []

        lines.append(f"# {folder_name}")
        lines.append("")
        lines.append(f"**{len(bookmarks)} bookmark(s)**")
        lines.append("")

        references = []

        for i, bookmark in enumerate(bookmarks):
            bookmark_lines, ref = self._format_bookmark(bookmark, f"ref-{i}")
            lines.extend(bookmark_lines)
            if ref:
                references.append(ref)

        if self.link_style == "reference" and references:
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.extend(references)

        return "\n".join(lines)

    def _build_index_content(self, by_folder: Dict[str, List[Bookmark]]) -> str:
        """Build index/README content for directory mode."""
        lines = []

        total = sum(len(b) for b in by_folder.values())

        lines.append("# Bookmarks Index")
        lines.append("")
        lines.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")
        lines.append(f"**Total:** {total} bookmarks in {len(by_folder)} folders")
        lines.append("")
        lines.append("## Folders")
        lines.append("")

        for folder_name, folder_bookmarks in by_folder.items():
            # Create relative link to folder file
            folder_parts = folder_name.split("/")
            if len(folder_parts) > 1:
                link_path = "/".join(
                    self.sanitize_filename(p) for p in folder_parts[:-1]
                ) + "/" + self.sanitize_filename(folder_parts[-1]) + ".md"
            else:
                link_path = self.sanitize_filename(folder_name) + ".md"

            lines.append(f"- [{folder_name}]({link_path}) ({len(folder_bookmarks)} bookmarks)")

        return "\n".join(lines)

    def _format_bookmark(
        self,
        bookmark: Bookmark,
        ref_id: str
    ) -> tuple:
        """
        Format a single bookmark as Markdown.

        Returns:
            Tuple of (lines, reference_line or None)
        """
        lines = []
        reference = None

        title = bookmark.get_effective_title()
        url = bookmark.url

        # Build the link
        if self.use_checkboxes:
            prefix = "- [ ] "
        else:
            prefix = "- "

        if self.link_style == "inline":
            link = f"[{title}]({url})"
        else:
            link = f"[{title}][{ref_id}]"
            reference = f"[{ref_id}]: {url}"

        lines.append(f"{prefix}{link}")

        # Add description
        if self.include_descriptions:
            description = bookmark.get_effective_description()
            if description:
                # Truncate long descriptions
                if len(description) > 200:
                    description = description[:197] + "..."
                lines.append(f"  > {description}")

        # Add tags
        if self.include_tags:
            tags = bookmark.get_final_tags()
            if tags:
                tag_str = " ".join(f"`{tag}`" for tag in tags)
                lines.append(f"  Tags: {tag_str}")

        # Add date
        if self.include_dates and bookmark.created:
            date_str = bookmark.created.strftime("%Y-%m-%d")
            lines.append(f"  *Created: {date_str}*")

        lines.append("")

        return lines, reference

    def _folder_to_anchor(self, folder_name: str) -> str:
        """Convert folder name to Markdown anchor."""
        import re
        anchor = folder_name.lower()
        anchor = re.sub(r'[^a-z0-9\s-]', '', anchor)
        anchor = re.sub(r'\s+', '-', anchor)
        return anchor
