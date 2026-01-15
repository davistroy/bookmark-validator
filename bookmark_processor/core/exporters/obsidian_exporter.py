"""
Obsidian vault bookmark exporter.

Exports bookmarks to Obsidian-compatible Markdown files with YAML frontmatter.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import BookmarkExporter, ExportResult, ExportError
from ..data_models import Bookmark


class ObsidianExporter(BookmarkExporter):
    """
    Export bookmarks to Obsidian vault format.

    Creates individual note files for each bookmark with YAML frontmatter
    containing metadata. Supports Obsidian-specific features like wikilinks
    and tags.

    Example:
        >>> exporter = ObsidianExporter(use_wikilinks=True)
        >>> result = exporter.export(bookmarks, Path("vault/bookmarks/"))
    """

    def __init__(
        self,
        use_wikilinks: bool = True,
        include_aliases: bool = True,
        tags_in_frontmatter: bool = True,
        create_folder_notes: bool = True,
        create_moc: bool = True,
        date_format: str = "%Y-%m-%d",
        template: Optional[str] = None
    ):
        """
        Initialize the Obsidian exporter.

        Args:
            use_wikilinks: Use [[wikilinks]] for internal links
            include_aliases: Add aliases in frontmatter for search
            tags_in_frontmatter: Put tags in frontmatter (vs inline #tags)
            create_folder_notes: Create folder index notes
            create_moc: Create a Map of Content note linking all bookmarks
            date_format: Format for dates in frontmatter
            template: Custom template for notes (uses default if None)
        """
        super().__init__()
        self.use_wikilinks = use_wikilinks
        self.include_aliases = include_aliases
        self.tags_in_frontmatter = tags_in_frontmatter
        self.create_folder_notes = create_folder_notes
        self.create_moc = create_moc
        self.date_format = date_format
        self.template = template

    @property
    def format_name(self) -> str:
        return "Obsidian"

    @property
    def file_extension(self) -> str:
        return "md"

    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks to an Obsidian vault.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path to the Obsidian vault folder

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

        # Prepare vault directory
        vault_path = self.prepare_output_path(output_path, is_directory=True)

        try:
            # Group bookmarks by folder
            by_folder = self._group_by_folder(bookmarks)
            files_created = 0

            # Create folder structure and export bookmarks
            for folder_name, folder_bookmarks in by_folder.items():
                folder_path = self._create_folder_structure(vault_path, folder_name)

                # Create individual bookmark notes
                for bookmark in folder_bookmarks:
                    note_path = self._create_bookmark_note(folder_path, bookmark)
                    files_created += 1

                # Create folder index note
                if self.create_folder_notes:
                    self._create_folder_note(folder_path, folder_name, folder_bookmarks)
                    files_created += 1

            # Create Map of Content
            if self.create_moc:
                self._create_moc_note(vault_path, by_folder)
                files_created += 1

            self.logger.info(
                f"Exported {len(bookmarks)} bookmarks to {files_created} notes in {vault_path}"
            )

            return ExportResult(
                path=vault_path,
                count=len(bookmarks),
                format_name=self.format_name,
                additional_info={
                    "files_created": files_created,
                    "folders": len(by_folder),
                    "moc_created": self.create_moc,
                    "folder_notes_created": self.create_folder_notes
                },
                warnings=warnings
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export to Obsidian vault: {e}",
                format_name=self.format_name,
                path=vault_path,
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

        return dict(sorted(by_folder.items()))

    def _create_folder_structure(self, vault_path: Path, folder_name: str) -> Path:
        """Create folder structure in the vault."""
        # Handle nested folders
        folder_parts = folder_name.split("/")
        safe_parts = [self.sanitize_filename(part) for part in folder_parts]

        folder_path = vault_path / "/".join(safe_parts)
        folder_path.mkdir(parents=True, exist_ok=True)

        return folder_path

    def _create_bookmark_note(self, folder_path: Path, bookmark: Bookmark) -> Path:
        """Create a note file for a bookmark."""
        title = bookmark.get_effective_title()
        safe_title = self.sanitize_filename(title)

        # Ensure unique filename
        note_path = folder_path / f"{safe_title}.md"
        counter = 1
        while note_path.exists():
            note_path = folder_path / f"{safe_title} ({counter}).md"
            counter += 1

        # Generate note content
        content = self._generate_note_content(bookmark)

        with open(note_path, "w", encoding="utf-8") as f:
            f.write(content)

        return note_path

    def _generate_note_content(self, bookmark: Bookmark) -> str:
        """Generate the content for a bookmark note."""
        lines = []

        # YAML frontmatter
        lines.append("---")
        lines.extend(self._generate_frontmatter(bookmark))
        lines.append("---")
        lines.append("")

        # Title
        lines.append(f"# {bookmark.get_effective_title()}")
        lines.append("")

        # URL
        lines.append(f"**URL:** {bookmark.url}")
        lines.append("")

        # Description
        description = bookmark.get_effective_description()
        if description:
            lines.append("## Description")
            lines.append("")
            lines.append(description)
            lines.append("")

        # Tags (inline format if not in frontmatter)
        if not self.tags_in_frontmatter:
            tags = bookmark.get_final_tags()
            if tags:
                tag_str = " ".join(f"#{tag.replace(' ', '-')}" for tag in tags)
                lines.append(f"Tags: {tag_str}")
                lines.append("")

        # Metadata section
        lines.append("## Metadata")
        lines.append("")
        if bookmark.folder:
            lines.append(f"- **Folder:** {bookmark.folder}")
        if bookmark.created:
            lines.append(f"- **Created:** {bookmark.created.strftime(self.date_format)}")
        if bookmark.favorite:
            lines.append("- **Favorite:** Yes")

        return "\n".join(lines)

    def _generate_frontmatter(self, bookmark: Bookmark) -> List[str]:
        """Generate YAML frontmatter for a bookmark note."""
        lines = []

        # URL
        lines.append(f"url: \"{bookmark.url}\"")

        # Title
        title = bookmark.get_effective_title()
        lines.append(f"title: \"{self._escape_yaml_string(title)}\"")

        # Aliases
        if self.include_aliases:
            aliases = [title]
            # Add domain as alias
            from urllib.parse import urlparse
            try:
                domain = urlparse(bookmark.url).netloc
                if domain and domain not in aliases:
                    aliases.append(domain)
            except Exception:
                pass
            aliases_str = ", ".join(f'"{self._escape_yaml_string(a)}"' for a in aliases)
            lines.append(f"aliases: [{aliases_str}]")

        # Tags
        if self.tags_in_frontmatter:
            tags = bookmark.get_final_tags()
            if tags:
                # Obsidian tags should not have spaces
                safe_tags = [tag.replace(" ", "-") for tag in tags]
                tags_str = ", ".join(f'"{t}"' for t in safe_tags)
                lines.append(f"tags: [{tags_str}]")

        # Date
        if bookmark.created:
            lines.append(f"created: {bookmark.created.strftime(self.date_format)}")

        # Custom fields
        lines.append(f"exported: {datetime.now().strftime(self.date_format)}")
        lines.append("type: bookmark")

        if bookmark.folder:
            lines.append(f"folder: \"{self._escape_yaml_string(bookmark.folder)}\"")

        if bookmark.favorite:
            lines.append("favorite: true")

        return lines

    def _create_folder_note(
        self,
        folder_path: Path,
        folder_name: str,
        bookmarks: List[Bookmark]
    ) -> None:
        """Create an index note for a folder."""
        # Use folder name as note title
        folder_parts = folder_name.split("/")
        note_name = folder_parts[-1] if folder_parts else "Unsorted"

        note_path = folder_path / f"{self.sanitize_filename(note_name)} (Index).md"

        lines = []

        # Frontmatter
        lines.append("---")
        lines.append(f"title: \"{self._escape_yaml_string(note_name)} Index\"")
        lines.append("type: folder-index")
        lines.append("tags: [index, bookmarks]")
        lines.append(f"created: {datetime.now().strftime(self.date_format)}")
        lines.append("---")
        lines.append("")

        # Title
        lines.append(f"# {note_name}")
        lines.append("")
        lines.append(f"*{len(bookmarks)} bookmarks*")
        lines.append("")

        # List of bookmarks
        lines.append("## Bookmarks")
        lines.append("")

        for bookmark in bookmarks:
            title = bookmark.get_effective_title()
            safe_title = self.sanitize_filename(title)

            if self.use_wikilinks:
                link = f"[[{safe_title}]]"
            else:
                link = f"[{title}]({safe_title}.md)"

            lines.append(f"- {link}")

        with open(note_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _create_moc_note(
        self,
        vault_path: Path,
        by_folder: Dict[str, List[Bookmark]]
    ) -> None:
        """Create a Map of Content note."""
        note_path = vault_path / "Bookmarks MOC.md"

        total_bookmarks = sum(len(b) for b in by_folder.values())

        lines = []

        # Frontmatter
        lines.append("---")
        lines.append("title: \"Bookmarks Map of Content\"")
        lines.append("type: moc")
        lines.append("tags: [moc, bookmarks]")
        lines.append(f"created: {datetime.now().strftime(self.date_format)}")
        lines.append("---")
        lines.append("")

        # Title
        lines.append("# Bookmarks Map of Content")
        lines.append("")
        lines.append(f"*{total_bookmarks} bookmarks in {len(by_folder)} folders*")
        lines.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")

        # Folders section
        lines.append("## Folders")
        lines.append("")

        for folder_name, folder_bookmarks in by_folder.items():
            folder_parts = folder_name.split("/")
            safe_parts = [self.sanitize_filename(p) for p in folder_parts]

            # Link to folder index
            folder_display = folder_name.replace("/", " > ")
            index_name = f"{safe_parts[-1]} (Index)"
            index_path = "/".join(safe_parts) + f"/{index_name}"

            if self.use_wikilinks:
                folder_link = f"[[{index_path}|{folder_display}]]"
            else:
                folder_link = f"[{folder_display}]({index_path}.md)"

            lines.append(f"- {folder_link} ({len(folder_bookmarks)} bookmarks)")

        lines.append("")

        # Recent bookmarks section
        lines.append("## Recent Bookmarks")
        lines.append("")

        # Get most recent bookmarks (with dates)
        dated_bookmarks = [
            b for b in sum(by_folder.values(), [])
            if b.created
        ]
        dated_bookmarks.sort(key=lambda b: b.created, reverse=True)

        for bookmark in dated_bookmarks[:10]:
            title = bookmark.get_effective_title()
            safe_title = self.sanitize_filename(title)
            folder_parts = (bookmark.folder or "Unsorted").split("/")
            safe_folder = "/".join(self.sanitize_filename(p) for p in folder_parts)

            if self.use_wikilinks:
                link = f"[[{safe_folder}/{safe_title}|{title}]]"
            else:
                link = f"[{title}]({safe_folder}/{safe_title}.md)"

            date_str = bookmark.created.strftime(self.date_format)
            lines.append(f"- {date_str}: {link}")

        with open(note_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _escape_yaml_string(self, s: str) -> str:
        """Escape special characters in YAML strings."""
        if not s:
            return ""
        # Escape double quotes and backslashes
        return s.replace("\\", "\\\\").replace('"', '\\"')
