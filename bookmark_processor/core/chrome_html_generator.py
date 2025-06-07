"""
Chrome HTML Bookmark Generator

This module generates Chrome-compatible HTML bookmark files from processed bookmark data.
Follows the Netscape-Bookmark-file-1 format specification as defined in CHROME_SPEC.md.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .data_models import Bookmark


class ChromeHTMLGeneratorError(Exception):
    """Base exception for Chrome HTML generation errors."""

    pass


class FolderNode:
    """
    Represents a folder in the bookmark hierarchy.

    Used to build the hierarchical structure before HTML generation.
    """

    def __init__(self, name: str, parent: Optional["FolderNode"] = None):
        """
        Initialize folder node.

        Args:
            name: Folder name
            parent: Parent folder node (None for root)
        """
        self.name = name
        self.parent = parent
        self.children: Dict[str, "FolderNode"] = {}
        self.bookmarks: List[Bookmark] = []
        self.created_timestamp = int(time.time())
        self.modified_timestamp = int(time.time())

    def add_child(self, name: str) -> "FolderNode":
        """Add a child folder and return it."""
        if name not in self.children:
            self.children[name] = FolderNode(name, parent=self)
        return self.children[name]

    def add_bookmark(self, bookmark: Bookmark) -> None:
        """Add a bookmark to this folder."""
        self.bookmarks.append(bookmark)
        self.modified_timestamp = int(time.time())

    def get_path(self) -> str:
        """Get the full path to this folder."""
        if self.parent is None:
            return ""
        path_parts = []
        current = self
        while current.parent is not None:
            path_parts.append(current.name)
            current = current.parent
        return "/".join(reversed(path_parts))

    def is_bookmarks_bar(self) -> bool:
        """Check if this is the Bookmarks Bar folder."""
        return self.name.lower() in ["bookmarks bar", "bookmarks_bar", "toolbar"]


class ChromeHTMLGenerator:
    """
    Generator for Chrome-compatible HTML bookmark files.

    Creates properly structured HTML files that can be imported into Chrome
    following the Netscape-Bookmark-file-1 format specification.
    """

    def __init__(self):
        """Initialize the Chrome HTML generator."""
        self.logger = logging.getLogger(__name__)

    def generate_html(
        self,
        bookmarks: List[Bookmark],
        output_path: Union[str, Path],
        title: str = "Bookmarks",
    ) -> None:
        """
        Generate Chrome HTML bookmark file from bookmarks.

        Args:
            bookmarks: List of processed bookmarks
            output_path: Path where to save the HTML file
            title: Title for the bookmark file

        Raises:
            ChromeHTMLGeneratorError: If generation fails
        """
        output_path = Path(output_path)

        try:
            self.logger.info(f"Generating Chrome HTML bookmark file: {output_path}")
            self.logger.info(f"Processing {len(bookmarks)} bookmarks")

            # Build folder hierarchy
            root = self._build_folder_hierarchy(bookmarks)

            # Generate HTML content
            html_content = self._generate_html_content(root, title)

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            self.logger.info(f"Successfully generated Chrome HTML file: {output_path}")

        except Exception as e:
            error_msg = f"Failed to generate Chrome HTML file: {str(e)}"
            self.logger.error(error_msg)
            raise ChromeHTMLGeneratorError(error_msg) from e

    def _build_folder_hierarchy(self, bookmarks: List[Bookmark]) -> FolderNode:
        """
        Build hierarchical folder structure from bookmarks.

        Args:
            bookmarks: List of bookmarks to organize

        Returns:
            Root folder node containing the hierarchy
        """
        root = FolderNode("Root")

        # Create default folders
        bookmarks_bar = root.add_child("Bookmarks bar")
        other_bookmarks = root.add_child("Other bookmarks")

        for bookmark in bookmarks:
            # Determine target folder
            if bookmark.folder:
                # Use AI-generated folder structure
                folder_path = bookmark.folder
                target_parent = other_bookmarks  # Default to "Other bookmarks"

                # Check if this should go in bookmarks bar
                path_parts = folder_path.split("/")
                if path_parts[0].lower() in [
                    "bookmarks bar",
                    "bookmarks_bar",
                    "toolbar",
                ]:
                    target_parent = bookmarks_bar
                    path_parts = path_parts[1:]  # Remove "Bookmarks bar" prefix

                # Create nested folder structure
                current_folder = target_parent
                for part in path_parts:
                    if part.strip():  # Skip empty parts
                        current_folder = current_folder.add_child(part.strip())

                current_folder.add_bookmark(bookmark)
            else:
                # No folder specified, put in "Other bookmarks"
                other_bookmarks.add_bookmark(bookmark)

        self.logger.info(f"Built folder hierarchy with {len(bookmarks)} bookmarks")
        self._log_folder_stats(root)

        return root

    def _log_folder_stats(self, root: FolderNode) -> None:
        """Log statistics about the folder structure."""

        def count_folders_and_bookmarks(node: FolderNode) -> tuple:
            folder_count = len(node.children)
            bookmark_count = len(node.bookmarks)

            for child in node.children.values():
                child_folders, child_bookmarks = count_folders_and_bookmarks(child)
                folder_count += child_folders
                bookmark_count += child_bookmarks

            return folder_count, bookmark_count

        total_folders, total_bookmarks = count_folders_and_bookmarks(root)
        self.logger.info(
            f"Folder statistics: {total_folders} folders, {total_bookmarks} bookmarks"
        )

    def _generate_html_content(self, root: FolderNode, title: str) -> str:
        """
        Generate the complete HTML content.

        Args:
            root: Root folder node
            title: HTML document title

        Returns:
            Complete HTML content as string
        """
        # HTML header
        html_parts = [
            "<!DOCTYPE NETSCAPE-Bookmark-file-1>",
            "<!-- This is an automatically generated file.",
            "     It will be read and overwritten.",
            "     DO NOT EDIT! -->",
            '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
            f"<TITLE>{self._escape_html(title)}</TITLE>",
            f"<H1>{self._escape_html(title)}</H1>",
            "<DL><p>",
        ]

        # Generate folder content
        for child in root.children.values():
            if child.name in ["Bookmarks bar", "Other bookmarks"]:
                html_parts.extend(self._generate_folder_html(child, is_top_level=True))

        # HTML footer
        html_parts.append("</DL><p>")

        return "\n".join(html_parts)

    def _generate_folder_html(
        self, folder: FolderNode, is_top_level: bool = False
    ) -> List[str]:
        """
        Generate HTML for a folder and its contents.

        Args:
            folder: Folder node to generate HTML for
            is_top_level: Whether this is a top-level folder

        Returns:
            List of HTML lines
        """
        html_lines = []

        # Folder header
        folder_attrs = [
            f'ADD_DATE="{folder.created_timestamp}"',
            f'LAST_MODIFIED="{folder.modified_timestamp}"',
        ]

        # Special attribute for Bookmarks Bar
        if folder.is_bookmarks_bar():
            folder_attrs.append('PERSONAL_TOOLBAR_FOLDER="true"')

        folder_tag = (
            f'<H3 {" ".join(folder_attrs)}>{self._escape_html(folder.name)}</H3>'
        )
        html_lines.append(f"    <DT>{folder_tag}")

        # Folder content
        html_lines.append("    <DL><p>")

        # Add sub-folders first
        for child_folder in sorted(
            folder.children.values(), key=lambda f: f.name.lower()
        ):
            child_lines = self._generate_folder_html(child_folder, is_top_level=False)
            html_lines.extend([f"    {line}" for line in child_lines])

        # Add bookmarks
        for bookmark in sorted(folder.bookmarks, key=lambda b: b.title.lower()):
            bookmark_html = self._generate_bookmark_html(bookmark)
            html_lines.append(f"        <DT>{bookmark_html}")

        html_lines.append("    </DL><p>")

        return html_lines

    def _generate_bookmark_html(self, bookmark: Bookmark) -> str:
        """
        Generate HTML for a single bookmark.

        Args:
            bookmark: Bookmark to generate HTML for

        Returns:
            HTML anchor tag as string
        """
        # Build attributes
        attrs = [f'HREF="{self._escape_html(bookmark.url)}"']

        # Add timestamp if available
        if bookmark.created:
            timestamp = int(bookmark.created.timestamp())
            attrs.append(f'ADD_DATE="{timestamp}"')
        else:
            # Use current time as fallback
            attrs.append(f'ADD_DATE="{int(time.time())}"')

        # Note: We don't include ICON attribute since we don't have favicon data
        # Chrome will fetch favicons automatically when importing

        attr_string = " ".join(attrs)
        title = self._escape_html(bookmark.title or bookmark.url)

        return f"<A {attr_string}>{title}</A>"

    def _escape_html(self, text: str) -> str:
        """
        Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            HTML-escaped text
        """
        if not text:
            return ""

        # Basic HTML escaping
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&#x27;")

        return text

    def get_generation_info(self, bookmarks: List[Bookmark]) -> Dict[str, Any]:
        """
        Get information about what would be generated.

        Args:
            bookmarks: List of bookmarks to analyze

        Returns:
            Dictionary with generation information
        """
        # Build temporary hierarchy for analysis
        root = self._build_folder_hierarchy(bookmarks)

        def analyze_folder(node: FolderNode) -> Dict[str, Any]:
            info = {
                "name": node.name,
                "bookmark_count": len(node.bookmarks),
                "subfolder_count": len(node.children),
                "subfolders": {},
            }

            for child_name, child_node in node.children.items():
                info["subfolders"][child_name] = analyze_folder(child_node)

            return info

        return {
            "total_bookmarks": len(bookmarks),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "folder_structure": {
                name: analyze_folder(child) for name, child in root.children.items()
            },
        }
