"""
OPML bookmark exporter.

Exports bookmarks to OPML (Outline Processor Markup Language) format
compatible with RSS readers and other outline-based tools.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from xml.dom import minidom

from .base import BookmarkExporter, ExportResult, ExportError
from ..data_models import Bookmark


class OPMLExporter(BookmarkExporter):
    """
    Export bookmarks to OPML format.

    Creates an OPML file that can be imported into RSS readers like
    Feedly, Inoreader, NewsBlur, etc. Bookmarks are organized by folder
    in a hierarchical outline structure.

    Example:
        >>> exporter = OPMLExporter(title="My Bookmarks")
        >>> result = exporter.export(bookmarks, Path("bookmarks.opml"))
    """

    def __init__(
        self,
        title: str = "Bookmarks Export",
        owner_name: Optional[str] = None,
        owner_email: Optional[str] = None,
        include_descriptions: bool = True,
        include_tags_as_category: bool = True,
        use_html_url: bool = True,
        pretty_print: bool = True
    ):
        """
        Initialize the OPML exporter.

        Args:
            title: Title for the OPML document
            owner_name: Optional owner name for the document
            owner_email: Optional owner email for the document
            include_descriptions: Include bookmark descriptions in title attribute
            include_tags_as_category: Include tags in the category attribute
            use_html_url: Use htmlUrl attribute (for web pages) vs xmlUrl (for feeds)
            pretty_print: Format XML with indentation
        """
        super().__init__()
        self.title = title
        self.owner_name = owner_name
        self.owner_email = owner_email
        self.include_descriptions = include_descriptions
        self.include_tags_as_category = include_tags_as_category
        self.use_html_url = use_html_url
        self.pretty_print = pretty_print

    @property
    def format_name(self) -> str:
        return "OPML"

    @property
    def file_extension(self) -> str:
        return "opml"

    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks to an OPML file.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for the OPML file

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

        if not str(path).lower().endswith(".opml"):
            path = path.with_suffix(".opml")

        try:
            # Build OPML document
            opml = self._build_opml(bookmarks)

            # Convert to string
            xml_string = self._to_xml_string(opml)

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(xml_string)

            self.logger.info(f"Exported {len(bookmarks)} bookmarks to {path}")

            # Count folders
            by_folder = self._group_by_folder(bookmarks)

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=self.format_name,
                additional_info={
                    "folders": len(by_folder),
                    "file_size": path.stat().st_size
                },
                warnings=warnings
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export OPML: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )

    def _build_opml(self, bookmarks: List[Bookmark]) -> ET.Element:
        """Build the OPML XML tree."""
        # Root element
        opml = ET.Element("opml", version="2.0")

        # Head section
        head = ET.SubElement(opml, "head")

        title_elem = ET.SubElement(head, "title")
        title_elem.text = self.title

        date_created = ET.SubElement(head, "dateCreated")
        date_created.text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")

        if self.owner_name:
            owner_name_elem = ET.SubElement(head, "ownerName")
            owner_name_elem.text = self.owner_name

        if self.owner_email:
            owner_email_elem = ET.SubElement(head, "ownerEmail")
            owner_email_elem.text = self.owner_email

        # Body section
        body = ET.SubElement(opml, "body")

        # Group bookmarks by folder
        by_folder = self._group_by_folder(bookmarks)

        # Create folder structure
        for folder_name, folder_bookmarks in by_folder.items():
            self._add_folder_outline(body, folder_name, folder_bookmarks)

        return opml

    def _group_by_folder(self, bookmarks: List[Bookmark]) -> Dict[str, List[Bookmark]]:
        """Group bookmarks by folder."""
        by_folder: Dict[str, List[Bookmark]] = {}

        for bookmark in bookmarks:
            folder = bookmark.folder or "Unsorted"
            if folder not in by_folder:
                by_folder[folder] = []
            by_folder[folder].append(bookmark)

        return dict(sorted(by_folder.items()))

    def _add_folder_outline(
        self,
        parent: ET.Element,
        folder_name: str,
        bookmarks: List[Bookmark]
    ) -> None:
        """Add a folder outline with its bookmarks."""
        # Handle nested folders
        folder_parts = folder_name.split("/")

        current_parent = parent

        # Create nested folder structure
        for i, part in enumerate(folder_parts):
            # Check if this folder level already exists
            existing = None
            for child in current_parent:
                if child.get("text") == part and child.get("type") == "folder":
                    existing = child
                    break

            if existing is not None:
                current_parent = existing
            else:
                # Create new folder outline
                folder_outline = ET.SubElement(
                    current_parent,
                    "outline",
                    text=part,
                    type="folder"
                )
                current_parent = folder_outline

        # Add bookmarks to the deepest folder
        for bookmark in bookmarks:
            self._add_bookmark_outline(current_parent, bookmark)

    def _add_bookmark_outline(
        self,
        parent: ET.Element,
        bookmark: Bookmark
    ) -> None:
        """Add a bookmark as an outline element."""
        attribs = {
            "type": "link",
            "text": bookmark.get_effective_title()
        }

        # URL attribute
        if self.use_html_url:
            attribs["htmlUrl"] = bookmark.url
        else:
            attribs["xmlUrl"] = bookmark.url

        # Description
        if self.include_descriptions:
            description = bookmark.get_effective_description()
            if description:
                # Truncate long descriptions for OPML compatibility
                if len(description) > 500:
                    description = description[:497] + "..."
                attribs["title"] = description

        # Tags as category
        if self.include_tags_as_category:
            tags = bookmark.get_final_tags()
            if tags:
                attribs["category"] = ",".join(tags)

        # Created date
        if bookmark.created:
            attribs["created"] = bookmark.created.strftime("%a, %d %b %Y %H:%M:%S %z")

        ET.SubElement(parent, "outline", **attribs)

    def _to_xml_string(self, element: ET.Element) -> str:
        """Convert XML element to string."""
        rough_string = ET.tostring(element, encoding="unicode", method="xml")

        if self.pretty_print:
            # Use minidom for pretty printing
            dom = minidom.parseString(rough_string)  # nosec B318 - parsing self-generated XML
            pretty_xml = dom.toprettyxml(indent="  ", encoding=None)

            # Remove extra blank lines and XML declaration
            lines = pretty_xml.split("\n")
            # Remove XML declaration added by minidom (we'll add our own)
            if lines[0].startswith("<?xml"):
                lines = lines[1:]
            # Remove empty lines
            lines = [line for line in lines if line.strip()]

            xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines)
        else:
            xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + rough_string

        return xml_string

    def export_flat(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult:
        """
        Export bookmarks in a flat structure (no folder hierarchy).

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for the OPML file

        Returns:
            ExportResult with export details
        """
        path = self.prepare_output_path(output_path)

        if not str(path).lower().endswith(".opml"):
            path = path.with_suffix(".opml")

        try:
            # Build OPML with flat structure
            opml = ET.Element("opml", version="2.0")

            # Head
            head = ET.SubElement(opml, "head")
            title_elem = ET.SubElement(head, "title")
            title_elem.text = self.title

            # Body with flat list
            body = ET.SubElement(opml, "body")

            for bookmark in bookmarks:
                self._add_bookmark_outline(body, bookmark)

            # Write file
            xml_string = self._to_xml_string(opml)

            with open(path, "w", encoding="utf-8") as f:
                f.write(xml_string)

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=f"{self.format_name} (flat)",
                additional_info={"flat": True}
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export flat OPML: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )

    def export_for_rss_reader(
        self,
        bookmarks: List[Bookmark],
        output_path: Path,
        feed_urls: Optional[Dict[str, str]] = None
    ) -> ExportResult:
        """
        Export bookmarks optimized for RSS reader import.

        This version uses xmlUrl for any bookmarks that have associated
        RSS feeds, making it easier to subscribe to feeds directly.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path for the OPML file
            feed_urls: Optional mapping of bookmark URLs to RSS feed URLs

        Returns:
            ExportResult with export details
        """
        feed_urls = feed_urls or {}
        path = self.prepare_output_path(output_path)

        if not str(path).lower().endswith(".opml"):
            path = path.with_suffix(".opml")

        try:
            opml = ET.Element("opml", version="2.0")

            head = ET.SubElement(opml, "head")
            title_elem = ET.SubElement(head, "title")
            title_elem.text = f"{self.title} - RSS Feeds"

            body = ET.SubElement(opml, "body")

            # Group bookmarks
            by_folder = self._group_by_folder(bookmarks)

            feeds_count = 0

            for folder_name, folder_bookmarks in by_folder.items():
                # Create folder outline
                folder_parts = folder_name.split("/")
                folder_outline = body

                for part in folder_parts:
                    new_outline = ET.SubElement(folder_outline, "outline", text=part)
                    folder_outline = new_outline

                # Add bookmarks
                for bookmark in folder_bookmarks:
                    attribs = {"text": bookmark.get_effective_title()}

                    # Check for RSS feed URL
                    feed_url = feed_urls.get(bookmark.url)
                    if feed_url:
                        attribs["type"] = "rss"
                        attribs["xmlUrl"] = feed_url
                        attribs["htmlUrl"] = bookmark.url
                        feeds_count += 1
                    else:
                        attribs["type"] = "link"
                        attribs["htmlUrl"] = bookmark.url

                    ET.SubElement(folder_outline, "outline", **attribs)

            xml_string = self._to_xml_string(opml)

            with open(path, "w", encoding="utf-8") as f:
                f.write(xml_string)

            return ExportResult(
                path=path,
                count=len(bookmarks),
                format_name=f"{self.format_name} (RSS)",
                additional_info={
                    "feeds_count": feeds_count,
                    "links_count": len(bookmarks) - feeds_count
                }
            )

        except Exception as e:
            raise ExportError(
                f"Failed to export RSS OPML: {e}",
                format_name=self.format_name,
                path=path,
                original_error=e
            )
