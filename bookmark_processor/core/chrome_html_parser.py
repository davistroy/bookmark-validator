"""
Chrome HTML bookmark parser module.

This module handles parsing Chrome HTML bookmark exports according to the
CHROME_SPEC.md specification. It processes the Netscape bookmark file format
and extracts bookmarks with their metadata for processing.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

from bs4 import BeautifulSoup

from .data_models import Bookmark


class ChromeHTMLError(Exception):
    """Base exception for Chrome HTML parsing errors."""

    pass


class ChromeHTMLStructureError(ChromeHTMLError):
    """Raised when HTML structure doesn't match Chrome bookmark format."""

    pass


class ChromeHTMLParser:
    """
    Parser for Chrome HTML bookmark exports.

    Handles the Netscape bookmark file format used by Chrome following
    the CHROME_SPEC.md specification.
    """

    # Chrome bookmark file format constants
    DOCTYPE_PATTERN = r"<!DOCTYPE\s+NETSCAPE-Bookmark-file-1>"
    SUPPORTED_ENCODINGS = ["utf-8", "utf-16", "iso-8859-1"]

    def __init__(self):
        """Initialize the Chrome HTML parser."""
        self.logger = logging.getLogger(__name__)

    def parse_file(self, file_path: Union[str, Path]) -> List[Bookmark]:
        """
        Parse a Chrome HTML bookmark export file.

        Args:
            file_path: Path to the Chrome HTML bookmark file

        Returns:
            List of Bookmark objects extracted from the file

        Raises:
            ChromeHTMLError: If parsing fails
            ChromeHTMLStructureError: If file format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ChromeHTMLError(f"File not found: {file_path}")

        try:
            # Read file with encoding detection
            html_content = self._read_html_file(file_path)

            # Validate and parse HTML structure
            soup = self._parse_html_content(html_content)

            # Extract bookmarks from the DOM structure
            bookmarks = self._extract_bookmarks(soup)

            self.logger.info(
                f"Successfully parsed {len(bookmarks)} bookmarks from {file_path}"
            )
            return bookmarks

        except Exception as e:
            self.logger.error(f"Error parsing Chrome HTML file {file_path}: {str(e)}")
            raise ChromeHTMLError(f"Failed to parse Chrome HTML file: {str(e)}") from e

    def _read_html_file(self, file_path: Path) -> str:
        """
        Read HTML file with proper encoding detection.

        Args:
            file_path: Path to the HTML file

        Returns:
            HTML content as string

        Raises:
            ChromeHTMLError: If file cannot be read
        """
        try:
            # Try UTF-8 first (most common)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Validate this is a Chrome bookmark file
            if not re.search(self.DOCTYPE_PATTERN, content):
                # Try other encodings if UTF-8 didn't work
                for encoding in self.SUPPORTED_ENCODINGS[1:]:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                        if re.search(self.DOCTYPE_PATTERN, content):
                            break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ChromeHTMLStructureError(
                        "File does not appear to be a Chrome bookmark export (missing DOCTYPE)"
                    )

            return content

        except UnicodeDecodeError as e:
            raise ChromeHTMLError(
                f"Unable to read file with supported encodings: {str(e)}"
            )
        except IOError as e:
            raise ChromeHTMLError(f"Error reading file: {str(e)}")

    def _parse_html_content(self, html_content: str) -> BeautifulSoup:
        """
        Parse HTML content and validate structure.

        Args:
            html_content: Raw HTML content

        Returns:
            BeautifulSoup object

        Raises:
            ChromeHTMLStructureError: If HTML structure is invalid
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Validate required elements exist
            if not soup.find("dl"):
                raise ChromeHTMLStructureError(
                    "No bookmark data found (missing <DL> elements)"
                )

            # Look for the main bookmarks header
            h1 = soup.find("h1")
            if not h1 or h1.get_text().strip().lower() != "bookmarks":
                self.logger.warning("Expected <H1>Bookmarks</H1> header not found")

            return soup

        except Exception as e:
            raise ChromeHTMLStructureError(f"Invalid HTML structure: {str(e)}")

    def _extract_bookmarks(self, soup: BeautifulSoup) -> List[Bookmark]:
        """
        Extract bookmarks from the parsed HTML structure.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of Bookmark objects
        """
        bookmarks = []

        # Find the root DL element
        root_dl = soup.find("dl")
        if not root_dl:
            return bookmarks

        # Process all DT elements recursively
        self._process_dl_element(root_dl, bookmarks, folder_path="")

        return bookmarks

    def _process_dl_element(
        self, dl_element, bookmarks: List[Bookmark], folder_path: str
    ):
        """
        Recursively process DL elements to extract bookmarks and folders.

        Args:
            dl_element: BeautifulSoup DL element
            bookmarks: List to append bookmarks to
            folder_path: Current folder path
        """
        # Find all DT children (may be nested under P elements)
        dt_elements = dl_element.find_all("dt")

        for dt in dt_elements:
            # Check if this DT contains a folder (H3) or bookmark (A)
            h3 = dt.find("h3", recursive=False)
            a_tag = dt.find("a", recursive=False)

            if h3:
                # This is a folder
                folder_name = h3.get_text().strip()
                new_folder_path = self._build_folder_path(folder_path, folder_name)

                # Check if this is a special folder
                is_toolbar = h3.get("personal_toolbar_folder") == "true"
                if is_toolbar and not folder_path:
                    # This is the bookmarks bar - use a standardized name
                    new_folder_path = "Bookmarks Bar"

                # Find the next DL element after this DT
                next_dl = self._find_next_dl(dt)
                if next_dl:
                    self._process_dl_element(next_dl, bookmarks, new_folder_path)

            elif a_tag:
                # This is a bookmark
                bookmark = self._parse_bookmark_link(a_tag, folder_path)
                if bookmark:
                    bookmarks.append(bookmark)

    def _find_next_dl(self, dt_element):
        """
        Find the DL element that follows a DT containing a folder.

        Args:
            dt_element: DT element that should be followed by a DL

        Returns:
            Next DL element or None
        """
        # Look for DL as next sibling
        next_sibling = dt_element.find_next_sibling()
        while next_sibling:
            if next_sibling.name and next_sibling.name.lower() == "dl":
                return next_sibling
            next_sibling = next_sibling.find_next_sibling()

        # Look for DL as child of the DT
        dl_child = dt_element.find("dl")
        return dl_child

    def _build_folder_path(self, current_path: str, folder_name: str) -> str:
        """
        Build hierarchical folder path.

        Args:
            current_path: Current folder path
            folder_name: Name of the folder to add

        Returns:
            New folder path
        """
        if not current_path:
            return folder_name
        return f"{current_path}/{folder_name}"

    def _parse_bookmark_link(self, a_tag, folder_path: str) -> Optional[Bookmark]:
        """
        Parse a bookmark link element into a Bookmark object.

        Args:
            a_tag: BeautifulSoup A element
            folder_path: Current folder path

        Returns:
            Bookmark object or None if parsing fails
        """
        try:
            # Extract required URL
            url = a_tag.get("href")
            if not url:
                self.logger.warning("Bookmark found without URL, skipping")
                return None

            # Clean up URL encoding
            url = unquote(url)

            # Extract title
            title = a_tag.get_text().strip()
            if not title:
                title = url  # Fallback to URL if no title

            # Extract timestamps
            add_date = self._parse_timestamp(a_tag.get("add_date"))
            last_modified = self._parse_timestamp(a_tag.get("last_modified"))

            # Use add_date as created time, fallback to current time
            created = add_date if add_date else datetime.now(timezone.utc)

            # Create Bookmark object
            bookmark = Bookmark(
                url=url,
                title=title,
                folder=folder_path,
                note="",  # Chrome HTML doesn't include notes
                tags=[],  # Chrome HTML doesn't include tags
                created=created,
                excerpt="",  # Will be extracted later if needed
            )

            return bookmark

        except Exception as e:
            self.logger.warning(f"Error parsing bookmark: {str(e)}")
            return None

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """
        Parse Unix timestamp string to datetime object.

        Args:
            timestamp_str: Unix timestamp as string

        Returns:
            datetime object or None if parsing fails
        """
        if not timestamp_str:
            return None

        try:
            # Chrome uses Unix timestamps (seconds since epoch)
            timestamp = int(timestamp_str)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OverflowError) as e:
            self.logger.warning(f"Invalid timestamp format: {timestamp_str} - {str(e)}")
            return None

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if a file is a Chrome HTML bookmark export.

        Args:
            file_path: Path to the file to validate

        Returns:
            True if file appears to be a Chrome bookmark export
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists() or file_path.suffix.lower() != ".html":
                return False

            # Read first few lines to check DOCTYPE
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(1024)  # Read first 1KB

            return bool(re.search(self.DOCTYPE_PATTERN, header))

        except Exception:
            return False

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Get information about a Chrome HTML bookmark file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)

        info = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size_bytes": 0,
            "is_chrome_bookmarks": False,
            "estimated_bookmark_count": 0,
        }

        if not file_path.exists():
            return info

        try:
            info["size_bytes"] = file_path.stat().st_size
            info["is_chrome_bookmarks"] = self.validate_file(file_path)

            if info["is_chrome_bookmarks"]:
                # Estimate bookmark count by counting <A HREF= occurrences
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                info["estimated_bookmark_count"] = len(
                    re.findall(r"<A\s+HREF=", content, re.IGNORECASE)
                )

        except Exception as e:
            self.logger.warning(f"Error getting file info for {file_path}: {str(e)}")

        return info
