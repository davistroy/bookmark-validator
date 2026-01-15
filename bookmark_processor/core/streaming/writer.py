"""
Streaming Bookmark Writer.

Provides incremental writing of bookmarks to CSV files,
enabling output of large datasets without memory issues.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..data_models import Bookmark


class StreamingBookmarkWriter:
    """
    Write bookmarks incrementally to a CSV file.

    This class provides incremental writing capabilities, flushing
    data to disk as it's written to avoid memory buildup and ensure
    durability in case of interruption.

    Attributes:
        output_path: Path to the output CSV file
        written_count: Number of bookmarks written so far

    Example:
        >>> with StreamingBookmarkWriter(Path("output.csv")) as writer:
        ...     for bookmark in bookmarks:
        ...         writer.write(bookmark)

        >>> # Or write batches
        >>> with StreamingBookmarkWriter(Path("output.csv")) as writer:
        ...     writer.write_batch(batch1)
        ...     writer.write_batch(batch2)
    """

    # Output columns for raindrop.io import format
    IMPORT_COLUMNS = ["url", "folder", "title", "note", "tags", "created"]

    def __init__(
        self,
        output_path: Union[str, Path],
        encoding: str = "utf-8-sig",
        flush_interval: int = 10
    ):
        """
        Initialize the streaming writer.

        Args:
            output_path: Path to the output CSV file
            encoding: File encoding (default utf-8-sig for compatibility)
            flush_interval: How often to flush to disk (every N writes)
        """
        self.output_path = Path(output_path)
        self.encoding = encoding
        self.flush_interval = flush_interval
        self.logger = logging.getLogger(__name__)

        self._file = None
        self._writer = None
        self._written_count = 0
        self._is_open = False

    @property
    def written_count(self) -> int:
        """Get the number of bookmarks written."""
        return self._written_count

    def __enter__(self) -> "StreamingBookmarkWriter":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """
        Open the output file for writing.

        Creates parent directories if they don't exist.
        """
        if self._is_open:
            return

        # Create parent directories
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._file = open(
                self.output_path,
                "w",
                newline="",
                encoding=self.encoding
            )
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.IMPORT_COLUMNS,
                quoting=csv.QUOTE_ALL,
                extrasaction="ignore"
            )
            self._writer.writeheader()
            self._file.flush()
            self._is_open = True
            self._written_count = 0

            self.logger.info(f"Opened output file: {self.output_path}")

        except Exception as e:
            self.logger.error(f"Failed to open output file: {e}")
            if self._file:
                self._file.close()
                self._file = None
            raise

    def close(self) -> None:
        """Close the output file."""
        if self._file:
            try:
                self._file.flush()
                self._file.close()
                self.logger.info(
                    f"Closed output file: {self.output_path} "
                    f"({self._written_count} bookmarks written)"
                )
            except Exception as e:
                self.logger.error(f"Error closing output file: {e}")
            finally:
                self._file = None
                self._writer = None
                self._is_open = False

    def _ensure_open(self) -> None:
        """Ensure the writer is open."""
        if not self._is_open:
            raise RuntimeError(
                "Writer is not open. Use 'with' statement or call open() first."
            )

    def _bookmark_to_row(self, bookmark: Bookmark) -> Dict[str, str]:
        """
        Convert a Bookmark to a CSV row dictionary.

        Args:
            bookmark: Bookmark to convert

        Returns:
            Dictionary suitable for csv.DictWriter
        """
        # Get final tags
        tags = bookmark.get_final_tags()

        # Format tags according to raindrop.io requirements
        if len(tags) == 0:
            formatted_tags = ""
        elif len(tags) == 1:
            formatted_tags = tags[0]
        else:
            formatted_tags = f'"{", ".join(tags)}"'

        # Format created date
        if bookmark.created:
            if hasattr(bookmark.created, "isoformat"):
                created_str = bookmark.created.isoformat()
            else:
                created_str = str(bookmark.created)
        else:
            created_str = ""

        return {
            "url": bookmark.url or "",
            "folder": bookmark.get_folder_path() or bookmark.folder or "",
            "title": bookmark.get_effective_title() or "",
            "note": bookmark.get_effective_description() or "",
            "tags": formatted_tags,
            "created": created_str,
        }

    def write(self, bookmark: Bookmark) -> None:
        """
        Write a single bookmark to the output file.

        Args:
            bookmark: Bookmark to write
        """
        self._ensure_open()

        if not bookmark or not bookmark.url:
            self.logger.debug("Skipping invalid bookmark (no URL)")
            return

        try:
            row = self._bookmark_to_row(bookmark)
            self._writer.writerow(row)
            self._written_count += 1

            # Periodic flush for durability
            if self._written_count % self.flush_interval == 0:
                self._file.flush()

        except Exception as e:
            self.logger.error(f"Error writing bookmark {bookmark.url}: {e}")
            raise

    def write_batch(self, bookmarks: List[Bookmark]) -> int:
        """
        Write a batch of bookmarks to the output file.

        Args:
            bookmarks: List of bookmarks to write

        Returns:
            Number of bookmarks successfully written
        """
        self._ensure_open()

        written = 0
        for bookmark in bookmarks:
            if bookmark and bookmark.url:
                try:
                    row = self._bookmark_to_row(bookmark)
                    self._writer.writerow(row)
                    written += 1
                except Exception as e:
                    self.logger.debug(f"Error writing bookmark {bookmark.url}: {e}")

        self._written_count += written

        # Flush after batch
        self._file.flush()

        self.logger.debug(f"Wrote batch of {written} bookmarks")
        return written

    def flush(self) -> None:
        """Force flush data to disk."""
        self._ensure_open()
        self._file.flush()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get writing statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "output_path": str(self.output_path),
            "written_count": self._written_count,
            "is_open": self._is_open,
            "encoding": self.encoding,
        }

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return (
            f"StreamingBookmarkWriter(path={self.output_path}, "
            f"written={self._written_count}, status={status})"
        )


class AppendingBookmarkWriter(StreamingBookmarkWriter):
    """
    Streaming writer that appends to existing file.

    Useful for resuming interrupted processing or adding
    to existing output files.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        encoding: str = "utf-8-sig",
        flush_interval: int = 10
    ):
        """
        Initialize the appending writer.

        Args:
            output_path: Path to the output CSV file
            encoding: File encoding
            flush_interval: How often to flush to disk
        """
        super().__init__(output_path, encoding, flush_interval)
        self._header_written = False

    def open(self) -> None:
        """
        Open the output file for appending.

        If file exists, appends without writing header.
        If file doesn't exist, creates new file with header.
        """
        if self._is_open:
            return

        # Create parent directories
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and has content
        file_exists = self.output_path.exists() and self.output_path.stat().st_size > 0

        try:
            mode = "a" if file_exists else "w"
            self._file = open(
                self.output_path,
                mode,
                newline="",
                encoding=self.encoding
            )
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.IMPORT_COLUMNS,
                quoting=csv.QUOTE_ALL,
                extrasaction="ignore"
            )

            # Only write header for new files
            if not file_exists:
                self._writer.writeheader()
                self._header_written = True
            else:
                self._header_written = False

            self._file.flush()
            self._is_open = True
            self._written_count = 0

            action = "Appending to" if file_exists else "Created"
            self.logger.info(f"{action} output file: {self.output_path}")

        except Exception as e:
            self.logger.error(f"Failed to open output file: {e}")
            if self._file:
                self._file.close()
                self._file = None
            raise
