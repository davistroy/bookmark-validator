"""
Streaming Bookmark Reader.

Provides generator-based reading of bookmarks from CSV files,
enabling processing of large datasets without loading all into memory.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import chardet

from ..data_models import Bookmark


class StreamingBookmarkReader:
    """
    Read bookmarks as a stream instead of loading all into memory.

    This class provides generator-based reading of CSV files, yielding
    bookmarks one at a time or in batches. This enables processing of
    very large datasets (100k+ bookmarks) without memory issues.

    Attributes:
        input_path: Path to the input CSV file
        encoding: File encoding (auto-detected if not specified)
        total_count: Total number of rows (set after first full pass or count_rows())

    Example:
        >>> reader = StreamingBookmarkReader(Path("bookmarks.csv"))
        >>> for bookmark in reader.stream():
        ...     process(bookmark)

        >>> # Or process in batches
        >>> for batch in reader.stream_batches(batch_size=100):
        ...     process_batch(batch)
    """

    # Expected column names for raindrop.io export format
    EXPORT_COLUMNS = [
        "id", "title", "note", "excerpt", "url", "folder",
        "tags", "created", "cover", "highlights", "favorite"
    ]

    def __init__(
        self,
        input_path: Union[str, Path],
        encoding: Optional[str] = None,
        skip_invalid: bool = True
    ):
        """
        Initialize the streaming reader.

        Args:
            input_path: Path to the input CSV file
            encoding: File encoding (auto-detected if not provided)
            skip_invalid: Whether to skip invalid rows (default True)
        """
        self.input_path = Path(input_path)
        self.encoding = encoding
        self.skip_invalid = skip_invalid
        self.logger = logging.getLogger(__name__)
        self._total_count: Optional[int] = None
        self._detected_encoding: Optional[str] = None

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if not self.input_path.is_file():
            raise ValueError(f"Path is not a file: {self.input_path}")

    @property
    def total_count(self) -> Optional[int]:
        """Get total row count (None if not yet counted)."""
        return self._total_count

    def _detect_encoding(self) -> str:
        """
        Detect file encoding using chardet.

        Returns:
            Detected encoding string
        """
        if self._detected_encoding:
            return self._detected_encoding

        try:
            with open(self.input_path, "rb") as f:
                sample = f.read(65536)
                result = chardet.detect(sample)

            encoding = result.get("encoding", "utf-8")
            confidence = result.get("confidence", 0.0)

            self.logger.debug(
                f"Detected encoding: {encoding} (confidence: {confidence:.2f})"
            )

            if confidence < 0.7:
                self.logger.warning(
                    f"Low encoding confidence ({confidence:.2f}), using utf-8"
                )
                encoding = "utf-8"

            self._detected_encoding = encoding
            return encoding

        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            self._detected_encoding = "utf-8"
            return "utf-8"

    def _get_encoding(self) -> str:
        """Get the encoding to use for reading."""
        if self.encoding:
            return self.encoding
        return self._detect_encoding()

    def count_rows(self) -> int:
        """
        Count total rows in the file without loading all data.

        Returns:
            Total number of data rows (excluding header)
        """
        if self._total_count is not None:
            return self._total_count

        encoding = self._get_encoding()
        count = 0

        try:
            with open(self.input_path, "r", encoding=encoding, errors="replace") as f:
                # Skip header
                next(f, None)
                for _ in f:
                    count += 1

            self._total_count = count
            self.logger.info(f"Counted {count} rows in {self.input_path}")
            return count

        except Exception as e:
            self.logger.error(f"Error counting rows: {e}")
            return 0

    def _parse_row_to_bookmark(self, row: Dict[str, str]) -> Optional[Bookmark]:
        """
        Parse a CSV row dict into a Bookmark object.

        Args:
            row: Dictionary from csv.DictReader

        Returns:
            Bookmark object or None if parsing fails
        """
        try:
            # Extract URL first - it's required
            url = self._clean_string(row.get("url", ""))
            if not url:
                if not self.skip_invalid:
                    self.logger.warning("Row missing URL")
                return None

            # Parse tags
            tags = self._parse_tags(row.get("tags", ""))

            # Parse created date
            created = self._parse_datetime(row.get("created", ""))

            # Parse favorite boolean
            favorite = self._parse_boolean(row.get("favorite", "false"))

            return Bookmark(
                id=self._clean_string(row.get("id", "")),
                title=self._clean_string(row.get("title", "")),
                note=self._clean_string(row.get("note", "")),
                excerpt=self._clean_string(row.get("excerpt", "")),
                url=url,
                folder=self._clean_string(row.get("folder", "")),
                tags=tags,
                created=created,
                cover=self._clean_string(row.get("cover", "")),
                highlights=self._clean_string(row.get("highlights", "")),
                favorite=favorite,
            )

        except Exception as e:
            self.logger.debug(f"Error parsing row: {e}")
            if not self.skip_invalid:
                raise
            return None

    def _clean_string(self, value: Any) -> str:
        """Clean and normalize string value."""
        if value is None:
            return ""
        return str(value).strip()

    def _parse_tags(self, value: str) -> List[str]:
        """Parse tags string into list."""
        if not value or not value.strip():
            return []

        tags_str = value.strip()

        # Handle quoted tags
        if tags_str.startswith('"') and tags_str.endswith('"'):
            tags_str = tags_str[1:-1]

        # Split and clean
        tags = []
        for tag in tags_str.split(","):
            tag = tag.strip().strip("\"'")
            if tag:
                tags.append(tag)

        return tags

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        """Parse datetime string."""
        if not value or not value.strip():
            return None

        datetime_str = value.strip()

        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S+00:00",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue

        # Try fromisoformat as fallback
        try:
            clean_str = datetime_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
            return dt.replace(tzinfo=None)
        except (ValueError, AttributeError):
            pass

        return None

    def _parse_boolean(self, value: str) -> bool:
        """Parse boolean string."""
        if not value:
            return False
        return str(value).lower().strip() in ("true", "1", "yes", "on")

    def stream(self) -> Generator[Bookmark, None, None]:
        """
        Yield bookmarks one at a time.

        This generator reads the CSV file line by line, yielding
        each valid bookmark without loading all data into memory.

        Yields:
            Bookmark objects one at a time

        Example:
            >>> reader = StreamingBookmarkReader(Path("bookmarks.csv"))
            >>> for bookmark in reader.stream():
            ...     print(bookmark.url)
        """
        encoding = self._get_encoding()
        processed = 0
        skipped = 0

        self.logger.info(f"Starting to stream bookmarks from {self.input_path}")

        try:
            with open(self.input_path, "r", encoding=encoding, errors="replace", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    bookmark = self._parse_row_to_bookmark(row)
                    if bookmark:
                        processed += 1
                        yield bookmark
                    else:
                        skipped += 1

            self._total_count = processed + skipped
            self.logger.info(
                f"Streaming complete: {processed} bookmarks yielded, {skipped} skipped"
            )

        except Exception as e:
            self.logger.error(f"Error streaming bookmarks: {e}")
            raise

    def stream_batches(
        self,
        batch_size: int = 100
    ) -> Generator[List[Bookmark], None, None]:
        """
        Yield bookmarks in batches.

        This generator reads the CSV file and yields batches of bookmarks,
        which is useful for batch processing operations.

        Args:
            batch_size: Number of bookmarks per batch (default 100)

        Yields:
            Lists of Bookmark objects

        Example:
            >>> reader = StreamingBookmarkReader(Path("bookmarks.csv"))
            >>> for batch in reader.stream_batches(batch_size=50):
            ...     process_batch(batch)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batch: List[Bookmark] = []
        batch_count = 0

        for bookmark in self.stream():
            batch.append(bookmark)

            if len(batch) >= batch_size:
                batch_count += 1
                self.logger.debug(f"Yielding batch {batch_count} ({len(batch)} bookmarks)")
                yield batch
                batch = []

        # Yield remaining bookmarks
        if batch:
            batch_count += 1
            self.logger.debug(f"Yielding final batch {batch_count} ({len(batch)} bookmarks)")
            yield batch

        self.logger.info(f"Streamed {batch_count} batches total")

    def stream_with_index(self) -> Generator[tuple[int, Bookmark], None, None]:
        """
        Yield bookmarks with their index.

        Useful for progress tracking and checkpointing.

        Yields:
            Tuple of (index, Bookmark)
        """
        for index, bookmark in enumerate(self.stream()):
            yield index, bookmark

    def peek(self, count: int = 5) -> List[Bookmark]:
        """
        Preview first N bookmarks without consuming the stream.

        Args:
            count: Number of bookmarks to preview

        Returns:
            List of first N bookmarks
        """
        bookmarks = []
        for bookmark in self.stream():
            bookmarks.append(bookmark)
            if len(bookmarks) >= count:
                break
        return bookmarks

    def get_sample(self, count: int = 10, skip: int = 0) -> List[Bookmark]:
        """
        Get a sample of bookmarks from the file.

        Args:
            count: Number of bookmarks to return
            skip: Number of bookmarks to skip first

        Returns:
            List of sampled bookmarks
        """
        bookmarks = []
        skipped = 0

        for bookmark in self.stream():
            if skipped < skip:
                skipped += 1
                continue

            bookmarks.append(bookmark)
            if len(bookmarks) >= count:
                break

        return bookmarks

    def __iter__(self) -> Iterator[Bookmark]:
        """Make the reader iterable."""
        return iter(self.stream())

    def __repr__(self) -> str:
        count_str = str(self._total_count) if self._total_count else "unknown"
        return f"StreamingBookmarkReader(path={self.input_path}, count={count_str})"
