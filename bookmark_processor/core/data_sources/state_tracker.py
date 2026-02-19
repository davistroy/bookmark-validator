"""
Processing State Tracker for Incremental Updates.

This module provides SQLite-based tracking of bookmark processing state,
enabling incremental updates by remembering which bookmarks have been
processed and detecting content changes.
"""

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from ..data_models import Bookmark


class ProcessingStateTracker:
    """
    Track bookmark processing state for incremental updates.

    This class maintains a SQLite database to track:
    - Which bookmarks have been processed
    - Content hashes to detect changes
    - Processing run history
    - AI engine used for each bookmark

    This enables incremental processing by only processing bookmarks
    that are new or have changed since the last run.

    Attributes:
        db_path: Path to the SQLite database file

    Example:
        >>> tracker = ProcessingStateTracker(Path(".bookmark_state.db"))
        >>> unprocessed = tracker.get_unprocessed(bookmarks)
        >>> for bookmark in unprocessed:
        ...     # Process bookmark...
        ...     tracker.mark_processed(bookmark, content_hash, "claude")
    """

    DB_SCHEMA = """
    CREATE TABLE IF NOT EXISTS processed_bookmarks (
        url TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        processed_at TIMESTAMP NOT NULL,
        ai_engine TEXT,
        description TEXT,
        tags TEXT,
        folder TEXT,
        title TEXT
    );

    CREATE TABLE IF NOT EXISTS processing_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        source TEXT NOT NULL,
        total_processed INTEGER DEFAULT 0,
        total_succeeded INTEGER DEFAULT 0,
        total_failed INTEGER DEFAULT 0,
        config_hash TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_processed_at
        ON processed_bookmarks(processed_at);

    CREATE INDEX IF NOT EXISTS idx_content_hash
        ON processed_bookmarks(content_hash);

    CREATE INDEX IF NOT EXISTS idx_runs_started
        ON processing_runs(started_at);
    """

    def __init__(
        self,
        db_path: Union[str, Path] = Path(".bookmark_processor_state.db")
    ):
        """
        Initialize the processing state tracker.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None
        self._current_run_id: Optional[int] = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        try:
            with self._get_connection() as conn:
                conn.executescript(self.DB_SCHEMA)
                conn.commit()
            self.logger.debug(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection with proper error handling.

        Yields:
            SQLite connection
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()

    def _compute_hash(self, bookmark: Bookmark) -> str:
        """
        Compute content hash for change detection.

        The hash is based on fields that, when changed, should trigger
        reprocessing of the bookmark.

        Args:
            bookmark: The bookmark to hash

        Returns:
            MD5 hash string
        """
        # Include fields that would warrant reprocessing if changed
        content_parts = [
            bookmark.url or "",
            bookmark.title or "",
            bookmark.note or "",
            bookmark.excerpt or "",
            bookmark.folder or "",
            ",".join(sorted(bookmark.tags)) if bookmark.tags else ""
        ]
        content = "|".join(content_parts)
        # Use surrogatepass to handle any malformed unicode
        return hashlib.md5(content.encode("utf-8", errors="surrogatepass"), usedforsecurity=False).hexdigest()

    def mark_processed(
        self,
        bookmark: Bookmark,
        content_hash: Optional[str] = None,
        ai_engine: str = "local"
    ) -> None:
        """
        Mark a bookmark as processed.

        Args:
            bookmark: The processed bookmark
            content_hash: Optional pre-computed hash (computed if not provided)
            ai_engine: The AI engine used for processing
        """
        if content_hash is None:
            content_hash = self._compute_hash(bookmark)

        tags_str = ",".join(bookmark.optimized_tags or bookmark.tags or [])

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO processed_bookmarks
                    (url, content_hash, processed_at, ai_engine, description, tags, folder, title)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bookmark.url,
                        content_hash,
                        datetime.now().isoformat(),
                        ai_engine,
                        bookmark.enhanced_description or bookmark.get_effective_description(),
                        tags_str,
                        bookmark.folder,
                        bookmark.get_effective_title()
                    )
                )
                conn.commit()
                self.logger.debug(f"Marked as processed: {bookmark.url}")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to mark bookmark as processed: {e}")
            raise

    def needs_processing(self, bookmark: Bookmark) -> bool:
        """
        Check if a bookmark needs (re)processing.

        A bookmark needs processing if:
        - It has never been processed
        - Its content hash has changed since last processing

        Args:
            bookmark: The bookmark to check

        Returns:
            True if bookmark needs processing
        """
        current_hash = self._compute_hash(bookmark)

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT content_hash FROM processed_bookmarks WHERE url = ?",
                    (bookmark.url,)
                )
                row = cursor.fetchone()

                if row is None:
                    return True  # Never processed

                return row["content_hash"] != current_hash  # Content changed

        except sqlite3.Error as e:
            self.logger.error(f"Error checking processing status: {e}")
            return True  # Assume needs processing on error

    def get_unprocessed(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """
        Filter to only bookmarks that need processing.

        This is more efficient than calling needs_processing() for each
        bookmark individually as it uses a single database query.

        Args:
            bookmarks: List of bookmarks to filter

        Returns:
            List of bookmarks that need processing
        """
        if not bookmarks:
            return []

        # Compute hashes for all bookmarks
        bookmark_hashes = {
            bookmark.url: self._compute_hash(bookmark)
            for bookmark in bookmarks
        }

        try:
            with self._get_connection() as conn:
                # Get existing hashes from database
                placeholders = ",".join("?" * len(bookmark_hashes))
                cursor = conn.execute(
                    f"SELECT url, content_hash FROM processed_bookmarks WHERE url IN ({placeholders})",  # nosec B608 - parameterized query with ? placeholders
                    list(bookmark_hashes.keys())
                )

                # Build dict of URL -> stored hash
                stored_hashes = {
                    row["url"]: row["content_hash"]
                    for row in cursor.fetchall()
                }

            # Filter bookmarks that need processing
            unprocessed = []
            for bookmark in bookmarks:
                stored_hash = stored_hashes.get(bookmark.url)
                current_hash = bookmark_hashes[bookmark.url]

                if stored_hash is None or stored_hash != current_hash:
                    unprocessed.append(bookmark)

            self.logger.info(
                f"Found {len(unprocessed)} unprocessed bookmarks "
                f"out of {len(bookmarks)} total"
            )
            return unprocessed

        except sqlite3.Error as e:
            self.logger.error(f"Error getting unprocessed bookmarks: {e}")
            return bookmarks  # Return all on error

    def get_processed_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get processing information for a bookmark URL.

        Args:
            url: The bookmark URL

        Returns:
            Dictionary with processing info, or None if not processed
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT url, content_hash, processed_at, ai_engine,
                           description, tags, folder, title
                    FROM processed_bookmarks WHERE url = ?
                    """,
                    (url,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return dict(row)

        except sqlite3.Error as e:
            self.logger.error(f"Error getting processed info: {e}")
            return None

    def start_processing_run(
        self,
        source: str,
        config_hash: Optional[str] = None
    ) -> int:
        """
        Start a new processing run.

        Args:
            source: Name of the data source
            config_hash: Optional hash of configuration for tracking

        Returns:
            Run ID for this processing run
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO processing_runs (started_at, source, config_hash)
                    VALUES (?, ?, ?)
                    """,
                    (datetime.now().isoformat(), source, config_hash)
                )
                conn.commit()
                self._current_run_id = cursor.lastrowid
                self.logger.info(f"Started processing run {self._current_run_id}")
                return self._current_run_id

        except sqlite3.Error as e:
            self.logger.error(f"Failed to start processing run: {e}")
            raise

    def complete_processing_run(
        self,
        run_id: Optional[int] = None,
        total_processed: int = 0,
        total_succeeded: int = 0,
        total_failed: int = 0
    ) -> None:
        """
        Complete a processing run with statistics.

        Args:
            run_id: The run ID (uses current run if not specified)
            total_processed: Total bookmarks processed
            total_succeeded: Number of successful processing
            total_failed: Number of failed processing
        """
        run_id = run_id or self._current_run_id
        if run_id is None:
            self.logger.warning("No active run to complete")
            return

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE processing_runs
                    SET completed_at = ?, total_processed = ?,
                        total_succeeded = ?, total_failed = ?
                    WHERE id = ?
                    """,
                    (
                        datetime.now().isoformat(),
                        total_processed,
                        total_succeeded,
                        total_failed,
                        run_id
                    )
                )
                conn.commit()
                self.logger.info(f"Completed processing run {run_id}")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to complete processing run: {e}")
            raise

    def get_last_run(self, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about the last processing run.

        Args:
            source: Optional filter by source name

        Returns:
            Dictionary with run info, or None if no runs
        """
        try:
            with self._get_connection() as conn:
                if source:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processing_runs
                        WHERE source = ?
                        ORDER BY started_at DESC LIMIT 1
                        """,
                        (source,)
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processing_runs
                        ORDER BY started_at DESC LIMIT 1
                        """
                    )
                row = cursor.fetchone()
                return dict(row) if row else None

        except sqlite3.Error as e:
            self.logger.error(f"Error getting last run: {e}")
            return None

    def get_run_history(
        self,
        limit: int = 10,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get processing run history.

        Args:
            limit: Maximum number of runs to return
            source: Optional filter by source name

        Returns:
            List of run dictionaries, most recent first
        """
        try:
            with self._get_connection() as conn:
                if source:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processing_runs
                        WHERE source = ?
                        ORDER BY started_at DESC LIMIT ?
                        """,
                        (source, limit)
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processing_runs
                        ORDER BY started_at DESC LIMIT ?
                        """,
                        (limit,)
                    )
                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error getting run history: {e}")
            return []

    def get_processed_count(self) -> int:
        """
        Get the total number of processed bookmarks.

        Returns:
            Count of processed bookmarks
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM processed_bookmarks"
                )
                row = cursor.fetchone()
                return row["count"] if row else 0

        except sqlite3.Error as e:
            self.logger.error(f"Error getting processed count: {e}")
            return 0

    def get_processed_urls(self) -> List[str]:
        """
        Get all processed URLs.

        Returns:
            List of processed URLs
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT url FROM processed_bookmarks")
                return [row["url"] for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error getting processed URLs: {e}")
            return []

    def clear_processing_state(self, older_than: Optional[datetime] = None) -> int:
        """
        Clear processing state from the database.

        Args:
            older_than: If provided, only clear state older than this date

        Returns:
            Number of records cleared
        """
        try:
            with self._get_connection() as conn:
                if older_than:
                    cursor = conn.execute(
                        "DELETE FROM processed_bookmarks WHERE processed_at < ?",
                        (older_than.isoformat(),)
                    )
                else:
                    cursor = conn.execute("DELETE FROM processed_bookmarks")

                count = cursor.rowcount
                conn.commit()
                self.logger.info(f"Cleared {count} processing state records")
                return count

        except sqlite3.Error as e:
            self.logger.error(f"Error clearing processing state: {e}")
            return 0

    def remove_bookmark_state(self, url: str) -> bool:
        """
        Remove processing state for a specific bookmark.

        Args:
            url: The bookmark URL

        Returns:
            True if state was removed
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM processed_bookmarks WHERE url = ?",
                    (url,)
                )
                conn.commit()
                return cursor.rowcount > 0

        except sqlite3.Error as e:
            self.logger.error(f"Error removing bookmark state: {e}")
            return False

    def export_state(self, output_path: Path) -> None:
        """
        Export processing state to a JSON file.

        Args:
            output_path: Path to write JSON export
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM processed_bookmarks")
                bookmarks = [dict(row) for row in cursor.fetchall()]

                cursor = conn.execute("SELECT * FROM processing_runs")
                runs = [dict(row) for row in cursor.fetchall()]

            export_data = {
                "exported_at": datetime.now().isoformat(),
                "processed_bookmarks": bookmarks,
                "processing_runs": runs
            }

            output_path.write_text(json.dumps(export_data, indent=2))
            self.logger.info(f"Exported state to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting state: {e}")
            raise

    def import_state(self, input_path: Path) -> Tuple[int, int]:
        """
        Import processing state from a JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Tuple of (bookmarks imported, runs imported)
        """
        try:
            data = json.loads(input_path.read_text())

            bookmarks_imported = 0
            runs_imported = 0

            with self._get_connection() as conn:
                # Import bookmarks
                for bookmark in data.get("processed_bookmarks", []):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO processed_bookmarks
                        (url, content_hash, processed_at, ai_engine, description, tags, folder, title)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            bookmark.get("url"),
                            bookmark.get("content_hash"),
                            bookmark.get("processed_at"),
                            bookmark.get("ai_engine"),
                            bookmark.get("description"),
                            bookmark.get("tags"),
                            bookmark.get("folder"),
                            bookmark.get("title")
                        )
                    )
                    bookmarks_imported += 1

                # Import runs (skip existing IDs)
                for run in data.get("processing_runs", []):
                    try:
                        conn.execute(
                            """
                            INSERT INTO processing_runs
                            (started_at, completed_at, source, total_processed,
                             total_succeeded, total_failed, config_hash)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run.get("started_at"),
                                run.get("completed_at"),
                                run.get("source"),
                                run.get("total_processed"),
                                run.get("total_succeeded"),
                                run.get("total_failed"),
                                run.get("config_hash")
                            )
                        )
                        runs_imported += 1
                    except sqlite3.IntegrityError:
                        pass  # Skip duplicate runs

                conn.commit()

            self.logger.info(
                f"Imported {bookmarks_imported} bookmarks and {runs_imported} runs"
            )
            return (bookmarks_imported, runs_imported)

        except Exception as e:
            self.logger.error(f"Error importing state: {e}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Cleanup on destruction."""
        self.close()

    def __repr__(self) -> str:
        return f"ProcessingStateTracker(db_path={self.db_path})"
