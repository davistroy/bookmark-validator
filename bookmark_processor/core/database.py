"""
Database-Backed State Management.

Provides full database backing for bookmark processing state, history,
and query capabilities for advanced processing workflows.
"""

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from .data_models import Bookmark


@dataclass
class ProcessingRun:
    """Represents a single processing run."""
    id: int
    started_at: datetime
    completed_at: Optional[datetime]
    source: str
    total_processed: int
    total_succeeded: int
    total_failed: int
    config_hash: Optional[str]
    duration_seconds: Optional[float] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ProcessingRun":
        """Create from database row."""
        started = datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
        completed = datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None

        duration = None
        if started and completed:
            duration = (completed - started).total_seconds()

        return cls(
            id=row["id"],
            started_at=started,
            completed_at=completed,
            source=row["source"],
            total_processed=row["total_processed"] or 0,
            total_succeeded=row["total_succeeded"] or 0,
            total_failed=row["total_failed"] or 0,
            config_hash=row["config_hash"],
            duration_seconds=duration
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source": self.source,
            "total_processed": self.total_processed,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "config_hash": self.config_hash,
            "duration_seconds": self.duration_seconds
        }


@dataclass
class BookmarkRecord:
    """Represents a bookmark record in the database."""
    url: str
    content_hash: str
    processed_at: datetime
    ai_engine: Optional[str]
    description: Optional[str]
    tags: List[str]
    folder: Optional[str]
    title: Optional[str]
    status: str = "processed"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "BookmarkRecord":
        """Create from database row."""
        tags = []
        if row["tags"]:
            tags = [t.strip() for t in row["tags"].split(",") if t.strip()]

        # sqlite3.Row doesn't have .get(), so check keys() instead
        row_keys = row.keys()
        status = row["status"] if "status" in row_keys and row["status"] else "processed"

        return cls(
            url=row["url"],
            content_hash=row["content_hash"],
            processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else datetime.now(),
            ai_engine=row["ai_engine"],
            description=row["description"],
            tags=tags,
            folder=row["folder"],
            title=row["title"],
            status=status
        )

    def to_bookmark(self) -> Bookmark:
        """Convert to Bookmark object."""
        return Bookmark(
            url=self.url,
            title=self.title or "",
            note=self.description or "",
            folder=self.folder or "",
            tags=self.tags,
            enhanced_description=self.description or ""
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "content_hash": self.content_hash,
            "processed_at": self.processed_at.isoformat(),
            "ai_engine": self.ai_engine,
            "description": self.description,
            "tags": self.tags,
            "folder": self.folder,
            "title": self.title,
            "status": self.status
        }


@dataclass
class RunComparison:
    """Comparison between two processing runs."""
    run1: ProcessingRun
    run2: ProcessingRun
    new_bookmarks: List[str]
    removed_bookmarks: List[str]
    changed_bookmarks: List[str]
    unchanged_bookmarks: List[str]

    @property
    def total_changes(self) -> int:
        return len(self.new_bookmarks) + len(self.removed_bookmarks) + len(self.changed_bookmarks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run1_id": self.run1.id,
            "run2_id": self.run2.id,
            "new_bookmarks_count": len(self.new_bookmarks),
            "removed_bookmarks_count": len(self.removed_bookmarks),
            "changed_bookmarks_count": len(self.changed_bookmarks),
            "unchanged_bookmarks_count": len(self.unchanged_bookmarks),
            "total_changes": self.total_changes,
            "new_bookmarks": self.new_bookmarks[:100],  # Limit for large results
            "removed_bookmarks": self.removed_bookmarks[:100],
            "changed_bookmarks": self.changed_bookmarks[:100],
        }


class BookmarkDatabase:
    """
    Full database backing for processing state and history.

    This class extends the basic ProcessingStateTracker with:
    - Advanced query capabilities
    - Full-text search
    - Processing history comparison
    - Status-based filtering
    - Date range queries

    Example:
        >>> db = BookmarkDatabase(Path("bookmarks.db"))
        >>> failed = db.query_failed()
        >>> recent = db.query_by_date(start=datetime.now() - timedelta(days=7))
        >>> comparison = db.compare_runs(1, 2)
    """

    ENHANCED_SCHEMA = """
    -- Main bookmark processing table
    CREATE TABLE IF NOT EXISTS processed_bookmarks (
        url TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        processed_at TIMESTAMP NOT NULL,
        ai_engine TEXT,
        description TEXT,
        tags TEXT,
        folder TEXT,
        title TEXT,
        status TEXT DEFAULT 'processed'
    );

    -- Processing runs table
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

    -- Bookmark history (tracks changes over time)
    CREATE TABLE IF NOT EXISTS bookmark_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        run_id INTEGER,
        content_hash TEXT NOT NULL,
        description TEXT,
        tags TEXT,
        folder TEXT,
        title TEXT,
        changed_at TIMESTAMP NOT NULL,
        change_type TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES processing_runs(id)
    );

    -- Indexes for efficient queries
    CREATE INDEX IF NOT EXISTS idx_processed_at ON processed_bookmarks(processed_at);
    CREATE INDEX IF NOT EXISTS idx_content_hash ON processed_bookmarks(content_hash);
    CREATE INDEX IF NOT EXISTS idx_status ON processed_bookmarks(status);
    CREATE INDEX IF NOT EXISTS idx_folder ON processed_bookmarks(folder);
    CREATE INDEX IF NOT EXISTS idx_runs_started ON processing_runs(started_at);
    CREATE INDEX IF NOT EXISTS idx_history_url ON bookmark_history(url);
    CREATE INDEX IF NOT EXISTS idx_history_run ON bookmark_history(run_id);

    -- Views for common queries
    CREATE VIEW IF NOT EXISTS failed_bookmarks AS
        SELECT * FROM processed_bookmarks WHERE status = 'failed';

    CREATE VIEW IF NOT EXISTS recent_bookmarks AS
        SELECT * FROM processed_bookmarks
        ORDER BY processed_at DESC LIMIT 100;

    CREATE VIEW IF NOT EXISTS run_summary AS
        SELECT
            id,
            started_at,
            completed_at,
            source,
            total_processed,
            total_succeeded,
            total_failed,
            ROUND(CAST(total_succeeded AS FLOAT) / NULLIF(total_processed, 0) * 100, 2) as success_rate,
            ROUND((julianday(completed_at) - julianday(started_at)) * 86400, 2) as duration_seconds
        FROM processing_runs
        WHERE completed_at IS NOT NULL;
    """

    FTS_SCHEMA = """
    -- Full-text search table
    CREATE VIRTUAL TABLE IF NOT EXISTS bookmark_fts USING fts5(
        url,
        title,
        description,
        tags,
        content='processed_bookmarks',
        content_rowid='rowid'
    );

    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS bookmark_fts_insert AFTER INSERT ON processed_bookmarks BEGIN
        INSERT INTO bookmark_fts(rowid, url, title, description, tags)
        VALUES (NEW.rowid, NEW.url, NEW.title, NEW.description, NEW.tags);
    END;

    CREATE TRIGGER IF NOT EXISTS bookmark_fts_update AFTER UPDATE ON processed_bookmarks BEGIN
        INSERT INTO bookmark_fts(bookmark_fts, rowid, url, title, description, tags)
        VALUES ('delete', OLD.rowid, OLD.url, OLD.title, OLD.description, OLD.tags);
        INSERT INTO bookmark_fts(rowid, url, title, description, tags)
        VALUES (NEW.rowid, NEW.url, NEW.title, NEW.description, NEW.tags);
    END;

    CREATE TRIGGER IF NOT EXISTS bookmark_fts_delete AFTER DELETE ON processed_bookmarks BEGIN
        INSERT INTO bookmark_fts(bookmark_fts, rowid, url, title, description, tags)
        VALUES ('delete', OLD.rowid, OLD.url, OLD.title, OLD.description, OLD.tags);
    END;
    """

    def __init__(
        self,
        db_path: Union[str, Path] = Path(".bookmark_database.db"),
        enable_fts: bool = True
    ):
        """
        Initialize the bookmark database.

        Args:
            db_path: Path to SQLite database file
            enable_fts: Enable full-text search (default True)
        """
        self.db_path = Path(db_path)
        self.enable_fts = enable_fts
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None
        self._current_run_id: Optional[int] = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        try:
            with self._get_connection() as conn:
                conn.executescript(self.ENHANCED_SCHEMA)

                if self.enable_fts:
                    try:
                        conn.executescript(self.FTS_SCHEMA)
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            self.logger.warning(f"FTS setup failed: {e}")

                conn.commit()
            self.logger.debug(f"Database initialized at {self.db_path}")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()

    def _compute_hash(self, bookmark: Bookmark) -> str:
        """Compute content hash for a bookmark."""
        content_parts = [
            bookmark.url or "",
            bookmark.title or "",
            bookmark.note or "",
            bookmark.excerpt or "",
            bookmark.folder or "",
            ",".join(sorted(bookmark.tags)) if bookmark.tags else ""
        ]
        content = "|".join(content_parts)
        return hashlib.md5(content.encode("utf-8", errors="surrogatepass")).hexdigest()

    # ============ Query Methods ============

    def query_failed(self) -> List[BookmarkRecord]:
        """
        Query all failed bookmarks.

        Returns:
            List of BookmarkRecord objects with failed status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM processed_bookmarks WHERE status = 'failed'"
                )
                return [BookmarkRecord.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error querying failed bookmarks: {e}")
            return []

    def query_by_date(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[BookmarkRecord]:
        """
        Query bookmarks processed within a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of BookmarkRecord objects
        """
        try:
            with self._get_connection() as conn:
                if start and end:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processed_bookmarks
                        WHERE processed_at >= ? AND processed_at <= ?
                        ORDER BY processed_at DESC
                        """,
                        (start.isoformat(), end.isoformat())
                    )
                elif start:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processed_bookmarks
                        WHERE processed_at >= ?
                        ORDER BY processed_at DESC
                        """,
                        (start.isoformat(),)
                    )
                elif end:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processed_bookmarks
                        WHERE processed_at <= ?
                        ORDER BY processed_at DESC
                        """,
                        (end.isoformat(),)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM processed_bookmarks ORDER BY processed_at DESC"
                    )

                return [BookmarkRecord.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error querying by date: {e}")
            return []

    def query_by_status(self, status: str) -> List[BookmarkRecord]:
        """
        Query bookmarks by processing status.

        Args:
            status: Status to filter by (e.g., 'processed', 'failed', 'pending')

        Returns:
            List of BookmarkRecord objects
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM processed_bookmarks WHERE status = ?",
                    (status,)
                )
                return [BookmarkRecord.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error querying by status: {e}")
            return []

    def query_by_folder(self, folder: str, exact: bool = False) -> List[BookmarkRecord]:
        """
        Query bookmarks by folder.

        Args:
            folder: Folder path to search
            exact: If True, match exact folder; if False, match prefix

        Returns:
            List of BookmarkRecord objects
        """
        try:
            with self._get_connection() as conn:
                if exact:
                    cursor = conn.execute(
                        "SELECT * FROM processed_bookmarks WHERE folder = ?",
                        (folder,)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM processed_bookmarks WHERE folder LIKE ?",
                        (f"{folder}%",)
                    )

                return [BookmarkRecord.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error querying by folder: {e}")
            return []

    def query_by_tag(self, tag: str) -> List[BookmarkRecord]:
        """
        Query bookmarks that have a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of BookmarkRecord objects
        """
        try:
            with self._get_connection() as conn:
                # SQLite LIKE for searching within comma-separated tags
                cursor = conn.execute(
                    """
                    SELECT * FROM processed_bookmarks
                    WHERE tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR tags = ?
                    """,
                    (f"{tag},%", f"%, {tag},%", f"%, {tag}", tag)
                )
                return [BookmarkRecord.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error querying by tag: {e}")
            return []

    def search_content(self, query: str, limit: int = 100) -> List[BookmarkRecord]:
        """
        Full-text search across bookmark content.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of BookmarkRecord objects matching the query
        """
        if not self.enable_fts:
            self.logger.warning("Full-text search not enabled")
            return self._fallback_search(query, limit)

        try:
            with self._get_connection() as conn:
                # Use FTS5 MATCH syntax
                cursor = conn.execute(
                    """
                    SELECT p.* FROM processed_bookmarks p
                    INNER JOIN bookmark_fts ON p.rowid = bookmark_fts.rowid
                    WHERE bookmark_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit)
                )
                return [BookmarkRecord.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.warning(f"FTS search failed, using fallback: {e}")
            return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> List[BookmarkRecord]:
        """Fallback search using LIKE."""
        try:
            with self._get_connection() as conn:
                pattern = f"%{query}%"
                cursor = conn.execute(
                    """
                    SELECT * FROM processed_bookmarks
                    WHERE title LIKE ? OR description LIKE ? OR tags LIKE ? OR url LIKE ?
                    LIMIT ?
                    """,
                    (pattern, pattern, pattern, pattern, limit)
                )
                return [BookmarkRecord.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Fallback search failed: {e}")
            return []

    # ============ History Methods ============

    def get_processing_history(self, url: str) -> List[Dict[str, Any]]:
        """
        Get processing history for a specific URL.

        Args:
            url: URL to get history for

        Returns:
            List of history records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT h.*, r.source, r.started_at as run_started
                    FROM bookmark_history h
                    LEFT JOIN processing_runs r ON h.run_id = r.id
                    WHERE h.url = ?
                    ORDER BY h.changed_at DESC
                    """,
                    (url,)
                )
                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error getting processing history: {e}")
            return []

    def compare_runs(self, run1_id: int, run2_id: int) -> Optional[RunComparison]:
        """
        Compare two processing runs to see changes.

        Args:
            run1_id: First run ID (earlier)
            run2_id: Second run ID (later)

        Returns:
            RunComparison object or None if runs not found
        """
        try:
            with self._get_connection() as conn:
                # Get run info
                run1_row = conn.execute(
                    "SELECT * FROM processing_runs WHERE id = ?",
                    (run1_id,)
                ).fetchone()
                run2_row = conn.execute(
                    "SELECT * FROM processing_runs WHERE id = ?",
                    (run2_id,)
                ).fetchone()

                if not run1_row or not run2_row:
                    return None

                run1 = ProcessingRun.from_row(run1_row)
                run2 = ProcessingRun.from_row(run2_row)

                # Get URLs from each run's history
                run1_urls = set()
                run1_hashes = {}
                cursor = conn.execute(
                    "SELECT url, content_hash FROM bookmark_history WHERE run_id = ?",
                    (run1_id,)
                )
                for row in cursor.fetchall():
                    run1_urls.add(row["url"])
                    run1_hashes[row["url"]] = row["content_hash"]

                run2_urls = set()
                run2_hashes = {}
                cursor = conn.execute(
                    "SELECT url, content_hash FROM bookmark_history WHERE run_id = ?",
                    (run2_id,)
                )
                for row in cursor.fetchall():
                    run2_urls.add(row["url"])
                    run2_hashes[row["url"]] = row["content_hash"]

                # Calculate differences
                new_bookmarks = list(run2_urls - run1_urls)
                removed_bookmarks = list(run1_urls - run2_urls)

                # Find changed (same URL, different hash)
                common_urls = run1_urls & run2_urls
                changed_bookmarks = [
                    url for url in common_urls
                    if run1_hashes.get(url) != run2_hashes.get(url)
                ]
                unchanged_bookmarks = [
                    url for url in common_urls
                    if run1_hashes.get(url) == run2_hashes.get(url)
                ]

                return RunComparison(
                    run1=run1,
                    run2=run2,
                    new_bookmarks=new_bookmarks,
                    removed_bookmarks=removed_bookmarks,
                    changed_bookmarks=changed_bookmarks,
                    unchanged_bookmarks=unchanged_bookmarks
                )

        except sqlite3.Error as e:
            self.logger.error(f"Error comparing runs: {e}")
            return None

    def get_run_history(
        self,
        limit: int = 10,
        source: Optional[str] = None
    ) -> List[ProcessingRun]:
        """
        Get processing run history.

        Args:
            limit: Maximum runs to return
            source: Optional filter by source

        Returns:
            List of ProcessingRun objects
        """
        try:
            with self._get_connection() as conn:
                if source:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processing_runs
                        WHERE source = ?
                        ORDER BY started_at DESC
                        LIMIT ?
                        """,
                        (source, limit)
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM processing_runs
                        ORDER BY started_at DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )

                return [ProcessingRun.from_row(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error getting run history: {e}")
            return []

    # ============ State Management Methods ============

    def mark_processed(
        self,
        bookmark: Bookmark,
        content_hash: Optional[str] = None,
        ai_engine: str = "local",
        status: str = "processed",
        run_id: Optional[int] = None
    ) -> None:
        """
        Mark a bookmark as processed.

        Args:
            bookmark: Processed bookmark
            content_hash: Optional pre-computed hash
            ai_engine: AI engine used
            status: Processing status
            run_id: Optional run ID for history tracking
        """
        if content_hash is None:
            content_hash = self._compute_hash(bookmark)

        tags_str = ",".join(bookmark.optimized_tags or bookmark.tags or [])
        now = datetime.now().isoformat()

        try:
            with self._get_connection() as conn:
                # Update or insert main record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO processed_bookmarks
                    (url, content_hash, processed_at, ai_engine, description, tags, folder, title, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bookmark.url,
                        content_hash,
                        now,
                        ai_engine,
                        bookmark.enhanced_description or bookmark.get_effective_description(),
                        tags_str,
                        bookmark.folder,
                        bookmark.get_effective_title(),
                        status
                    )
                )

                # Add to history if run_id provided
                if run_id:
                    conn.execute(
                        """
                        INSERT INTO bookmark_history
                        (url, run_id, content_hash, description, tags, folder, title, changed_at, change_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            bookmark.url,
                            run_id,
                            content_hash,
                            bookmark.enhanced_description or bookmark.get_effective_description(),
                            tags_str,
                            bookmark.folder,
                            bookmark.get_effective_title(),
                            now,
                            "processed"
                        )
                    )

                conn.commit()

        except sqlite3.Error as e:
            self.logger.error(f"Error marking bookmark as processed: {e}")
            raise

    def mark_failed(
        self,
        url: str,
        error_message: str,
        run_id: Optional[int] = None
    ) -> None:
        """
        Mark a URL as failed processing.

        Args:
            url: URL that failed
            error_message: Error description
            run_id: Optional run ID
        """
        now = datetime.now().isoformat()

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO processed_bookmarks
                    (url, content_hash, processed_at, status, description)
                    VALUES (?, ?, ?, 'failed', ?)
                    """,
                    (url, "", now, error_message)
                )

                if run_id:
                    conn.execute(
                        """
                        INSERT INTO bookmark_history
                        (url, run_id, content_hash, description, changed_at, change_type)
                        VALUES (?, ?, '', ?, ?, 'failed')
                        """,
                        (url, run_id, error_message, now)
                    )

                conn.commit()

        except sqlite3.Error as e:
            self.logger.error(f"Error marking URL as failed: {e}")

    def needs_processing(self, bookmark: Bookmark) -> bool:
        """Check if a bookmark needs processing."""
        current_hash = self._compute_hash(bookmark)

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT content_hash, status FROM processed_bookmarks WHERE url = ?",
                    (bookmark.url,)
                )
                row = cursor.fetchone()

                if row is None:
                    return True

                # Needs reprocessing if hash changed or previous run failed
                return row["content_hash"] != current_hash or row["status"] == "failed"

        except sqlite3.Error as e:
            self.logger.error(f"Error checking processing status: {e}")
            return True

    def get_unprocessed(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """Filter to only bookmarks that need processing."""
        if not bookmarks:
            return []

        return [b for b in bookmarks if self.needs_processing(b)]

    # ============ Run Management Methods ============

    def start_processing_run(
        self,
        source: str,
        config_hash: Optional[str] = None
    ) -> int:
        """
        Start a new processing run.

        Args:
            source: Source identifier
            config_hash: Optional configuration hash

        Returns:
            Run ID
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
                return self._current_run_id

        except sqlite3.Error as e:
            self.logger.error(f"Error starting run: {e}")
            raise

    def complete_processing_run(
        self,
        run_id: Optional[int] = None,
        total_processed: int = 0,
        total_succeeded: int = 0,
        total_failed: int = 0
    ) -> None:
        """Complete a processing run."""
        run_id = run_id or self._current_run_id
        if not run_id:
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

        except sqlite3.Error as e:
            self.logger.error(f"Error completing run: {e}")

    # ============ Statistics Methods ============

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with various statistics
        """
        try:
            with self._get_connection() as conn:
                stats = {}

                # Total bookmarks
                cursor = conn.execute("SELECT COUNT(*) as count FROM processed_bookmarks")
                stats["total_bookmarks"] = cursor.fetchone()["count"]

                # By status
                cursor = conn.execute(
                    """
                    SELECT status, COUNT(*) as count
                    FROM processed_bookmarks
                    GROUP BY status
                    """
                )
                stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

                # Total runs
                cursor = conn.execute("SELECT COUNT(*) as count FROM processing_runs")
                stats["total_runs"] = cursor.fetchone()["count"]

                # Recent activity
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as count FROM processed_bookmarks
                    WHERE processed_at >= datetime('now', '-7 days')
                    """
                )
                stats["processed_last_7_days"] = cursor.fetchone()["count"]

                # Unique folders
                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT folder) as count FROM processed_bookmarks"
                )
                stats["unique_folders"] = cursor.fetchone()["count"]

                # Average tags per bookmark
                cursor = conn.execute(
                    """
                    SELECT AVG(LENGTH(tags) - LENGTH(REPLACE(tags, ',', '')) + 1) as avg
                    FROM processed_bookmarks WHERE tags != '' AND tags IS NOT NULL
                    """
                )
                result = cursor.fetchone()
                stats["avg_tags_per_bookmark"] = round(result["avg"] or 0, 2)

                return stats

        except sqlite3.Error as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}

    def vacuum(self) -> None:
        """Optimize database storage."""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
            self.logger.info("Database vacuumed successfully")

        except sqlite3.Error as e:
            self.logger.error(f"Error vacuuming database: {e}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __repr__(self) -> str:
        return f"BookmarkDatabase(db_path={self.db_path})"
