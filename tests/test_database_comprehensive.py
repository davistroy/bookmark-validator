"""
Comprehensive tests for Database-Backed State Management.

Target: 90%+ code coverage for bookmark_processor/core/database.py

Tests cover:
- BookmarkDatabase class - initialization, connection management
- CRUD operations for bookmarks
- FTS5 full-text search functionality
- Run comparison features
- Transaction handling
- Error handling for database operations
- Connection pooling and cleanup
- Data model classes (ProcessingRun, BookmarkRecord, RunComparison)
"""

import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.database import (
    BookmarkDatabase,
    BookmarkRecord,
    ProcessingRun,
    RunComparison,
)


# ============ Fixtures ============


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_comprehensive.db"


@pytest.fixture
def db(tmp_path: Path) -> Generator[BookmarkDatabase, None, None]:
    """Create a file-based database for testing."""
    db_path = tmp_path / "test_db.db"
    db = BookmarkDatabase(db_path, enable_fts=True)
    yield db
    db.close()


@pytest.fixture
def db_no_fts(tmp_path: Path) -> Generator[BookmarkDatabase, None, None]:
    """Create a file-based database without FTS."""
    db_path = tmp_path / "test_db_no_fts.db"
    db = BookmarkDatabase(db_path, enable_fts=False)
    yield db
    db.close()


@pytest.fixture
def sample_bookmark() -> Bookmark:
    """Create a sample bookmark for testing."""
    return Bookmark(
        id="test-1",
        url="https://example.com/test",
        title="Test Bookmark",
        note="Test note for the bookmark",
        excerpt="Test excerpt",
        folder="Tech/Testing",
        tags=["test", "example", "python"],
        enhanced_description="Enhanced test description",
        optimized_tags=["testing", "automation"],
    )


@pytest.fixture
def sample_bookmarks() -> list:
    """Create multiple sample bookmarks."""
    return [
        Bookmark(
            id="1",
            url="https://example.com/article1",
            title="Python Tutorial",
            note="Learn Python basics",
            folder="Programming/Python",
            tags=["python", "tutorial", "beginner"],
            enhanced_description="Comprehensive Python learning resource",
        ),
        Bookmark(
            id="2",
            url="https://example.com/article2",
            title="Machine Learning Guide",
            note="ML fundamentals",
            folder="AI/ML",
            tags=["machine-learning", "ai", "data-science"],
            enhanced_description="Introduction to machine learning concepts",
        ),
        Bookmark(
            id="3",
            url="https://example.com/article3",
            title="Web Development",
            note="Frontend development",
            folder="Programming/Web",
            tags=["web", "frontend", "javascript"],
            enhanced_description="Modern web development practices",
        ),
        Bookmark(
            id="4",
            url="https://example.com/article4",
            title="Database Design",
            note="SQL and NoSQL",
            folder="Programming/Database",
            tags=["database", "sql", "design"],
            enhanced_description="Database architecture patterns",
        ),
    ]


@pytest.fixture
def populated_db(db: BookmarkDatabase, sample_bookmarks: list) -> BookmarkDatabase:
    """Create a database populated with sample data."""
    run_id = db.start_processing_run("test_source", config_hash="test_config")

    for i, bookmark in enumerate(sample_bookmarks):
        if i == 3:  # Mark one as failed
            db.mark_failed(bookmark.url, "Test error", run_id=run_id)
        else:
            db.mark_processed(bookmark, ai_engine="test", run_id=run_id)

    db.complete_processing_run(
        run_id=run_id,
        total_processed=4,
        total_succeeded=3,
        total_failed=1
    )

    return db


# ============ ProcessingRun Dataclass Tests ============


class TestProcessingRunDataclass:
    """Tests for ProcessingRun dataclass."""

    def test_from_row_complete(self):
        """Test from_row with complete data."""
        # Create a sqlite3.Row-like object
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                id INTEGER, started_at TEXT, completed_at TEXT,
                source TEXT, total_processed INTEGER, total_succeeded INTEGER,
                total_failed INTEGER, config_hash TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "2024-01-15T10:00:00", "2024-01-15T10:30:00", "test_source",
             100, 95, 5, "abc123")
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        run = ProcessingRun.from_row(row)

        assert run.id == 1
        assert run.source == "test_source"
        assert run.total_processed == 100
        assert run.total_succeeded == 95
        assert run.total_failed == 5
        assert run.config_hash == "abc123"
        assert run.duration_seconds == 1800.0  # 30 minutes

    def test_from_row_incomplete(self):
        """Test from_row with None values."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                id INTEGER, started_at TEXT, completed_at TEXT,
                source TEXT, total_processed INTEGER, total_succeeded INTEGER,
                total_failed INTEGER, config_hash TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (1, '2024-01-15T10:00:00', NULL, 'test', NULL, NULL, NULL, NULL)"
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        run = ProcessingRun.from_row(row)

        assert run.id == 1
        assert run.completed_at is None
        assert run.total_processed == 0
        assert run.total_succeeded == 0
        assert run.total_failed == 0
        assert run.config_hash is None
        assert run.duration_seconds is None

    def test_from_row_no_started_at(self):
        """Test from_row when started_at is None."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                id INTEGER, started_at TEXT, completed_at TEXT,
                source TEXT, total_processed INTEGER, total_succeeded INTEGER,
                total_failed INTEGER, config_hash TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (1, NULL, NULL, 'test', 0, 0, 0, NULL)"
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        run = ProcessingRun.from_row(row)

        assert run.started_at is None
        assert run.duration_seconds is None

    def test_to_dict_complete(self):
        """Test to_dict with complete data."""
        run = ProcessingRun(
            id=1,
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            completed_at=datetime(2024, 1, 15, 10, 30, 0),
            source="test",
            total_processed=100,
            total_succeeded=95,
            total_failed=5,
            config_hash="abc123",
            duration_seconds=1800.0,
        )

        d = run.to_dict()

        assert d["id"] == 1
        assert d["started_at"] == "2024-01-15T10:00:00"
        assert d["completed_at"] == "2024-01-15T10:30:00"
        assert d["source"] == "test"
        assert d["duration_seconds"] == 1800.0

    def test_to_dict_none_dates(self):
        """Test to_dict with None dates."""
        run = ProcessingRun(
            id=1,
            started_at=None,
            completed_at=None,
            source="test",
            total_processed=0,
            total_succeeded=0,
            total_failed=0,
            config_hash=None,
        )

        d = run.to_dict()

        assert d["started_at"] is None
        assert d["completed_at"] is None


# ============ BookmarkRecord Dataclass Tests ============


class TestBookmarkRecordDataclass:
    """Tests for BookmarkRecord dataclass."""

    def test_from_row_complete(self):
        """Test from_row with complete data."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                url TEXT, content_hash TEXT, processed_at TEXT,
                ai_engine TEXT, description TEXT, tags TEXT,
                folder TEXT, title TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("https://example.com", "hash123", "2024-01-15T10:00:00",
             "claude", "Test description", "tag1, tag2, tag3",
             "Tech", "Test Title", "processed")
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        record = BookmarkRecord.from_row(row)

        assert record.url == "https://example.com"
        assert record.content_hash == "hash123"
        assert record.ai_engine == "claude"
        assert record.description == "Test description"
        assert record.tags == ["tag1", "tag2", "tag3"]
        assert record.folder == "Tech"
        assert record.title == "Test Title"
        assert record.status == "processed"

    def test_from_row_empty_tags(self):
        """Test from_row with empty tags."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                url TEXT, content_hash TEXT, processed_at TEXT,
                ai_engine TEXT, description TEXT, tags TEXT,
                folder TEXT, title TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("https://example.com", "hash123", "2024-01-15T10:00:00",
             "claude", "Test", "", "Tech", "Test", "processed")
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        record = BookmarkRecord.from_row(row)

        assert record.tags == []

    def test_from_row_null_tags(self):
        """Test from_row with NULL tags."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                url TEXT, content_hash TEXT, processed_at TEXT,
                ai_engine TEXT, description TEXT, tags TEXT,
                folder TEXT, title TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("https://example.com", "hash123", "2024-01-15T10:00:00",
             "claude", "Test", None, "Tech", "Test", "processed")
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        record = BookmarkRecord.from_row(row)

        assert record.tags == []

    def test_from_row_no_status_column(self):
        """Test from_row when status column is missing."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                url TEXT, content_hash TEXT, processed_at TEXT,
                ai_engine TEXT, description TEXT, tags TEXT,
                folder TEXT, title TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("https://example.com", "hash123", "2024-01-15T10:00:00",
             "claude", "Test", "tag1", "Tech", "Test")
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        record = BookmarkRecord.from_row(row)

        assert record.status == "processed"

    def test_from_row_null_status(self):
        """Test from_row with NULL status."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                url TEXT, content_hash TEXT, processed_at TEXT,
                ai_engine TEXT, description TEXT, tags TEXT,
                folder TEXT, title TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("https://example.com", "hash123", "2024-01-15T10:00:00",
             "claude", "Test", "tag1", "Tech", "Test", None)
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        record = BookmarkRecord.from_row(row)

        assert record.status == "processed"

    def test_from_row_null_processed_at(self):
        """Test from_row with NULL processed_at defaults to now."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                url TEXT, content_hash TEXT, processed_at TEXT,
                ai_engine TEXT, description TEXT, tags TEXT,
                folder TEXT, title TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("https://example.com", "hash123", None,
             "claude", "Test", "tag1", "Tech", "Test", "processed")
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        before = datetime.now()
        record = BookmarkRecord.from_row(row)
        after = datetime.now()

        assert before <= record.processed_at <= after

    def test_from_row_whitespace_tags(self):
        """Test from_row strips whitespace from tags."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE test (
                url TEXT, content_hash TEXT, processed_at TEXT,
                ai_engine TEXT, description TEXT, tags TEXT,
                folder TEXT, title TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("https://example.com", "hash123", "2024-01-15T10:00:00",
             "claude", "Test", "  tag1  ,  ,  tag2  ", "Tech", "Test", "processed")
        )
        row = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        record = BookmarkRecord.from_row(row)

        assert record.tags == ["tag1", "tag2"]

    def test_to_bookmark(self):
        """Test conversion to Bookmark object."""
        record = BookmarkRecord(
            url="https://example.com",
            content_hash="hash123",
            processed_at=datetime.now(),
            ai_engine="claude",
            description="Test description",
            tags=["tag1", "tag2"],
            folder="Tech",
            title="Test Title",
            status="processed",
        )

        bookmark = record.to_bookmark()

        assert bookmark.url == "https://example.com"
        assert bookmark.title == "Test Title"
        assert bookmark.note == "Test description"
        assert bookmark.folder == "Tech"
        assert bookmark.tags == ["tag1", "tag2"]
        assert bookmark.enhanced_description == "Test description"

    def test_to_bookmark_none_values(self):
        """Test conversion with None values."""
        record = BookmarkRecord(
            url="https://example.com",
            content_hash="hash123",
            processed_at=datetime.now(),
            ai_engine=None,
            description=None,
            tags=[],
            folder=None,
            title=None,
            status="processed",
        )

        bookmark = record.to_bookmark()

        assert bookmark.title == ""
        assert bookmark.note == ""
        assert bookmark.folder == ""
        assert bookmark.enhanced_description == ""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        record = BookmarkRecord(
            url="https://example.com",
            content_hash="hash123",
            processed_at=now,
            ai_engine="claude",
            description="Test",
            tags=["tag1"],
            folder="Tech",
            title="Test",
            status="processed",
        )

        d = record.to_dict()

        assert d["url"] == "https://example.com"
        assert d["processed_at"] == "2024-01-15T10:00:00"
        assert d["tags"] == ["tag1"]


# ============ RunComparison Dataclass Tests ============


class TestRunComparisonDataclass:
    """Tests for RunComparison dataclass."""

    def test_total_changes(self):
        """Test total_changes property."""
        run1 = ProcessingRun(
            id=1, started_at=datetime.now(), completed_at=datetime.now(),
            source="test", total_processed=0, total_succeeded=0,
            total_failed=0, config_hash=None
        )
        run2 = ProcessingRun(
            id=2, started_at=datetime.now(), completed_at=datetime.now(),
            source="test", total_processed=0, total_succeeded=0,
            total_failed=0, config_hash=None
        )

        comparison = RunComparison(
            run1=run1,
            run2=run2,
            new_bookmarks=["a", "b", "c"],
            removed_bookmarks=["d", "e"],
            changed_bookmarks=["f"],
            unchanged_bookmarks=["g", "h", "i", "j"],
        )

        assert comparison.total_changes == 6  # 3 new + 2 removed + 1 changed

    def test_total_changes_empty(self):
        """Test total_changes with empty lists."""
        run1 = ProcessingRun(
            id=1, started_at=datetime.now(), completed_at=datetime.now(),
            source="test", total_processed=0, total_succeeded=0,
            total_failed=0, config_hash=None
        )
        run2 = ProcessingRun(
            id=2, started_at=datetime.now(), completed_at=datetime.now(),
            source="test", total_processed=0, total_succeeded=0,
            total_failed=0, config_hash=None
        )

        comparison = RunComparison(
            run1=run1,
            run2=run2,
            new_bookmarks=[],
            removed_bookmarks=[],
            changed_bookmarks=[],
            unchanged_bookmarks=["a", "b"],
        )

        assert comparison.total_changes == 0

    def test_to_dict_limits_results(self):
        """Test to_dict limits large result lists to 100."""
        run1 = ProcessingRun(
            id=1, started_at=datetime.now(), completed_at=datetime.now(),
            source="test", total_processed=0, total_succeeded=0,
            total_failed=0, config_hash=None
        )
        run2 = ProcessingRun(
            id=2, started_at=datetime.now(), completed_at=datetime.now(),
            source="test", total_processed=0, total_succeeded=0,
            total_failed=0, config_hash=None
        )

        # Create lists larger than 100
        large_list = [f"url_{i}" for i in range(150)]

        comparison = RunComparison(
            run1=run1,
            run2=run2,
            new_bookmarks=large_list.copy(),
            removed_bookmarks=large_list.copy(),
            changed_bookmarks=large_list.copy(),
            unchanged_bookmarks=[],
        )

        d = comparison.to_dict()

        # Counts should reflect actual totals
        assert d["new_bookmarks_count"] == 150
        assert d["removed_bookmarks_count"] == 150
        assert d["changed_bookmarks_count"] == 150
        # But lists should be limited to 100
        assert len(d["new_bookmarks"]) == 100
        assert len(d["removed_bookmarks"]) == 100
        assert len(d["changed_bookmarks"]) == 100


# ============ BookmarkDatabase Initialization Tests ============


class TestBookmarkDatabaseInit:
    """Tests for BookmarkDatabase initialization."""

    def test_init_creates_file(self, temp_db_path: Path):
        """Test database creates file."""
        db = BookmarkDatabase(temp_db_path)
        assert temp_db_path.exists()
        db.close()

    def test_init_with_string_path(self, tmp_path: Path):
        """Test initialization with string path."""
        path_str = str(tmp_path / "string_path.db")
        db = BookmarkDatabase(path_str)
        assert db.db_path == Path(path_str)
        db.close()

    def test_init_creates_tables(self, db: BookmarkDatabase):
        """Test that initialization creates all required tables."""
        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        assert "processed_bookmarks" in tables
        assert "processing_runs" in tables
        assert "bookmark_history" in tables

    def test_init_creates_indexes(self, db: BookmarkDatabase):
        """Test that initialization creates indexes."""
        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = {row[0] for row in cursor.fetchall()}

        assert "idx_processed_at" in indexes
        assert "idx_content_hash" in indexes
        assert "idx_status" in indexes

    def test_init_creates_views(self, db: BookmarkDatabase):
        """Test that initialization creates views."""
        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            )
            views = {row[0] for row in cursor.fetchall()}

        assert "failed_bookmarks" in views
        assert "recent_bookmarks" in views
        assert "run_summary" in views

    def test_init_with_fts_enabled(self, db: BookmarkDatabase):
        """Test FTS tables are created when enabled."""
        assert db.enable_fts is True

        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%fts%'"
            )
            fts_tables = {row[0] for row in cursor.fetchall()}

        assert "bookmark_fts" in fts_tables

    def test_init_without_fts(self, db_no_fts: BookmarkDatabase):
        """Test FTS tables are not created when disabled."""
        assert db_no_fts.enable_fts is False

        with db_no_fts._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%fts%'"
            )
            fts_tables = list(cursor.fetchall())

        assert len(fts_tables) == 0

    def test_repr(self, db: BookmarkDatabase):
        """Test string representation."""
        repr_str = repr(db)
        assert "BookmarkDatabase" in repr_str


# ============ Connection Management Tests ============


class TestConnectionManagement:
    """Tests for database connection management."""

    def test_get_connection_context_manager(self, db: BookmarkDatabase):
        """Test connection context manager."""
        with db._get_connection() as conn:
            assert conn is not None
            cursor = conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1

    def test_get_connection_row_factory(self, db: BookmarkDatabase):
        """Test connection has Row factory set."""
        with db._get_connection() as conn:
            cursor = conn.execute("SELECT 1 as value")
            row = cursor.fetchone()
            assert row["value"] == 1

    def test_close_method(self, temp_db_path: Path):
        """Test close method."""
        db = BookmarkDatabase(temp_db_path)
        db.close()
        assert db._conn is None

    def test_close_when_no_connection(self, db: BookmarkDatabase):
        """Test close when no persistent connection exists."""
        db._conn = None
        db.close()  # Should not raise
        assert db._conn is None


# ============ Compute Hash Tests ============


class TestComputeHash:
    """Tests for _compute_hash method."""

    def test_compute_hash_basic(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test basic hash computation."""
        hash1 = db._compute_hash(sample_bookmark)
        assert hash1 is not None
        assert len(hash1) == 32  # MD5 hex digest length

    def test_compute_hash_deterministic(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test hash is deterministic."""
        hash1 = db._compute_hash(sample_bookmark)
        hash2 = db._compute_hash(sample_bookmark)
        assert hash1 == hash2

    def test_compute_hash_different_for_different_content(self, db: BookmarkDatabase):
        """Test different content produces different hashes."""
        bookmark1 = Bookmark(url="https://example1.com", title="Title 1")
        bookmark2 = Bookmark(url="https://example2.com", title="Title 2")

        hash1 = db._compute_hash(bookmark1)
        hash2 = db._compute_hash(bookmark2)

        assert hash1 != hash2

    def test_compute_hash_empty_fields(self, db: BookmarkDatabase):
        """Test hash computation with empty fields."""
        bookmark = Bookmark(url="", title="", note="", excerpt="", folder="", tags=[])
        hash_val = db._compute_hash(bookmark)
        assert hash_val is not None
        assert len(hash_val) == 32

    def test_compute_hash_none_fields(self, db: BookmarkDatabase):
        """Test hash computation handles None-like values."""
        bookmark = Bookmark()
        hash_val = db._compute_hash(bookmark)
        assert hash_val is not None

    def test_compute_hash_tags_order_independent(self, db: BookmarkDatabase):
        """Test hash is same regardless of tag order."""
        bookmark1 = Bookmark(url="https://example.com", tags=["a", "b", "c"])
        bookmark2 = Bookmark(url="https://example.com", tags=["c", "a", "b"])

        hash1 = db._compute_hash(bookmark1)
        hash2 = db._compute_hash(bookmark2)

        assert hash1 == hash2


# ============ CRUD Operations Tests ============


class TestMarkProcessed:
    """Tests for mark_processed method."""

    def test_mark_processed_basic(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test basic mark_processed."""
        db.mark_processed(sample_bookmark, ai_engine="claude")

        records = db.query_by_status("processed")
        assert len(records) == 1
        assert records[0].url == sample_bookmark.url

    def test_mark_processed_with_custom_hash(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test mark_processed with pre-computed hash."""
        db.mark_processed(sample_bookmark, content_hash="custom_hash_123")

        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT content_hash FROM processed_bookmarks WHERE url = ?",
                (sample_bookmark.url,)
            )
            row = cursor.fetchone()

        assert row["content_hash"] == "custom_hash_123"

    def test_mark_processed_with_run_id(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test mark_processed creates history when run_id provided."""
        run_id = db.start_processing_run("test")
        db.mark_processed(sample_bookmark, run_id=run_id)

        history = db.get_processing_history(sample_bookmark.url)
        assert len(history) == 1
        assert history[0]["run_id"] == run_id
        assert history[0]["change_type"] == "processed"

    def test_mark_processed_updates_existing(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test mark_processed updates existing record."""
        db.mark_processed(sample_bookmark, ai_engine="openai")
        db.mark_processed(sample_bookmark, ai_engine="claude")

        with db._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM processed_bookmarks")
            count = cursor.fetchone()[0]
            cursor = conn.execute(
                "SELECT ai_engine FROM processed_bookmarks WHERE url = ?",
                (sample_bookmark.url,)
            )
            engine = cursor.fetchone()["ai_engine"]

        assert count == 1
        assert engine == "claude"

    def test_mark_processed_stores_tags(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test tags are stored correctly."""
        db.mark_processed(sample_bookmark)

        records = db.query_by_status("processed")
        # optimized_tags should be used if available
        assert "testing" in records[0].tags or "automation" in records[0].tags

    def test_mark_processed_status_custom(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test mark_processed with custom status."""
        db.mark_processed(sample_bookmark, status="pending")

        records = db.query_by_status("pending")
        assert len(records) == 1


class TestMarkFailed:
    """Tests for mark_failed method."""

    def test_mark_failed_basic(self, db: BookmarkDatabase):
        """Test basic mark_failed."""
        db.mark_failed("https://failed.com", "Connection timeout")

        failed = db.query_failed()
        assert len(failed) == 1
        assert failed[0].url == "https://failed.com"
        assert "timeout" in failed[0].description

    def test_mark_failed_with_run_id(self, db: BookmarkDatabase):
        """Test mark_failed creates history."""
        run_id = db.start_processing_run("test")
        db.mark_failed("https://failed.com", "Error", run_id=run_id)

        history = db.get_processing_history("https://failed.com")
        assert len(history) == 1
        assert history[0]["change_type"] == "failed"

    def test_mark_failed_updates_existing(self, db: BookmarkDatabase):
        """Test mark_failed updates existing record."""
        db.mark_failed("https://failed.com", "Error 1")
        db.mark_failed("https://failed.com", "Error 2")

        failed = db.query_failed()
        assert len(failed) == 1
        assert "Error 2" in failed[0].description


# ============ Query Methods Tests ============


class TestQueryMethods:
    """Tests for query methods."""

    def test_query_failed(self, populated_db: BookmarkDatabase):
        """Test query_failed returns only failed bookmarks."""
        failed = populated_db.query_failed()
        assert len(failed) == 1
        assert failed[0].status == "failed"

    def test_query_by_date_with_both_bounds(self, populated_db: BookmarkDatabase):
        """Test query_by_date with start and end."""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        results = populated_db.query_by_date(start=start, end=end)
        assert len(results) == 4

    def test_query_by_date_start_only(self, populated_db: BookmarkDatabase):
        """Test query_by_date with start only."""
        start = datetime.now() - timedelta(hours=1)

        results = populated_db.query_by_date(start=start)
        assert len(results) == 4

    def test_query_by_date_end_only(self, populated_db: BookmarkDatabase):
        """Test query_by_date with end only."""
        end = datetime.now() + timedelta(hours=1)

        results = populated_db.query_by_date(end=end)
        assert len(results) == 4

    def test_query_by_date_no_bounds(self, populated_db: BookmarkDatabase):
        """Test query_by_date with no bounds returns all."""
        results = populated_db.query_by_date()
        assert len(results) == 4

    def test_query_by_date_no_results(self, populated_db: BookmarkDatabase):
        """Test query_by_date with future dates returns empty."""
        start = datetime.now() + timedelta(days=1)

        results = populated_db.query_by_date(start=start)
        assert len(results) == 0

    def test_query_by_status(self, populated_db: BookmarkDatabase):
        """Test query_by_status."""
        processed = populated_db.query_by_status("processed")
        failed = populated_db.query_by_status("failed")

        assert len(processed) == 3
        assert len(failed) == 1

    def test_query_by_folder_exact(self, populated_db: BookmarkDatabase):
        """Test query_by_folder with exact match."""
        results = populated_db.query_by_folder("Programming/Python", exact=True)
        assert len(results) == 1
        assert results[0].folder == "Programming/Python"

    def test_query_by_folder_prefix(self, populated_db: BookmarkDatabase):
        """Test query_by_folder with prefix match."""
        results = populated_db.query_by_folder("Programming", exact=False)
        assert len(results) >= 2

    def test_query_by_tag(self, populated_db: BookmarkDatabase):
        """Test query_by_tag."""
        # Search for a tag that should exist
        results = populated_db.query_by_tag("python")
        assert len(results) >= 1

    def test_query_by_tag_no_match(self, populated_db: BookmarkDatabase):
        """Test query_by_tag with no matches."""
        results = populated_db.query_by_tag("nonexistent_tag_xyz")
        assert len(results) == 0


# ============ Full-Text Search Tests ============


class TestFullTextSearch:
    """Tests for FTS5 full-text search."""

    def test_search_content_basic(self, populated_db: BookmarkDatabase):
        """Test basic FTS search."""
        results = populated_db.search_content("Python")
        assert len(results) >= 1

    def test_search_content_with_limit(self, populated_db: BookmarkDatabase):
        """Test FTS search with limit."""
        results = populated_db.search_content("tutorial", limit=1)
        assert len(results) <= 1

    def test_search_content_no_results(self, populated_db: BookmarkDatabase):
        """Test FTS search with no matches."""
        results = populated_db.search_content("xyznonexistent123")
        assert len(results) == 0

    def test_search_content_fts_disabled_uses_fallback(self, db_no_fts: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test search falls back to LIKE when FTS disabled."""
        db_no_fts.mark_processed(sample_bookmark)

        results = db_no_fts.search_content("example")
        assert len(results) >= 1

    def test_fallback_search(self, db_no_fts: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test _fallback_search directly."""
        db_no_fts.mark_processed(sample_bookmark)

        results = db_no_fts._fallback_search("Test", limit=10)
        assert len(results) >= 1

    def test_fallback_search_with_limit(self, db_no_fts: BookmarkDatabase, sample_bookmarks: list):
        """Test fallback search respects limit."""
        for bookmark in sample_bookmarks:
            db_no_fts.mark_processed(bookmark)

        results = db_no_fts._fallback_search("example", limit=2)
        assert len(results) <= 2


# ============ Processing History Tests ============


class TestProcessingHistory:
    """Tests for processing history."""

    def test_get_processing_history(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test getting processing history."""
        run_id = db.start_processing_run("test")
        db.mark_processed(sample_bookmark, run_id=run_id)

        history = db.get_processing_history(sample_bookmark.url)

        assert len(history) == 1
        assert history[0]["url"] == sample_bookmark.url
        assert history[0]["source"] == "test"

    def test_get_processing_history_multiple_runs(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test history accumulates across runs."""
        # First run
        run_id1 = db.start_processing_run("run1")
        db.mark_processed(sample_bookmark, run_id=run_id1)
        db.complete_processing_run(run_id1)

        # Second run
        run_id2 = db.start_processing_run("run2")
        sample_bookmark.enhanced_description = "Updated description"
        db.mark_processed(sample_bookmark, run_id=run_id2)

        history = db.get_processing_history(sample_bookmark.url)
        assert len(history) == 2

    def test_get_processing_history_no_history(self, db: BookmarkDatabase):
        """Test getting history for URL with no records."""
        history = db.get_processing_history("https://nonexistent.com")
        assert len(history) == 0


# ============ Run Comparison Tests ============


class TestRunComparison:
    """Tests for run comparison."""

    def test_compare_runs_basic(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test basic run comparison."""
        # First run: bookmarks 0 and 1
        run_id1 = db.start_processing_run("run1")
        db.mark_processed(sample_bookmarks[0], run_id=run_id1)
        db.mark_processed(sample_bookmarks[1], run_id=run_id1)
        db.complete_processing_run(run_id1, total_processed=2)

        # Second run: bookmarks 0 and 2 (bookmark 1 removed, 2 added)
        run_id2 = db.start_processing_run("run2")
        db.mark_processed(sample_bookmarks[0], run_id=run_id2)
        db.mark_processed(sample_bookmarks[2], run_id=run_id2)
        db.complete_processing_run(run_id2, total_processed=2)

        comparison = db.compare_runs(run_id1, run_id2)

        assert comparison is not None
        assert sample_bookmarks[2].url in comparison.new_bookmarks
        assert sample_bookmarks[1].url in comparison.removed_bookmarks

    def test_compare_runs_with_changes(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test run comparison detects content changes."""
        # First run
        run_id1 = db.start_processing_run("run1")
        db.mark_processed(sample_bookmarks[0], run_id=run_id1)
        db.complete_processing_run(run_id1)

        # Second run with modified bookmark - modify title which affects the hash
        run_id2 = db.start_processing_run("run2")
        sample_bookmarks[0].title = "Modified Title in Run 2"
        db.mark_processed(sample_bookmarks[0], run_id=run_id2)
        db.complete_processing_run(run_id2)

        comparison = db.compare_runs(run_id1, run_id2)

        assert comparison is not None
        assert sample_bookmarks[0].url in comparison.changed_bookmarks

    def test_compare_runs_unchanged(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test run comparison detects unchanged bookmarks."""
        # First run
        run_id1 = db.start_processing_run("run1")
        db.mark_processed(sample_bookmarks[0], run_id=run_id1)
        db.complete_processing_run(run_id1)

        # Second run with same bookmark (unchanged)
        run_id2 = db.start_processing_run("run2")
        db.mark_processed(sample_bookmarks[0], run_id=run_id2)
        db.complete_processing_run(run_id2)

        comparison = db.compare_runs(run_id1, run_id2)

        assert comparison is not None
        assert sample_bookmarks[0].url in comparison.unchanged_bookmarks

    def test_compare_runs_nonexistent(self, db: BookmarkDatabase):
        """Test comparing non-existent runs returns None."""
        comparison = db.compare_runs(9999, 10000)
        assert comparison is None

    def test_compare_runs_one_nonexistent(self, db: BookmarkDatabase):
        """Test comparing when one run doesn't exist."""
        run_id = db.start_processing_run("test")
        db.complete_processing_run(run_id)

        comparison = db.compare_runs(run_id, 9999)
        assert comparison is None


# ============ Run Management Tests ============


class TestRunManagement:
    """Tests for run management."""

    def test_start_processing_run(self, db: BookmarkDatabase):
        """Test starting a processing run."""
        run_id = db.start_processing_run("test_source", config_hash="config123")

        assert run_id is not None
        assert run_id > 0
        assert db._current_run_id == run_id

    def test_complete_processing_run(self, db: BookmarkDatabase):
        """Test completing a processing run."""
        run_id = db.start_processing_run("test")
        db.complete_processing_run(
            run_id=run_id,
            total_processed=100,
            total_succeeded=95,
            total_failed=5
        )

        runs = db.get_run_history(limit=1)
        assert len(runs) == 1
        assert runs[0].total_processed == 100
        assert runs[0].total_succeeded == 95
        assert runs[0].total_failed == 5
        assert runs[0].completed_at is not None

    def test_complete_processing_run_uses_current(self, db: BookmarkDatabase):
        """Test complete_processing_run uses current run_id if none provided."""
        run_id = db.start_processing_run("test")
        db.complete_processing_run(total_processed=50)

        runs = db.get_run_history(limit=1)
        assert runs[0].total_processed == 50

    def test_complete_processing_run_no_run_id(self, db: BookmarkDatabase):
        """Test complete_processing_run does nothing if no run_id."""
        db._current_run_id = None
        db.complete_processing_run(total_processed=50)  # Should not raise

        runs = db.get_run_history()
        assert len(runs) == 0

    def test_get_run_history(self, db: BookmarkDatabase):
        """Test getting run history."""
        for i in range(5):
            run_id = db.start_processing_run(f"source_{i}")
            db.complete_processing_run(run_id, total_processed=i * 10)

        runs = db.get_run_history(limit=3)
        assert len(runs) == 3

    def test_get_run_history_by_source(self, db: BookmarkDatabase):
        """Test filtering run history by source."""
        db.start_processing_run("source_a")
        db.start_processing_run("source_b")
        db.start_processing_run("source_a")

        runs = db.get_run_history(source="source_a")
        assert len(runs) == 2
        assert all(r.source == "source_a" for r in runs)


# ============ State Tracking Tests ============


class TestStateTracking:
    """Tests for state tracking methods."""

    def test_needs_processing_new_bookmark(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test needs_processing returns True for new bookmark."""
        assert db.needs_processing(sample_bookmark) is True

    def test_needs_processing_already_processed(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test needs_processing returns False for processed bookmark."""
        db.mark_processed(sample_bookmark)
        assert db.needs_processing(sample_bookmark) is False

    def test_needs_processing_content_changed(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test needs_processing returns True for changed bookmark."""
        db.mark_processed(sample_bookmark)

        sample_bookmark.title = "Modified Title"
        assert db.needs_processing(sample_bookmark) is True

    def test_needs_processing_failed_bookmark(self, db: BookmarkDatabase):
        """Test needs_processing returns True for previously failed bookmark."""
        db.mark_failed("https://test.com", "Error")

        bookmark = Bookmark(url="https://test.com", title="Test")
        assert db.needs_processing(bookmark) is True

    def test_get_unprocessed(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test get_unprocessed filters correctly."""
        # Process first two
        db.mark_processed(sample_bookmarks[0])
        db.mark_processed(sample_bookmarks[1])

        unprocessed = db.get_unprocessed(sample_bookmarks)

        assert len(unprocessed) == 2
        urls = [b.url for b in unprocessed]
        assert sample_bookmarks[0].url not in urls
        assert sample_bookmarks[1].url not in urls
        assert sample_bookmarks[2].url in urls
        assert sample_bookmarks[3].url in urls

    def test_get_unprocessed_empty_list(self, db: BookmarkDatabase):
        """Test get_unprocessed with empty list."""
        result = db.get_unprocessed([])
        assert result == []


# ============ Statistics Tests ============


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_statistics(self, populated_db: BookmarkDatabase):
        """Test getting database statistics."""
        stats = populated_db.get_statistics()

        assert stats["total_bookmarks"] == 4
        assert stats["total_runs"] >= 1
        assert "by_status" in stats
        assert "processed" in stats["by_status"]
        assert stats["by_status"]["processed"] == 3
        assert stats["by_status"]["failed"] == 1
        assert "unique_folders" in stats

    def test_get_statistics_empty_database(self, db: BookmarkDatabase):
        """Test statistics on empty database."""
        stats = db.get_statistics()

        assert stats["total_bookmarks"] == 0
        assert stats["total_runs"] == 0
        assert stats.get("by_status", {}) == {}

    def test_get_statistics_avg_tags(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test average tags calculation."""
        for bookmark in sample_bookmarks:
            db.mark_processed(bookmark)

        stats = db.get_statistics()

        assert "avg_tags_per_bookmark" in stats
        assert stats["avg_tags_per_bookmark"] >= 0


# ============ Utility Methods Tests ============


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_vacuum(self, populated_db: BookmarkDatabase):
        """Test vacuum operation."""
        # Should not raise
        populated_db.vacuum()

    def test_vacuum_empty_db(self, db: BookmarkDatabase):
        """Test vacuum on empty database."""
        db.vacuum()  # Should not raise


# ============ Error Handling Tests ============


class TestErrorHandling:
    """Tests for error handling."""

    def test_query_failed_handles_error(self, db: BookmarkDatabase):
        """Test query_failed handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.query_failed()

        assert result == []

    def test_query_by_date_handles_error(self, db: BookmarkDatabase):
        """Test query_by_date handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.query_by_date()

        assert result == []

    def test_query_by_status_handles_error(self, db: BookmarkDatabase):
        """Test query_by_status handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.query_by_status("processed")

        assert result == []

    def test_query_by_folder_handles_error(self, db: BookmarkDatabase):
        """Test query_by_folder handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.query_by_folder("Tech")

        assert result == []

    def test_query_by_tag_handles_error(self, db: BookmarkDatabase):
        """Test query_by_tag handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.query_by_tag("python")

        assert result == []

    def test_get_processing_history_handles_error(self, db: BookmarkDatabase):
        """Test get_processing_history handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.get_processing_history("https://test.com")

        assert result == []

    def test_compare_runs_handles_error(self, db: BookmarkDatabase):
        """Test compare_runs handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.compare_runs(1, 2)

        assert result is None

    def test_get_run_history_handles_error(self, db: BookmarkDatabase):
        """Test get_run_history handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.get_run_history()

        assert result == []

    def test_needs_processing_handles_error(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test needs_processing returns True on error."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.needs_processing(sample_bookmark)

        assert result is True

    def test_mark_processed_raises_on_error(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test mark_processed raises exception on database error."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            with pytest.raises(sqlite3.Error):
                db.mark_processed(sample_bookmark)

    def test_mark_failed_handles_error(self, db: BookmarkDatabase):
        """Test mark_failed logs error but doesn't raise."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            # Should not raise
            db.mark_failed("https://test.com", "Error")

    def test_start_processing_run_raises_on_error(self, db: BookmarkDatabase):
        """Test start_processing_run raises exception on database error."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            with pytest.raises(sqlite3.Error):
                db.start_processing_run("test")

    def test_complete_processing_run_handles_error(self, db: BookmarkDatabase):
        """Test complete_processing_run logs error but doesn't raise."""
        run_id = db.start_processing_run("test")

        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            # Should not raise
            db.complete_processing_run(run_id, total_processed=50)

    def test_get_statistics_handles_error(self, db: BookmarkDatabase):
        """Test get_statistics handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db.get_statistics()

        assert result == {}

    def test_vacuum_handles_error(self, db: BookmarkDatabase):
        """Test vacuum handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            # Should not raise
            db.vacuum()

    def test_fallback_search_handles_error(self, db: BookmarkDatabase):
        """Test _fallback_search handles database errors gracefully."""
        with patch.object(db, '_get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Test error")
            result = db._fallback_search("test", limit=10)

        assert result == []


# ============ FTS Trigger Tests ============


class TestFTSTriggers:
    """Tests for FTS synchronization triggers."""

    def test_fts_insert_trigger(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test FTS is updated on insert."""
        db.mark_processed(sample_bookmark)

        # Search should find the bookmark via FTS
        results = db.search_content("Test Bookmark")
        assert len(results) >= 1

    def test_fts_update_trigger(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test FTS is updated on update."""
        db.mark_processed(sample_bookmark)

        # Update the bookmark
        sample_bookmark.title = "Unique Updated Title XYZ"
        db.mark_processed(sample_bookmark)

        # Search should find the updated content
        results = db.search_content("Unique Updated Title XYZ")
        assert len(results) >= 1

    def test_fts_search_multiple_fields(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test FTS searches across multiple fields."""
        db.mark_processed(sample_bookmark)

        # Search by URL
        results = db.search_content("example.com")
        assert len(results) >= 1

        # Search by description
        results = db.search_content("Enhanced test description")
        assert len(results) >= 1


# ============ Database Schema Tests ============


class TestDatabaseSchema:
    """Tests for database schema integrity."""

    def test_schema_processed_bookmarks_columns(self, db: BookmarkDatabase):
        """Test processed_bookmarks table has correct columns."""
        with db._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(processed_bookmarks)")
            columns = {row[1] for row in cursor.fetchall()}

        expected = {"url", "content_hash", "processed_at", "ai_engine",
                   "description", "tags", "folder", "title", "status"}
        assert expected.issubset(columns)

    def test_schema_processing_runs_columns(self, db: BookmarkDatabase):
        """Test processing_runs table has correct columns."""
        with db._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(processing_runs)")
            columns = {row[1] for row in cursor.fetchall()}

        expected = {"id", "started_at", "completed_at", "source",
                   "total_processed", "total_succeeded", "total_failed", "config_hash"}
        assert expected.issubset(columns)

    def test_schema_bookmark_history_columns(self, db: BookmarkDatabase):
        """Test bookmark_history table has correct columns."""
        with db._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(bookmark_history)")
            columns = {row[1] for row in cursor.fetchall()}

        expected = {"id", "url", "run_id", "content_hash", "description",
                   "tags", "folder", "title", "changed_at", "change_type"}
        assert expected.issubset(columns)


# ============ Integration Tests ============


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_processing_workflow(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test complete processing workflow."""
        # Start run
        run_id = db.start_processing_run("integration_test", config_hash="test123")

        # Process bookmarks
        succeeded = 0
        failed = 0
        for i, bookmark in enumerate(sample_bookmarks):
            if i % 3 == 0:
                db.mark_failed(bookmark.url, "Test failure", run_id=run_id)
                failed += 1
            else:
                db.mark_processed(bookmark, ai_engine="test", run_id=run_id)
                succeeded += 1

        # Complete run
        db.complete_processing_run(
            run_id=run_id,
            total_processed=len(sample_bookmarks),
            total_succeeded=succeeded,
            total_failed=failed
        )

        # Verify results
        stats = db.get_statistics()
        assert stats["total_bookmarks"] == len(sample_bookmarks)

        runs = db.get_run_history()
        assert len(runs) == 1
        assert runs[0].total_processed == len(sample_bookmarks)

    def test_multiple_runs_comparison(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test comparing multiple processing runs."""
        # First run: process all bookmarks
        run_id1 = db.start_processing_run("run1")
        for bookmark in sample_bookmarks:
            db.mark_processed(bookmark, run_id=run_id1)
        db.complete_processing_run(run_id1, total_processed=len(sample_bookmarks))

        # Second run: modify one, remove one, add none
        run_id2 = db.start_processing_run("run2")
        for i, bookmark in enumerate(sample_bookmarks[:-1]):  # Skip last
            if i == 0:
                bookmark.enhanced_description = "Modified in run 2"
            db.mark_processed(bookmark, run_id=run_id2)
        db.complete_processing_run(run_id2, total_processed=len(sample_bookmarks) - 1)

        # Compare runs
        comparison = db.compare_runs(run_id1, run_id2)

        assert comparison is not None
        assert len(comparison.removed_bookmarks) == 1
        assert sample_bookmarks[-1].url in comparison.removed_bookmarks

    def test_search_after_processing(self, db: BookmarkDatabase, sample_bookmarks: list):
        """Test search functionality after processing."""
        for bookmark in sample_bookmarks:
            db.mark_processed(bookmark)

        # Search by various criteria
        python_results = db.search_content("Python")
        assert len(python_results) >= 1

        ml_results = db.search_content("machine learning")
        assert len(ml_results) >= 1

    def test_state_tracking_across_runs(self, db: BookmarkDatabase, sample_bookmark: Bookmark):
        """Test state tracking persists across runs."""
        # First run
        run_id1 = db.start_processing_run("run1")
        db.mark_processed(sample_bookmark, run_id=run_id1)
        db.complete_processing_run(run_id1)

        # Verify bookmark is marked as processed
        assert db.needs_processing(sample_bookmark) is False

        # Start new run - same bookmark shouldn't need processing
        run_id2 = db.start_processing_run("run2")
        assert db.needs_processing(sample_bookmark) is False

        # Modify bookmark - should now need processing
        sample_bookmark.title = "Modified Title"
        assert db.needs_processing(sample_bookmark) is True
