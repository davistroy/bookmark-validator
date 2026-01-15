"""
Tests for Database-Backed State (Phase 8.3).

Tests cover:
- BookmarkDatabase: Full database operations
- Query methods: by date, status, folder, tag
- Full-text search
- Processing history and run comparison
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

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
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_bookmarks.db"


@pytest.fixture
def db(temp_db_path):
    """Create a BookmarkDatabase instance."""
    return BookmarkDatabase(temp_db_path, enable_fts=True)


@pytest.fixture
def sample_bookmarks():
    """Create sample Bookmark objects."""
    return [
        Bookmark(
            id="1",
            url="https://example.com/1",
            title="Test Site 1",
            note="Note 1",
            folder="Tech",
            tags=["test", "example"],
            enhanced_description="Enhanced description 1"
        ),
        Bookmark(
            id="2",
            url="https://example.com/2",
            title="Test Site 2",
            note="Note 2",
            folder="Tech/AI",
            tags=["ai", "ml"],
            enhanced_description="Enhanced description 2"
        ),
        Bookmark(
            id="3",
            url="https://example.com/3",
            title="Test Site 3",
            folder="Science",
            tags=["science"],
            enhanced_description="Enhanced description 3"
        ),
    ]


@pytest.fixture
def populated_db(db, sample_bookmarks):
    """Create a database with sample data."""
    run_id = db.start_processing_run("test_source")

    for bookmark in sample_bookmarks:
        db.mark_processed(bookmark, ai_engine="local", run_id=run_id)

    db.complete_processing_run(
        run_id=run_id,
        total_processed=3,
        total_succeeded=3,
        total_failed=0
    )

    return db


# ============ BookmarkDatabase Tests ============


class TestBookmarkDatabase:
    """Tests for BookmarkDatabase."""

    def test_init(self, temp_db_path):
        """Test database initialization."""
        db = BookmarkDatabase(temp_db_path)
        assert db.db_path == temp_db_path
        assert temp_db_path.exists()

    def test_init_creates_schema(self, temp_db_path):
        """Test that initialization creates schema."""
        db = BookmarkDatabase(temp_db_path)

        # Verify tables exist
        import sqlite3
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "processed_bookmarks" in tables
        assert "processing_runs" in tables
        assert "bookmark_history" in tables

    def test_init_without_fts(self, temp_db_path):
        """Test initialization without FTS."""
        db = BookmarkDatabase(temp_db_path, enable_fts=False)
        assert db.enable_fts is False


# ============ Mark Processed Tests ============


class TestMarkProcessed:
    """Tests for mark_processed method."""

    def test_mark_processed_basic(self, db, sample_bookmarks):
        """Test basic mark_processed functionality."""
        db.mark_processed(sample_bookmarks[0], ai_engine="claude")

        records = db.query_by_status("processed")
        assert len(records) == 1
        assert records[0].url == "https://example.com/1"

    def test_mark_processed_with_run_id(self, db, sample_bookmarks):
        """Test mark_processed with run_id creates history."""
        run_id = db.start_processing_run("test")
        db.mark_processed(sample_bookmarks[0], run_id=run_id)

        history = db.get_processing_history(sample_bookmarks[0].url)
        assert len(history) == 1
        assert history[0]["run_id"] == run_id

    def test_mark_processed_updates_existing(self, db, sample_bookmarks):
        """Test that mark_processed updates existing records."""
        db.mark_processed(sample_bookmarks[0])

        # Update with different description
        sample_bookmarks[0].enhanced_description = "Updated description"
        db.mark_processed(sample_bookmarks[0])

        records = db.query_by_status("processed")
        assert len(records) == 1  # Still just one record
        assert "Updated" in records[0].description


class TestMarkFailed:
    """Tests for mark_failed method."""

    def test_mark_failed(self, db):
        """Test marking a URL as failed."""
        db.mark_failed("https://failed.com", "Connection timeout")

        failed = db.query_failed()
        assert len(failed) == 1
        assert failed[0].url == "https://failed.com"
        assert "timeout" in failed[0].description


# ============ Query Methods Tests ============


class TestQueryMethods:
    """Tests for query methods."""

    def test_query_failed(self, db, sample_bookmarks):
        """Test query_failed returns failed bookmarks."""
        # Add a failed bookmark
        db.mark_failed("https://failed.com", "Error")

        # Add a successful bookmark
        db.mark_processed(sample_bookmarks[0])

        failed = db.query_failed()
        assert len(failed) == 1
        assert failed[0].url == "https://failed.com"

    def test_query_by_date_range(self, populated_db):
        """Test query_by_date with range."""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        results = populated_db.query_by_date(start=start, end=end)
        assert len(results) == 3

    def test_query_by_date_start_only(self, populated_db):
        """Test query_by_date with start only."""
        start = datetime.now() - timedelta(hours=1)

        results = populated_db.query_by_date(start=start)
        assert len(results) == 3

    def test_query_by_date_end_only(self, populated_db):
        """Test query_by_date with end only."""
        end = datetime.now() + timedelta(hours=1)

        results = populated_db.query_by_date(end=end)
        assert len(results) == 3

    def test_query_by_date_no_results(self, populated_db):
        """Test query_by_date with no matching results."""
        start = datetime.now() + timedelta(days=1)

        results = populated_db.query_by_date(start=start)
        assert len(results) == 0

    def test_query_by_status_processed(self, populated_db):
        """Test query_by_status for processed bookmarks."""
        results = populated_db.query_by_status("processed")
        assert len(results) == 3

    def test_query_by_status_failed(self, populated_db):
        """Test query_by_status for failed bookmarks."""
        results = populated_db.query_by_status("failed")
        assert len(results) == 0

    def test_query_by_folder_exact(self, populated_db):
        """Test query_by_folder with exact match."""
        results = populated_db.query_by_folder("Tech", exact=True)
        assert len(results) == 1
        assert results[0].url == "https://example.com/1"

    def test_query_by_folder_prefix(self, populated_db):
        """Test query_by_folder with prefix match."""
        results = populated_db.query_by_folder("Tech", exact=False)
        assert len(results) == 2  # Tech and Tech/AI

    def test_query_by_tag(self, populated_db):
        """Test query_by_tag."""
        results = populated_db.query_by_tag("ai")
        assert len(results) == 1
        assert results[0].url == "https://example.com/2"


# ============ Full-Text Search Tests ============


class TestFullTextSearch:
    """Tests for full-text search."""

    def test_search_content_title(self, populated_db):
        """Test searching by title."""
        results = populated_db.search_content("Site 1")
        assert len(results) >= 1
        assert any(r.url == "https://example.com/1" for r in results)

    def test_search_content_description(self, populated_db):
        """Test searching by description."""
        results = populated_db.search_content("Enhanced")
        assert len(results) >= 1

    def test_search_content_no_results(self, populated_db):
        """Test search with no results."""
        results = populated_db.search_content("nonexistent_term_xyz")
        assert len(results) == 0

    def test_search_content_limit(self, populated_db):
        """Test search with limit."""
        results = populated_db.search_content("description", limit=1)
        assert len(results) <= 1

    def test_fallback_search_no_fts(self, temp_db_path, sample_bookmarks):
        """Test fallback search when FTS disabled."""
        db = BookmarkDatabase(temp_db_path, enable_fts=False)
        db.mark_processed(sample_bookmarks[0])

        # Should use LIKE-based search
        results = db.search_content("example")
        assert len(results) >= 1


# ============ Processing History Tests ============


class TestProcessingHistory:
    """Tests for processing history."""

    def test_get_processing_history(self, db, sample_bookmarks):
        """Test getting processing history for a URL."""
        run_id = db.start_processing_run("test")
        db.mark_processed(sample_bookmarks[0], run_id=run_id)

        history = db.get_processing_history(sample_bookmarks[0].url)

        assert len(history) == 1
        assert history[0]["url"] == sample_bookmarks[0].url
        assert history[0]["change_type"] == "processed"

    def test_get_processing_history_multiple_runs(self, db, sample_bookmarks):
        """Test history with multiple processing runs."""
        # First run
        run_id1 = db.start_processing_run("test1")
        db.mark_processed(sample_bookmarks[0], run_id=run_id1)
        db.complete_processing_run(run_id1)

        # Second run
        run_id2 = db.start_processing_run("test2")
        sample_bookmarks[0].enhanced_description = "Updated"
        db.mark_processed(sample_bookmarks[0], run_id=run_id2)

        history = db.get_processing_history(sample_bookmarks[0].url)
        assert len(history) == 2


# ============ Run Comparison Tests ============


class TestRunComparison:
    """Tests for run comparison."""

    def test_compare_runs(self, db, sample_bookmarks):
        """Test comparing two processing runs."""
        # First run with 2 bookmarks
        run_id1 = db.start_processing_run("test1")
        db.mark_processed(sample_bookmarks[0], run_id=run_id1)
        db.mark_processed(sample_bookmarks[1], run_id=run_id1)
        db.complete_processing_run(run_id1, total_processed=2)

        # Second run with different bookmarks (remove one, add one)
        run_id2 = db.start_processing_run("test2")
        db.mark_processed(sample_bookmarks[0], run_id=run_id2)
        db.mark_processed(sample_bookmarks[2], run_id=run_id2)
        db.complete_processing_run(run_id2, total_processed=2)

        comparison = db.compare_runs(run_id1, run_id2)

        assert comparison is not None
        assert len(comparison.new_bookmarks) == 1  # bookmark 3 is new
        assert len(comparison.removed_bookmarks) == 1  # bookmark 2 was removed
        assert sample_bookmarks[2].url in comparison.new_bookmarks
        assert sample_bookmarks[1].url in comparison.removed_bookmarks

    def test_compare_runs_nonexistent(self, db):
        """Test comparing non-existent runs."""
        comparison = db.compare_runs(999, 1000)
        assert comparison is None


# ============ Run Management Tests ============


class TestRunManagement:
    """Tests for run management."""

    def test_start_processing_run(self, db):
        """Test starting a processing run."""
        run_id = db.start_processing_run("test_source", config_hash="abc123")

        assert run_id is not None
        assert run_id > 0

    def test_complete_processing_run(self, db):
        """Test completing a processing run."""
        run_id = db.start_processing_run("test")
        db.complete_processing_run(
            run_id=run_id,
            total_processed=100,
            total_succeeded=90,
            total_failed=10
        )

        runs = db.get_run_history(limit=1)
        assert len(runs) == 1
        assert runs[0].total_processed == 100
        assert runs[0].total_succeeded == 90

    def test_get_run_history(self, db):
        """Test getting run history."""
        # Create multiple runs
        for i in range(5):
            run_id = db.start_processing_run(f"source_{i}")
            db.complete_processing_run(run_id, total_processed=i * 10)

        runs = db.get_run_history(limit=3)
        assert len(runs) == 3

    def test_get_run_history_by_source(self, db):
        """Test getting run history filtered by source."""
        db.start_processing_run("source_a")
        db.start_processing_run("source_b")
        db.start_processing_run("source_a")

        runs = db.get_run_history(source="source_a")
        assert len(runs) == 2


# ============ State Tracking Tests ============


class TestStateTracking:
    """Tests for state tracking."""

    def test_needs_processing_new(self, db, sample_bookmarks):
        """Test needs_processing for new bookmark."""
        assert db.needs_processing(sample_bookmarks[0]) is True

    def test_needs_processing_processed(self, db, sample_bookmarks):
        """Test needs_processing for processed bookmark."""
        db.mark_processed(sample_bookmarks[0])
        assert db.needs_processing(sample_bookmarks[0]) is False

    def test_needs_processing_changed(self, db, sample_bookmarks):
        """Test needs_processing for changed bookmark."""
        db.mark_processed(sample_bookmarks[0])

        # Modify the bookmark
        sample_bookmarks[0].title = "Modified Title"
        assert db.needs_processing(sample_bookmarks[0]) is True

    def test_needs_processing_failed(self, db):
        """Test needs_processing for failed bookmark."""
        db.mark_failed("https://test.com", "Error")

        bookmark = Bookmark(url="https://test.com")
        assert db.needs_processing(bookmark) is True

    def test_get_unprocessed(self, db, sample_bookmarks):
        """Test get_unprocessed filtering."""
        # Process first two
        db.mark_processed(sample_bookmarks[0])
        db.mark_processed(sample_bookmarks[1])

        unprocessed = db.get_unprocessed(sample_bookmarks)
        assert len(unprocessed) == 1
        assert unprocessed[0].url == sample_bookmarks[2].url


# ============ Statistics Tests ============


class TestStatistics:
    """Tests for statistics."""

    def test_get_statistics(self, populated_db):
        """Test getting database statistics."""
        stats = populated_db.get_statistics()

        assert stats["total_bookmarks"] == 3
        assert stats["total_runs"] >= 1
        assert "by_status" in stats
        assert "processed" in stats["by_status"]

    def test_get_statistics_empty_db(self, db):
        """Test statistics on empty database."""
        stats = db.get_statistics()

        assert stats["total_bookmarks"] == 0
        assert stats["total_runs"] == 0


# ============ Data Model Tests ============


class TestBookmarkRecord:
    """Tests for BookmarkRecord dataclass."""

    def test_to_bookmark(self):
        """Test converting to Bookmark object."""
        record = BookmarkRecord(
            url="https://example.com",
            content_hash="abc123",
            processed_at=datetime.now(),
            ai_engine="claude",
            description="Test description",
            tags=["tag1", "tag2"],
            folder="Tech",
            title="Test Title"
        )

        bookmark = record.to_bookmark()

        assert bookmark.url == "https://example.com"
        assert bookmark.title == "Test Title"
        assert bookmark.folder == "Tech"

    def test_to_dict(self):
        """Test converting to dictionary."""
        record = BookmarkRecord(
            url="https://example.com",
            content_hash="abc123",
            processed_at=datetime.now(),
            ai_engine="claude",
            description="Test",
            tags=["tag1"],
            folder="Tech",
            title="Test"
        )

        d = record.to_dict()

        assert d["url"] == "https://example.com"
        assert d["ai_engine"] == "claude"


class TestProcessingRun:
    """Tests for ProcessingRun dataclass."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        run = ProcessingRun(
            id=1,
            started_at=datetime(2024, 1, 1, 0, 0, 0),
            completed_at=datetime(2024, 1, 1, 0, 1, 0),
            source="test",
            total_processed=100,
            total_succeeded=90,
            total_failed=10,
            config_hash="abc123"
        )

        d = run.to_dict()

        assert d["id"] == 1
        assert d["source"] == "test"
        assert d["total_processed"] == 100


class TestRunComparison:
    """Tests for RunComparison dataclass."""

    def test_total_changes(self):
        """Test total_changes calculation."""
        run1 = ProcessingRun(
            id=1,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            source="test",
            total_processed=0,
            total_succeeded=0,
            total_failed=0,
            config_hash=None
        )
        run2 = ProcessingRun(
            id=2,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            source="test",
            total_processed=0,
            total_succeeded=0,
            total_failed=0,
            config_hash=None
        )

        comparison = RunComparison(
            run1=run1,
            run2=run2,
            new_bookmarks=["a", "b"],
            removed_bookmarks=["c"],
            changed_bookmarks=["d", "e", "f"],
            unchanged_bookmarks=["g"]
        )

        assert comparison.total_changes == 6  # 2 new + 1 removed + 3 changed

    def test_to_dict(self):
        """Test converting to dictionary."""
        run1 = ProcessingRun(
            id=1,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            source="test",
            total_processed=0,
            total_succeeded=0,
            total_failed=0,
            config_hash=None
        )
        run2 = ProcessingRun(
            id=2,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            source="test",
            total_processed=0,
            total_succeeded=0,
            total_failed=0,
            config_hash=None
        )

        comparison = RunComparison(
            run1=run1,
            run2=run2,
            new_bookmarks=["a"],
            removed_bookmarks=["b"],
            changed_bookmarks=["c"],
            unchanged_bookmarks=["d"]
        )

        d = comparison.to_dict()

        assert d["run1_id"] == 1
        assert d["run2_id"] == 2
        assert d["new_bookmarks_count"] == 1
        assert d["total_changes"] == 3


# ============ Utility Tests ============


class TestDatabaseUtilities:
    """Tests for database utility methods."""

    def test_vacuum(self, populated_db):
        """Test vacuum operation."""
        # Should not raise
        populated_db.vacuum()

    def test_close(self, db):
        """Test close method."""
        db.close()
        # Connection should be closed
        assert db._conn is None

    def test_repr(self, temp_db_path):
        """Test string representation."""
        db = BookmarkDatabase(temp_db_path)
        assert "BookmarkDatabase" in repr(db)
        assert str(temp_db_path) in repr(db)
