"""
Unit tests for the ProcessingStateTracker.

Tests SQLite-based state tracking for incremental bookmark processing.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.data_sources import ProcessingStateTracker


class TestProcessingStateTrackerBasics:
    """Test basic ProcessingStateTracker functionality."""

    def test_initialization(self, temp_dir):
        """Test ProcessingStateTracker initialization."""
        db_path = temp_dir / "test_state.db"
        tracker = ProcessingStateTracker(db_path)

        assert tracker.db_path == db_path
        assert db_path.exists()

    def test_initialization_with_string_path(self, temp_dir):
        """Test initialization with string path."""
        db_path = str(temp_dir / "test_state.db")
        tracker = ProcessingStateTracker(db_path)

        assert isinstance(tracker.db_path, Path)

    def test_default_path(self, temp_dir, monkeypatch):
        """Test default database path."""
        monkeypatch.chdir(temp_dir)
        tracker = ProcessingStateTracker()

        assert tracker.db_path.name == ".bookmark_processor_state.db"

    def test_repr(self, temp_dir):
        """Test string representation."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        repr_str = repr(tracker)
        assert "ProcessingStateTracker" in repr_str
        assert "db_path=" in repr_str


class TestMarkingBookmarksProcessed:
    """Test marking bookmarks as processed."""

    def test_mark_processed(self, temp_dir):
        """Test marking a bookmark as processed."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(url="http://test.com", title="Test")

        tracker.mark_processed(bookmark, ai_engine="claude")

        info = tracker.get_processed_info("http://test.com")
        assert info is not None
        assert info["url"] == "http://test.com"
        assert info["ai_engine"] == "claude"

    def test_mark_processed_with_hash(self, temp_dir):
        """Test marking processed with custom hash."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(url="http://test.com", title="Test")

        tracker.mark_processed(
            bookmark,
            content_hash="custom_hash_value",
            ai_engine="openai"
        )

        info = tracker.get_processed_info("http://test.com")
        assert info["content_hash"] == "custom_hash_value"
        assert info["ai_engine"] == "openai"

    def test_mark_processed_stores_details(self, temp_dir):
        """Test that mark_processed stores bookmark details."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(
            url="http://test.com",
            title="Test Title",
            folder="Test/Folder",
            tags=["tag1", "tag2"],
            enhanced_description="Enhanced description"
        )
        bookmark.optimized_tags = ["optimized1", "optimized2"]

        tracker.mark_processed(bookmark, ai_engine="local")

        info = tracker.get_processed_info("http://test.com")
        assert info["title"] == "Test Title"
        assert info["folder"] == "Test/Folder"
        assert "optimized1" in info["tags"]
        assert info["description"] == "Enhanced description"

    def test_mark_processed_updates_existing(self, temp_dir):
        """Test that marking processed updates existing record."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(url="http://test.com", title="Original")

        # First processing
        tracker.mark_processed(bookmark, ai_engine="local")

        # Update bookmark and reprocess
        bookmark.title = "Updated"
        tracker.mark_processed(bookmark, ai_engine="claude")

        info = tracker.get_processed_info("http://test.com")
        assert info["title"] == "Updated"
        assert info["ai_engine"] == "claude"


class TestNeedsProcessing:
    """Test needs_processing detection."""

    def test_unprocessed_bookmark_needs_processing(self, temp_dir):
        """Test that unprocessed bookmark needs processing."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(url="http://test.com", title="Test")

        assert tracker.needs_processing(bookmark) is True

    def test_processed_bookmark_no_longer_needs_processing(self, temp_dir):
        """Test that processed bookmark doesn't need reprocessing."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(url="http://test.com", title="Test")

        tracker.mark_processed(bookmark, ai_engine="local")

        assert tracker.needs_processing(bookmark) is False

    def test_changed_bookmark_needs_reprocessing(self, temp_dir):
        """Test that changed bookmark needs reprocessing."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(url="http://test.com", title="Original")

        tracker.mark_processed(bookmark, ai_engine="local")

        # Change the bookmark
        bookmark.title = "Changed Title"

        assert tracker.needs_processing(bookmark) is True

    def test_hash_includes_relevant_fields(self, temp_dir):
        """Test that content hash considers relevant fields."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmark = Bookmark(
            url="http://test.com",
            title="Test",
            note="Note",
            excerpt="Excerpt",
            folder="Folder",
            tags=["tag1", "tag2"]
        )

        tracker.mark_processed(bookmark, ai_engine="local")

        # Changes to any of these should trigger reprocessing
        fields_to_test = [
            ("title", "New Title"),
            ("note", "New Note"),
            ("excerpt", "New Excerpt"),
            ("folder", "New/Folder"),
            ("tags", ["different", "tags"]),
        ]

        for field, new_value in fields_to_test:
            test_bookmark = Bookmark(
                url="http://test.com",
                title="Test",
                note="Note",
                excerpt="Excerpt",
                folder="Folder",
                tags=["tag1", "tag2"]
            )
            setattr(test_bookmark, field, new_value)

            assert tracker.needs_processing(test_bookmark) is True, (
                f"Change to {field} should trigger reprocessing"
            )


class TestGetUnprocessed:
    """Test get_unprocessed bulk operation."""

    def test_get_unprocessed_all_new(self, temp_dir):
        """Test get_unprocessed with all new bookmarks."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmarks = [
            Bookmark(url="http://test1.com", title="Test 1"),
            Bookmark(url="http://test2.com", title="Test 2"),
            Bookmark(url="http://test3.com", title="Test 3"),
        ]

        unprocessed = tracker.get_unprocessed(bookmarks)

        assert len(unprocessed) == 3

    def test_get_unprocessed_some_processed(self, temp_dir):
        """Test get_unprocessed with some already processed."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # Mark one as processed
        processed = Bookmark(url="http://test1.com", title="Test 1")
        tracker.mark_processed(processed, ai_engine="local")

        bookmarks = [
            Bookmark(url="http://test1.com", title="Test 1"),  # Processed
            Bookmark(url="http://test2.com", title="Test 2"),  # New
            Bookmark(url="http://test3.com", title="Test 3"),  # New
        ]

        unprocessed = tracker.get_unprocessed(bookmarks)

        assert len(unprocessed) == 2
        urls = {b.url for b in unprocessed}
        assert "http://test1.com" not in urls
        assert "http://test2.com" in urls
        assert "http://test3.com" in urls

    def test_get_unprocessed_includes_changed(self, temp_dir):
        """Test that get_unprocessed includes changed bookmarks."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # Mark bookmark as processed
        original = Bookmark(url="http://test1.com", title="Original")
        tracker.mark_processed(original, ai_engine="local")

        # Create changed version
        changed = Bookmark(url="http://test1.com", title="Changed")

        unprocessed = tracker.get_unprocessed([changed])

        assert len(unprocessed) == 1
        assert unprocessed[0].title == "Changed"

    def test_get_unprocessed_empty_list(self, temp_dir):
        """Test get_unprocessed with empty list."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        unprocessed = tracker.get_unprocessed([])

        assert unprocessed == []


class TestProcessingRuns:
    """Test processing run tracking."""

    def test_start_processing_run(self, temp_dir):
        """Test starting a processing run."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        run_id = tracker.start_processing_run(
            source="CSV File",
            config_hash="abc123"
        )

        assert run_id > 0
        assert tracker._current_run_id == run_id

    def test_complete_processing_run(self, temp_dir):
        """Test completing a processing run."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        run_id = tracker.start_processing_run(source="CSV File")
        tracker.complete_processing_run(
            run_id=run_id,
            total_processed=100,
            total_succeeded=95,
            total_failed=5
        )

        last_run = tracker.get_last_run()
        assert last_run["id"] == run_id
        assert last_run["total_processed"] == 100
        assert last_run["total_succeeded"] == 95
        assert last_run["total_failed"] == 5
        assert last_run["completed_at"] is not None

    def test_get_last_run(self, temp_dir):
        """Test getting last processing run."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # No runs yet
        assert tracker.get_last_run() is None

        # Add a run
        run_id = tracker.start_processing_run(source="CSV File")
        tracker.complete_processing_run(run_id=run_id, total_processed=50)

        last_run = tracker.get_last_run()
        assert last_run is not None
        assert last_run["source"] == "CSV File"

    def test_get_last_run_by_source(self, temp_dir):
        """Test getting last run filtered by source."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # Add runs from different sources
        run1 = tracker.start_processing_run(source="Source A")
        tracker.complete_processing_run(run1, total_processed=10)

        run2 = tracker.start_processing_run(source="Source B")
        tracker.complete_processing_run(run2, total_processed=20)

        run3 = tracker.start_processing_run(source="Source A")
        tracker.complete_processing_run(run3, total_processed=30)

        # Get last run for Source A
        last_a = tracker.get_last_run(source="Source A")
        assert last_a["id"] == run3
        assert last_a["total_processed"] == 30

        # Get last run for Source B
        last_b = tracker.get_last_run(source="Source B")
        assert last_b["id"] == run2
        assert last_b["total_processed"] == 20

    def test_get_run_history(self, temp_dir):
        """Test getting run history."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # Add multiple runs
        for i in range(5):
            run_id = tracker.start_processing_run(source="Test")
            tracker.complete_processing_run(run_id, total_processed=i * 10)

        history = tracker.get_run_history(limit=3)

        assert len(history) == 3
        # Most recent first
        assert history[0]["total_processed"] == 40
        assert history[1]["total_processed"] == 30
        assert history[2]["total_processed"] == 20


class TestStateManagement:
    """Test state management operations."""

    def test_get_processed_count(self, temp_dir):
        """Test getting processed count."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        assert tracker.get_processed_count() == 0

        # Add some processed bookmarks
        for i in range(5):
            bookmark = Bookmark(url=f"http://test{i}.com", title=f"Test {i}")
            tracker.mark_processed(bookmark, ai_engine="local")

        assert tracker.get_processed_count() == 5

    def test_get_processed_urls(self, temp_dir):
        """Test getting all processed URLs."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        urls = ["http://test1.com", "http://test2.com", "http://test3.com"]
        for url in urls:
            bookmark = Bookmark(url=url, title="Test")
            tracker.mark_processed(bookmark, ai_engine="local")

        processed_urls = tracker.get_processed_urls()

        assert set(processed_urls) == set(urls)

    def test_clear_processing_state(self, temp_dir):
        """Test clearing all processing state."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # Add some state
        for i in range(5):
            bookmark = Bookmark(url=f"http://test{i}.com", title=f"Test {i}")
            tracker.mark_processed(bookmark, ai_engine="local")

        assert tracker.get_processed_count() == 5

        cleared = tracker.clear_processing_state()

        assert cleared == 5
        assert tracker.get_processed_count() == 0

    def test_clear_processing_state_older_than(self, temp_dir):
        """Test clearing state older than date."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # Add bookmark and mark processed
        bookmark = Bookmark(url="http://test.com", title="Test")
        tracker.mark_processed(bookmark, ai_engine="local")

        # Clear state older than tomorrow (should not delete)
        cleared = tracker.clear_processing_state(
            older_than=datetime.now() + timedelta(days=1)
        )

        assert cleared == 1
        assert tracker.get_processed_count() == 0

    def test_remove_bookmark_state(self, temp_dir):
        """Test removing state for specific bookmark."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        bookmark = Bookmark(url="http://test.com", title="Test")
        tracker.mark_processed(bookmark, ai_engine="local")

        assert tracker.get_processed_info("http://test.com") is not None

        result = tracker.remove_bookmark_state("http://test.com")

        assert result is True
        assert tracker.get_processed_info("http://test.com") is None

    def test_remove_nonexistent_bookmark_state(self, temp_dir):
        """Test removing state for non-existent bookmark."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        result = tracker.remove_bookmark_state("http://nonexistent.com")

        assert result is False


class TestExportImport:
    """Test state export and import functionality."""

    def test_export_state(self, temp_dir):
        """Test exporting state to JSON."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # Add some data
        bookmark = Bookmark(url="http://test.com", title="Test")
        tracker.mark_processed(bookmark, ai_engine="claude")

        run_id = tracker.start_processing_run(source="Test")
        tracker.complete_processing_run(run_id, total_processed=10)

        # Export
        export_path = temp_dir / "export.json"
        tracker.export_state(export_path)

        assert export_path.exists()

        # Verify export content
        data = json.loads(export_path.read_text())
        assert "exported_at" in data
        assert "processed_bookmarks" in data
        assert "processing_runs" in data
        assert len(data["processed_bookmarks"]) == 1
        assert len(data["processing_runs"]) == 1

    def test_import_state(self, temp_dir):
        """Test importing state from JSON."""
        # Create export file
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "processed_bookmarks": [
                {
                    "url": "http://test.com",
                    "content_hash": "abc123",
                    "processed_at": datetime.now().isoformat(),
                    "ai_engine": "claude",
                    "description": "Test description",
                    "tags": "tag1,tag2",
                    "folder": "Test",
                    "title": "Test Title"
                }
            ],
            "processing_runs": [
                {
                    "started_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "source": "Test",
                    "total_processed": 100,
                    "total_succeeded": 95,
                    "total_failed": 5,
                    "config_hash": "xyz789"
                }
            ]
        }

        import_path = temp_dir / "import.json"
        import_path.write_text(json.dumps(export_data))

        # Import into new tracker
        tracker = ProcessingStateTracker(temp_dir / "state.db")
        bookmarks_imported, runs_imported = tracker.import_state(import_path)

        assert bookmarks_imported == 1
        assert runs_imported == 1

        # Verify imported data
        info = tracker.get_processed_info("http://test.com")
        assert info is not None
        assert info["ai_engine"] == "claude"

        history = tracker.get_run_history()
        assert len(history) == 1

    def test_export_import_roundtrip(self, temp_dir):
        """Test export-import roundtrip preserves data."""
        # Create and populate first tracker
        tracker1 = ProcessingStateTracker(temp_dir / "state1.db")

        for i in range(3):
            bookmark = Bookmark(
                url=f"http://test{i}.com",
                title=f"Test {i}",
                tags=[f"tag{i}"]
            )
            tracker1.mark_processed(bookmark, ai_engine="local")

        run_id = tracker1.start_processing_run(source="Test Source")
        tracker1.complete_processing_run(run_id, total_processed=3)

        # Export
        export_path = temp_dir / "export.json"
        tracker1.export_state(export_path)

        # Import into second tracker
        tracker2 = ProcessingStateTracker(temp_dir / "state2.db")
        tracker2.import_state(export_path)

        # Verify data preserved
        assert tracker2.get_processed_count() == tracker1.get_processed_count()

        for i in range(3):
            info1 = tracker1.get_processed_info(f"http://test{i}.com")
            info2 = tracker2.get_processed_info(f"http://test{i}.com")
            assert info1["content_hash"] == info2["content_hash"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unicode_in_bookmark_data(self, temp_dir):
        """Test handling Unicode characters."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        bookmark = Bookmark(
            url="http://test.com/\u00e9\u00e0\u00fc",
            title="\u4e2d\u6587\u6807\u9898",  # Chinese characters
            note="Emoji test with special chars: \u00e9\u00e0\u00fc"  # Valid unicode
        )

        tracker.mark_processed(bookmark, ai_engine="local")

        info = tracker.get_processed_info(bookmark.url)
        assert info is not None
        assert "\u4e2d" in info["title"]

    def test_very_long_url(self, temp_dir):
        """Test handling very long URLs."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        long_url = "http://test.com/" + "a" * 5000
        bookmark = Bookmark(url=long_url, title="Test")

        tracker.mark_processed(bookmark, ai_engine="local")

        info = tracker.get_processed_info(long_url)
        assert info is not None
        assert info["url"] == long_url

    def test_bookmark_with_empty_fields(self, temp_dir):
        """Test handling bookmarks with empty/None fields."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        bookmark = Bookmark(url="http://test.com", title="")
        bookmark.note = None
        bookmark.tags = []

        tracker.mark_processed(bookmark, ai_engine="local")

        assert tracker.needs_processing(bookmark) is False

    def test_concurrent_access_safety(self, temp_dir):
        """Test that multiple trackers can access same database."""
        db_path = temp_dir / "shared_state.db"

        tracker1 = ProcessingStateTracker(db_path)
        tracker2 = ProcessingStateTracker(db_path)

        # Both should be able to read/write
        bookmark1 = Bookmark(url="http://test1.com", title="Test 1")
        tracker1.mark_processed(bookmark1, ai_engine="local")

        bookmark2 = Bookmark(url="http://test2.com", title="Test 2")
        tracker2.mark_processed(bookmark2, ai_engine="local")

        # Both should see all data
        assert tracker1.get_processed_count() == 2
        assert tracker2.get_processed_count() == 2


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_incremental_processing_workflow(self, temp_dir):
        """Test typical incremental processing workflow."""
        tracker = ProcessingStateTracker(temp_dir / "state.db")

        # First run - process all bookmarks
        first_batch = [
            Bookmark(url="http://a.com", title="A"),
            Bookmark(url="http://b.com", title="B"),
            Bookmark(url="http://c.com", title="C"),
        ]

        run1_id = tracker.start_processing_run(source="CSV")
        unprocessed = tracker.get_unprocessed(first_batch)

        assert len(unprocessed) == 3

        for bookmark in unprocessed:
            bookmark.enhanced_description = f"Processed: {bookmark.title}"
            tracker.mark_processed(bookmark, ai_engine="claude")

        tracker.complete_processing_run(
            run1_id,
            total_processed=3,
            total_succeeded=3
        )

        # Second run - same bookmarks, none should need processing
        run2_id = tracker.start_processing_run(source="CSV")
        unprocessed = tracker.get_unprocessed(first_batch)

        assert len(unprocessed) == 0

        # Third run - add new bookmark and modify existing
        third_batch = first_batch + [
            Bookmark(url="http://d.com", title="D")  # New
        ]
        third_batch[0].title = "A Modified"  # Modified

        run3_id = tracker.start_processing_run(source="CSV")
        unprocessed = tracker.get_unprocessed(third_batch)

        # Should have new bookmark and modified bookmark
        assert len(unprocessed) == 2
        urls = {b.url for b in unprocessed}
        assert "http://a.com" in urls  # Modified
        assert "http://d.com" in urls  # New

        tracker.complete_processing_run(run3_id, total_processed=2)

        # Verify run history
        history = tracker.get_run_history()
        assert len(history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
