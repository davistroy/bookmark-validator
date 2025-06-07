"""
Unit tests for checkpoint manager and related components.

Tests for checkpoint management, resumption functionality, and data persistence.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, mock_open, patch

import pytest

from bookmark_processor.core.checkpoint_manager import (
    CheckpointManager,
    ProcessingStage,
    ProcessingState,
)
from bookmark_processor.core.data_models import Bookmark, ProcessingResults
from tests.fixtures.test_data import create_sample_bookmark_objects


class TestProcessingState:
    """Test ProcessingState class."""

    def test_init_default(self):
        """Test ProcessingState initialization with required parameters."""
        data = ProcessingState(
            input_file="test.csv",
            output_file="output.csv",
            total_bookmarks=100,
            config_hash="abc123",
            current_stage=ProcessingStage.INITIALIZATION,
            stage_progress=0,
        )

        assert data.input_file == "test.csv"
        assert data.output_file == "output.csv"
        assert data.total_bookmarks == 100
        assert data.processed_urls == set()
        assert data.processing_stats == {}

    def test_init_custom(self):
        """Test ProcessingState initialization with custom values."""
        bookmarks = create_sample_bookmark_objects()
        stats = {"processed": 5, "errors": 1}
        created_time = datetime.now(timezone.utc)

        data = ProcessingState(
            processed_bookmarks=bookmarks,
            last_processed_index=3,
            processing_stats=stats,
            created_at=created_time,
        )

        assert data.processed_bookmarks == bookmarks
        assert data.last_processed_index == 3
        assert data.processing_stats == stats
        assert data.created_at == created_time

    def test_to_dict(self):
        """Test converting ProcessingState to dictionary."""
        bookmarks = create_sample_bookmark_objects()[:2]
        data = ProcessingState(
            processed_bookmarks=bookmarks,
            last_processed_index=2,
            processing_stats={"processed": 2},
        )

        dict_data = data.to_dict()

        assert isinstance(dict_data, dict)
        assert "processed_bookmarks" in dict_data
        assert "last_processed_index" in dict_data
        assert "processing_stats" in dict_data
        assert "created_at" in dict_data
        assert "format_version" in dict_data

        assert dict_data["last_processed_index"] == 2
        assert len(dict_data["processed_bookmarks"]) == 2

    def test_from_dict(self):
        """Test creating ProcessingState from dictionary."""
        bookmarks = create_sample_bookmark_objects()[:2]
        original_data = ProcessingState(
            processed_bookmarks=bookmarks,
            last_processed_index=2,
            processing_stats={"processed": 2},
        )

        dict_data = original_data.to_dict()
        restored_data = ProcessingState.from_dict(dict_data)

        assert restored_data.last_processed_index == original_data.last_processed_index
        assert restored_data.processing_stats == original_data.processing_stats
        assert len(restored_data.processed_bookmarks) == len(
            original_data.processed_bookmarks
        )

    def test_is_valid(self):
        """Test checkpoint data validation."""
        # Valid data
        valid_data = ProcessingState(
            processed_bookmarks=create_sample_bookmark_objects(), last_processed_index=2
        )
        assert valid_data.is_valid() is True

        # Invalid data - negative index
        invalid_data = ProcessingState(
            processed_bookmarks=create_sample_bookmark_objects(),
            last_processed_index=-1,
        )
        assert invalid_data.is_valid() is False

        # Invalid data - index beyond bookmarks
        invalid_data2 = ProcessingState(
            processed_bookmarks=create_sample_bookmark_objects()[:2],
            last_processed_index=5,
        )
        assert invalid_data2.is_valid() is False

    def test_get_summary(self):
        """Test getting checkpoint summary."""
        bookmarks = create_sample_bookmark_objects()
        data = ProcessingState(
            processed_bookmarks=bookmarks,
            last_processed_index=3,
            processing_stats={"processed": 3, "errors": 1},
        )

        summary = data.get_summary()

        assert isinstance(summary, dict)
        assert summary["total_bookmarks"] == len(bookmarks)
        assert summary["last_processed_index"] == 3
        assert summary["processed_count"] == 3
        assert summary["error_count"] == 1
        assert "created_at" in summary


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_init_default(self, temp_dir):
        """Test CheckpointManager initialization with defaults."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        assert manager.checkpoint_dir == Path(temp_dir)
        assert manager.auto_cleanup is True
        assert manager.max_checkpoints == 10
        assert manager.compression_enabled is True

    def test_init_custom(self, temp_dir):
        """Test CheckpointManager initialization with custom values."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir),
            auto_cleanup=False,
            max_checkpoints=5,
            compression_enabled=False,
        )

        assert manager.auto_cleanup is False
        assert manager.max_checkpoints == 5
        assert manager.compression_enabled is False

    def test_create_checkpoint_directory(self, temp_dir):
        """Test checkpoint directory creation."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

        # Directory should be created automatically
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_generate_checkpoint_id(self, temp_dir):
        """Test checkpoint ID generation."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        checkpoint_id = manager._generate_checkpoint_id("test_session")

        assert isinstance(checkpoint_id, str)
        assert "test_session" in checkpoint_id
        assert len(checkpoint_id) > len("test_session")

    def test_save_checkpoint_success(self, temp_dir):
        """Test successful checkpoint saving."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        bookmarks = create_sample_bookmark_objects()
        checkpoint_data = ProcessingState(
            processed_bookmarks=bookmarks,
            last_processed_index=2,
            processing_stats={"processed": 2},
        )

        checkpoint_id = manager.save_checkpoint("test_session", checkpoint_data)

        assert isinstance(checkpoint_id, str)
        assert checkpoint_id in manager.active_checkpoints

        # Verify file was created
        checkpoint_files = list(manager.checkpoint_dir.glob("*.checkpoint"))
        assert len(checkpoint_files) > 0

    def test_save_checkpoint_with_compression(self, temp_dir):
        """Test checkpoint saving with compression."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir), compression_enabled=True
        )

        bookmarks = create_sample_bookmark_objects()
        checkpoint_data = ProcessingState(
            processed_bookmarks=bookmarks, last_processed_index=2
        )

        checkpoint_id = manager.save_checkpoint("test_session", checkpoint_data)

        # Verify compressed file exists
        checkpoint_path = manager._get_checkpoint_path(checkpoint_id)
        assert checkpoint_path.exists()

        # File should be smaller due to compression (for larger datasets)
        assert checkpoint_path.stat().st_size > 0

    def test_load_checkpoint_success(self, temp_dir):
        """Test successful checkpoint loading."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # First save a checkpoint
        original_bookmarks = create_sample_bookmark_objects()
        original_data = ProcessingState(
            processed_bookmarks=original_bookmarks,
            last_processed_index=3,
            processing_stats={"processed": 3},
        )

        checkpoint_id = manager.save_checkpoint("test_session", original_data)

        # Then load it
        loaded_data = manager.load_checkpoint(checkpoint_id)

        assert loaded_data is not None
        assert loaded_data.last_processed_index == 3
        assert loaded_data.processing_stats["processed"] == 3
        assert len(loaded_data.processed_bookmarks) == len(original_bookmarks)

    def test_load_checkpoint_nonexistent(self, temp_dir):
        """Test loading non-existent checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        loaded_data = manager.load_checkpoint("nonexistent_checkpoint")

        assert loaded_data is None

    def test_load_checkpoint_corrupted(self, temp_dir):
        """Test loading corrupted checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Create a corrupted checkpoint file
        checkpoint_path = manager.checkpoint_dir / "corrupted.checkpoint"
        checkpoint_path.write_text("corrupted data")

        # Add to active checkpoints
        manager.active_checkpoints["corrupted"] = {
            "file_path": checkpoint_path,
            "created_at": datetime.now(timezone.utc),
        }

        loaded_data = manager.load_checkpoint("corrupted")

        assert loaded_data is None

    def test_has_checkpoint(self, temp_dir):
        """Test checking for checkpoint existence."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Should not exist initially
        assert manager.has_checkpoint("test_session") is False

        # Save a checkpoint
        checkpoint_data = ProcessingState(
            processed_bookmarks=create_sample_bookmark_objects(), last_processed_index=1
        )
        checkpoint_id = manager.save_checkpoint("test_session", checkpoint_data)

        # Should exist now
        assert manager.has_checkpoint(checkpoint_id) is True
        assert manager.has_checkpoint("nonexistent") is False

    def test_list_checkpoints_empty(self, temp_dir):
        """Test listing checkpoints when none exist."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        checkpoints = manager.list_checkpoints()

        assert isinstance(checkpoints, list)
        assert len(checkpoints) == 0

    def test_list_checkpoints_with_data(self, temp_dir):
        """Test listing checkpoints with saved data."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save multiple checkpoints
        for i in range(3):
            checkpoint_data = ProcessingState(
                processed_bookmarks=create_sample_bookmark_objects(),
                last_processed_index=i,
            )
            manager.save_checkpoint(f"session_{i}", checkpoint_data)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3
        for checkpoint in checkpoints:
            assert "checkpoint_id" in checkpoint
            assert "session_name" in checkpoint
            assert "created_at" in checkpoint
            assert "summary" in checkpoint

    def test_delete_checkpoint_success(self, temp_dir):
        """Test successful checkpoint deletion."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save a checkpoint
        checkpoint_data = ProcessingState(
            processed_bookmarks=create_sample_bookmark_objects(), last_processed_index=1
        )
        checkpoint_id = manager.save_checkpoint("test_session", checkpoint_data)

        # Verify it exists
        assert manager.has_checkpoint(checkpoint_id) is True

        # Delete it
        success = manager.delete_checkpoint(checkpoint_id)

        assert success is True
        assert manager.has_checkpoint(checkpoint_id) is False
        assert checkpoint_id not in manager.active_checkpoints

    def test_delete_checkpoint_nonexistent(self, temp_dir):
        """Test deleting non-existent checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        success = manager.delete_checkpoint("nonexistent")

        assert success is False

    def test_cleanup_old_checkpoints(self, temp_dir):
        """Test cleaning up old checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir), max_checkpoints=2)

        # Save more checkpoints than the limit
        checkpoint_ids = []
        for i in range(4):
            checkpoint_data = ProcessingState(
                processed_bookmarks=create_sample_bookmark_objects(),
                last_processed_index=i,
            )
            checkpoint_id = manager.save_checkpoint(f"session_{i}", checkpoint_data)
            checkpoint_ids.append(checkpoint_id)

        # Should only have max_checkpoints remaining
        remaining = manager.list_checkpoints()
        assert len(remaining) <= manager.max_checkpoints

        # Newest checkpoints should be preserved
        latest_ids = [cp["checkpoint_id"] for cp in remaining]
        assert checkpoint_ids[-1] in latest_ids  # Most recent should remain

    def test_cleanup_all_checkpoints(self, temp_dir):
        """Test cleaning up all checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save multiple checkpoints
        for i in range(3):
            checkpoint_data = ProcessingState(
                processed_bookmarks=create_sample_bookmark_objects(),
                last_processed_index=i,
            )
            manager.save_checkpoint(f"session_{i}", checkpoint_data)

        # Verify they exist
        assert len(manager.list_checkpoints()) == 3

        # Clean up all
        deleted_count = manager.cleanup_checkpoints()

        assert deleted_count == 3
        assert len(manager.list_checkpoints()) == 0
        assert len(manager.active_checkpoints) == 0

    def test_get_latest_checkpoint(self, temp_dir):
        """Test getting the latest checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # No checkpoints initially
        latest = manager.get_latest_checkpoint("test_session")
        assert latest is None

        # Save multiple checkpoints with slight delays
        import time

        checkpoint_ids = []
        for i in range(3):
            checkpoint_data = ProcessingState(
                processed_bookmarks=create_sample_bookmark_objects(),
                last_processed_index=i,
            )
            checkpoint_id = manager.save_checkpoint("test_session", checkpoint_data)
            checkpoint_ids.append(checkpoint_id)
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Get latest should return the most recent
        latest = manager.get_latest_checkpoint("test_session")
        assert latest is not None
        assert latest["checkpoint_id"] == checkpoint_ids[-1]

    def test_validate_checkpoint_data(self, temp_dir):
        """Test checkpoint data validation."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Valid data
        valid_data = ProcessingState(
            processed_bookmarks=create_sample_bookmark_objects(), last_processed_index=2
        )
        assert manager._validate_checkpoint_data(valid_data) is True

        # Invalid data
        invalid_data = ProcessingState(
            processed_bookmarks=create_sample_bookmark_objects(),
            last_processed_index=-1,
        )
        assert manager._validate_checkpoint_data(invalid_data) is False

    def test_get_checkpoint_statistics(self, temp_dir):
        """Test getting checkpoint statistics."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save some checkpoints
        for i in range(3):
            checkpoint_data = ProcessingState(
                processed_bookmarks=create_sample_bookmark_objects(),
                last_processed_index=i,
            )
            manager.save_checkpoint(f"session_{i}", checkpoint_data)

        stats = manager.get_checkpoint_statistics()

        assert isinstance(stats, dict)
        assert stats["total_checkpoints"] == 3
        assert stats["total_size_mb"] > 0
        assert "oldest_checkpoint" in stats
        assert "newest_checkpoint" in stats

    def test_checkpoint_with_large_dataset(self, temp_dir):
        """Test checkpoint handling with larger dataset."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Create a larger dataset
        large_bookmarks = []
        for i in range(100):
            bookmark = Bookmark(
                id=str(i),
                title=f"Bookmark {i}",
                url=f"https://example.com/{i}",
                note=f"This is bookmark number {i} with some content",
            )
            large_bookmarks.append(bookmark)

        checkpoint_data = ProcessingState(
            processed_bookmarks=large_bookmarks,
            last_processed_index=50,
            processing_stats={"processed": 50, "remaining": 50},
        )

        # Save and load
        checkpoint_id = manager.save_checkpoint("large_session", checkpoint_data)
        loaded_data = manager.load_checkpoint(checkpoint_id)

        assert loaded_data is not None
        assert len(loaded_data.processed_bookmarks) == 100
        assert loaded_data.last_processed_index == 50
        assert loaded_data.processing_stats["processed"] == 50


class TestCheckpointError:
    """Test CheckpointError exception."""

    def test_checkpoint_error_creation(self):
        """Test CheckpointError exception creation."""
        error = CheckpointError("Test checkpoint error")

        assert str(error) == "Test checkpoint error"
        assert isinstance(error, Exception)

    def test_checkpoint_error_with_checkpoint_id(self):
        """Test CheckpointError with checkpoint ID."""
        error = CheckpointError("Checkpoint failed", checkpoint_id="test_checkpoint")

        assert str(error) == "Checkpoint failed"
        assert error.checkpoint_id == "test_checkpoint"


class TestCheckpointIntegration:
    """Integration tests for checkpoint functionality."""

    def test_resume_processing_workflow(self, temp_dir):
        """Test complete resume processing workflow."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Simulate initial processing session
        all_bookmarks = create_sample_bookmark_objects()

        # Process first 3 bookmarks
        processed_bookmarks = all_bookmarks[:3]
        for bookmark in processed_bookmarks:
            bookmark.enhanced_description = f"Processed: {bookmark.title}"
            bookmark.processing_status.ai_processed = True

        # Save checkpoint
        checkpoint_data = ProcessingState(
            processed_bookmarks=processed_bookmarks,
            last_processed_index=3,
            processing_stats={"processed": 3, "remaining": 2},
        )

        checkpoint_id = manager.save_checkpoint("resume_test", checkpoint_data)

        # Simulate resuming processing
        loaded_data = manager.load_checkpoint(checkpoint_id)
        assert loaded_data is not None

        # Continue processing from where we left off
        remaining_bookmarks = all_bookmarks[loaded_data.last_processed_index :]

        # Verify we have the right remaining bookmarks
        assert len(remaining_bookmarks) == 2
        assert remaining_bookmarks[0].id != processed_bookmarks[-1].id

        # Simulate processing remaining bookmarks
        for bookmark in remaining_bookmarks:
            bookmark.enhanced_description = f"Resumed: {bookmark.title}"
            bookmark.processing_status.ai_processed = True

        # Combine all processed bookmarks
        final_bookmarks = loaded_data.processed_bookmarks + remaining_bookmarks
        assert len(final_bookmarks) == len(all_bookmarks)

        # Clean up checkpoint after successful completion
        manager.delete_checkpoint(checkpoint_id)
        assert not manager.has_checkpoint(checkpoint_id)

    def test_checkpoint_recovery_after_failure(self, temp_dir):
        """Test checkpoint recovery after processing failure."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Simulate processing with periodic checkpoints
        all_bookmarks = create_sample_bookmark_objects()
        processed_bookmarks = []

        # Process bookmarks with checkpoints every 2 items
        for i, bookmark in enumerate(all_bookmarks):
            # Simulate processing
            bookmark.enhanced_description = f"Processed: {bookmark.title}"
            processed_bookmarks.append(bookmark)

            # Save checkpoint every 2 items
            if (i + 1) % 2 == 0:
                checkpoint_data = ProcessingState(
                    processed_bookmarks=processed_bookmarks.copy(),
                    last_processed_index=i + 1,
                    processing_stats={"processed": i + 1},
                )
                checkpoint_id = manager.save_checkpoint(f"batch_{i}", checkpoint_data)

                # Simulate failure at index 3 (after 4 items processed)
                if i == 3:
                    break

        # Find the latest checkpoint
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) > 0

        latest = max(checkpoints, key=lambda x: x["created_at"])
        loaded_data = manager.load_checkpoint(latest["checkpoint_id"])

        # Should have recovered state up to the latest checkpoint
        assert loaded_data is not None
        assert len(loaded_data.processed_bookmarks) >= 2
        assert loaded_data.last_processed_index >= 2

    def test_checkpoint_size_optimization(self, temp_dir):
        """Test checkpoint size optimization with compression."""
        manager_compressed = CheckpointManager(
            checkpoint_dir=str(temp_dir / "compressed"), compression_enabled=True
        )

        manager_uncompressed = CheckpointManager(
            checkpoint_dir=str(temp_dir / "uncompressed"), compression_enabled=False
        )

        # Create identical large checkpoint data
        large_bookmarks = []
        for i in range(50):
            bookmark = Bookmark(
                id=str(i),
                title=f"Large Bookmark {i}",
                url=f"https://example.com/{i}",
                note="This is a longer note with repeated content " * 10,
                enhanced_description="This is an enhanced description " * 5,
            )
            large_bookmarks.append(bookmark)

        checkpoint_data = ProcessingState(
            processed_bookmarks=large_bookmarks, last_processed_index=50
        )

        # Save with both managers
        compressed_id = manager_compressed.save_checkpoint(
            "large_test", checkpoint_data
        )
        uncompressed_id = manager_uncompressed.save_checkpoint(
            "large_test", checkpoint_data
        )

        # Compare file sizes
        compressed_path = manager_compressed._get_checkpoint_path(compressed_id)
        uncompressed_path = manager_uncompressed._get_checkpoint_path(uncompressed_id)

        compressed_size = compressed_path.stat().st_size
        uncompressed_size = uncompressed_path.stat().st_size

        # Compressed should be smaller (for repetitive data)
        assert compressed_size < uncompressed_size

        # Both should load correctly
        compressed_loaded = manager_compressed.load_checkpoint(compressed_id)
        uncompressed_loaded = manager_uncompressed.load_checkpoint(uncompressed_id)

        assert compressed_loaded is not None
        assert uncompressed_loaded is not None
        assert len(compressed_loaded.processed_bookmarks) == len(
            uncompressed_loaded.processed_bookmarks
        )


if __name__ == "__main__":
    pytest.main([__file__])
