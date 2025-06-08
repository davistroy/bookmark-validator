"""
Unit tests for checkpoint manager and related components.

Tests for checkpoint management, resumption functionality, and data persistence.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from bookmark_processor.core.checkpoint_manager import (
    CheckpointManager,
    ProcessingStage,
    ProcessingState,
)
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.url_validator import ValidationResult
from bookmark_processor.core.ai_processor import AIProcessingResult
from bookmark_processor.core.content_analyzer import ContentData


@pytest.fixture
def sample_processing_state():
    """Create a sample ProcessingState for testing."""
    return ProcessingState(
        input_file="test_input.csv",
        output_file="test_output.csv",
        total_bookmarks=100,
        config_hash="test_hash_123",
        current_stage=ProcessingStage.URL_VALIDATION,
        stage_progress=50,
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        id="test_1",
        title="Test Bookmark",
        url="https://example.com",
        note="Test note",
        excerpt="Test excerpt",
        folder="Test Folder",
        tags=["test", "bookmark"],
        created="2024-01-01T00:00:00Z",
        cover="",
        highlights="",
        favorite=False,
    )


@pytest.fixture
def sample_validation_result():
    """Create a sample ValidationResult for testing."""
    return ValidationResult(
        url="https://example.com",
        is_valid=True,
        status_code=200,
        response_time=0.5,
    )


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
        assert data.current_stage == ProcessingStage.INITIALIZATION
        assert data.stage_progress == 0
        assert data.processed_urls == set()
        assert data.processing_stats == {}
        assert isinstance(data.start_time, datetime)

    def test_init_with_data(self):
        """Test ProcessingState initialization with some data."""
        data = ProcessingState(
            input_file="test.csv",
            output_file="output.csv", 
            total_bookmarks=50,
            config_hash="xyz789",
            current_stage=ProcessingStage.AI_PROCESSING,
            stage_progress=25,
        )
        
        # Add some processed URLs
        data.processed_urls.add("https://example.com")
        data.processed_urls.add("https://test.com")
        
        # Add some processing stats
        data.processing_stats["valid_urls"] = 20
        data.processing_stats["failed_urls"] = 5

        assert len(data.processed_urls) == 2
        assert data.processing_stats["valid_urls"] == 20
        assert data.current_stage == ProcessingStage.AI_PROCESSING

    def test_to_dict(self):
        """Test converting ProcessingState to dictionary."""
        data = ProcessingState(
            input_file="input.csv",
            output_file="output.csv",
            total_bookmarks=10,
            config_hash="hash123",
            current_stage=ProcessingStage.CONTENT_ANALYSIS,
            stage_progress=5,
        )
        
        # Add some test data
        data.processed_urls.add("https://example.com")
        data.processing_stats["test"] = "value"

        dict_data = data.to_dict()

        assert isinstance(dict_data, dict)
        assert dict_data["input_file"] == "input.csv"
        assert dict_data["output_file"] == "output.csv"
        assert dict_data["total_bookmarks"] == 10
        assert dict_data["current_stage"] == ProcessingStage.CONTENT_ANALYSIS.value
        assert dict_data["processed_urls"] == ["https://example.com"]
        assert isinstance(dict_data["start_time"], str)

    def test_from_dict(self):
        """Test creating ProcessingState from dictionary."""
        original_data = ProcessingState(
            input_file="test.csv",
            output_file="out.csv",
            total_bookmarks=25,
            config_hash="original_hash",
            current_stage=ProcessingStage.TAG_OPTIMIZATION,
            stage_progress=15,
        )
        
        original_data.processed_urls.add("https://test.com")
        original_data.processing_stats["count"] = 42

        dict_data = original_data.to_dict()
        restored_data = ProcessingState.from_dict(dict_data)

        assert restored_data.input_file == original_data.input_file
        assert restored_data.output_file == original_data.output_file
        assert restored_data.total_bookmarks == original_data.total_bookmarks
        assert restored_data.current_stage == original_data.current_stage
        assert restored_data.processed_urls == original_data.processed_urls
        assert restored_data.processing_stats == original_data.processing_stats


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_init_default(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization with defaults."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))

        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.save_interval == 50
        assert manager.auto_cleanup is True
        assert manager.compression is True

    def test_init_custom(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization with custom values."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            save_interval=25,
            auto_cleanup=False,
            compression=False,
        )

        assert manager.save_interval == 25
        assert manager.auto_cleanup is False
        assert manager.compression is False

    def test_create_checkpoint_directory(self, temp_checkpoint_dir):
        """Test checkpoint directory creation."""
        checkpoint_dir = temp_checkpoint_dir / "checkpoints"
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

        # Directory should be created automatically
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_initialize_processing(self, temp_checkpoint_dir):
        """Test initializing new processing session."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        config = {"ai_engine": "local", "batch_size": 50}
        state = manager.initialize_processing(
            input_file="input.csv",
            output_file="output.csv",
            total_bookmarks=100,
            config=config,
        )

        assert isinstance(state, ProcessingState)
        assert state.input_file == "input.csv"
        assert state.output_file == "output.csv"
        assert state.total_bookmarks == 100
        assert state.current_stage == ProcessingStage.INITIALIZATION
        assert manager.current_state is not None

    def test_save_checkpoint(self, temp_checkpoint_dir):
        """Test saving checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        # Initialize processing
        state = manager.initialize_processing(
            input_file="test.csv",
            output_file="out.csv", 
            total_bookmarks=50,
            config={"test": "config"},
        )
        
        # Add some processed URLs to trigger save
        for i in range(60):  # More than save_interval (50)
            state.processed_urls.add(f"https://example{i}.com")
        
        # Should save automatically due to interval
        success = manager.save_checkpoint(force=True)
        assert success is True
        
        # Verify checkpoint file exists
        checkpoint_files = list(manager.checkpoint_dir.glob("checkpoint_*.json"))
        assert len(checkpoint_files) > 0

    def test_load_checkpoint(self, temp_checkpoint_dir):
        """Test loading checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        # Initialize and save a checkpoint
        state = manager.initialize_processing(
            input_file="input.csv",
            output_file="output.csv",
            total_bookmarks=100,
            config={"test": "value"},
        )
        
        state.processed_urls.add("https://test.com")
        state.processing_stats["test_stat"] = 42
        manager.save_checkpoint(force=True)
        
        # Create new manager and load checkpoint
        manager2 = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        loaded_state = manager2.load_checkpoint("input.csv")
        
        assert loaded_state is not None
        assert loaded_state.input_file == "input.csv"
        assert "https://test.com" in loaded_state.processed_urls
        assert loaded_state.processing_stats["test_stat"] == 42

    def test_has_checkpoint(self, temp_checkpoint_dir):
        """Test checking for existing checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        # No checkpoint initially
        assert manager.has_checkpoint("nonexistent.csv") is False
        
        # Create checkpoint
        manager.initialize_processing(
            input_file="test.csv",
            output_file="out.csv",
            total_bookmarks=10,
            config={},
        )
        manager.save_checkpoint(force=True)
        
        # Should find checkpoint
        assert manager.has_checkpoint("test.csv") is True

    def test_update_stage(self, temp_checkpoint_dir):
        """Test updating processing stage."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        state = manager.initialize_processing(
            input_file="test.csv",
            output_file="out.csv",
            total_bookmarks=10,
            config={},
        )
        
        assert state.current_stage == ProcessingStage.INITIALIZATION
        
        manager.update_stage(ProcessingStage.URL_VALIDATION, 25)
        assert manager.current_state.current_stage == ProcessingStage.URL_VALIDATION

    def test_add_validated_bookmark(self, temp_checkpoint_dir, sample_bookmark, sample_validation_result):
        """Test adding validated bookmark to checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        manager.initialize_processing(
            input_file="test.csv",
            output_file="out.csv",
            total_bookmarks=10,
            config={},
        )
        
        manager.add_validated_bookmark(sample_validation_result, sample_bookmark)
        
        assert len(manager.current_state.validated_bookmarks) == 1
        assert sample_bookmark.url in manager.current_state.processed_urls

    def test_get_processing_progress(self, temp_checkpoint_dir):
        """Test getting processing progress."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        state = manager.initialize_processing(
            input_file="test.csv",
            output_file="out.csv",
            total_bookmarks=100,
            config={},
        )
        
        # Add some processed URLs
        state.processed_urls.add("https://example1.com")
        state.processed_urls.add("https://example2.com")
        
        progress = manager.get_processing_progress()
        
        assert isinstance(progress, dict)
        assert progress["total_bookmarks"] == 100
        assert progress["processed_count"] == 2
        assert progress["progress_percentage"] == 2.0
        assert "current_stage" in progress

    def test_clear_checkpoint(self, temp_checkpoint_dir):
        """Test clearing checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        # Create checkpoint
        manager.initialize_processing(
            input_file="test.csv",
            output_file="out.csv",
            total_bookmarks=10,
            config={},
        )
        manager.save_checkpoint(force=True)
        
        # Verify checkpoint exists
        assert manager.checkpoint_file is not None
        assert manager.checkpoint_file.exists()
        
        # Clear checkpoint
        success = manager.clear_checkpoint()
        assert success is True
        assert manager.current_state is None
        assert manager.checkpoint_file is None

    def test_cleanup_old_checkpoints(self, temp_checkpoint_dir):
        """Test cleaning up old checkpoint files."""
        manager = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        # Create multiple checkpoint files
        for i in range(5):
            checkpoint_file = temp_checkpoint_dir / f"checkpoint_old_{i}.json"
            checkpoint_file.write_text(json.dumps({"test": f"data_{i}"}))
        
        # Run cleanup
        manager.cleanup_old_checkpoints(keep_count=2)
        
        # Should only have 2 files left
        remaining_files = list(temp_checkpoint_dir.glob("checkpoint_*.json"))
        assert len(remaining_files) == 2


class TestCheckpointIntegration:
    """Test integrated checkpoint functionality."""

    def test_resume_processing_workflow(self, temp_checkpoint_dir, sample_bookmark):
        """Test complete resume processing workflow."""
        # First processing session
        manager1 = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        state = manager1.initialize_processing(
            input_file="large_dataset.csv",
            output_file="processed.csv",
            total_bookmarks=1000,
            config={"batch_size": 50},
        )
        
        # Simulate partial processing
        state.current_stage = ProcessingStage.URL_VALIDATION
        for i in range(300):  # Process 300 out of 1000
            state.processed_urls.add(f"https://example{i}.com")
        
        state.processing_stats["validated"] = 280
        state.processing_stats["failed"] = 20
        
        manager1.save_checkpoint(force=True)
        
        # Simulate application restart - new manager
        manager2 = CheckpointManager(checkpoint_dir=str(temp_checkpoint_dir))
        
        # Resume processing
        resumed_state = manager2.load_checkpoint("large_dataset.csv")
        
        assert resumed_state is not None
        assert resumed_state.total_bookmarks == 1000
        assert len(resumed_state.processed_urls) == 300
        assert resumed_state.current_stage == ProcessingStage.URL_VALIDATION
        assert resumed_state.processing_stats["validated"] == 280
        
        # Continue processing
        for i in range(300, 350):  # Process 50 more
            resumed_state.processed_urls.add(f"https://example{i}.com")
        
        assert len(resumed_state.processed_urls) == 350
        
        # Progress should be calculated correctly
        progress = manager2.get_processing_progress()
        assert progress["processed_count"] == 350
        assert progress["progress_percentage"] == 35.0


if __name__ == "__main__":
    pytest.main([__file__])