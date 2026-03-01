"""
Comprehensive Tests for Streaming Pipeline Module.

This module provides comprehensive tests for bookmark_processor/core/streaming/pipeline.py
to achieve 90%+ code coverage.

Tests cover:
1. StreamingPipelineConfig - Configuration dataclass
2. ProcessingStats - Statistics tracking and calculations
3. StreamingPipelineResults - Results container
4. StreamingPipeline - Main pipeline class with all methods
5. Generator-based processing
6. Memory-efficient batch handling
7. Pipeline stage composition
8. Error handling during streaming
9. Progress tracking integration
10. Lazy component initialization
"""

import csv
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import pytest

from bookmark_processor.core.data_models import Bookmark, ProcessingStatus
from bookmark_processor.core.streaming.pipeline import (
    ProcessingStats,
    StreamingPipeline,
    StreamingPipelineConfig,
    StreamingPipelineResults,
)
from bookmark_processor.core.streaming.reader import StreamingBookmarkReader
from bookmark_processor.core.streaming.writer import StreamingBookmarkWriter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_config(tmp_path):
    """Create a basic StreamingPipelineConfig for testing."""
    return StreamingPipelineConfig(
        input_file=tmp_path / "input.csv",
        output_file=tmp_path / "output.csv",
        batch_size=10,
        flush_interval=5,
        url_timeout=10.0,
        max_concurrent_requests=5,
        verify_ssl=True,
        ai_enabled=False,
        max_description_length=150,
        target_tag_count=100,
        max_tags_per_bookmark=5,
        use_state_tracker=False,
        state_db_path=None,
        checkpoint_interval=50,
        checkpoint_dir=".bookmark_checkpoints",
        progress_callback=None,
        verbose=False,
    )


@pytest.fixture
def sample_csv_data():
    """Sample CSV data in raindrop.io export format."""
    return [
        ["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"],
        ["1", "Site One", "Note 1", "Excerpt 1", "https://example.com/1", "Tech", "python, web", "2024-01-01T00:00:00Z", "", "", "false"],
        ["2", "Site Two", "Note 2", "Excerpt 2", "https://example.com/2", "Tech/AI", "ai, ml", "2024-01-02T00:00:00Z", "", "", "true"],
        ["3", "Site Three", "", "", "https://example.com/3", "Science", "science", "2024-01-03T00:00:00Z", "", "", "false"],
        ["4", "Site Four", "Note 4", "", "https://example.com/4", "News", "news", "2024-01-04T00:00:00Z", "", "", "false"],
        ["5", "Site Five", "", "Excerpt 5", "https://example.com/5", "Dev", "dev, code", "2024-01-05T00:00:00Z", "", "", "true"],
    ]


@pytest.fixture
def sample_input_file(tmp_path, sample_csv_data):
    """Create a sample input CSV file."""
    csv_file = tmp_path / "input.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(sample_csv_data)
    return csv_file


@pytest.fixture
def large_input_file(tmp_path):
    """Create a large input CSV file for batch testing."""
    csv_file = tmp_path / "large_input.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"])
        for i in range(350):  # Create 350 bookmarks to test multiple batches
            writer.writerow([
                str(i),
                f"Test Site {i}",
                f"Note {i}",
                f"Excerpt {i}",
                f"https://example{i}.com/page",
                f"Folder{i % 10}",
                f"tag{i % 20}",
                "2024-01-01T00:00:00Z",
                "",
                "",
                "false"
            ])
    return csv_file


@pytest.fixture
def sample_bookmarks():
    """Create sample Bookmark objects for testing."""
    return [
        Bookmark(
            id="1",
            url="https://example.com/1",
            title="Test Site 1",
            note="Note 1",
            folder="Tech",
            tags=["python", "web"],
        ),
        Bookmark(
            id="2",
            url="https://example.com/2",
            title="Test Site 2",
            note="Note 2",
            folder="Tech/AI",
            tags=["ai", "ml"],
        ),
        Bookmark(
            id="3",
            url="https://example.com/3",
            title="Test Site 3",
            folder="Science",
            tags=["science"],
        ),
    ]


@pytest.fixture
def mock_url_validator():
    """Create a mock URL validator."""
    mock = MagicMock()
    mock.validate_url.return_value = MagicMock(is_valid=True, error_message=None)
    mock.close = MagicMock()
    return mock


@pytest.fixture
def mock_content_analyzer():
    """Create a mock content analyzer."""
    mock = MagicMock()
    mock.analyze_content.return_value = {
        "content": "Sample content text",
        "title": "Extracted Title",
        "description": "Extracted description",
    }
    mock.close = MagicMock()
    return mock


@pytest.fixture
def mock_ai_processor():
    """Create a mock AI processor."""
    mock = MagicMock()
    mock_result = MagicMock()
    mock_result.enhanced_description = "AI enhanced description"
    mock.process_single.return_value = mock_result
    mock.close = MagicMock()
    return mock


@pytest.fixture
def mock_tag_generator():
    """Create a mock tag generator."""
    mock = MagicMock()
    mock.generate_for_single_bookmark.return_value = ["generated", "tags", "here"]
    return mock


@pytest.fixture
def mock_state_tracker():
    """Create a mock state tracker."""
    mock = MagicMock()
    mock.start_processing_run.return_value = 1
    mock.get_unprocessed.return_value = []  # Will be overridden in tests
    mock.mark_processed = MagicMock()
    mock.complete_processing_run = MagicMock()
    return mock


# ============================================================================
# StreamingPipelineConfig Tests
# ============================================================================


class TestStreamingPipelineConfig:
    """Tests for StreamingPipelineConfig dataclass."""

    def test_config_default_values(self, tmp_path):
        """Test that config has correct default values."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
        )

        assert config.batch_size == 100
        assert config.flush_interval == 10
        assert config.url_timeout == 30.0
        assert config.max_concurrent_requests == 10
        assert config.verify_ssl is True
        assert config.ai_enabled is True
        assert config.max_description_length == 150
        assert config.target_tag_count == 150
        assert config.max_tags_per_bookmark == 5
        assert config.use_state_tracker is True
        assert config.state_db_path is None
        assert config.checkpoint_interval == 50
        assert config.checkpoint_dir == ".bookmark_checkpoints"
        assert config.progress_callback is None
        assert config.verbose is False

    def test_config_custom_values(self, tmp_path):
        """Test config with custom values."""
        callback = MagicMock()
        config = StreamingPipelineConfig(
            input_file=tmp_path / "custom_input.csv",
            output_file=tmp_path / "custom_output.csv",
            batch_size=50,
            flush_interval=20,
            url_timeout=60.0,
            max_concurrent_requests=20,
            verify_ssl=False,
            ai_enabled=False,
            max_description_length=200,
            target_tag_count=200,
            max_tags_per_bookmark=10,
            use_state_tracker=False,
            state_db_path=tmp_path / "state.db",
            checkpoint_interval=100,
            checkpoint_dir="custom_checkpoints",
            progress_callback=callback,
            verbose=True,
        )

        assert config.batch_size == 50
        assert config.flush_interval == 20
        assert config.url_timeout == 60.0
        assert config.max_concurrent_requests == 20
        assert config.verify_ssl is False
        assert config.ai_enabled is False
        assert config.max_description_length == 200
        assert config.target_tag_count == 200
        assert config.max_tags_per_bookmark == 10
        assert config.use_state_tracker is False
        assert config.state_db_path == tmp_path / "state.db"
        assert config.checkpoint_interval == 100
        assert config.checkpoint_dir == "custom_checkpoints"
        assert config.progress_callback is callback
        assert config.verbose is True

    def test_config_with_path_objects(self, tmp_path):
        """Test that config accepts Path objects for file paths."""
        config = StreamingPipelineConfig(
            input_file=Path(tmp_path) / "input.csv",
            output_file=Path(tmp_path) / "output.csv",
        )

        assert isinstance(config.input_file, Path)
        assert isinstance(config.output_file, Path)

    def test_config_with_string_paths(self, tmp_path):
        """Test that config accepts string paths."""
        config = StreamingPipelineConfig(
            input_file=str(tmp_path / "input.csv"),
            output_file=str(tmp_path / "output.csv"),
        )

        assert config.input_file == str(tmp_path / "input.csv")
        assert config.output_file == str(tmp_path / "output.csv")


# ============================================================================
# ProcessingStats Tests
# ============================================================================


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_stats_default_values(self):
        """Test ProcessingStats has correct default values."""
        stats = ProcessingStats()

        assert stats.total_read == 0
        assert stats.total_processed == 0
        assert stats.total_written == 0
        assert stats.total_skipped == 0
        assert stats.total_errors == 0
        assert stats.validation_success == 0
        assert stats.validation_failed == 0
        assert stats.content_extracted == 0
        assert stats.ai_processed == 0
        assert stats.tags_generated == 0
        assert stats.start_time is None
        assert stats.end_time is None
        assert stats.errors == []

    def test_processing_time_with_start_and_end(self):
        """Test processing_time calculation with both times set."""
        stats = ProcessingStats()
        stats.start_time = datetime(2024, 1, 1, 0, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 0, 5, 30)  # 5 minutes 30 seconds

        assert stats.processing_time == timedelta(minutes=5, seconds=30)
        assert stats.processing_time.total_seconds() == 330

    def test_processing_time_with_only_start(self):
        """Test processing_time when only start_time is set."""
        stats = ProcessingStats()
        stats.start_time = datetime.now() - timedelta(seconds=10)

        # Should return time since start
        assert stats.processing_time.total_seconds() >= 10

    def test_processing_time_with_no_times(self):
        """Test processing_time when no times are set."""
        stats = ProcessingStats()

        assert stats.processing_time == timedelta(0)

    def test_success_rate_normal(self):
        """Test success_rate calculation with normal values."""
        stats = ProcessingStats()
        stats.total_read = 100
        stats.total_processed = 80

        assert stats.success_rate == 80.0

    def test_success_rate_zero_total(self):
        """Test success_rate when total_read is zero."""
        stats = ProcessingStats()
        stats.total_read = 0
        stats.total_processed = 0

        assert stats.success_rate == 0.0

    def test_success_rate_all_processed(self):
        """Test success_rate when all items processed."""
        stats = ProcessingStats()
        stats.total_read = 50
        stats.total_processed = 50

        assert stats.success_rate == 100.0

    def test_throughput_normal(self):
        """Test throughput calculation with normal values."""
        stats = ProcessingStats()
        stats.total_processed = 100
        stats.start_time = datetime(2024, 1, 1, 0, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 0, 0, 10)  # 10 seconds

        assert stats.throughput == 10.0  # 100 items / 10 seconds

    def test_throughput_zero_time(self):
        """Test throughput when processing time is zero."""
        stats = ProcessingStats()
        stats.total_processed = 100
        # No time set, so processing_time is zero

        assert stats.throughput == 0.0

    def test_throughput_fast_processing(self):
        """Test throughput with fast processing."""
        stats = ProcessingStats()
        stats.total_processed = 1000
        stats.start_time = datetime(2024, 1, 1, 0, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 0, 0, 1)  # 1 second

        assert stats.throughput == 1000.0

    def test_to_dict(self):
        """Test to_dict conversion."""
        stats = ProcessingStats()
        stats.total_read = 100
        stats.total_processed = 80
        stats.total_written = 75
        stats.total_skipped = 10
        stats.total_errors = 5
        stats.validation_success = 75
        stats.validation_failed = 25
        stats.content_extracted = 70
        stats.ai_processed = 65
        stats.tags_generated = 60
        stats.start_time = datetime(2024, 1, 1, 0, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 0, 1, 0)  # 60 seconds
        stats.errors = [{"error": "test error"}]

        result = stats.to_dict()

        assert result["total_read"] == 100
        assert result["total_processed"] == 80
        assert result["total_written"] == 75
        assert result["total_skipped"] == 10
        assert result["total_errors"] == 5
        assert result["validation_success"] == 75
        assert result["validation_failed"] == 25
        assert result["content_extracted"] == 70
        assert result["ai_processed"] == 65
        assert result["tags_generated"] == 60
        assert result["processing_time_seconds"] == 60.0
        assert result["success_rate"] == 80.0
        assert result["throughput"] == pytest.approx(80 / 60, rel=0.01)
        assert result["error_count"] == 1

    def test_errors_list_mutability(self):
        """Test that errors list is properly initialized and mutable."""
        stats = ProcessingStats()
        stats.errors.append({"url": "https://test.com", "error": "test"})

        assert len(stats.errors) == 1
        assert stats.errors[0]["url"] == "https://test.com"


# ============================================================================
# StreamingPipelineResults Tests
# ============================================================================


class TestStreamingPipelineResults:
    """Tests for StreamingPipelineResults dataclass."""

    def test_results_properties(self, basic_config):
        """Test results property accessors."""
        stats = ProcessingStats()
        stats.total_read = 100
        stats.validation_success = 80
        stats.validation_failed = 20
        stats.total_processed = 75
        stats.start_time = datetime(2024, 1, 1, 0, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 0, 1, 0)

        results = StreamingPipelineResults(
            stats=stats,
            config=basic_config,
            completed=True,
            error_message=None,
        )

        assert results.total_bookmarks == 100
        assert results.valid_bookmarks == 80
        assert results.invalid_bookmarks == 20
        assert results.processed_bookmarks == 75
        assert results.processing_time == 60.0

    def test_results_with_error(self, basic_config):
        """Test results with error message."""
        stats = ProcessingStats()

        results = StreamingPipelineResults(
            stats=stats,
            config=basic_config,
            completed=False,
            error_message="Test error occurred",
        )

        assert results.completed is False
        assert results.error_message == "Test error occurred"

    def test_results_to_dict(self, basic_config):
        """Test results to_dict conversion."""
        stats = ProcessingStats()
        stats.total_read = 50
        stats.total_processed = 45

        results = StreamingPipelineResults(
            stats=stats,
            config=basic_config,
            completed=True,
        )

        result_dict = results.to_dict()

        assert result_dict["completed"] is True
        assert result_dict["error_message"] is None
        assert "input_file" in result_dict
        assert "output_file" in result_dict
        assert "stats" in result_dict
        assert result_dict["stats"]["total_read"] == 50

    def test_results_default_values(self, basic_config):
        """Test results with default values."""
        stats = ProcessingStats()

        results = StreamingPipelineResults(
            stats=stats,
            config=basic_config,
        )

        assert results.completed is False
        assert results.error_message is None


# ============================================================================
# StreamingPipeline Initialization Tests
# ============================================================================


class TestStreamingPipelineInit:
    """Tests for StreamingPipeline initialization."""

    def test_init_basic(self, basic_config):
        """Test basic initialization."""
        pipeline = StreamingPipeline(basic_config)

        assert pipeline.config == basic_config
        assert pipeline._url_validator is None
        assert pipeline._content_analyzer is None
        assert pipeline._ai_processor is None
        assert pipeline._tag_generator is None
        assert pipeline._state_tracker is None

    def test_init_with_components(self, basic_config, mock_url_validator,
                                   mock_content_analyzer, mock_ai_processor,
                                   mock_tag_generator, mock_state_tracker):
        """Test initialization with injected components."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            content_analyzer=mock_content_analyzer,
            ai_processor=mock_ai_processor,
            tag_generator=mock_tag_generator,
            state_tracker=mock_state_tracker,
        )

        assert pipeline._url_validator is mock_url_validator
        assert pipeline._content_analyzer is mock_content_analyzer
        assert pipeline._ai_processor is mock_ai_processor
        assert pipeline._tag_generator is mock_tag_generator
        assert pipeline._state_tracker is mock_state_tracker

    def test_init_stats_empty(self, basic_config):
        """Test that stats are initialized to empty on init."""
        pipeline = StreamingPipeline(basic_config)

        assert pipeline.stats.total_read == 0
        assert pipeline.stats.total_processed == 0

    def test_repr(self, basic_config):
        """Test string representation."""
        pipeline = StreamingPipeline(basic_config)
        repr_str = repr(pipeline)

        assert "StreamingPipeline" in repr_str
        assert "input=" in repr_str
        assert "output=" in repr_str


# ============================================================================
# StreamingPipeline Lazy Initialization Tests
# ============================================================================


class TestStreamingPipelineLazyInit:
    """Tests for lazy component initialization."""

    @patch('bookmark_processor.core.url_validator.URLValidator')
    def test_get_url_validator_lazy_init(self, mock_url_validator_class, basic_config):
        """Test lazy initialization of URL validator."""
        pipeline = StreamingPipeline(basic_config)

        # First call should create validator
        validator = pipeline._get_url_validator()

        mock_url_validator_class.assert_called_once_with(
            timeout=basic_config.url_timeout,
            max_concurrent=basic_config.max_concurrent_requests,
            verify_ssl=basic_config.verify_ssl,
        )
        assert validator is not None

    @patch('bookmark_processor.core.url_validator.URLValidator')
    def test_get_url_validator_cached(self, mock_url_validator_class, basic_config):
        """Test that URL validator is cached after first call."""
        pipeline = StreamingPipeline(basic_config)

        # Call twice
        validator1 = pipeline._get_url_validator()
        validator2 = pipeline._get_url_validator()

        # Should only create once
        assert mock_url_validator_class.call_count == 1
        assert validator1 is validator2

    def test_get_url_validator_pre_injected(self, basic_config, mock_url_validator):
        """Test that injected validator is used."""
        pipeline = StreamingPipeline(basic_config, url_validator=mock_url_validator)

        validator = pipeline._get_url_validator()

        assert validator is mock_url_validator

    @patch('bookmark_processor.core.content_analyzer.ContentAnalyzer')
    def test_get_content_analyzer_lazy_init(self, mock_analyzer_class, basic_config):
        """Test lazy initialization of content analyzer."""
        pipeline = StreamingPipeline(basic_config)

        analyzer = pipeline._get_content_analyzer()

        mock_analyzer_class.assert_called_once_with(
            timeout=basic_config.url_timeout
        )
        assert analyzer is not None

    @patch('bookmark_processor.core.ai_processor.EnhancedAIProcessor')
    def test_get_ai_processor_lazy_init(self, mock_processor_class, tmp_path):
        """Test lazy initialization of AI processor."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            ai_enabled=True,
            max_description_length=200,
        )
        pipeline = StreamingPipeline(config)

        processor = pipeline._get_ai_processor()

        mock_processor_class.assert_called_once_with(
            max_description_length=200
        )
        assert processor is not None

    def test_get_ai_processor_disabled(self, basic_config):
        """Test AI processor returns None when disabled."""
        # basic_config has ai_enabled=False
        pipeline = StreamingPipeline(basic_config)

        processor = pipeline._get_ai_processor()

        assert processor is None

    @patch('bookmark_processor.core.tag_generator.CorpusAwareTagGenerator')
    def test_get_tag_generator_lazy_init(self, mock_generator_class, basic_config):
        """Test lazy initialization of tag generator."""
        pipeline = StreamingPipeline(basic_config)

        generator = pipeline._get_tag_generator()

        mock_generator_class.assert_called_once_with(
            target_tag_count=basic_config.target_tag_count,
            max_tags_per_bookmark=basic_config.max_tags_per_bookmark,
        )
        assert generator is not None

    @patch('bookmark_processor.core.streaming.pipeline.ProcessingStateTracker')
    def test_get_state_tracker_lazy_init(self, mock_tracker_class, tmp_path):
        """Test lazy initialization of state tracker."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            use_state_tracker=True,
            state_db_path=tmp_path / "state.db",
        )
        pipeline = StreamingPipeline(config)

        tracker = pipeline._get_state_tracker()

        mock_tracker_class.assert_called_once()
        assert tracker is not None

    @patch('bookmark_processor.core.streaming.pipeline.ProcessingStateTracker')
    def test_get_state_tracker_default_path(self, mock_tracker_class, tmp_path):
        """Test state tracker with default path."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            use_state_tracker=True,
            state_db_path=None,
        )
        pipeline = StreamingPipeline(config)

        tracker = pipeline._get_state_tracker()

        mock_tracker_class.assert_called_once_with(
            db_path=".bookmark_processor_state.db"
        )

    def test_get_state_tracker_disabled(self, basic_config):
        """Test state tracker returns None when disabled."""
        # basic_config has use_state_tracker=False
        pipeline = StreamingPipeline(basic_config)

        tracker = pipeline._get_state_tracker()

        assert tracker is None


# ============================================================================
# StreamingPipeline Processing Stage Tests
# ============================================================================


class TestStreamingPipelineValidateUrl:
    """Tests for URL validation stage."""

    def test_validate_url_success(self, basic_config, mock_url_validator):
        """Test successful URL validation."""
        pipeline = StreamingPipeline(basic_config, url_validator=mock_url_validator)
        mock_url_validator.validate_url.return_value = MagicMock(
            is_valid=True,
            error_message=None
        )

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._validate_url(bookmark)

        assert result is True
        assert bookmark.processing_status.url_validated is True
        assert bookmark.processing_status.url_validation_error is None

    def test_validate_url_failure(self, basic_config, mock_url_validator):
        """Test failed URL validation."""
        pipeline = StreamingPipeline(basic_config, url_validator=mock_url_validator)
        mock_url_validator.validate_url.return_value = MagicMock(
            is_valid=False,
            error_message="Connection timeout"
        )

        bookmark = Bookmark(url="https://invalid.example.com", title="Test")
        result = pipeline._validate_url(bookmark)

        assert result is False
        assert bookmark.processing_status.url_validated is True
        assert bookmark.processing_status.url_validation_error == "Connection timeout"

    def test_validate_url_empty(self, basic_config, mock_url_validator):
        """Test validation with empty URL."""
        pipeline = StreamingPipeline(basic_config, url_validator=mock_url_validator)

        bookmark = Bookmark(url="", title="Test")
        result = pipeline._validate_url(bookmark)

        assert result is False

    def test_validate_url_none(self, basic_config, mock_url_validator):
        """Test validation with None URL."""
        pipeline = StreamingPipeline(basic_config, url_validator=mock_url_validator)

        bookmark = Bookmark(url=None, title="Test")
        result = pipeline._validate_url(bookmark)

        assert result is False

    def test_validate_url_no_validator(self, basic_config):
        """Test validation when no validator is available."""
        pipeline = StreamingPipeline(basic_config)
        pipeline._url_validator = None
        # Patch to return None
        pipeline._get_url_validator = MagicMock(return_value=None)

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._validate_url(bookmark)

        # Should assume valid if no validator
        assert result is True

    def test_validate_url_exception(self, basic_config, mock_url_validator):
        """Test validation when exception occurs."""
        pipeline = StreamingPipeline(basic_config, url_validator=mock_url_validator)
        mock_url_validator.validate_url.side_effect = Exception("Network error")

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._validate_url(bookmark)

        assert result is False
        assert bookmark.processing_status.url_validation_error == "Network error"


class TestStreamingPipelineAnalyzeContent:
    """Tests for content analysis stage."""

    def test_analyze_content_success(self, basic_config, mock_content_analyzer):
        """Test successful content analysis."""
        pipeline = StreamingPipeline(basic_config, content_analyzer=mock_content_analyzer)
        mock_content_analyzer.analyze_content.return_value = {
            "content": "Sample content",
            "title": "Page Title",
        }

        bookmark = Bookmark(
            url="https://example.com",
            title="Original Title",
            note="Original Note",
            excerpt="Original Excerpt",
        )
        result = pipeline._analyze_content(bookmark)

        assert result is not None
        assert result["content"] == "Sample content"
        assert bookmark.processing_status.content_extracted is True

    def test_analyze_content_no_analyzer(self, basic_config):
        """Test content analysis when no analyzer is available."""
        pipeline = StreamingPipeline(basic_config)
        pipeline._get_content_analyzer = MagicMock(return_value=None)

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._analyze_content(bookmark)

        assert result is None

    def test_analyze_content_exception(self, basic_config, mock_content_analyzer):
        """Test content analysis when exception occurs."""
        pipeline = StreamingPipeline(basic_config, content_analyzer=mock_content_analyzer)
        mock_content_analyzer.analyze_content.side_effect = Exception("Parse error")

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._analyze_content(bookmark)

        assert result is None
        assert bookmark.processing_status.content_extraction_error == "Parse error"


class TestStreamingPipelineProcessAI:
    """Tests for AI processing stage."""

    def test_process_ai_success(self, tmp_path, mock_ai_processor):
        """Test successful AI processing."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            ai_enabled=True,
        )
        pipeline = StreamingPipeline(config, ai_processor=mock_ai_processor)

        mock_result = MagicMock()
        mock_result.enhanced_description = "AI enhanced description"
        mock_ai_processor.process_single.return_value = mock_result

        bookmark = Bookmark(url="https://example.com", title="Test")
        content_data = {"content": "Sample content text"}

        result = pipeline._process_ai(bookmark, content_data)

        assert result == "AI enhanced description"
        assert bookmark.enhanced_description == "AI enhanced description"
        assert bookmark.processing_status.ai_processed is True

    def test_process_ai_no_content(self, tmp_path, mock_ai_processor):
        """Test AI processing without content data."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            ai_enabled=True,
        )
        pipeline = StreamingPipeline(config, ai_processor=mock_ai_processor)

        mock_result = MagicMock()
        mock_result.enhanced_description = "Description from title"
        mock_ai_processor.process_single.return_value = mock_result

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._process_ai(bookmark, None)

        assert result == "Description from title"

    def test_process_ai_no_processor(self, basic_config):
        """Test AI processing when processor is not available."""
        pipeline = StreamingPipeline(basic_config)  # ai_enabled=False

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._process_ai(bookmark, {"content": "test"})

        assert result is None

    def test_process_ai_exception(self, tmp_path, mock_ai_processor):
        """Test AI processing when exception occurs."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            ai_enabled=True,
        )
        pipeline = StreamingPipeline(config, ai_processor=mock_ai_processor)
        mock_ai_processor.process_single.side_effect = Exception("AI error")

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._process_ai(bookmark, {"content": "test"})

        assert result is None
        assert bookmark.processing_status.ai_processing_error == "AI error"

    def test_process_ai_no_result(self, tmp_path, mock_ai_processor):
        """Test AI processing when result has no description."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            ai_enabled=True,
        )
        pipeline = StreamingPipeline(config, ai_processor=mock_ai_processor)

        mock_result = MagicMock()
        mock_result.enhanced_description = None
        mock_ai_processor.process_single.return_value = mock_result

        bookmark = Bookmark(url="https://example.com", title="Test")
        result = pipeline._process_ai(bookmark, {"content": "test"})

        assert result is None


class TestStreamingPipelineGenerateTags:
    """Tests for tag generation stage."""

    def test_generate_tags_success(self, basic_config, mock_tag_generator):
        """Test successful tag generation."""
        pipeline = StreamingPipeline(basic_config, tag_generator=mock_tag_generator)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["ai", "python", "web"]

        bookmark = Bookmark(url="https://example.com", title="Test", tags=["original"])
        content_data = {"content": "Sample content"}

        result = pipeline._generate_tags(bookmark, content_data)

        assert result == ["ai", "python", "web"]
        assert bookmark.processing_status.tags_optimized is True

    def test_generate_tags_no_generator(self, basic_config):
        """Test tag generation when no generator is available."""
        pipeline = StreamingPipeline(basic_config)
        pipeline._get_tag_generator = MagicMock(return_value=None)

        bookmark = Bookmark(url="https://example.com", title="Test", tags=["original", "tags"])
        result = pipeline._generate_tags(bookmark, None)

        # Should return original tags
        assert result == ["original", "tags"]

    def test_generate_tags_exception(self, basic_config, mock_tag_generator):
        """Test tag generation when exception occurs."""
        pipeline = StreamingPipeline(basic_config, tag_generator=mock_tag_generator)
        mock_tag_generator.generate_for_single_bookmark.side_effect = Exception("Tag error")

        bookmark = Bookmark(url="https://example.com", title="Test", tags=["original"])
        result = pipeline._generate_tags(bookmark, {"content": "test"})

        # Should return original tags on error
        assert result == ["original"]


# ============================================================================
# StreamingPipeline Batch Processing Tests
# ============================================================================


class TestStreamingPipelineProcessBatch:
    """Tests for batch processing."""

    def test_process_batch_empty(self, basic_config, mock_url_validator):
        """Test processing empty batch."""
        pipeline = StreamingPipeline(basic_config, url_validator=mock_url_validator)

        result = pipeline._process_batch([])

        assert result == []

    def test_process_batch_all_valid(self, basic_config, mock_url_validator,
                                      mock_tag_generator, sample_bookmarks):
        """Test processing batch where all bookmarks are valid."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag1"]

        result = pipeline._process_batch(sample_bookmarks)

        assert len(result) == 3
        assert pipeline.stats.validation_success == 3
        assert pipeline.stats.validation_failed == 0
        assert pipeline.stats.total_processed == 3

    def test_process_batch_some_invalid(self, basic_config, mock_url_validator,
                                         mock_tag_generator, sample_bookmarks):
        """Test processing batch where some bookmarks fail validation."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        # First two valid, third invalid
        mock_url_validator.validate_url.side_effect = [
            MagicMock(is_valid=True),
            MagicMock(is_valid=False, error_message="Timeout"),
            MagicMock(is_valid=True),
        ]
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag1"]

        result = pipeline._process_batch(sample_bookmarks)

        assert len(result) == 2  # Only 2 passed validation
        assert pipeline.stats.validation_success == 2
        assert pipeline.stats.validation_failed == 1

    def test_process_batch_with_content_extraction(self, basic_config, mock_url_validator,
                                                    mock_content_analyzer, mock_tag_generator,
                                                    sample_bookmarks):
        """Test batch processing with content extraction."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            content_analyzer=mock_content_analyzer,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_content_analyzer.analyze_content.return_value = {"content": "extracted"}
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag1"]

        result = pipeline._process_batch(sample_bookmarks)

        assert len(result) == 3
        assert pipeline.stats.content_extracted == 3

    def test_process_batch_with_ai(self, tmp_path, mock_url_validator,
                                    mock_ai_processor, mock_content_analyzer,
                                    mock_tag_generator, sample_bookmarks):
        """Test batch processing with AI enhancement."""
        config = StreamingPipelineConfig(
            input_file=tmp_path / "input.csv",
            output_file=tmp_path / "output.csv",
            ai_enabled=True,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            content_analyzer=mock_content_analyzer,
            ai_processor=mock_ai_processor,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_content_analyzer.analyze_content.return_value = {"content": "test content"}
        mock_result = MagicMock()
        mock_result.enhanced_description = "Enhanced"
        mock_ai_processor.process_single.return_value = mock_result
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag1"]

        result = pipeline._process_batch(sample_bookmarks)

        assert len(result) == 3
        assert pipeline.stats.ai_processed == 3

    def test_process_batch_with_errors(self, basic_config, mock_url_validator,
                                        mock_tag_generator, sample_bookmarks):
        """Test batch processing when errors occur during processing."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        # First succeeds, second throws exception (caught in _validate_url, returns False), third succeeds
        # Exceptions in _validate_url are caught and return False, incrementing validation_failed
        mock_url_validator.validate_url.side_effect = [
            MagicMock(is_valid=True),
            Exception("Unexpected error"),  # This causes validation_failed, not total_errors
            MagicMock(is_valid=True),
        ]
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag1"]

        result = pipeline._process_batch(sample_bookmarks)

        assert len(result) == 2  # Only 2 processed successfully
        # Note: Exceptions in _validate_url are caught internally, so they count as validation_failed
        assert pipeline.stats.validation_failed == 1
        assert pipeline.stats.validation_success == 2

    def test_process_batch_tag_generation_error_graceful(self, basic_config, mock_url_validator,
                                                          mock_tag_generator, sample_bookmarks):
        """Test batch processing handles tag generation errors gracefully."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        # Tag generator throws exception which is caught by _generate_tags
        # and falls back to original tags
        mock_tag_generator.generate_for_single_bookmark.side_effect = [
            ["tag1"],
            Exception("Tag generation error"),  # Caught internally, returns original tags
            ["tag1"],
        ]

        result = pipeline._process_batch(sample_bookmarks)

        # All 3 succeed because tag generation errors are caught gracefully
        assert len(result) == 3
        assert pipeline.stats.validation_success == 3
        # Second bookmark should have its original tags (ai, ml)
        assert result[1].tags == ["ai", "ml"]

    def test_process_batch_tags_assigned(self, basic_config, mock_url_validator,
                                          mock_tag_generator, sample_bookmarks):
        """Test that generated tags are assigned to bookmarks."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["new", "tags"]

        result = pipeline._process_batch(sample_bookmarks)

        for bookmark in result:
            assert bookmark.optimized_tags == ["new", "tags"]
        assert pipeline.stats.tags_generated == 3


# ============================================================================
# StreamingPipeline Execute Tests
# ============================================================================


class TestStreamingPipelineExecute:
    """Tests for pipeline execution."""

    def test_execute_basic(self, sample_input_file, tmp_path, mock_url_validator,
                           mock_tag_generator):
        """Test basic pipeline execution."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag1"]

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 5
        assert results.stats.total_processed == 5
        assert (tmp_path / "output.csv").exists()

    def test_execute_creates_output_file(self, sample_input_file, tmp_path,
                                          mock_url_validator, mock_tag_generator):
        """Test that execute creates output file."""
        output_path = tmp_path / "subdir" / "output.csv"
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=output_path,
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = []

        results = pipeline.execute()

        assert output_path.exists()


class TestStreamingPipelineExecuteStreaming:
    """Tests for execute_streaming method."""

    def test_execute_streaming_basic(self, sample_input_file, tmp_path,
                                      mock_url_validator, mock_tag_generator):
        """Test execute_streaming with reader and writer."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=2,
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag"]

        reader = StreamingBookmarkReader(sample_input_file)
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        with writer:
            results = pipeline.execute_streaming(reader, writer)

        assert results.completed is True
        assert results.stats.total_read == 5
        assert results.stats.total_processed == 5

    def test_execute_streaming_with_state_tracker(self, sample_input_file, tmp_path,
                                                   mock_url_validator, mock_tag_generator,
                                                   mock_state_tracker):
        """Test execute_streaming with state tracker."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=5,
            ai_enabled=False,
            use_state_tracker=True,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
            state_tracker=mock_state_tracker,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag"]
        # Return all bookmarks as unprocessed
        mock_state_tracker.get_unprocessed.side_effect = lambda x: x

        reader = StreamingBookmarkReader(sample_input_file)
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        with writer:
            results = pipeline.execute_streaming(reader, writer)

        assert results.completed is True
        mock_state_tracker.start_processing_run.assert_called_once()
        mock_state_tracker.complete_processing_run.assert_called_once()

    def test_execute_streaming_skips_processed(self, sample_input_file, tmp_path,
                                                mock_url_validator, mock_tag_generator,
                                                mock_state_tracker):
        """Test that already processed bookmarks are skipped."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=5,
            ai_enabled=False,
            use_state_tracker=True,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
            state_tracker=mock_state_tracker,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag"]
        # Return only 2 bookmarks as unprocessed (skip 3)
        mock_state_tracker.get_unprocessed.side_effect = lambda x: x[:2] if len(x) >= 2 else x

        reader = StreamingBookmarkReader(sample_input_file)
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        with writer:
            results = pipeline.execute_streaming(reader, writer)

        assert results.completed is True
        assert results.stats.total_skipped == 3

    def test_execute_streaming_with_progress_callback(self, sample_input_file, tmp_path,
                                                       mock_url_validator, mock_tag_generator):
        """Test execute_streaming with progress callback."""
        callback = MagicMock()
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=2,
            ai_enabled=False,
            use_state_tracker=False,
            progress_callback=callback,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag"]

        reader = StreamingBookmarkReader(sample_input_file)
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        with writer:
            results = pipeline.execute_streaming(reader, writer)

        # Callback should be called for each batch
        assert callback.call_count >= 1

    def test_execute_streaming_exception_handling(self, sample_input_file, tmp_path):
        """Test execute_streaming handles exceptions gracefully."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(config)

        # Mock _process_batch to raise exception
        pipeline._process_batch = MagicMock(side_effect=Exception("Critical error"))

        reader = StreamingBookmarkReader(sample_input_file)
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        with writer:
            results = pipeline.execute_streaming(reader, writer)

        assert results.completed is False
        assert "Critical error" in results.error_message

    def test_execute_streaming_timing(self, sample_input_file, tmp_path,
                                       mock_url_validator, mock_tag_generator):
        """Test that timing is recorded correctly."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = []

        reader = StreamingBookmarkReader(sample_input_file)
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        with writer:
            results = pipeline.execute_streaming(reader, writer)

        assert results.stats.start_time is not None
        assert results.stats.end_time is not None
        assert results.stats.end_time >= results.stats.start_time


class TestStreamingPipelineLargeDataset:
    """Tests for processing large datasets."""

    def test_execute_large_dataset(self, large_input_file, tmp_path,
                                    mock_url_validator, mock_tag_generator):
        """Test processing large dataset in batches."""
        config = StreamingPipelineConfig(
            input_file=large_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=50,  # Process in batches of 50
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag"]

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 350
        assert results.stats.total_processed == 350

    def test_checkpoint_interval(self, large_input_file, tmp_path,
                                  mock_url_validator, mock_tag_generator):
        """Test that checkpoints are saved at intervals."""
        config = StreamingPipelineConfig(
            input_file=large_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=50,
            checkpoint_interval=50,
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = []

        # Mock checkpoint saving
        pipeline._save_checkpoint = MagicMock()

        results = pipeline.execute()

        assert results.completed is True
        # Checkpoints should be called multiple times
        assert pipeline._save_checkpoint.call_count >= 1


# ============================================================================
# StreamingPipeline Utility Method Tests
# ============================================================================


class TestStreamingPipelineUtilities:
    """Tests for utility methods."""

    def test_get_statistics(self, basic_config):
        """Test get_statistics method."""
        pipeline = StreamingPipeline(basic_config)
        pipeline.stats.total_read = 100
        pipeline.stats.total_processed = 90
        pipeline.stats.validation_success = 85

        stats = pipeline.get_statistics()

        assert stats["total_read"] == 100
        assert stats["total_processed"] == 90
        assert stats["validation_success"] == 85

    def test_close_with_components(self, basic_config, mock_url_validator,
                                    mock_content_analyzer, mock_ai_processor):
        """Test close method cleans up components."""
        pipeline = StreamingPipeline(
            basic_config,
            url_validator=mock_url_validator,
            content_analyzer=mock_content_analyzer,
            ai_processor=mock_ai_processor,
        )

        pipeline.close()

        mock_url_validator.close.assert_called_once()
        mock_content_analyzer.close.assert_called_once()
        mock_ai_processor.close.assert_called_once()

    def test_close_without_components(self, basic_config):
        """Test close method when no components are initialized."""
        pipeline = StreamingPipeline(basic_config)

        # Should not raise
        pipeline.close()

    def test_close_components_without_close_method(self, basic_config):
        """Test close handles components without close method."""
        mock_validator = MagicMock(spec=[])  # No close method
        pipeline = StreamingPipeline(basic_config, url_validator=mock_validator)

        # Should not raise
        pipeline.close()

    def test_save_checkpoint(self, basic_config):
        """Test _save_checkpoint method."""
        pipeline = StreamingPipeline(basic_config)
        pipeline.stats.total_processed = 100

        # Should not raise (currently just logs)
        pipeline._save_checkpoint()


# ============================================================================
# Integration Tests
# ============================================================================


class TestStreamingPipelineIntegration:
    """Integration tests for complete pipeline flow."""

    def test_full_pipeline_flow(self, sample_input_file, tmp_path,
                                 mock_url_validator, mock_content_analyzer,
                                 mock_ai_processor, mock_tag_generator):
        """Test complete pipeline flow with all components."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=2,
            ai_enabled=True,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            content_analyzer=mock_content_analyzer,
            ai_processor=mock_ai_processor,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_content_analyzer.analyze_content.return_value = {"content": "test"}
        mock_result = MagicMock()
        mock_result.enhanced_description = "Enhanced"
        mock_ai_processor.process_single.return_value = mock_result
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag1", "tag2"]

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 5
        assert results.stats.validation_success == 5
        assert results.stats.content_extracted == 5
        assert results.stats.ai_processed == 5
        assert results.stats.tags_generated == 5

        # Verify output file
        output_path = tmp_path / "output.csv"
        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5

    def test_pipeline_partial_failures(self, sample_input_file, tmp_path,
                                        mock_url_validator, mock_tag_generator):
        """Test pipeline handles partial failures gracefully."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=5,
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        # Make some validations fail
        mock_url_validator.validate_url.side_effect = [
            MagicMock(is_valid=True),
            MagicMock(is_valid=False, error_message="404"),
            MagicMock(is_valid=True),
            MagicMock(is_valid=False, error_message="Timeout"),
            MagicMock(is_valid=True),
        ]
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag"]

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 5
        assert results.stats.validation_success == 3
        assert results.stats.validation_failed == 2
        assert results.stats.total_processed == 3

    def test_pipeline_with_mixed_content(self, tmp_path, mock_url_validator,
                                          mock_tag_generator):
        """Test pipeline with varied bookmark content."""
        # Create CSV with varied content
        csv_file = tmp_path / "varied_input.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows([
                ["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"],
                ["1", "Complete", "Full note", "Full excerpt", "https://complete.com", "Complete", "a, b, c", "2024-01-01T00:00:00Z", "cover.jpg", "highlight", "true"],
                ["2", "Minimal", "", "", "https://minimal.com", "", "", "", "", "", "false"],
                ["3", "Unicode Title", "Unicode note", "", "https://unicode.com", "", "", "", "", "", "false"],
                ["4", "Special <>&", "Note with <special> chars", "", "https://special.com", "", "", "", "", "", "false"],
            ])

        config = StreamingPipelineConfig(
            input_file=csv_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = []

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 4


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestStreamingPipelineEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_input_file(self, tmp_path, mock_url_validator):
        """Test handling of empty input file."""
        csv_file = tmp_path / "empty.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"])

        config = StreamingPipelineConfig(
            input_file=csv_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(config, url_validator=mock_url_validator)

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 0
        assert results.stats.total_processed == 0

    def test_single_bookmark(self, tmp_path, mock_url_validator, mock_tag_generator):
        """Test processing single bookmark."""
        csv_file = tmp_path / "single.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows([
                ["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"],
                ["1", "Single", "Note", "", "https://single.com", "Folder", "tag", "2024-01-01T00:00:00Z", "", "", "false"],
            ])

        config = StreamingPipelineConfig(
            input_file=csv_file,
            output_file=tmp_path / "output.csv",
            batch_size=100,  # Larger than data
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = ["tag"]

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 1
        assert results.stats.total_processed == 1

    def test_batch_size_larger_than_data(self, sample_input_file, tmp_path,
                                          mock_url_validator, mock_tag_generator):
        """Test when batch size is larger than total data."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            batch_size=1000,  # Much larger than 5 bookmarks
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = []

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 5

    def test_all_bookmarks_fail_validation(self, sample_input_file, tmp_path,
                                            mock_url_validator, mock_tag_generator):
        """Test when all bookmarks fail validation."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(
            is_valid=False,
            error_message="All fail"
        )

        results = pipeline.execute()

        assert results.completed is True
        assert results.stats.total_read == 5
        assert results.stats.validation_failed == 5
        assert results.stats.total_processed == 0
        assert results.stats.total_written == 0

    def test_stats_reset_on_execute(self, sample_input_file, tmp_path,
                                     mock_url_validator, mock_tag_generator):
        """Test that stats are reset on each execute call."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        mock_url_validator.validate_url.return_value = MagicMock(is_valid=True)
        mock_tag_generator.generate_for_single_bookmark.return_value = []

        # First execution
        results1 = pipeline.execute()

        # Second execution - stats should be fresh
        results2 = pipeline.execute()

        # Both should have same counts (data didn't change)
        assert results1.stats.total_read == results2.stats.total_read
        assert results1.stats.total_processed == results2.stats.total_processed


# ============================================================================
# Logging Tests
# ============================================================================


class TestStreamingPipelineLogging:
    """Tests for logging behavior."""

    def test_logging_setup(self, basic_config, caplog):
        """Test that pipeline sets up logging correctly."""
        with caplog.at_level(logging.DEBUG):
            pipeline = StreamingPipeline(basic_config)

            assert pipeline.logger is not None
            assert pipeline.logger.name == "bookmark_processor.core.streaming.pipeline"

    def test_error_logging(self, sample_input_file, tmp_path, mock_url_validator,
                           mock_tag_generator, caplog):
        """Test that validation errors are handled appropriately."""
        config = StreamingPipelineConfig(
            input_file=sample_input_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False,
        )
        pipeline = StreamingPipeline(
            config,
            url_validator=mock_url_validator,
            tag_generator=mock_tag_generator,
        )
        # First succeeds, second has validation error exception (caught internally), rest succeed
        mock_url_validator.validate_url.side_effect = [
            MagicMock(is_valid=True),
            Exception("Validation error"),  # Caught in _validate_url, returns False
            MagicMock(is_valid=True),
            MagicMock(is_valid=True),
            MagicMock(is_valid=True),
        ]
        mock_tag_generator.generate_for_single_bookmark.return_value = []

        with caplog.at_level(logging.DEBUG):
            results = pipeline.execute()

        # Validation exception is caught internally and counts as validation_failed
        assert results.stats.validation_failed == 1
        assert results.stats.validation_success == 4
        assert results.stats.total_processed == 4
