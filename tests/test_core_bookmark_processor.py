"""
Unit tests for bookmark processor and pipeline components.

Tests for bookmark processor, batch processor, and processing pipeline.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.batch_processor import BatchProcessor
from bookmark_processor.core.bookmark_processor import BookmarkProcessor, ProcessingResults
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.pipeline import BookmarkProcessingPipeline
from bookmark_processor.core.pipeline.config import PipelineConfig, PipelineResults
from tests.fixtures.mock_utilities import (
    MockAIProcessor,
    MockContentAnalyzer,
    MockEnhancedAIProcessor,
    MockRequestsSession,
    create_mock_pipeline_context,
)
from tests.fixtures.test_data import (
    create_expected_import_dataframe,
    create_sample_bookmark_objects,
    create_sample_export_dataframe,
)


class TestBookmarkProcessor:
    """Test BookmarkProcessor class."""

    def test_init_with_config(self):
        """Test BookmarkProcessor initialization with config."""
        config = Configuration()
        processor = BookmarkProcessor(config)

        assert processor.config is not None
        assert processor.checkpoint_manager is None  # Not initialized until processing
        assert processor.progress_tracker is None

    def test_process_bookmarks_creates_pipeline(self, temp_dir):
        """Test that process_bookmarks creates and executes a pipeline."""
        config = Configuration()
        processor = BookmarkProcessor(config)

        # Create test input file
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        # Mock the pipeline to avoid actual processing
        with patch.object(BookmarkProcessingPipeline, 'execute') as mock_execute:
            mock_execute.return_value = PipelineResults(
                total_bookmarks=5,
                valid_bookmarks=4,
                invalid_bookmarks=1,
                ai_processed=3,
                tagged_bookmarks=4,
                unique_tags=10,
                processing_time=1.5,
                stages_completed=["validation", "ai_processing"],
                error_summary={},
                statistics={},
            )

            results = processor.process_bookmarks(
                input_file=input_file,
                output_file=output_file,
                resume=False,
            )

            assert isinstance(results, ProcessingResults)
            assert results.total_bookmarks == 5
            assert results.valid_bookmarks == 4

    def test_processing_results_container(self):
        """Test ProcessingResults container with PipelineResults."""
        pipeline_results = PipelineResults(
            total_bookmarks=100,
            valid_bookmarks=95,
            invalid_bookmarks=5,
            ai_processed=90,
            tagged_bookmarks=95,
            unique_tags=50,
            processing_time=120.5,
            stages_completed=["validation", "content_extraction", "ai_processing"],
            error_summary={"timeout": 3, "not_found": 2},
            statistics={"avg_time": 1.2},
        )

        results = ProcessingResults(pipeline_results)

        assert results.total_bookmarks == 100
        assert results.valid_bookmarks == 95
        assert results.invalid_bookmarks == 5
        assert results.ai_processed == 90
        assert results.processing_time == 120.5
        assert "validation" in results.stages_completed

    def test_processing_results_empty(self):
        """Test ProcessingResults with no pipeline results."""
        results = ProcessingResults()

        assert results.total_bookmarks == 0
        assert results.valid_bookmarks == 0
        assert results.processing_time == 0.0
        assert results.errors == []


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_init_with_ai_manager(self):
        """Test BatchProcessor initialization with AIManager."""
        mock_ai_manager = Mock()
        mock_ai_manager.get_current_provider.return_value = "local"

        processor = BatchProcessor(ai_manager=mock_ai_manager)

        assert processor.ai_manager == mock_ai_manager
        assert processor.cost_tracker is not None
        assert processor.verbose is False
        assert processor.max_concurrent == 10

    def test_init_custom_params(self):
        """Test BatchProcessor initialization with custom parameters."""
        mock_ai_manager = Mock()
        mock_cost_tracker = Mock()

        processor = BatchProcessor(
            ai_manager=mock_ai_manager,
            cost_tracker=mock_cost_tracker,
            verbose=True,
            max_concurrent=5,
        )

        assert processor.ai_manager == mock_ai_manager
        assert processor.cost_tracker == mock_cost_tracker
        assert processor.verbose is True
        assert processor.max_concurrent == 5

    def test_get_batch_size_by_provider(self):
        """Test batch size selection by provider."""
        mock_ai_manager = Mock()
        processor = BatchProcessor(ai_manager=mock_ai_manager)

        assert processor.get_batch_size("local") == 50
        assert processor.get_batch_size("claude") == 10
        assert processor.get_batch_size("openai") == 20
        assert processor.get_batch_size("unknown") == 10  # Default

    @pytest.mark.asyncio
    async def test_process_bookmarks_async(self):
        """Test async bookmark processing."""
        mock_ai_manager = Mock()
        mock_ai_manager.get_current_provider.return_value = "local"
        mock_ai_manager.generate_descriptions_batch = Mock(
            return_value=[
                ("Enhanced description", {"success": True}),
            ]
        )

        processor = BatchProcessor(ai_manager=mock_ai_manager, verbose=False)

        bookmarks = create_sample_bookmark_objects()[:2]

        # Since process_bookmarks is async, we need to mock it properly
        with patch.object(processor, '_process_batch') as mock_batch:
            mock_batch.return_value = [
                ("Enhanced description", {"success": True})
                for _ in bookmarks
            ]

            results, stats = await processor.process_bookmarks(bookmarks)

            assert stats["total_bookmarks"] == len(bookmarks)
            assert "provider" in stats

    def test_get_rate_limit_status(self):
        """Test rate limit status retrieval."""
        mock_ai_manager = Mock()
        mock_ai_manager.get_current_provider.return_value = "local"

        processor = BatchProcessor(ai_manager=mock_ai_manager)

        status = processor.get_rate_limit_status()

        assert status["provider"] == "local"
        assert status["status"] == "unlimited"

    def test_reset_session(self):
        """Test session reset."""
        mock_ai_manager = Mock()
        mock_ai_manager.get_current_provider.return_value = "local"
        mock_cost_tracker = Mock()

        processor = BatchProcessor(
            ai_manager=mock_ai_manager,
            cost_tracker=mock_cost_tracker,
        )

        # Set some state
        processor.processed_count = 100
        processor.failed_count = 5

        processor.reset_session()

        assert processor.processed_count == 0
        assert processor.failed_count == 0
        mock_cost_tracker.reset_session.assert_called_once()


class TestBookmarkProcessingPipeline:
    """Test BookmarkProcessingPipeline class."""

    def test_init_with_config(self, temp_dir):
        """Test BookmarkProcessingPipeline initialization with config."""
        config = PipelineConfig(
            input_file=str(temp_dir / "input.csv"),
            output_file=str(temp_dir / "output.csv"),
        )

        pipeline = BookmarkProcessingPipeline(config)

        assert pipeline.config == config
        assert pipeline.csv_handler is not None
        assert pipeline.url_validator is not None

    def test_init_custom_config(self, temp_dir):
        """Test BookmarkProcessingPipeline initialization with custom config."""
        config = PipelineConfig(
            input_file=str(temp_dir / "input.csv"),
            output_file=str(temp_dir / "output.csv"),
            batch_size=50,
            max_retries=5,
            verbose=True,
            detect_duplicates=False,
            generate_folders=False,
        )

        pipeline = BookmarkProcessingPipeline(config)

        assert pipeline.config.batch_size == 50
        assert pipeline.config.max_retries == 5
        assert pipeline.config.verbose is True
        assert pipeline.duplicate_detector is None  # Not created when disabled
        assert pipeline.folder_generator is None  # Not created when disabled

    def test_init_with_injected_dependencies(self, temp_dir):
        """Test pipeline initialization with dependency injection."""
        config = PipelineConfig(
            input_file=str(temp_dir / "input.csv"),
            output_file=str(temp_dir / "output.csv"),
        )

        mock_url_validator = Mock()
        mock_content_analyzer = Mock()

        pipeline = BookmarkProcessingPipeline(
            config,
            url_validator=mock_url_validator,
            content_analyzer=mock_content_analyzer,
        )

        assert pipeline.url_validator == mock_url_validator
        assert pipeline.content_analyzer == mock_content_analyzer

    def test_execute_pipeline(self, temp_dir):
        """Test pipeline execution setup."""
        # Setup mock CSV handler
        mock_handler = Mock()
        sample_df = create_sample_export_dataframe()
        mock_handler.read_raindrop_export.return_value = sample_df
        mock_handler.write_raindrop_import.return_value = True

        config = PipelineConfig(
            input_file=str(temp_dir / "input.csv"),
            output_file=str(temp_dir / "output.csv"),
            detect_duplicates=False,
            generate_folders=False,
            ai_enabled=False,
        )

        # Inject the mock CSV handler via constructor
        pipeline = BookmarkProcessingPipeline(config, csv_handler=mock_handler)

        # Mock URL validation
        pipeline.url_validator = Mock()
        pipeline.url_validator.validate_batch.return_value = [
            Mock(is_valid=True, status_code=200)
            for _ in range(len(sample_df))
        ]

        # Verify the pipeline is configured correctly
        assert pipeline.config.input_file == str(temp_dir / "input.csv")
        assert pipeline.config.output_file == str(temp_dir / "output.csv")
        assert pipeline.csv_handler == mock_handler
        assert pipeline.ai_processor is None  # ai_enabled=False


class TestProcessingIntegration:
    """Integration tests for processing components."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig(
            input_file="input.csv",
            output_file="output.csv",
        )

        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.resume_enabled is True
        assert config.url_timeout == 30.0
        assert config.ai_enabled is True
        assert config.detect_duplicates is True
        assert config.generate_folders is True

    def test_pipeline_results_structure(self):
        """Test PipelineResults data structure."""
        results = PipelineResults(
            total_bookmarks=100,
            valid_bookmarks=95,
            invalid_bookmarks=5,
            ai_processed=90,
            tagged_bookmarks=95,
            unique_tags=50,
            processing_time=60.0,
            stages_completed=["validation", "ai_processing", "tagging"],
            error_summary={"timeout": 3},
            statistics={"avg_processing_time": 0.6},
        )

        assert results.total_bookmarks == 100
        assert results.valid_bookmarks == 95
        assert len(results.stages_completed) == 3
        assert results.error_summary["timeout"] == 3

    def test_mock_pipeline_context_creation(self):
        """Test that mock pipeline context is properly created."""
        context = create_mock_pipeline_context()

        assert "requests_session" in context
        assert "ai_processor" in context
        assert "content_analyzer" in context
        assert "checkpoint_manager" in context
        assert "progress_tracker" in context

    def test_sample_bookmark_creation(self):
        """Test that sample bookmarks are properly created."""
        bookmarks = create_sample_bookmark_objects()

        assert len(bookmarks) == 5
        assert all(isinstance(b, Bookmark) for b in bookmarks)
        assert bookmarks[0].url == "https://docs.python.org/3/"

    def test_sample_dataframe_creation(self):
        """Test that sample DataFrames are properly created."""
        export_df = create_sample_export_dataframe()
        import_df = create_expected_import_dataframe()

        # Export should have 11 columns
        assert len(export_df.columns) == 11
        assert "id" in export_df.columns
        assert "cover" in export_df.columns

        # Import should have 6 columns
        assert len(import_df.columns) == 6
        assert "url" in import_df.columns
        assert "id" not in import_df.columns


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="bookmark_test_") as tmpdir:
        yield Path(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__])
