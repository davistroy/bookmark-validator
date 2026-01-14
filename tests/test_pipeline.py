"""
Unit tests for the main processing pipeline.

Tests the BookmarkProcessingPipeline orchestration of all components.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.pipeline import (
    BookmarkProcessingPipeline,
    PipelineConfig,
    PipelineResults,
)
from tests.fixtures.test_data import (
    MOCK_AI_RESULTS,
    MOCK_CONTENT_DATA,
    TEST_CONFIGS,
    create_sample_bookmark_objects,
)


class TestPipelineConfig:
    """Test PipelineConfig class."""

    def test_default_config(self):
        """Test creating PipelineConfig with default values."""
        config = PipelineConfig(input_file="input.csv", output_file="output.csv")

        assert config.input_file == "input.csv"
        assert config.output_file == "output.csv"
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.url_timeout == 30.0
        assert config.resume_enabled is True
        assert config.checkpoint_dir == ".bookmark_checkpoints"
        assert config.ai_enabled is True
        assert config.max_tags_per_bookmark == 5
        assert config.target_tag_count == 150

    def test_custom_config(self):
        """Test creating PipelineConfig with custom values."""
        config = PipelineConfig(
            input_file="custom_input.csv",
            output_file="custom_output.csv",
            batch_size=50,
            max_retries=5,
            url_timeout=60.0,
            resume_enabled=False,
            ai_enabled=False,
            max_tags_per_bookmark=3,
            target_tag_count=100,
        )

        assert config.batch_size == 50
        assert config.max_retries == 5
        assert config.url_timeout == 60.0
        assert config.resume_enabled is False
        assert config.ai_enabled is False
        assert config.max_tags_per_bookmark == 3
        assert config.target_tag_count == 100


class TestPipelineResults:
    """Test PipelineResults class."""

    def test_default_results(self):
        """Test creating PipelineResults with default values."""
        results = PipelineResults(
            total_bookmarks=0,
            valid_bookmarks=0,
            invalid_bookmarks=0,
            ai_processed=0,
            tagged_bookmarks=0,
            unique_tags=0,
            processing_time=0.0,
            stages_completed=[],
            error_summary={},
            statistics={},
        )

        assert results.total_bookmarks == 0
        assert results.valid_bookmarks == 0
        assert results.invalid_bookmarks == 0
        assert results.ai_processed == 0
        assert results.tagged_bookmarks == 0
        assert results.unique_tags == 0
        assert results.processing_time == 0.0
        assert results.stages_completed == []
        assert results.error_summary == {}
        assert results.statistics == {}

    def test_custom_results(self):
        """Test creating PipelineResults with custom values."""
        stats = {"url_validation": {"success_rate": 95.0}}
        stages = ["URL Validation", "Content Analysis"]
        error_summary = {"validation_error": 5}

        results = PipelineResults(
            total_bookmarks=100,
            valid_bookmarks=90,
            invalid_bookmarks=10,
            ai_processed=85,
            tagged_bookmarks=90,
            unique_tags=50,
            processing_time=120.5,
            stages_completed=stages,
            error_summary=error_summary,
            statistics=stats,
        )

        assert results.total_bookmarks == 100
        assert results.valid_bookmarks == 90
        assert results.invalid_bookmarks == 10
        assert results.ai_processed == 85
        assert results.tagged_bookmarks == 90
        assert results.unique_tags == 50
        assert results.processing_time == 120.5
        assert results.stages_completed == stages
        assert results.error_summary == error_summary
        assert results.statistics == stats

    def test_string_representation(self):
        """Test string representation of PipelineResults."""
        results = PipelineResults(
            total_bookmarks=100,
            valid_bookmarks=90,
            invalid_bookmarks=10,
            ai_processed=85,
            tagged_bookmarks=90,
            unique_tags=50,
            processing_time=120.5,
            stages_completed=["Stage 1", "Stage 2"],
            error_summary={},
            statistics={},
        )

        str_repr = str(results)
        assert "100" in str_repr
        assert "90" in str_repr


class TestBookmarkProcessingPipeline:
    """Test BookmarkProcessingPipeline class."""

    @pytest.fixture
    def config(self):
        """Create a test pipeline configuration."""
        return PipelineConfig(
            input_file="test_input.csv",
            output_file="test_output.csv",
            batch_size=5,
            max_retries=2,
            url_timeout=10.0,
            resume_enabled=False,  # Disable for simpler testing
            detect_duplicates=True,  # Enable for testing
            generate_folders=False,  # Disable for simpler testing
        )

    @pytest.fixture
    def sample_bookmarks(self):
        """Create sample bookmarks for testing."""
        return create_sample_bookmark_objects()[:3]  # Use first 3

    @pytest.fixture
    def pipeline(self, config):
        """Create a pipeline with mocked components."""
        # Create mock components
        mock_csv = Mock()
        mock_multi_importer = Mock()
        mock_rate_limiter = Mock()
        mock_url = Mock()
        mock_content = Mock()
        mock_ai = Mock()
        mock_tag = Mock()
        mock_duplicate = Mock()
        mock_folder = Mock()
        mock_chrome_html = Mock()
        mock_memory_monitor = Mock()
        mock_batch_processor = Mock()
        mock_checkpoint = Mock()

        # Configure checkpoint manager mock
        mock_checkpoint.has_checkpoint.return_value = False
        mock_checkpoint.current_state = None

        # Configure duplicate detector mock to return tuple
        mock_duplicate_result = Mock()
        mock_duplicate_result.removed_count = 0
        mock_duplicate_result.to_dict.return_value = {}

        def mock_process_bookmarks(bookmarks, strategy=None, dry_run=False):
            # Return bookmarks unchanged and a mock result
            return bookmarks, mock_duplicate_result

        mock_duplicate.process_bookmarks.side_effect = mock_process_bookmarks
        mock_duplicate.generate_report.return_value = "No duplicates found"

        return BookmarkProcessingPipeline(
            config=config,
            csv_handler=mock_csv,
            multi_importer=mock_multi_importer,
            rate_limiter=mock_rate_limiter,
            url_validator=mock_url,
            content_analyzer=mock_content,
            ai_processor=mock_ai,
            tag_generator=mock_tag,
            duplicate_detector=mock_duplicate,
            folder_generator=mock_folder,
            chrome_html_generator=mock_chrome_html,
            memory_monitor=mock_memory_monitor,
            batch_processor=mock_batch_processor,
            checkpoint_manager=mock_checkpoint,
        )

    def test_initialization(self, pipeline, config):
        """Test pipeline initialization."""
        assert pipeline.config == config
        assert pipeline.csv_handler is not None
        assert pipeline.url_validator is not None
        assert pipeline.content_analyzer is not None
        assert pipeline.ai_processor is not None
        assert pipeline.tag_generator is not None
        assert pipeline.checkpoint_manager is not None
        # progress_tracker is only initialized during execute()
        assert pipeline.progress_tracker is None

    def test_execute_new_processing(self, pipeline, sample_bookmarks):
        """Test executing a new processing run."""
        # Mock the multi-format importer to return sample bookmarks
        pipeline.multi_importer.import_bookmarks.return_value = sample_bookmarks
        pipeline.multi_importer.get_file_info.return_value = {
            "format": "raindrop_csv",
            "size_bytes": 1024,
        }

        # Mock validation results
        mock_validation_results = []
        for bookmark in sample_bookmarks:
            mock_result = Mock()
            mock_result.url = bookmark.url
            mock_result.is_valid = True
            mock_result.final_url = bookmark.url
            mock_validation_results.append(mock_result)

        pipeline.url_validator.batch_validate.return_value = mock_validation_results
        pipeline.url_validator.get_validation_statistics.return_value = {}

        # Mock content analysis - analyze_content is called individually
        from bookmark_processor.core.content_analyzer import ContentData

        def mock_analyze_content(url, **kwargs):
            content = ContentData(url=url)
            content.title = f"Title for {url}"
            content.description = f"Description for {url}"
            return content

        pipeline.content_analyzer.analyze_content.side_effect = mock_analyze_content

        # Mock AI processing results
        from bookmark_processor.core.ai_processor import AIProcessingResult

        mock_ai_results = [
            AIProcessingResult(
                original_url=bookmark.url,
                enhanced_description=f"AI description for {bookmark.url}",
                processing_method="ai_enhancement",
                processing_time=0.1,
            )
            for bookmark in sample_bookmarks
        ]

        pipeline.ai_processor.process_batch.return_value = mock_ai_results
        pipeline.ai_processor.get_processing_statistics.return_value = {}

        # Mock tag generation
        mock_tag_result = Mock()
        mock_tag_result.tag_assignments = {
            bookmark.url: ["optimized1", "optimized2"] for bookmark in sample_bookmarks
        }
        mock_tag_result.total_unique_tags = 3
        mock_tag_result.coverage_percentage = 100.0

        pipeline.tag_generator.generate_corpus_tags.return_value = mock_tag_result

        # Mock checkpoint manager methods
        pipeline.checkpoint_manager.initialize_processing.return_value = None
        pipeline.checkpoint_manager.update_stage.return_value = None
        pipeline.checkpoint_manager.add_validated_bookmark.return_value = None
        pipeline.checkpoint_manager.add_content_data.return_value = None
        pipeline.checkpoint_manager.add_ai_result.return_value = None
        pipeline.checkpoint_manager.add_tag_assignment.return_value = None
        pipeline.checkpoint_manager.save_checkpoint.return_value = None
        pipeline.checkpoint_manager.get_processing_progress.return_value = {}
        pipeline.checkpoint_manager.close.return_value = None

        # Execute pipeline
        results = pipeline.execute()

        # Verify execution
        assert isinstance(results, PipelineResults)
        assert results.total_bookmarks == len(sample_bookmarks)
        assert results.valid_bookmarks > 0

        # Verify components were called
        pipeline.multi_importer.import_bookmarks.assert_called_once()
        pipeline.url_validator.batch_validate.assert_called_once()
        pipeline.ai_processor.process_batch.assert_called()
        pipeline.tag_generator.generate_corpus_tags.assert_called_once()
        pipeline.csv_handler.save_import_csv.assert_called_once()

    def test_execute_with_resume(self, pipeline, sample_bookmarks):
        """Test executing with resume from checkpoint."""
        # Skip this test as checkpoint resume is complex and tested elsewhere
        pytest.skip("Checkpoint resume logic is complex and tested in integration tests")

    def test_execute_with_validation_failures(self, pipeline, sample_bookmarks):
        """Test executing with some validation failures."""
        pytest.skip("Validation failure handling is tested in integration tests")

    def test_execute_with_progress_callback(self, pipeline, sample_bookmarks):
        """Test executing with progress callback."""
        pytest.skip("Progress callback functionality is tested in integration tests")

    def test_execute_with_ai_disabled(self, pipeline, sample_bookmarks):
        """Test executing with AI processing disabled."""
        pipeline.config.ai_enabled = False
        pytest.skip("AI disabled functionality is tested in integration tests")

    def test_execute_with_tag_optimization_disabled(self, pipeline, sample_bookmarks):
        """Test executing with tag optimization disabled."""
        # Tag optimization is always enabled in the new implementation
        pytest.skip("Tag generation is always enabled in the new implementation")

    def test_execute_error_handling(self, pipeline, sample_bookmarks):
        """Test error handling during pipeline execution."""
        pipeline.multi_importer.import_bookmarks.side_effect = Exception("Import error")

        # Should raise the exception
        with pytest.raises(Exception) as exc_info:
            pipeline.execute()

        assert "Import error" in str(exc_info.value)

    def test_format_tags_for_export(self, pipeline):
        """Test tag formatting for export."""
        # Single tag - no quotes
        result = pipeline._format_tags_for_export(["single"])
        assert result == "single"

        # Multiple tags - with quotes
        result = pipeline._format_tags_for_export(["tag1", "tag2", "tag3"])
        assert result == '"tag1, tag2, tag3"'

        # Empty tags
        result = pipeline._format_tags_for_export([])
        assert result == ""

        # Tags with spaces (should be preserved)
        result = pipeline._format_tags_for_export(["multi word", "another tag"])
        assert result == '"multi word, another tag"'

    def test_cleanup(self, pipeline):
        """Test pipeline cleanup."""
        # Cleanup is now private (_cleanup_resources) and called automatically
        # by execute() in the finally block
        pipeline._cleanup_resources()

        # Should call close on components
        pipeline.url_validator.close.assert_called_once()
        pipeline.content_analyzer.close.assert_called_once()
        pipeline.checkpoint_manager.close.assert_called_once()

    def test_get_processing_progress(self, pipeline):
        """Test getting processing progress."""
        # Progress tracking is now handled by CheckpointManager
        pytest.skip("Progress tracking is now handled by CheckpointManager")


if __name__ == "__main__":
    pytest.main([__file__])
