"""
Unit tests for the main processing pipeline.

Tests the BookmarkProcessingPipeline orchestration of all components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime

from bookmark_processor.core.pipeline import (
    BookmarkProcessingPipeline,
    PipelineConfig,
    PipelineResults
)
from bookmark_processor.core.data_models import Bookmark
from tests.fixtures.test_data import (
    create_sample_bookmark_objects,
    MOCK_CONTENT_DATA,
    MOCK_AI_RESULTS,
    TEST_CONFIGS
)


class TestPipelineConfig:
    """Test PipelineConfig class."""
    
    def test_default_config(self):
        """Test creating PipelineConfig with default values."""
        config = PipelineConfig(
            input_file="input.csv",
            output_file="output.csv"
        )
        
        assert config.input_file == "input.csv"
        assert config.output_file == "output.csv"
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.enable_checkpoints is True
        assert config.checkpoint_dir == ".bookmark_checkpoints"
        assert config.enable_ai_processing is True
        assert config.enable_tag_optimization is True
        assert config.max_tags_per_bookmark == 5
        assert config.target_unique_tags == 150
    
    def test_custom_config(self):
        """Test creating PipelineConfig with custom values."""
        config = PipelineConfig(
            input_file="custom_input.csv",
            output_file="custom_output.csv",
            batch_size=50,
            max_retries=5,
            timeout=60,
            enable_checkpoints=False,
            enable_ai_processing=False,
            max_tags_per_bookmark=3,
            target_unique_tags=100
        )
        
        assert config.batch_size == 50
        assert config.max_retries == 5
        assert config.timeout == 60
        assert config.enable_checkpoints is False
        assert config.enable_ai_processing is False
        assert config.max_tags_per_bookmark == 3
        assert config.target_unique_tags == 100


class TestPipelineResults:
    """Test PipelineResults class."""
    
    def test_default_results(self):
        """Test creating PipelineResults with default values."""
        results = PipelineResults()
        
        assert results.total_bookmarks == 0
        assert results.processed_bookmarks == 0
        assert results.valid_bookmarks == 0
        assert results.invalid_bookmarks == 0
        assert results.processing_time == 0.0
        assert results.stages_completed == []
        assert results.statistics == {}
    
    def test_custom_results(self):
        """Test creating PipelineResults with custom values."""
        stats = {"url_validation": {"success_rate": 95.0}}
        stages = ["URL Validation", "Content Analysis"]
        
        results = PipelineResults(
            total_bookmarks=100,
            processed_bookmarks=95,
            valid_bookmarks=90,
            invalid_bookmarks=10,
            processing_time=120.5,
            stages_completed=stages,
            statistics=stats
        )
        
        assert results.total_bookmarks == 100
        assert results.processed_bookmarks == 95
        assert results.valid_bookmarks == 90
        assert results.invalid_bookmarks == 10
        assert results.processing_time == 120.5
        assert results.stages_completed == stages
        assert results.statistics == stats
    
    def test_string_representation(self):
        """Test string representation of PipelineResults."""
        results = PipelineResults(
            total_bookmarks=100,
            processed_bookmarks=95,
            valid_bookmarks=90,
            processing_time=120.5
        )
        
        str_repr = str(results)
        assert "100" in str_repr
        assert "95" in str_repr
        assert "90" in str_repr
        assert "120.50s" in str_repr


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
            timeout=10,
            enable_checkpoints=False  # Disable for simpler testing
        )
    
    @pytest.fixture
    def sample_bookmarks(self):
        """Create sample bookmarks for testing."""
        return create_sample_bookmark_objects()[:3]  # Use first 3
    
    @pytest.fixture
    def pipeline(self, config):
        """Create a pipeline with mocked components."""
        with patch('bookmark_processor.core.pipeline.RaindropCSVHandler') as mock_csv, \
             patch('bookmark_processor.core.pipeline.URLValidator') as mock_url, \
             patch('bookmark_processor.core.pipeline.ContentAnalyzer') as mock_content, \
             patch('bookmark_processor.core.pipeline.AIProcessor') as mock_ai, \
             patch('bookmark_processor.core.pipeline.CorpusAwareTagGenerator') as mock_tag, \
             patch('bookmark_processor.core.pipeline.CheckpointManager') as mock_checkpoint, \
             patch('bookmark_processor.core.pipeline.ProgressTracker') as mock_progress:
            
            return BookmarkProcessingPipeline(config)
    
    def test_initialization(self, pipeline, config):
        """Test pipeline initialization."""
        assert pipeline.config == config
        assert pipeline.csv_handler is not None
        assert pipeline.url_validator is not None
        assert pipeline.content_analyzer is not None
        assert pipeline.ai_processor is not None
        assert pipeline.tag_generator is not None
        assert pipeline.checkpoint_manager is not None
        assert pipeline.progress_tracker is not None
    
    def test_execute_new_processing(self, pipeline, sample_bookmarks):
        """Test executing a new processing run."""
        # Mock the CSV handler to return sample bookmarks
        pipeline.csv_handler.load_and_transform_csv.return_value = sample_bookmarks
        
        # Mock validation results
        mock_validation_results = []
        for bookmark in sample_bookmarks:
            mock_result = Mock()
            mock_result.is_valid = True
            mock_result.final_url = bookmark.url
            mock_validation_results.append(mock_result)
        
        pipeline.url_validator.batch_validate.return_value = mock_validation_results
        
        # Mock content analysis results
        mock_content_results = {}
        for bookmark in sample_bookmarks:
            mock_content = Mock()
            mock_content.title = f"Title for {bookmark.url}"
            mock_content.description = f"Description for {bookmark.url}"
            mock_content_results[bookmark.url] = mock_content
        
        pipeline.content_analyzer.analyze_batch.return_value = mock_content_results
        
        # Mock AI processing results
        mock_ai_results = {}
        for bookmark in sample_bookmarks:
            mock_ai_result = {
                'enhanced_description': f"AI description for {bookmark.url}",
                'generated_tags': ['tag1', 'tag2'],
                'processing_method': 'ai_enhancement'
            }
            mock_ai_results[bookmark.url] = mock_ai_result
        
        pipeline.ai_processor.process_batch.return_value = mock_ai_results
        
        # Mock tag generation
        mock_tag_result = Mock()
        mock_tag_result.tag_assignments = {
            bookmark.url: ['optimized1', 'optimized2'] for bookmark in sample_bookmarks
        }
        mock_tag_result.unique_tags = ['optimized1', 'optimized2', 'optimized3']
        mock_tag_result.processing_stats = {'total_tags': 3}
        
        pipeline.tag_generator.generate_corpus_tags.return_value = mock_tag_result
        
        # Mock checkpoint manager
        pipeline.checkpoint_manager.load_checkpoint.return_value = None
        
        # Execute pipeline
        results = pipeline.execute()
        
        # Verify execution
        assert isinstance(results, PipelineResults)
        assert results.total_bookmarks == len(sample_bookmarks)
        assert results.processed_bookmarks > 0
        
        # Verify components were called
        pipeline.csv_handler.load_and_transform_csv.assert_called_once()
        pipeline.url_validator.batch_validate.assert_called_once()
        pipeline.content_analyzer.analyze_batch.assert_called_once()
        pipeline.ai_processor.process_batch.assert_called_once()
        pipeline.tag_generator.generate_corpus_tags.assert_called_once()
        pipeline.csv_handler.save_import_csv.assert_called_once()
    
    def test_execute_with_resume(self, pipeline, sample_bookmarks):
        """Test executing with resume from checkpoint."""
        # Mock checkpoint data
        mock_checkpoint = {
            'bookmarks': [b.to_dict() for b in sample_bookmarks],
            'validation_results': {},
            'content_data': {},
            'ai_results': {},
            'processed_bookmarks': 2,
            'stage': 'ai_processing'
        }
        
        pipeline.checkpoint_manager.load_checkpoint.return_value = mock_checkpoint
        
        # Mock remaining processing
        mock_ai_results = {
            sample_bookmarks[2].url: {
                'enhanced_description': 'AI description',
                'generated_tags': ['tag1', 'tag2']
            }
        }
        pipeline.ai_processor.process_batch.return_value = mock_ai_results
        
        # Mock tag generation
        mock_tag_result = Mock()
        mock_tag_result.tag_assignments = {
            bookmark.url: ['tag1', 'tag2'] for bookmark in sample_bookmarks
        }
        mock_tag_result.unique_tags = ['tag1', 'tag2']
        
        pipeline.tag_generator.generate_corpus_tags.return_value = mock_tag_result
        
        # Execute pipeline
        results = pipeline.execute()
        
        # Should resume from checkpoint
        pipeline.checkpoint_manager.load_checkpoint.assert_called_once()
        # Should not reload CSV since resuming
        pipeline.csv_handler.load_and_transform_csv.assert_not_called()
    
    def test_execute_with_validation_failures(self, pipeline, sample_bookmarks):
        """Test executing with some validation failures."""
        pipeline.csv_handler.load_and_transform_csv.return_value = sample_bookmarks
        
        # Mock validation with some failures
        mock_validation_results = []
        for i, bookmark in enumerate(sample_bookmarks):
            mock_result = Mock()
            mock_result.is_valid = i < 2  # First 2 valid, last invalid
            mock_result.final_url = bookmark.url if i < 2 else None
            mock_validation_results.append(mock_result)
        
        pipeline.url_validator.batch_validate.return_value = mock_validation_results
        
        # Mock other components for valid URLs only
        valid_bookmarks = sample_bookmarks[:2]
        pipeline.content_analyzer.analyze_batch.return_value = {
            b.url: Mock() for b in valid_bookmarks
        }
        pipeline.ai_processor.process_batch.return_value = {
            b.url: {'enhanced_description': 'desc', 'generated_tags': []}
            for b in valid_bookmarks
        }
        
        mock_tag_result = Mock()
        mock_tag_result.tag_assignments = {b.url: [] for b in valid_bookmarks}
        mock_tag_result.unique_tags = []
        pipeline.tag_generator.generate_corpus_tags.return_value = mock_tag_result
        
        # Execute pipeline
        results = pipeline.execute()
        
        # Should process only valid bookmarks
        assert results.valid_bookmarks == 2
        assert results.invalid_bookmarks == 1
    
    def test_execute_with_progress_callback(self, pipeline, sample_bookmarks):
        """Test executing with progress callback."""
        progress_calls = []
        def progress_callback(info):
            progress_calls.append(info)
        
        pipeline.csv_handler.load_and_transform_csv.return_value = sample_bookmarks
        
        # Mock successful processing
        pipeline.url_validator.batch_validate.return_value = [
            Mock(is_valid=True, final_url=b.url) for b in sample_bookmarks
        ]
        pipeline.content_analyzer.analyze_batch.return_value = {
            b.url: Mock() for b in sample_bookmarks
        }
        pipeline.ai_processor.process_batch.return_value = {
            b.url: {'enhanced_description': 'desc', 'generated_tags': []}
            for b in sample_bookmarks
        }
        
        mock_tag_result = Mock()
        mock_tag_result.tag_assignments = {b.url: [] for b in sample_bookmarks}
        mock_tag_result.unique_tags = []
        pipeline.tag_generator.generate_corpus_tags.return_value = mock_tag_result
        
        # Execute with progress callback
        pipeline.execute(progress_callback=progress_callback)
        
        # Should have received progress updates
        assert len(progress_calls) > 0
    
    def test_execute_with_ai_disabled(self, pipeline, sample_bookmarks):
        """Test executing with AI processing disabled."""
        pipeline.config.enable_ai_processing = False
        pipeline.csv_handler.load_and_transform_csv.return_value = sample_bookmarks
        
        # Mock successful validation and content analysis
        pipeline.url_validator.batch_validate.return_value = [
            Mock(is_valid=True, final_url=b.url) for b in sample_bookmarks
        ]
        pipeline.content_analyzer.analyze_batch.return_value = {
            b.url: Mock() for b in sample_bookmarks
        }
        
        mock_tag_result = Mock()
        mock_tag_result.tag_assignments = {b.url: [] for b in sample_bookmarks}
        mock_tag_result.unique_tags = []
        pipeline.tag_generator.generate_corpus_tags.return_value = mock_tag_result
        
        # Execute pipeline
        results = pipeline.execute()
        
        # AI processor should not be called
        pipeline.ai_processor.process_batch.assert_not_called()
    
    def test_execute_with_tag_optimization_disabled(self, pipeline, sample_bookmarks):
        """Test executing with tag optimization disabled."""
        pipeline.config.enable_tag_optimization = False
        pipeline.csv_handler.load_and_transform_csv.return_value = sample_bookmarks
        
        # Mock successful processing
        pipeline.url_validator.batch_validate.return_value = [
            Mock(is_valid=True, final_url=b.url) for b in sample_bookmarks
        ]
        pipeline.content_analyzer.analyze_batch.return_value = {
            b.url: Mock() for b in sample_bookmarks
        }
        pipeline.ai_processor.process_batch.return_value = {
            b.url: {'enhanced_description': 'desc', 'generated_tags': ['tag1']}
            for b in sample_bookmarks
        }
        
        # Execute pipeline
        results = pipeline.execute()
        
        # Tag generator should not be called
        pipeline.tag_generator.generate_corpus_tags.assert_not_called()
    
    def test_execute_error_handling(self, pipeline, sample_bookmarks):
        """Test error handling during pipeline execution."""
        pipeline.csv_handler.load_and_transform_csv.side_effect = Exception("CSV error")
        
        # Should raise the exception
        with pytest.raises(Exception) as exc_info:
            pipeline.execute()
        
        assert "CSV error" in str(exc_info.value)
    
    def test_format_tags_for_export(self, pipeline):
        """Test tag formatting for export."""
        # Single tag - no quotes
        result = pipeline._format_tags_for_export(['single'])
        assert result == 'single'
        
        # Multiple tags - with quotes
        result = pipeline._format_tags_for_export(['tag1', 'tag2', 'tag3'])
        assert result == '"tag1, tag2, tag3"'
        
        # Empty tags
        result = pipeline._format_tags_for_export([])
        assert result == ''
        
        # Tags with spaces (should be preserved)
        result = pipeline._format_tags_for_export(['multi word', 'another tag'])
        assert result == '"multi word, another tag"'
    
    def test_cleanup(self, pipeline):
        """Test pipeline cleanup."""
        pipeline.cleanup()
        
        # Should call cleanup on components
        pipeline.url_validator.cleanup.assert_called_once()
        pipeline.content_analyzer.cleanup.assert_called_once()
        pipeline.checkpoint_manager.cleanup.assert_called_once()
    
    def test_get_processing_progress(self, pipeline):
        """Test getting processing progress."""
        pipeline.processed_bookmarks = 50
        pipeline.total_bookmarks = 100
        
        progress = pipeline._get_processing_progress()
        
        assert progress['processed'] == 50
        assert progress['total'] == 100
        assert progress['percentage'] == 50.0


if __name__ == "__main__":
    pytest.main([__file__])