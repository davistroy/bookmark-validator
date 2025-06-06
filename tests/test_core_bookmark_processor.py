"""
Unit tests for bookmark processor and pipeline components.

Tests for bookmark processor, batch processor, and processing pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from bookmark_processor.core.bookmark_processor import BookmarkProcessor
from bookmark_processor.core.batch_processor import BatchProcessor
from bookmark_processor.core.pipeline import BookmarkProcessingPipeline
from bookmark_processor.core.data_models import Bookmark, ProcessingResults

from tests.fixtures.test_data import (
    create_sample_bookmark_objects,
    create_sample_export_dataframe,
    create_expected_import_dataframe
)
from tests.fixtures.mock_utilities import (
    create_mock_pipeline_context,
    MockAIProcessor,
    MockContentAnalyzer,
    MockRequestsSession
)


class TestBookmarkProcessor:
    """Test BookmarkProcessor class."""
    
    def test_init_default(self):
        """Test BookmarkProcessor initialization with defaults."""
        processor = BookmarkProcessor()
        
        assert processor.batch_size == 50
        assert processor.max_retries == 3
        assert processor.timeout == 30
        assert processor.verbose is False
        assert processor.ai_engine == "local"
    
    def test_init_custom(self):
        """Test BookmarkProcessor initialization with custom values."""
        processor = BookmarkProcessor(
            batch_size=100,
            max_retries=5,
            timeout=60,
            verbose=True,
            ai_engine="claude",
            api_key="test-key"
        )
        
        assert processor.batch_size == 100
        assert processor.max_retries == 5
        assert processor.timeout == 60
        assert processor.verbose is True
        assert processor.ai_engine == "claude"
        assert processor.api_key == "test-key"
    
    def test_process_single_bookmark_success(self):
        """Test processing a single bookmark successfully."""
        # Setup mocks
        mock_context = create_mock_pipeline_context()
        
        processor = BookmarkProcessor()
        processor.url_validator = Mock()
        processor.url_validator.validate_url.return_value = (True, None)
        processor.content_analyzer = mock_context["content_analyzer"]
        processor.ai_processor = mock_context["ai_processor"]
        
        bookmark = Bookmark(
            title="Test Bookmark",
            url="https://example.com",
            note="Test note"
        )
        
        result = processor.process_bookmark(bookmark)
        
        assert result.processing_status.url_validated is True
        assert result.processing_status.content_extracted is True
        assert result.processing_status.ai_processed is True
        assert result.enhanced_description is not None
        assert len(result.optimized_tags) > 0
    
    def test_process_single_bookmark_url_validation_failure(self):
        """Test processing bookmark with URL validation failure."""
        processor = BookmarkProcessor()
        processor.url_validator = Mock()
        processor.url_validator.validate_url.return_value = (False, "Invalid URL")
        
        bookmark = Bookmark(
            title="Test Bookmark",
            url="invalid-url",
            note="Test note"
        )
        
        result = processor.process_bookmark(bookmark)
        
        assert result.processing_status.url_validated is False
        assert result.processing_status.url_validation_error == "Invalid URL"
        # Processing should continue even with invalid URL
        assert result is not None
    
    def test_process_single_bookmark_content_extraction_failure(self):
        """Test processing bookmark with content extraction failure."""
        mock_context = create_mock_pipeline_context()
        
        processor = BookmarkProcessor()
        processor.url_validator = Mock()
        processor.url_validator.validate_url.return_value = (True, None)
        
        # Mock content analyzer to return None (failure)
        processor.content_analyzer = Mock()
        processor.content_analyzer.extract_metadata.return_value = None
        
        processor.ai_processor = mock_context["ai_processor"]
        
        bookmark = Bookmark(
            title="Test Bookmark",
            url="https://example.com",
            note="Test note"
        )
        
        result = processor.process_bookmark(bookmark)
        
        assert result.processing_status.url_validated is True
        assert result.processing_status.content_extracted is False
        # AI processing should still work with existing content
        assert result.processing_status.ai_processed is True
    
    def test_process_batch_success(self):
        """Test processing a batch of bookmarks successfully."""
        mock_context = create_mock_pipeline_context()
        
        processor = BookmarkProcessor()
        processor.url_validator = Mock()
        processor.url_validator.validate_url.return_value = (True, None)
        processor.content_analyzer = mock_context["content_analyzer"]
        processor.ai_processor = mock_context["ai_processor"]
        processor.progress_tracker = mock_context["progress_tracker"]
        
        bookmarks = create_sample_bookmark_objects()
        
        results = processor.process_batch(bookmarks)
        
        assert len(results) == len(bookmarks)
        for bookmark in results:
            assert bookmark.processing_status.url_validated is True
            assert bookmark.enhanced_description is not None
    
    def test_process_batch_with_failures(self):
        """Test processing batch with some failures."""
        mock_context = create_mock_pipeline_context()
        
        processor = BookmarkProcessor()
        
        # Mock URL validator to fail for some URLs
        def mock_validate(url):
            if "invalid" in url:
                return (False, "Invalid URL")
            return (True, None)
        
        processor.url_validator = Mock()
        processor.url_validator.validate_url.side_effect = mock_validate
        processor.content_analyzer = mock_context["content_analyzer"]
        processor.ai_processor = mock_context["ai_processor"]
        processor.progress_tracker = mock_context["progress_tracker"]
        
        bookmarks = create_sample_bookmark_objects()
        # Add an invalid bookmark
        bookmarks.append(Bookmark(
            title="Invalid Bookmark",
            url="invalid-url",
            note="This should fail"
        ))
        
        results = processor.process_batch(bookmarks)
        
        assert len(results) == len(bookmarks)
        
        # Check that some succeeded and one failed
        valid_count = sum(1 for b in results if b.processing_status.url_validated)
        invalid_count = sum(1 for b in results if not b.processing_status.url_validated)
        
        assert valid_count > 0
        assert invalid_count > 0
    
    def test_generate_processing_report(self):
        """Test generating processing report."""
        processor = BookmarkProcessor()
        
        # Create sample results
        bookmarks = create_sample_bookmark_objects()
        for i, bookmark in enumerate(bookmarks):
            # Simulate processing status
            bookmark.processing_status.url_validated = i < 3  # 3 out of 5 valid
            bookmark.processing_status.content_extracted = i < 2  # 2 out of 5 extracted
            bookmark.processing_status.ai_processed = i < 4  # 4 out of 5 AI processed
        
        report = processor.generate_processing_report(bookmarks)
        
        assert isinstance(report, dict)
        assert report["total_bookmarks"] == len(bookmarks)
        assert report["url_validation_success"] == 3
        assert report["content_extraction_success"] == 2
        assert report["ai_processing_success"] == 4
        assert "processing_time" in report
        assert "error_summary" in report
    
    def test_save_intermediate_results(self, temp_dir):
        """Test saving intermediate processing results."""
        processor = BookmarkProcessor()
        
        bookmarks = create_sample_bookmark_objects()
        output_file = temp_dir / "intermediate.csv"
        
        processor.save_intermediate_results(bookmarks, str(output_file))
        
        assert output_file.exists()
        
        # Verify the saved file
        df = pd.read_csv(output_file)
        assert len(df) == len(bookmarks)
        assert "url" in df.columns
        assert "title" in df.columns


class TestBatchProcessor:
    """Test BatchProcessor class."""
    
    def test_init_default(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor()
        
        assert processor.batch_size == 50
        assert processor.max_workers == 4
        assert processor.progress_callback is None
    
    def test_init_custom(self):
        """Test BatchProcessor initialization with custom values."""
        progress_callback = Mock()
        
        processor = BatchProcessor(
            batch_size=100,
            max_workers=8,
            progress_callback=progress_callback
        )
        
        assert processor.batch_size == 100
        assert processor.max_workers == 8
        assert processor.progress_callback == progress_callback
    
    def test_create_batches(self):
        """Test creating batches from bookmark list."""
        processor = BatchProcessor(batch_size=2)
        
        bookmarks = create_sample_bookmark_objects()  # Usually 5 bookmarks
        batches = processor.create_batches(bookmarks)
        
        # Should create 3 batches (2 + 2 + 1)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1
    
    def test_process_batch_sequential(self):
        """Test sequential batch processing."""
        mock_context = create_mock_pipeline_context()
        
        processor = BatchProcessor(batch_size=2)
        bookmarks = create_sample_bookmark_objects()
        
        # Mock processing function
        def mock_process_bookmark(bookmark):
            bookmark.enhanced_description = f"Processed {bookmark.title}"
            return bookmark
        
        results = processor.process_batch_sequential(
            bookmarks,
            mock_process_bookmark,
            progress_callback=Mock()
        )
        
        assert len(results) == len(bookmarks)
        for bookmark in results:
            assert "Processed" in bookmark.enhanced_description
    
    def test_process_batch_parallel(self):
        """Test parallel batch processing."""
        processor = BatchProcessor(batch_size=2, max_workers=2)
        bookmarks = create_sample_bookmark_objects()
        
        # Mock processing function
        def mock_process_bookmark(bookmark):
            bookmark.enhanced_description = f"Processed {bookmark.title}"
            return bookmark
        
        results = processor.process_batch_parallel(
            bookmarks,
            mock_process_bookmark,
            progress_callback=Mock()
        )
        
        assert len(results) == len(bookmarks)
        for bookmark in results:
            assert "Processed" in bookmark.enhanced_description
    
    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        processor = BatchProcessor(batch_size=10)
        
        # Mock processing time data
        processor.processing_times = [1.0, 1.2, 0.8, 1.1, 0.9]  # 5 samples
        
        estimate = processor.estimate_processing_time(100)
        
        # Should estimate based on average time
        average_time = sum(processor.processing_times) / len(processor.processing_times)
        expected_batches = (100 + processor.batch_size - 1) // processor.batch_size
        expected_time = average_time * expected_batches
        
        assert abs(estimate - expected_time) < 0.1
    
    def test_get_processing_statistics(self):
        """Test getting processing statistics."""
        processor = BatchProcessor()
        
        # Simulate some processing
        processor.total_processed = 100
        processor.processing_times = [1.0, 1.2, 0.8, 1.1, 0.9]
        processor.errors = ["Error 1", "Error 2"]
        
        stats = processor.get_processing_statistics()
        
        assert isinstance(stats, dict)
        assert stats["total_processed"] == 100
        assert stats["total_batches"] == len(processor.processing_times)
        assert stats["average_batch_time"] == sum(processor.processing_times) / len(processor.processing_times)
        assert stats["total_errors"] == 2


class TestBookmarkProcessingPipeline:
    """Test BookmarkProcessingPipeline class."""
    
    def test_init_default(self):
        """Test BookmarkProcessingPipeline initialization."""
        pipeline = BookmarkProcessingPipeline()
        
        assert pipeline.batch_size == 50
        assert pipeline.enable_checkpoints is True
        assert pipeline.checkpoint_interval == 50
        assert pipeline.resume_from_checkpoint is False
    
    def test_init_custom(self):
        """Test BookmarkProcessingPipeline initialization with custom config."""
        config = {
            "batch_size": 100,
            "enable_checkpoints": False,
            "checkpoint_interval": 25,
            "verbose": True
        }
        
        pipeline = BookmarkProcessingPipeline(config)
        
        assert pipeline.batch_size == 100
        assert pipeline.enable_checkpoints is False
        assert pipeline.checkpoint_interval == 25
        assert pipeline.verbose is True
    
    @patch('bookmark_processor.core.csv_handler.RaindropCSVHandler')
    def test_run_pipeline_success(self, mock_csv_handler):
        """Test successful pipeline execution."""
        # Mock CSV handler
        mock_handler = Mock()
        sample_df = create_sample_export_dataframe()
        mock_handler.read_raindrop_export.return_value = sample_df
        mock_handler.write_raindrop_import.return_value = True
        mock_csv_handler.return_value = mock_handler
        
        # Mock components
        mock_context = create_mock_pipeline_context()
        
        pipeline = BookmarkProcessingPipeline({
            "batch_size": 2,
            "enable_checkpoints": False
        })
        
        # Inject mocks
        pipeline.bookmark_processor = Mock()
        pipeline.bookmark_processor.process_batch.return_value = create_sample_bookmark_objects()
        pipeline.progress_tracker = mock_context["progress_tracker"]
        
        results = pipeline.run(
            input_file="test_input.csv",
            output_file="test_output.csv"
        )
        
        assert isinstance(results, ProcessingResults)
        assert results.total_bookmarks > 0
        mock_handler.read_raindrop_export.assert_called_once()
        mock_handler.write_raindrop_import.assert_called_once()
    
    @patch('bookmark_processor.core.csv_handler.RaindropCSVHandler')
    def test_run_pipeline_with_checkpoints(self, mock_csv_handler):
        """Test pipeline execution with checkpoints enabled."""
        # Mock CSV handler
        mock_handler = Mock()
        sample_df = create_sample_export_dataframe()
        mock_handler.read_raindrop_export.return_value = sample_df
        mock_handler.write_raindrop_import.return_value = True
        mock_csv_handler.return_value = mock_handler
        
        # Mock checkpoint manager
        mock_checkpoint_manager = Mock()
        mock_checkpoint_manager.has_checkpoint.return_value = False
        
        pipeline = BookmarkProcessingPipeline({
            "batch_size": 2,
            "enable_checkpoints": True,
            "checkpoint_interval": 1
        })
        
        # Inject mocks
        pipeline.checkpoint_manager = mock_checkpoint_manager
        pipeline.bookmark_processor = Mock()
        pipeline.bookmark_processor.process_batch.return_value = create_sample_bookmark_objects()
        pipeline.progress_tracker = Mock()
        
        results = pipeline.run(
            input_file="test_input.csv",
            output_file="test_output.csv"
        )
        
        assert isinstance(results, ProcessingResults)
        # Should have attempted to save checkpoints
        mock_checkpoint_manager.save_checkpoint.assert_called()
    
    @patch('bookmark_processor.core.csv_handler.RaindropCSVHandler')
    def test_run_pipeline_resume_from_checkpoint(self, mock_csv_handler):
        """Test pipeline resuming from checkpoint."""
        # Mock CSV handler
        mock_handler = Mock()
        sample_df = create_sample_export_dataframe()
        mock_handler.read_raindrop_export.return_value = sample_df
        mock_handler.write_raindrop_import.return_value = True
        mock_csv_handler.return_value = mock_handler
        
        # Mock checkpoint manager with existing checkpoint
        mock_checkpoint_manager = Mock()
        mock_checkpoint_manager.has_checkpoint.return_value = True
        mock_checkpoint_manager.load_checkpoint.return_value = {
            "processed_bookmarks": create_sample_bookmark_objects()[:2],
            "last_processed_index": 2
        }
        
        pipeline = BookmarkProcessingPipeline({
            "enable_checkpoints": True,
            "resume_from_checkpoint": True
        })
        
        # Inject mocks
        pipeline.checkpoint_manager = mock_checkpoint_manager
        pipeline.bookmark_processor = Mock()
        pipeline.bookmark_processor.process_batch.return_value = create_sample_bookmark_objects()[2:]
        pipeline.progress_tracker = Mock()
        
        results = pipeline.run(
            input_file="test_input.csv",
            output_file="test_output.csv"
        )
        
        assert isinstance(results, ProcessingResults)
        # Should have loaded from checkpoint
        mock_checkpoint_manager.load_checkpoint.assert_called()
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        pipeline = BookmarkProcessingPipeline()
        
        # Valid configuration
        valid_config = {
            "batch_size": 50,
            "max_retries": 3,
            "timeout": 30
        }
        errors = pipeline.validate_configuration(valid_config)
        assert len(errors) == 0
        
        # Invalid configuration
        invalid_config = {
            "batch_size": 0,  # Invalid
            "max_retries": -1,  # Invalid
            "timeout": 0  # Invalid
        }
        errors = pipeline.validate_configuration(invalid_config)
        assert len(errors) > 0
    
    def test_prepare_for_processing(self):
        """Test preparation steps before processing."""
        pipeline = BookmarkProcessingPipeline()
        
        config = {
            "ai_engine": "local",
            "batch_size": 25,
            "verbose": True
        }
        
        pipeline.prepare_for_processing(config)
        
        assert pipeline.batch_size == 25
        assert pipeline.verbose is True
        assert pipeline.ai_processor is not None
        assert pipeline.url_validator is not None
    
    def test_create_processing_summary(self):
        """Test creating processing summary."""
        pipeline = BookmarkProcessingPipeline()
        
        bookmarks = create_sample_bookmark_objects()
        
        # Simulate processing results
        for i, bookmark in enumerate(bookmarks):
            bookmark.processing_status.url_validated = i < 4  # 4/5 success
            bookmark.processing_status.ai_processed = i < 3  # 3/5 success
        
        processing_time = 120.5
        summary = pipeline.create_processing_summary(bookmarks, processing_time)
        
        assert isinstance(summary, ProcessingResults)
        assert summary.total_bookmarks == len(bookmarks)
        assert summary.valid_bookmarks == 4  # Based on URL validation
        assert summary.processing_time == processing_time
    
    def test_cleanup_after_processing(self):
        """Test cleanup after processing completion."""
        pipeline = BookmarkProcessingPipeline({
            "enable_checkpoints": True
        })
        
        # Mock checkpoint manager
        mock_checkpoint_manager = Mock()
        pipeline.checkpoint_manager = mock_checkpoint_manager
        
        # Mock progress tracker
        mock_progress_tracker = Mock()
        pipeline.progress_tracker = mock_progress_tracker
        
        pipeline.cleanup_after_processing(success=True)
        
        # Should clean up checkpoints on success
        mock_checkpoint_manager.cleanup_checkpoints.assert_called()
        mock_progress_tracker.finish.assert_called()


class TestProcessingIntegration:
    """Integration tests for processing components."""
    
    def test_end_to_end_processing_workflow(self, temp_dir):
        """Test complete end-to-end processing workflow."""
        # Create test input file
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        
        sample_df.to_csv(input_file, index=False)
        
        # Create pipeline with minimal configuration
        config = {
            "batch_size": 2,
            "enable_checkpoints": False,
            "ai_engine": "local",
            "verbose": False
        }
        
        pipeline = BookmarkProcessingPipeline(config)
        
        # Mock external dependencies
        mock_context = create_mock_pipeline_context()
        pipeline.url_validator = Mock()
        pipeline.url_validator.validate_url.return_value = (True, None)
        pipeline.content_analyzer = mock_context["content_analyzer"]
        pipeline.ai_processor = mock_context["ai_processor"]
        pipeline.progress_tracker = mock_context["progress_tracker"]
        
        # Run pipeline
        results = pipeline.run(
            input_file=str(input_file),
            output_file=str(output_file)
        )
        
        # Verify results
        assert isinstance(results, ProcessingResults)
        assert results.total_bookmarks > 0
        assert output_file.exists()
        
        # Verify output file format
        output_df = pd.read_csv(output_file)
        expected_columns = ["url", "folder", "title", "note", "tags", "created"]
        assert list(output_df.columns) == expected_columns
    
    def test_large_batch_processing_simulation(self):
        """Test processing simulation with larger dataset."""
        # Create larger dataset
        from tests.fixtures.mock_utilities import create_performance_test_data
        
        large_df = create_performance_test_data(50)  # 50 bookmarks
        
        # Create processor with small batch size
        processor = BookmarkProcessor(batch_size=10)
        
        # Mock dependencies for fast processing
        processor.url_validator = Mock()
        processor.url_validator.validate_url.return_value = (True, None)
        processor.content_analyzer = MockContentAnalyzer()
        processor.ai_processor = MockAIProcessor()
        
        # Convert DataFrame to bookmark objects
        bookmarks = []
        for _, row in large_df.iterrows():
            bookmark = Bookmark.from_raindrop_export(row.to_dict())
            bookmarks.append(bookmark)
        
        # Process in batches
        batch_processor = BatchProcessor(batch_size=10)
        
        def process_single(bookmark):
            return processor.process_bookmark(bookmark)
        
        results = batch_processor.process_batch_sequential(
            bookmarks,
            process_single,
            progress_callback=Mock()
        )
        
        assert len(results) == len(bookmarks)
        # Verify all bookmarks were processed
        for bookmark in results:
            assert bookmark.enhanced_description is not None
            assert len(bookmark.optimized_tags) > 0


if __name__ == "__main__":
    pytest.main([__file__])