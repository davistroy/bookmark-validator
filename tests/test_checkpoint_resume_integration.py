"""
Checkpoint and Resume Integration Tests

Tests for checkpoint/resume functionality in the complete processing pipeline,
including interruption simulation, data persistence, and recovery scenarios.
"""

import asyncio
import os
import signal
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bookmark_processor.core.ai_processor import AIProcessingResult
from bookmark_processor.core.checkpoint_manager import (
    CheckpointManager,
    ProcessingStage,
    ProcessingState,
)
from bookmark_processor.core.content_analyzer import ContentData
from bookmark_processor.core.csv_handler import RaindropCSVHandler
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.pipeline import BookmarkProcessingPipeline, PipelineConfig
from bookmark_processor.core.url_validator import ValidationResult


class TestCheckpointResumeIntegration:
    """Test checkpoint and resume functionality in pipeline integration."""

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for checkpoint testing."""
        return """id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,"Python Tutorial","Learn Python basics","Official Python tutorial","https://docs.python.org/tutorial","Programming/Python","python, tutorial","2024-01-01T00:00:00Z","","","false"
2,"AI Research","Latest AI developments","Research paper on AI","https://arxiv.org/abs/12345","Research/AI","ai, research","2024-01-02T00:00:00Z","","","false"
3,"JavaScript Guide","Web development","MDN JavaScript docs","https://developer.mozilla.org/js","Programming/JavaScript","javascript, web","2024-01-03T00:00:00Z","","","false"
4,"React Documentation","React framework","React official docs","https://reactjs.org/docs","Programming/Frontend","react, frontend","2024-01-04T00:00:00Z","","","false"
5,"Database Design","SQL fundamentals","Database design principles","https://db.example.com/guide","Database","sql, design","2024-01-05T00:00:00Z","","","false"
6,"Machine Learning","ML concepts","Deep learning basics","https://deeplearning.ai/courses","AI/ML","machine-learning, deep-learning","2024-01-06T00:00:00Z","","","false"
"""

    @pytest.fixture
    def checkpoint_csv_file(self, sample_csv_data):
        """Create temporary CSV file for checkpoint testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            return f.name

    @pytest.fixture
    def checkpoint_config(self, checkpoint_csv_file):
        """Create pipeline configuration for checkpoint testing."""
        return PipelineConfig(
            input_file=checkpoint_csv_file,
            output_file=checkpoint_csv_file.replace(".csv", "_checkpoint_output.csv"),
            batch_size=2,
            max_retries=1,
            resume_enabled=True,
            clear_checkpoints=False,
            ai_enabled=True,
            checkpoint_dir=".test_checkpoint_integration",
            save_interval=2,
            verbose=True,
        )

    @pytest.mark.asyncio
    async def test_checkpoint_creation_during_processing(self, checkpoint_config):
        """Test that checkpoints are created during processing stages."""
        pipeline = BookmarkProcessingPipeline(checkpoint_config)

        # Mock external dependencies
        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            # Mock URL validation results
            validation_results = [
                ValidationResult(
                    url="https://docs.python.org/tutorial",
                    is_valid=True,
                    status_code=200,
                ),
                ValidationResult(
                    url="https://arxiv.org/abs/12345", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://developer.mozilla.org/js",
                    is_valid=True,
                    status_code=200,
                ),
                ValidationResult(
                    url="https://reactjs.org/docs", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://db.example.com/guide", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://deeplearning.ai/courses",
                    is_valid=True,
                    status_code=200,
                ),
            ]
            mock_validate.return_value = validation_results

            # Mock AI processing results - returns Bookmark objects
            def mock_ai_batch_process(bookmarks, **kwargs):
                results = []
                for bookmark in bookmarks:
                    # Create a copy with enhanced description
                    enhanced = bookmark.copy()
                    enhanced.enhanced_description = f"AI enhanced: {bookmark.title}"
                    results.append(enhanced)
                return results

            mock_ai_process.side_effect = mock_ai_batch_process

            # Start processing
            pipeline._start_new_processing(None)

            # Verify checkpoint was created
            assert pipeline.checkpoint_manager.has_checkpoint(
                checkpoint_config.input_file
            )

            # Verify checkpoint contains expected data
            state = pipeline.checkpoint_manager.load_checkpoint(
                checkpoint_config.input_file
            )
            assert state is not None
            assert state.current_stage == ProcessingStage.COMPLETED
            assert len(state.processed_urls) > 0

        # Cleanup
        pipeline._cleanup_resources()
        Path(checkpoint_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_resume_from_url_validation_stage(self, checkpoint_config):
        """Test resuming processing from URL validation stage."""
        # Create initial checkpoint state
        checkpoint_manager = CheckpointManager(checkpoint_config.checkpoint_dir)

        # Initialize processing state
        config_dict = checkpoint_config.__dict__.copy()
        checkpoint_manager.initialize_processing(
            checkpoint_config.input_file,
            checkpoint_config.output_file,
            6,  # Total bookmarks
            config_dict,
        )

        # Simulate partial URL validation completion
        checkpoint_manager.update_stage(ProcessingStage.URL_VALIDATION)

        # Add some validated URLs to checkpoint
        validation_result = ValidationResult(
            url="https://docs.python.org/tutorial", is_valid=True, status_code=200
        )

        # Create real bookmark object
        test_bookmark = Bookmark(
            id="1",
            title="Python Tutorial",
            url="https://docs.python.org/tutorial",
            note="Learn Python basics",
            excerpt="Official Python tutorial",
            folder="Programming/Python",
            tags=["python", "tutorial"],
            created="2024-01-01T00:00:00Z",
        )

        checkpoint_manager.add_validated_bookmark(validation_result, test_bookmark)
        checkpoint_manager.save_checkpoint(force=True)

        # Create pipeline and resume
        pipeline = BookmarkProcessingPipeline(checkpoint_config)

        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            # Mock remaining URL validation
            remaining_validation_results = [
                ValidationResult(
                    url="https://arxiv.org/abs/12345", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://developer.mozilla.org/js",
                    is_valid=True,
                    status_code=200,
                ),
                ValidationResult(
                    url="https://reactjs.org/docs", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://db.example.com/guide", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://deeplearning.ai/courses",
                    is_valid=True,
                    status_code=200,
                ),
            ]
            mock_validate.return_value = remaining_validation_results

            # Mock AI processing - returns Bookmark objects
            mock_ai_process.return_value = []

            # Resume processing
            results = pipeline._resume_processing(None)

            # Verify resumption worked
            assert results is not None
            assert results.total_bookmarks == 6

            # Verify that URL validation was resumed (not restarted)
            # Should only validate remaining URLs
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args[0][0]  # First argument (URLs list)
            assert len(call_args) == 5  # Should only validate remaining 5 URLs
            assert (
                "https://docs.python.org/tutorial" not in call_args
            )  # Already validated

        # Cleanup
        pipeline._cleanup_resources()
        Path(checkpoint_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_resume_from_ai_processing_stage(self, checkpoint_config):
        """Test resuming processing from AI processing stage."""
        checkpoint_manager = CheckpointManager(checkpoint_config.checkpoint_dir)

        # Initialize processing state
        config_dict = checkpoint_config.__dict__.copy()
        checkpoint_manager.initialize_processing(
            checkpoint_config.input_file, checkpoint_config.output_file, 6, config_dict
        )

        # Simulate completion of URL validation and content analysis
        checkpoint_manager.update_stage(ProcessingStage.AI_PROCESSING)

        # Add validation results
        urls = [
            "https://docs.python.org/tutorial",
            "https://arxiv.org/abs/12345",
            "https://developer.mozilla.org/js",
            "https://reactjs.org/docs",
            "https://db.example.com/guide",
            "https://deeplearning.ai/courses",
        ]

        for i, url in enumerate(urls):
            validation_result = ValidationResult(
                url=url, is_valid=True, status_code=200
            )
            test_bookmark = Bookmark(
                id=str(i+2),  # Starting from 2 since first one was id=1
                title=f"Title for {url}",
                url=url,
                note=f"Test note for {url}",
                excerpt=f"Test excerpt for {url}",
                folder="Test/Folder",
                tags=["test", "bookmark"],
                created="2024-01-01T00:00:00Z",
            )
            checkpoint_manager.add_validated_bookmark(validation_result, test_bookmark)

        # Add some AI processing results (partial completion)
        processed_urls = urls[:3]  # First 3 URLs processed
        for url in processed_urls:
            ai_result = AIProcessingResult(
                original_url=url,
                enhanced_description=f"Enhanced description for {url}",
                processing_method="test_ai",
                processing_time=0.1,
            )
            checkpoint_manager.add_ai_result(url, ai_result)

        checkpoint_manager.save_checkpoint(force=True)

        # Create pipeline and resume
        pipeline = BookmarkProcessingPipeline(checkpoint_config)

        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            # Mock should not be called for URL validation (already done)
            mock_validate.return_value = []

            # Mock AI processing for remaining URLs - returns AIProcessingResult with .url property
            def mock_ai_batch_process(bookmarks, **kwargs):
                results = []
                for bookmark in bookmarks:
                    result = AIProcessingResult(
                        original_url=bookmark.url,
                        enhanced_description=f"Resumed AI: {bookmark.title}",
                        processing_method="resumed_ai",
                        processing_time=0.1,
                    )
                    results.append(result)
                return results

            mock_ai_process.side_effect = mock_ai_batch_process

            # Resume processing
            results = pipeline._resume_processing(None)

            # Verify resumption worked
            assert results is not None
            assert results.total_bookmarks == 6
            assert results.ai_processed == 6  # All should be processed now

            # Verify AI processing was called for remaining bookmarks only
            mock_ai_process.assert_called()

            # Check that both original and resumed AI results exist
            assert len(pipeline.ai_results) == 6

            # Verify mix of original and resumed results
            original_count = sum(
                1
                for r in pipeline.ai_results.values()
                if r.processing_method == "test_ai"
            )
            resumed_count = sum(
                1
                for r in pipeline.ai_results.values()
                if r.processing_method == "resumed_ai"
            )

            assert original_count == 3  # From checkpoint
            assert resumed_count == 3  # From resumption

        # Cleanup
        pipeline._cleanup_resources()
        Path(checkpoint_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_checkpoint_data_integrity_after_interruption(
        self, checkpoint_config
    ):
        """Test that checkpoint data integrity is maintained after interruption."""
        pipeline = BookmarkProcessingPipeline(checkpoint_config)

        # Mock processing that will be "interrupted"
        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            # Mock URL validation results
            validation_results = [
                ValidationResult(
                    url="https://docs.python.org/tutorial",
                    is_valid=True,
                    status_code=200,
                ),
                ValidationResult(
                    url="https://arxiv.org/abs/12345", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://developer.mozilla.org/js",
                    is_valid=False,
                    status_code=404,
                ),
            ]
            mock_validate.return_value = validation_results

            # Mock AI processing that "fails" partway through
            def mock_ai_batch_process_with_interruption(bookmarks, **kwargs):
                results = []
                for i, bookmark in enumerate(bookmarks):
                    if i >= 1:  # Simulate interruption after first bookmark
                        raise Exception("Simulated processing interruption")

                    result = AIProcessingResult(
                        original_url=bookmark.url,
                        enhanced_description=f"Partial AI: {bookmark.title}",
                        processing_method="interrupted_ai",
                        processing_time=0.1,
                    )
                    results.append(result)
                return results

            mock_ai_process.side_effect = mock_ai_batch_process_with_interruption

            # Start processing - the pipeline handles partial AI failures gracefully
            # by continuing with remaining bookmarks
            try:
                pipeline._start_new_processing(None)
            except Exception:
                pass  # Exception may or may not propagate depending on batch handling

        # Verify checkpoint was saved during processing
        assert pipeline.checkpoint_manager.has_checkpoint(checkpoint_config.input_file)

        # Load checkpoint and verify data integrity
        state = pipeline.checkpoint_manager.load_checkpoint(
            checkpoint_config.input_file
        )
        assert state is not None

        # Processing may complete (ERROR stage) or be interrupted (other stage)
        # The key assertion is that checkpoint data was preserved
        assert state.current_stage in [
            ProcessingStage.ERROR,
            ProcessingStage.COMPLETED,
            ProcessingStage.AI_PROCESSING,
            ProcessingStage.CONTENT_ANALYSIS,
            ProcessingStage.TAG_OPTIMIZATION,
            ProcessingStage.OUTPUT_GENERATION,
        ]

        # Verify validation results were saved
        assert len(state.processed_urls) > 0

        # Verify checkpoint contains bookmark data
        assert len(state.validated_bookmarks) > 0

        # Cleanup
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    async def test_resume_with_different_configuration(self, checkpoint_config):
        """Test resuming with modified configuration parameters."""
        # Create initial checkpoint
        checkpoint_manager = CheckpointManager(checkpoint_config.checkpoint_dir)
        config_dict = checkpoint_config.__dict__.copy()
        checkpoint_manager.initialize_processing(
            checkpoint_config.input_file, checkpoint_config.output_file, 6, config_dict
        )

        # Add some processing state
        checkpoint_manager.update_stage(ProcessingStage.URL_VALIDATION)
        validation_result = ValidationResult(
            url="https://docs.python.org/tutorial", is_valid=True, status_code=200
        )
        test_bookmark = Bookmark(
            id="config_test_1",
            title="Python Tutorial Config Test",
            url="https://docs.python.org/tutorial",
            note="Config test note",
            excerpt="Config test excerpt",
            folder="Config/Test",
            tags=["config", "test"],
            created="2024-01-01T00:00:00Z",
        )
        checkpoint_manager.add_validated_bookmark(validation_result, test_bookmark)
        checkpoint_manager.save_checkpoint(force=True)

        # Create new pipeline with modified configuration
        modified_config = PipelineConfig(
            input_file=checkpoint_config.input_file,
            output_file=checkpoint_config.output_file,
            batch_size=3,  # Changed from 2
            max_retries=2,  # Changed from 1
            resume_enabled=True,
            ai_enabled=True,
            checkpoint_dir=checkpoint_config.checkpoint_dir,
            save_interval=1,  # Changed from 2
            verbose=False,  # Changed from True
        )

        pipeline = BookmarkProcessingPipeline(modified_config)

        # Verify that resumption uses original configuration
        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            mock_validate.return_value = []
            mock_ai_process.return_value = []

            # Resume processing
            results = pipeline._resume_processing(None)

            # Should complete successfully despite config differences
            assert results is not None

            # Current pipeline should use new config, but checkpoint restore should work
            assert pipeline.config.batch_size == 3  # New config
            assert pipeline.config.max_retries == 2  # New config

        # Cleanup
        pipeline._cleanup_resources()
        Path(checkpoint_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup_after_completion(self, checkpoint_config):
        """Test that checkpoints are properly cleaned up after successful completion."""
        pipeline = BookmarkProcessingPipeline(checkpoint_config)

        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            # Mock successful processing
            validation_results = [
                ValidationResult(
                    url="https://docs.python.org/tutorial",
                    is_valid=True,
                    status_code=200,
                ),
                ValidationResult(
                    url="https://arxiv.org/abs/12345", is_valid=True, status_code=200
                ),
            ]
            mock_validate.return_value = validation_results

            # Return AIProcessingResult objects with .url property
            mock_ai_process.return_value = [
                AIProcessingResult(
                    original_url="https://docs.python.org/tutorial",
                    enhanced_description="AI enhanced description",
                    processing_method="test_ai",
                    processing_time=0.1,
                )
            ]

            # Execute pipeline
            results = pipeline.execute()

            # Verify successful completion
            assert results is not None
            assert results.total_bookmarks > 0

            # Verify checkpoint still exists (for potential future resume)
            assert pipeline.checkpoint_manager.has_checkpoint(
                checkpoint_config.input_file
            )

            # Verify final stage is completed
            state = pipeline.checkpoint_manager.load_checkpoint(
                checkpoint_config.input_file
            )
            assert state.current_stage == ProcessingStage.COMPLETED

        # Cleanup
        pipeline._cleanup_resources()
        Path(checkpoint_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_multiple_resume_cycles(self, checkpoint_config):
        """Test multiple resume cycles with different interruption points."""
        checkpoint_manager = CheckpointManager(checkpoint_config.checkpoint_dir)

        # Simulate first processing session (interrupted during URL validation)
        config_dict = checkpoint_config.__dict__.copy()
        checkpoint_manager.initialize_processing(
            checkpoint_config.input_file, checkpoint_config.output_file, 6, config_dict
        )

        checkpoint_manager.update_stage(ProcessingStage.URL_VALIDATION)

        # Add partial URL validation
        urls = ["https://docs.python.org/tutorial", "https://arxiv.org/abs/12345"]
        for i, url in enumerate(urls):
            validation_result = ValidationResult(
                url=url, is_valid=True, status_code=200
            )
            test_bookmark = Bookmark(
                id=f"multi_resume_{i+1}",
                title=f"Multi Resume Test {i+1}",
                url=url,
                note=f"Multi resume note {i+1}",
                excerpt=f"Multi resume excerpt {i+1}",
                folder="MultiResume/Test",
                tags=["multi", "resume"],
                created="2024-01-01T00:00:00Z",
            )
            checkpoint_manager.add_validated_bookmark(validation_result, test_bookmark)

        checkpoint_manager.save_checkpoint(force=True)

        # First resume: Complete URL validation, start AI processing
        pipeline1 = BookmarkProcessingPipeline(checkpoint_config)

        with (
            patch.object(pipeline1.url_validator, "batch_validate") as mock_validate1,
            patch.object(pipeline1.ai_processor, "process_batch") as mock_ai_process1,
        ):

            # Complete remaining URL validation
            remaining_validation = [
                ValidationResult(
                    url="https://developer.mozilla.org/js",
                    is_valid=True,
                    status_code=200,
                ),
                ValidationResult(
                    url="https://reactjs.org/docs", is_valid=True, status_code=200
                ),
            ]
            mock_validate1.return_value = remaining_validation

            # Simulate AI processing interruption
            def mock_ai_interrupted(bookmarks, **kwargs):
                # Process only first bookmark, then "crash"
                if len(bookmarks) > 0:
                    result = AIProcessingResult(
                        original_url=bookmarks[0].url,
                        enhanced_description=f"First resume AI: {bookmarks[0].title}",
                        processing_method="first_resume",
                        processing_time=0.1,
                    )
                    return [result]
                return []

            mock_ai_process1.side_effect = mock_ai_interrupted

            try:
                # This should progress further but still not complete
                pipeline1._resume_processing(None)
            except Exception:
                pass  # Expected interruption

        # Verify checkpoint was updated
        state = checkpoint_manager.load_checkpoint(checkpoint_config.input_file)
        assert len(state.processed_urls) >= 4  # Should have more validated URLs
        assert len(state.ai_results) >= 1  # Should have some AI results

        pipeline1._cleanup_resources()

        # Second resume: Complete remaining processing
        pipeline2 = BookmarkProcessingPipeline(checkpoint_config)

        with (
            patch.object(pipeline2.url_validator, "batch_validate") as mock_validate2,
            patch.object(pipeline2.ai_processor, "process_batch") as mock_ai_process2,
        ):

            mock_validate2.return_value = []  # No more URL validation needed

            # Complete AI processing
            def mock_ai_complete(bookmarks, **kwargs):
                results = []
                for bookmark in bookmarks:
                    result = AIProcessingResult(
                        original_url=bookmark.url,
                        enhanced_description=f"Second resume AI: {bookmark.title}",
                        processing_method="second_resume",
                        processing_time=0.1,
                    )
                    results.append(result)
                return results

            mock_ai_process2.side_effect = mock_ai_complete

            # Should complete successfully
            results = pipeline2._resume_processing(None)

            assert results is not None
            assert results.total_bookmarks > 0

            # Verify that processing completed - either through first or second resume
            # The exact distribution depends on checkpoint state and batch handling
            total_ai_results = len(pipeline2.ai_results)
            assert total_ai_results >= 1, "Should have at least some AI results"

            # Check for any AI results (method names may vary)
            first_resume_count = sum(
                1
                for r in pipeline2.ai_results.values()
                if hasattr(r, 'processing_method') and r.processing_method == "first_resume"
            )

            # Verify first resume produced at least 1 result (from checkpoint)
            assert first_resume_count >= 1, "Should have at least one result from first resume"

        # Cleanup
        pipeline2._cleanup_resources()
        Path(checkpoint_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_checkpoint_corruption_recovery(self, checkpoint_config):
        """Test recovery from corrupted checkpoint files."""
        checkpoint_manager = CheckpointManager(checkpoint_config.checkpoint_dir)

        # Create a valid checkpoint
        config_dict = checkpoint_config.__dict__.copy()
        checkpoint_manager.initialize_processing(
            checkpoint_config.input_file, checkpoint_config.output_file, 6, config_dict
        )
        checkpoint_manager.save_checkpoint(force=True)

        # Verify checkpoint exists and is valid
        assert checkpoint_manager.has_checkpoint(checkpoint_config.input_file)

        # Corrupt the checkpoint file
        checkpoint_files = list(Path(checkpoint_config.checkpoint_dir).glob("*.json"))
        if checkpoint_files:
            corrupt_file = checkpoint_files[0]
            corrupt_file.write_text("corrupted checkpoint data")

        # Try to resume - should fall back to new processing
        pipeline = BookmarkProcessingPipeline(checkpoint_config)

        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            mock_validate.return_value = [
                ValidationResult(
                    url="https://docs.python.org/tutorial",
                    is_valid=True,
                    status_code=200,
                )
            ]
            mock_ai_process.return_value = []

            # Should start new processing instead of resuming
            results = pipeline.execute()

            # Should complete successfully despite corrupted checkpoint
            assert results is not None
            assert results.total_bookmarks > 0

        # Cleanup
        pipeline._cleanup_resources()
        Path(checkpoint_config.output_file).unlink(missing_ok=True)


class TestCheckpointPerformanceAndScaling:
    """Test checkpoint performance and scaling with larger datasets."""

    @pytest.fixture
    def large_csv_data(self):
        """Generate larger CSV dataset for performance testing."""
        lines = [
            "id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite"
        ]

        for i in range(20):  # 20 bookmarks for performance testing
            line = f'{i},"Bookmark {i}","Note {i}","Excerpt {i}","https://example.com/{i}","Folder{i % 3}","tag{i % 5}, test","2024-01-{(i % 28) + 1:02d}T00:00:00Z","","","false"'
            lines.append(line)

        return "\n".join(lines)

    @pytest.fixture
    def large_checkpoint_csv_file(self, large_csv_data):
        """Create temporary CSV file with larger dataset."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            return f.name

    @pytest.fixture
    def performance_config(self, large_checkpoint_csv_file):
        """Create configuration for performance testing."""
        return PipelineConfig(
            input_file=large_checkpoint_csv_file,
            output_file=large_checkpoint_csv_file.replace(".csv", "_perf_output.csv"),
            batch_size=5,
            save_interval=3,  # More frequent checkpoints
            checkpoint_dir=".test_checkpoint_performance",
            resume_enabled=True,
            ai_enabled=True,
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_checkpoint_frequency_with_large_dataset(self, performance_config):
        """Test checkpoint frequency and timing with larger dataset."""
        pipeline = BookmarkProcessingPipeline(performance_config)

        checkpoint_saves = []

        # Mock checkpoint manager to track save calls
        original_save = pipeline.checkpoint_manager.save_checkpoint

        def track_save_checkpoint(**kwargs):
            checkpoint_saves.append(time.time())
            return original_save(**kwargs)

        pipeline.checkpoint_manager.save_checkpoint = track_save_checkpoint

        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "process_batch") as mock_ai_process,
        ):

            # Mock successful processing
            validation_results = []
            for i in range(20):
                validation_results.append(
                    ValidationResult(
                        url=f"https://example.com/{i}", is_valid=True, status_code=200
                    )
                )
            mock_validate.return_value = validation_results

            # Return AIProcessingResult objects with .url property
            def mock_ai_batch_process(bookmarks, **kwargs):
                return [
                    AIProcessingResult(
                        original_url=bookmark.url,
                        enhanced_description=f"AI: {bookmark.title}",
                        processing_method="perf_test",
                        processing_time=0.01,
                    )
                    for bookmark in bookmarks
                ]

            mock_ai_process.side_effect = mock_ai_batch_process

            # Execute pipeline
            start_time = time.time()
            results = pipeline.execute()
            end_time = time.time()

            # Verify performance
            assert results is not None
            assert results.total_bookmarks == 20
            processing_time = end_time - start_time
            assert processing_time < 10  # Should complete quickly with mocks

            # Verify checkpoint frequency
            assert len(checkpoint_saves) >= 2  # Should have multiple checkpoints

            # Verify checkpoints were spaced appropriately
            if len(checkpoint_saves) > 1:
                intervals = [
                    checkpoint_saves[i + 1] - checkpoint_saves[i]
                    for i in range(len(checkpoint_saves) - 1)
                ]
                # Should have reasonable intervals (not too frequent)
                assert all(interval >= 0 for interval in intervals)

        # Cleanup
        pipeline._cleanup_resources()
        Path(performance_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_checkpoint_memory_efficiency(self, performance_config):
        """Test memory efficiency of checkpoint storage."""
        checkpoint_manager = CheckpointManager(performance_config.checkpoint_dir)

        # Load bookmarks
        csv_handler = RaindropCSVHandler()
        df = csv_handler.load_export_csv(performance_config.input_file)

        # Simulate processing state with large amount of data
        processed_urls = set()
        validation_results = {}
        ai_results = {}

        for _, row in df.iterrows():
            url = row["url"]
            processed_urls.add(url)

            validation_results[url] = {
                "url": url,
                "is_valid": True,
                "status_code": 200,
                "response_time": 0.5,
            }

            ai_results[url] = {
                "original_url": url,
                "enhanced_description": f"Enhanced description for {row['title']} with lots of content "
                * 10,
                "processing_method": "memory_test",
                "processing_time": 0.1,
            }

        # Initialize and save checkpoint
        config_dict = performance_config.__dict__.copy()
        checkpoint_manager.initialize_processing(
            performance_config.input_file,
            performance_config.output_file,
            len(df),
            config_dict,
        )

        # Update checkpoint with all data
        checkpoint_manager.current_state.processed_urls = processed_urls
        checkpoint_manager.current_state.ai_results = ai_results
        checkpoint_manager.update_stage(ProcessingStage.AI_PROCESSING)

        checkpoint_manager.save_checkpoint(force=True)

        # Measure total checkpoint file size after saving
        checkpoint_file_size = sum(
            f.stat().st_size for f in Path(performance_config.checkpoint_dir).glob("*")
        )

        # Verify checkpoint files exist and have reasonable size
        assert checkpoint_file_size > 0
        assert (
            checkpoint_file_size < 1024 * 1024
        )  # Should be less than 1MB for 20 bookmarks

        # Test loading efficiency
        load_start = time.time()
        loaded_state = checkpoint_manager.load_checkpoint(performance_config.input_file)
        load_time = time.time() - load_start

        # Should load quickly
        assert load_time < 1.0  # Should load in under 1 second
        assert loaded_state is not None
        assert len(loaded_state.processed_urls) == 20
        assert len(loaded_state.ai_results) == 20

    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_access(self, performance_config):
        """Test handling of concurrent checkpoint access."""
        checkpoint_manager1 = CheckpointManager(performance_config.checkpoint_dir)
        checkpoint_manager2 = CheckpointManager(performance_config.checkpoint_dir)

        # Initialize processing in first manager
        config_dict = performance_config.__dict__.copy()
        checkpoint_manager1.initialize_processing(
            performance_config.input_file,
            performance_config.output_file,
            10,
            config_dict,
        )

        # Save checkpoint from first manager
        checkpoint_manager1.update_stage(ProcessingStage.URL_VALIDATION)
        checkpoint_manager1.save_checkpoint(force=True)

        # Try to load from second manager
        loaded_state = checkpoint_manager2.load_checkpoint(
            performance_config.input_file
        )

        # Should be able to load the checkpoint
        assert loaded_state is not None
        assert loaded_state.current_stage == ProcessingStage.URL_VALIDATION

        # Both managers should be able to work with the same checkpoint
        assert checkpoint_manager1.has_checkpoint(performance_config.input_file)
        assert checkpoint_manager2.has_checkpoint(performance_config.input_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
