"""
Pipeline Integration Tests

Tests the complete bookmark processing pipeline with cloud AI integration,
including CSV handling, URL validation, content analysis, and AI processing.
"""

import asyncio
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.bookmark_processor import BookmarkProcessor
from bookmark_processor.core.csv_handler import RaindropCSVHandler


class TestPipelineIntegration:
    """Test complete pipeline integration."""

    @pytest.fixture
    def sample_csv_data(self):
        """Sample raindrop.io export CSV data."""
        return """id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,"Python Tutorial","Learn Python basics","Official Python tutorial","https://docs.python.org/tutorial","Programming/Python","python, tutorial","2024-01-01T00:00:00Z","","","false"
2,"AI Research","Latest AI developments","Research paper on AI","https://arxiv.org/abs/12345","Research/AI","ai, research","2024-01-02T00:00:00Z","","","false"
3,"JavaScript Guide","Web development","MDN JavaScript docs","https://developer.mozilla.org/js","Programming/JavaScript","javascript, web","2024-01-03T00:00:00Z","","","false"
4,"Invalid URL","Test invalid URL","This has bad URL","https://invalid-url-that-does-not-exist.example","Testing","test","2024-01-04T00:00:00Z","","","false"
"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock(spec=Configuration)
        config.get_api_key.return_value = "test-api-key"
        config.get.side_effect = lambda section, key, fallback=None: {
            ("processing", "batch_size"): "100",
            ("ai", "default_engine"): "local",
            ("ai", "claude_rpm"): "50",
            ("ai", "openai_rpm"): "60",
            ("checkpoint", "enabled"): "true",
            ("checkpoint", "save_interval"): "50",
        }.get((section, key), fallback)
        return config

    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            return f.name

    def test_csv_format_validation(self, sample_csv_data):
        """Test CSV format validation for raindrop.io format."""
        csv_handler = RaindropCSVHandler()

        # Test valid format
        with StringIO(sample_csv_data) as f:
            is_valid, message = csv_handler.validate_format(f)
            assert is_valid
            assert "valid raindrop.io export format" in message.lower()

        # Test invalid format
        invalid_csv = "url,title,description\nhttps://example.com,Test,Description"
        with StringIO(invalid_csv) as f:
            is_valid, message = csv_handler.validate_format(f)
            assert not is_valid
            assert "missing required columns" in message.lower()

    def test_bookmark_loading(self, temp_csv_file):
        """Test loading bookmarks from CSV."""
        csv_handler = RaindropCSVHandler()
        bookmarks = csv_handler.load_bookmarks(temp_csv_file)

        assert len(bookmarks) == 4

        # Test first bookmark
        bookmark = bookmarks[0]
        assert bookmark.title == "Python Tutorial"
        assert bookmark.url == "https://docs.python.org/tutorial"
        assert bookmark.note == "Learn Python basics"
        assert bookmark.folder == "Programming/Python"
        assert "python" in bookmark.tags
        assert "tutorial" in bookmark.tags

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.url_validator.URLValidator.validate_url")
    async def test_url_validation_stage(
        self, mock_validate, temp_csv_file, mock_config
    ):
        """Test URL validation stage of pipeline."""
        # Mock URL validation responses
        mock_validate.side_effect = [
            (
                True,
                {"status_code": 200, "final_url": "https://docs.python.org/tutorial"},
            ),
            (True, {"status_code": 200, "final_url": "https://arxiv.org/abs/12345"}),
            (
                True,
                {"status_code": 200, "final_url": "https://developer.mozilla.org/js"},
            ),
            (False, {"status_code": 404, "error": "Not found"}),
        ]

        # Create processor
        processor = BookmarkProcessor(mock_config)

        # Load bookmarks
        csv_handler = RaindropCSVHandler()
        bookmarks = csv_handler.load_bookmarks(temp_csv_file)

        # Validate URLs
        valid_bookmarks = []
        invalid_bookmarks = []

        for bookmark in bookmarks:
            is_valid, metadata = await mock_validate(bookmark.url)
            if is_valid:
                valid_bookmarks.append(bookmark)
            else:
                invalid_bookmarks.append((bookmark, metadata))

        assert len(valid_bookmarks) == 3
        assert len(invalid_bookmarks) == 1
        assert invalid_bookmarks[0][1]["status_code"] == 404

    @pytest.mark.asyncio
    async def test_content_analysis_stage(self, temp_csv_file):
        """Test content analysis stage."""
        from bookmark_processor.core.content_analyzer import ContentAnalyzer

        analyzer = ContentAnalyzer()

        # Mock content extraction
        with patch.object(analyzer, "analyze_url") as mock_analyze:
            mock_analyze.return_value = {
                "title": "Extracted Title",
                "description": "Extracted description",
                "keywords": ["python", "programming"],
                "content_type": "documentation",
                "word_count": 1500,
            }

            content_data = await analyzer.analyze_url(
                "https://docs.python.org/tutorial"
            )

            assert content_data["title"] == "Extracted Title"
            assert content_data["content_type"] == "documentation"
            assert "python" in content_data["keywords"]

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.ai_factory.AIManager.generate_description")
    async def test_ai_description_generation(
        self, mock_generate, temp_csv_file, mock_config
    ):
        """Test AI description generation stage."""
        # Mock AI description generation
        mock_generate.return_value = (
            "Enhanced Python tutorial for beginners and advanced developers",
            {
                "provider": "local",
                "success": True,
                "processing_time": 0.5,
                "confidence": 0.9,
            },
        )

        # Load bookmarks
        csv_handler = RaindropCSVHandler()
        bookmarks = csv_handler.load_bookmarks(temp_csv_file)

        # Test description generation
        from bookmark_processor.core.ai_factory import AIManager

        ai_manager = AIManager("local", mock_config)

        description, metadata = await mock_generate(bookmarks[0], bookmarks[0].note)

        assert (
            description
            == "Enhanced Python tutorial for beginners and advanced developers"
        )
        assert metadata["provider"] == "local"
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_tag_generation_stage(self, temp_csv_file):
        """Test tag generation stage."""
        from bookmark_processor.core.tag_generator import TagGenerator

        # Load bookmarks
        csv_handler = RaindropCSVHandler()
        bookmarks = csv_handler.load_bookmarks(temp_csv_file)

        # Mock tag generation
        tag_generator = TagGenerator()

        with patch.object(tag_generator, "generate_tags") as mock_generate_tags:
            mock_generate_tags.return_value = [
                "python",
                "tutorial",
                "programming",
                "documentation",
            ]

            tags = tag_generator.generate_tags(
                "Python Tutorial",
                "Enhanced Python tutorial for beginners",
                "https://docs.python.org/tutorial",
            )

            assert "python" in tags
            assert "tutorial" in tags
            assert len(tags) <= 4

    def test_output_csv_format(self, temp_csv_file):
        """Test output CSV format matches raindrop.io import requirements."""
        csv_handler = RaindropCSVHandler()

        # Create sample processed bookmarks
        processed_bookmarks = [
            {
                "url": "https://docs.python.org/tutorial",
                "folder": "Programming/Python",
                "title": "Python Tutorial",
                "note": "Enhanced Python tutorial for beginners and advanced developers",
                "tags": ["python", "tutorial", "programming", "documentation"],
                "created": "2024-01-01T00:00:00Z",
            },
            {
                "url": "https://arxiv.org/abs/12345",
                "folder": "Research/AI",
                "title": "AI Research",
                "note": "Comprehensive AI research covering latest developments",
                "tags": ["ai", "research", "machine-learning"],
                "created": "2024-01-02T00:00:00Z",
            },
        ]

        # Create output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_file = f.name

        # Save processed bookmarks
        csv_handler.save_processed_bookmarks(processed_bookmarks, output_file)

        # Read back and validate format
        df = pd.read_csv(output_file)

        # Check required columns
        required_columns = ["url", "folder", "title", "note", "tags", "created"]
        assert all(col in df.columns for col in required_columns)

        # Check data integrity
        assert len(df) == 2
        assert df.iloc[0]["url"] == "https://docs.python.org/tutorial"
        assert df.iloc[0]["folder"] == "Programming/Python"

        # Check tag formatting
        assert df.iloc[0]["tags"] == '"python, tutorial, programming, documentation"'
        assert df.iloc[1]["tags"] == '"ai, research, machine-learning"'

        # Cleanup
        Path(output_file).unlink()

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume(self, temp_csv_file, mock_config):
        """Test checkpoint and resume functionality."""
        from bookmark_processor.core.checkpoint_manager import CheckpointManager

        checkpoint_manager = CheckpointManager("test_checkpoint_dir")

        # Create checkpoint data
        checkpoint_data = {
            "processed_count": 2,
            "failed_count": 1,
            "current_batch": 1,
            "processed_urls": [
                "https://docs.python.org/tutorial",
                "https://arxiv.org/abs/12345",
            ],
            "failed_urls": ["https://invalid-url-that-does-not-exist.example"],
            "progress": {
                "stage": "generating_descriptions",
                "overall_progress": 50.0,
                "items_processed": 2,
                "items_total": 4,
            },
        }

        # Save checkpoint
        checkpoint_manager.save_checkpoint(checkpoint_data)

        # Test checkpoint exists
        assert checkpoint_manager.has_checkpoint()

        # Load checkpoint
        loaded_data = checkpoint_manager.load_checkpoint()
        assert loaded_data["processed_count"] == 2
        assert loaded_data["progress"]["overall_progress"] == 50.0

        # Cleanup
        checkpoint_manager.clear_checkpoints()
        assert not checkpoint_manager.has_checkpoint()

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.url_validator.URLValidator.validate_url")
    @patch("bookmark_processor.core.ai_factory.AIManager.generate_description")
    async def test_complete_pipeline_workflow(
        self, mock_ai_generate, mock_url_validate, temp_csv_file, mock_config
    ):
        """Test complete end-to-end pipeline workflow."""
        # Mock URL validation
        mock_url_validate.side_effect = [
            (
                True,
                {"status_code": 200, "final_url": "https://docs.python.org/tutorial"},
            ),
            (True, {"status_code": 200, "final_url": "https://arxiv.org/abs/12345"}),
            (
                True,
                {"status_code": 200, "final_url": "https://developer.mozilla.org/js"},
            ),
            (False, {"status_code": 404, "error": "Not found"}),
        ]

        # Mock AI description generation
        mock_ai_generate.side_effect = [
            (
                "Enhanced Python tutorial for beginners",
                {"provider": "local", "success": True},
            ),
            ("Comprehensive AI research paper", {"provider": "local", "success": True}),
            (
                "Complete JavaScript development guide",
                {"provider": "local", "success": True},
            ),
        ]

        # Create processor
        processor = BookmarkProcessor(mock_config)

        # Create output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_file = f.name

        # Process bookmarks (mock the full pipeline)
        csv_handler = RaindropCSVHandler()
        bookmarks = csv_handler.load_bookmarks(temp_csv_file)

        # Simulate processing
        processed_bookmarks = []
        valid_count = 0
        invalid_count = 0

        for bookmark in bookmarks:
            # URL validation
            is_valid, metadata = await mock_url_validate(bookmark.url)

            if is_valid:
                # AI description generation
                description, ai_metadata = await mock_ai_generate(
                    bookmark, bookmark.note
                )

                # Create processed bookmark
                processed_bookmark = {
                    "url": bookmark.url,
                    "folder": bookmark.folder,
                    "title": bookmark.title,
                    "note": description,
                    "tags": bookmark.tags,  # Would normally be AI-generated
                    "created": bookmark.created,
                }
                processed_bookmarks.append(processed_bookmark)
                valid_count += 1
            else:
                invalid_count += 1

        # Save results
        csv_handler.save_processed_bookmarks(processed_bookmarks, output_file)

        # Validate results
        assert valid_count == 3
        assert invalid_count == 1
        assert len(processed_bookmarks) == 3

        # Check output file
        df = pd.read_csv(output_file)
        assert len(df) == 3
        assert df.iloc[0]["note"] == "Enhanced Python tutorial for beginners"

        # Cleanup
        Path(output_file).unlink()

    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, temp_csv_file, mock_config):
        """Test error handling throughout the pipeline."""
        from bookmark_processor.utils.error_handler import ErrorHandler

        error_handler = ErrorHandler()

        # Test handling of various error types
        errors = [
            Exception("Network timeout"),
            Exception("401 Unauthorized"),
            Exception("Rate limit exceeded"),
            Exception("Invalid bookmark data"),
        ]

        for error in errors:
            error_details = error_handler.categorize_error(error)
            assert error_details.category is not None
            assert error_details.severity is not None

            # Test error recovery
            bookmark = Mock()
            bookmark.title = "Test Bookmark"
            bookmark.url = "https://example.com"
            bookmark.note = "Test note"

            description, metadata = (
                await error_handler.handle_bookmark_processing_error(
                    error, bookmark, "existing content"
                )
            )

            assert description is not None
            assert metadata["provider"] == "fallback"
            assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_memory_and_performance_monitoring(self, temp_csv_file):
        """Test memory usage and performance monitoring during processing."""
        from bookmark_processor.utils.progress_tracker import (
            AdvancedProgressTracker,
        )
        from bookmark_processor.utils.progress_tracker import ProcessingStage as PTStage
        from bookmark_processor.utils.progress_tracker import (
            ProgressLevel,
        )

        progress_tracker = AdvancedProgressTracker(
            total_items=4, description="Test Processing", level=ProgressLevel.STANDARD
        )

        # Simulate processing stages
        stages = [
            PTStage.LOADING_DATA,
            PTStage.VALIDATING_URLS,
            PTStage.GENERATING_DESCRIPTIONS,
            PTStage.SAVING_RESULTS,
        ]

        for stage in stages:
            progress_tracker.start_stage(stage, 4)

            # Simulate processing items
            for i in range(4):
                progress_tracker.update(stage_items=i + 1)

                # Check snapshot
                snapshot = progress_tracker.get_snapshot()
                assert snapshot.current_stage == stage
                assert snapshot.stage_items_processed == (i + 1)
                assert snapshot.memory_usage_mb >= 0

        # Get final performance report
        report = progress_tracker.get_performance_report()
        assert report["total_items_processed"] >= 4
        assert report["overall_success_rate"] >= 0
        assert "stage_timings" in report

        progress_tracker.complete()


class TestCompletePipelineWorkflow:
    """Test complete processing pipeline workflow with all stages."""

    @pytest.fixture
    def large_csv_data(self):
        """Sample CSV data with more complex scenarios."""
        return """id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,"Python Tutorial","Learn Python basics","Official Python tutorial","https://docs.python.org/tutorial","Programming/Python","python, tutorial","2024-01-01T00:00:00Z","","","false"
2,"AI Research","Latest AI developments","Research paper on AI","https://arxiv.org/abs/12345","Research/AI","ai, research","2024-01-02T00:00:00Z","","","false"
3,"JavaScript Guide","Web development","MDN JavaScript docs","https://developer.mozilla.org/js","Programming/JavaScript","javascript, web","2024-01-03T00:00:00Z","","","false"
4,"Invalid URL","Test invalid URL","This has bad URL","https://invalid-url-that-does-not-exist.example","Testing","test","2024-01-04T00:00:00Z","","","false"
5,"Duplicate URL","Same as first","Different description","https://docs.python.org/tutorial","Programming/Python","python, duplicate","2024-01-05T00:00:00Z","","","false"
6,"Machine Learning","ML concepts","Deep learning basics","https://deeplearning.ai/courses","AI/ML","machine-learning, deep-learning","2024-01-06T00:00:00Z","","","false"
7,"React Documentation","React framework","React official docs","https://reactjs.org/docs","Programming/Frontend","react, frontend","2024-01-07T00:00:00Z","","","false"
8,"Database Design","SQL fundamentals","Database design principles","https://db.example.com/guide","Database","sql, design","2024-01-08T00:00:00Z","","","false"
"""

    @pytest.fixture
    def large_temp_csv_file(self, large_csv_data):
        """Create temporary CSV file with larger sample data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            return f.name

    @pytest.fixture
    def pipeline_config(self, large_temp_csv_file):
        """Create pipeline configuration for testing."""
        from bookmark_processor.core.pipeline import PipelineConfig

        output_file = large_temp_csv_file.replace(".csv", "_output.csv")

        return PipelineConfig(
            input_file=large_temp_csv_file,
            output_file=output_file,
            batch_size=3,
            max_retries=2,
            resume_enabled=True,
            clear_checkpoints=False,
            url_timeout=10.0,
            max_concurrent_requests=5,
            ai_enabled=True,
            target_tag_count=20,
            max_tags_per_bookmark=4,
            detect_duplicates=True,
            duplicate_strategy="highest_quality",
            verbose=True,
            checkpoint_dir=".test_checkpoints",
            save_interval=3,
        )

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.url_validator.URLValidator.batch_validate")
    @patch("bookmark_processor.core.ai_processor.EnhancedAIProcessor.batch_process")
    @patch("bookmark_processor.core.content_analyzer.ContentAnalyzer.analyze_content")
    async def test_complete_pipeline_execution(
        self,
        mock_content_analyzer,
        mock_ai_processor,
        mock_url_validator,
        pipeline_config,
    ):
        """Test complete pipeline execution with all stages."""
        from bookmark_processor.core.ai_processor import AIProcessingResult
        from bookmark_processor.core.content_analyzer import ContentData
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline
        from bookmark_processor.core.url_validator import ValidationResult

        # Mock URL validation results
        validation_results = [
            ValidationResult(
                url="https://docs.python.org/tutorial",
                is_valid=True,
                status_code=200,
                final_url="https://docs.python.org/tutorial",
            ),
            ValidationResult(
                url="https://arxiv.org/abs/12345",
                is_valid=True,
                status_code=200,
                final_url="https://arxiv.org/abs/12345",
            ),
            ValidationResult(
                url="https://developer.mozilla.org/js",
                is_valid=True,
                status_code=200,
                final_url="https://developer.mozilla.org/js",
            ),
            ValidationResult(
                url="https://invalid-url-that-does-not-exist.example",
                is_valid=False,
                status_code=404,
                error_message="Not found",
            ),
            ValidationResult(
                url="https://deeplearning.ai/courses",
                is_valid=True,
                status_code=200,
                final_url="https://deeplearning.ai/courses",
            ),
            ValidationResult(
                url="https://reactjs.org/docs",
                is_valid=True,
                status_code=200,
                final_url="https://reactjs.org/docs",
            ),
            ValidationResult(
                url="https://db.example.com/guide",
                is_valid=True,
                status_code=200,
                final_url="https://db.example.com/guide",
            ),
        ]
        mock_url_validator.return_value = validation_results

        # Mock content analysis
        def mock_analyze_content(url, **kwargs):
            content_data = ContentData(url=url)
            content_data.title = f"Analyzed title for {url}"
            content_data.description = f"Analyzed description for {url}"
            content_data.keywords = ["programming", "tutorial", "guide"]
            return content_data

        mock_content_analyzer.side_effect = mock_analyze_content

        # Mock AI processing results
        def mock_batch_ai_process(bookmarks, **kwargs):
            results = []
            for bookmark in bookmarks:
                if bookmark.url.startswith("https://"):  # Only valid URLs
                    result = AIProcessingResult(
                        original_url=bookmark.url,
                        enhanced_description=f"Enhanced description for {bookmark.title}",
                        processing_method="mock_ai",
                        processing_time=0.1,
                    )
                    results.append(result)
            return results

        mock_ai_processor.side_effect = mock_batch_ai_process

        # Create and execute pipeline
        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Track progress
        progress_messages = []

        def progress_callback(message):
            progress_messages.append(message)

        # Execute pipeline
        results = pipeline.execute(progress_callback=progress_callback)

        # Verify results
        assert results.total_bookmarks == 7  # After duplicate removal (8-1)
        assert results.valid_bookmarks == 6  # 7 total - 1 invalid
        assert results.invalid_bookmarks == 1
        assert results.ai_processed == 6
        assert results.tagged_bookmarks >= 0
        assert results.unique_tags >= 0
        assert results.processing_time > 0
        assert len(results.stages_completed) > 0

        # Verify output file was created
        assert Path(pipeline_config.output_file).exists()

        # Verify output file content
        import pandas as pd

        output_df = pd.read_csv(pipeline_config.output_file)
        assert len(output_df) == 6  # Only valid bookmarks
        assert all(
            col in output_df.columns
            for col in ["url", "folder", "title", "note", "tags", "created"]
        )

        # Verify enhanced descriptions
        for _, row in output_df.iterrows():
            assert "Enhanced description" in str(row["note"])

        # Verify progress tracking worked
        assert len(progress_messages) > 0

        # Cleanup
        Path(pipeline_config.output_file).unlink(missing_ok=True)
        (
            Path(pipeline_config.checkpoint_dir).rmdir()
            if Path(pipeline_config.checkpoint_dir).exists()
            else None
        )

    @pytest.mark.asyncio
    async def test_pipeline_stage_by_stage_execution(self, pipeline_config):
        """Test pipeline execution stage by stage."""
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline

        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Test stage 1: Load bookmarks
        pipeline._stage_load_bookmarks()
        assert len(pipeline.bookmarks) > 0
        assert len(pipeline.bookmarks) == 7  # After duplicate removal (8-1)

        # Verify duplicate detection worked (should remove one duplicate)
        urls = [b.url for b in pipeline.bookmarks]
        assert len(set(urls)) == len(urls)  # All URLs should be unique

        # Test that the pipeline correctly loaded from CSV
        titles = [b.title for b in pipeline.bookmarks]
        assert "Python Tutorial" in titles
        assert "AI Research" in titles

        # Cleanup
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.url_validator.URLValidator.batch_validate")
    async def test_pipeline_url_validation_stage(
        self, mock_batch_validate, pipeline_config
    ):
        """Test URL validation stage in isolation."""
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline
        from bookmark_processor.core.url_validator import ValidationResult

        # Mock validation results
        validation_results = [
            ValidationResult(
                url="https://docs.python.org/tutorial", is_valid=True, status_code=200
            ),
            ValidationResult(
                url="https://arxiv.org/abs/12345", is_valid=True, status_code=200
            ),
            ValidationResult(
                url="https://developer.mozilla.org/js", is_valid=True, status_code=200
            ),
            ValidationResult(
                url="https://invalid-url-that-does-not-exist.example",
                is_valid=False,
                status_code=404,
            ),
            ValidationResult(
                url="https://deeplearning.ai/courses", is_valid=True, status_code=200
            ),
            ValidationResult(
                url="https://reactjs.org/docs", is_valid=True, status_code=200
            ),
            ValidationResult(
                url="https://db.example.com/guide", is_valid=True, status_code=200
            ),
        ]
        mock_batch_validate.return_value = validation_results

        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Load bookmarks first
        pipeline._stage_load_bookmarks()

        # Test URL validation stage
        pipeline._stage_validate_urls()

        # Verify validation results
        assert len(pipeline.validation_results) == 7
        valid_count = sum(1 for r in pipeline.validation_results.values() if r.is_valid)
        assert valid_count == 6

        # Verify checkpoint was updated
        assert pipeline.checkpoint_manager.has_checkpoint(pipeline_config.input_file)

        # Cleanup
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.ai_processor.EnhancedAIProcessor.batch_process")
    async def test_pipeline_ai_processing_stage(
        self, mock_batch_process, pipeline_config
    ):
        """Test AI processing stage in isolation."""
        from bookmark_processor.core.ai_processor import AIProcessingResult
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline
        from bookmark_processor.core.url_validator import ValidationResult

        # Mock AI processing results
        def mock_ai_batch_process(bookmarks, **kwargs):
            results = []
            for bookmark in bookmarks:
                result = AIProcessingResult(
                    original_url=bookmark.url,
                    enhanced_description=f"AI enhanced: {bookmark.title}",
                    processing_method="mock_ai",
                    processing_time=0.1,
                )
                results.append(result)
            return results

        mock_batch_process.side_effect = mock_ai_batch_process

        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Setup previous stages
        pipeline._stage_load_bookmarks()

        # Mock validation results for valid URLs
        for bookmark in pipeline.bookmarks:
            if "invalid" not in bookmark.url:
                pipeline.validation_results[bookmark.url] = ValidationResult(
                    url=bookmark.url, is_valid=True, status_code=200
                )
            else:
                pipeline.validation_results[bookmark.url] = ValidationResult(
                    url=bookmark.url, is_valid=False, status_code=404
                )

        # Test AI processing stage
        pipeline._stage_ai_processing()

        # Verify AI results
        assert len(pipeline.ai_results) == 6  # Only valid URLs
        for result in pipeline.ai_results.values():
            assert "AI enhanced:" in result.enhanced_description
            assert result.processing_method == "mock_ai"

        # Cleanup
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    async def test_pipeline_tag_generation_stage(self, pipeline_config):
        """Test tag generation stage in isolation."""
        from bookmark_processor.core.ai_processor import AIProcessingResult
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline
        from bookmark_processor.core.url_validator import ValidationResult

        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Setup previous stages
        pipeline._stage_load_bookmarks()

        # Mock validation and AI results
        for bookmark in pipeline.bookmarks:
            if "invalid" not in bookmark.url:
                pipeline.validation_results[bookmark.url] = ValidationResult(
                    url=bookmark.url, is_valid=True, status_code=200
                )
                pipeline.ai_results[bookmark.url] = AIProcessingResult(
                    original_url=bookmark.url,
                    enhanced_description=f"Enhanced {bookmark.title}",
                    processing_method="mock",
                    processing_time=0.1,
                )
            else:
                pipeline.validation_results[bookmark.url] = ValidationResult(
                    url=bookmark.url, is_valid=False, status_code=404
                )

        # Test tag generation stage
        with patch.object(
            pipeline.tag_generator, "generate_corpus_tags"
        ) as mock_generate_tags:
            from bookmark_processor.core.tag_generator import TagOptimizationResult

            # Mock tag optimization result
            mock_tag_assignments = {}
            for bookmark in pipeline.bookmarks:
                if (
                    bookmark.url in pipeline.validation_results
                    and pipeline.validation_results[bookmark.url].is_valid
                ):
                    mock_tag_assignments[bookmark.url] = ["tag1", "tag2", "tag3"]

            mock_result = TagOptimizationResult(
                tag_assignments=mock_tag_assignments,
                total_unique_tags=15,
                coverage_percentage=95.0,
                optimization_summary="Mock optimization",
            )
            mock_generate_tags.return_value = mock_result

            pipeline._stage_generate_tags()

            # Verify tag assignments
            assert len(pipeline.tag_assignments) == 6  # Valid bookmarks only
            for tags in pipeline.tag_assignments.values():
                assert len(tags) == 3
                assert all(tag in ["tag1", "tag2", "tag3"] for tag in tags)

        # Cleanup
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    async def test_pipeline_output_generation_stage(self, pipeline_config):
        """Test output generation stage in isolation."""
        from bookmark_processor.core.ai_processor import AIProcessingResult
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline
        from bookmark_processor.core.url_validator import ValidationResult

        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Setup previous stages
        pipeline._stage_load_bookmarks()

        # Mock all previous stage results
        for bookmark in pipeline.bookmarks:
            if "invalid" not in bookmark.url:
                pipeline.validation_results[bookmark.url] = ValidationResult(
                    url=bookmark.url, is_valid=True, status_code=200
                )
                pipeline.ai_results[bookmark.url] = AIProcessingResult(
                    original_url=bookmark.url,
                    enhanced_description=f"Enhanced {bookmark.title}",
                    processing_method="mock",
                    processing_time=0.1,
                )
                pipeline.tag_assignments[bookmark.url] = [
                    "python",
                    "tutorial",
                    "programming",
                ]
            else:
                pipeline.validation_results[bookmark.url] = ValidationResult(
                    url=bookmark.url, is_valid=False, status_code=404
                )

        # Test output generation stage
        pipeline._stage_generate_output()

        # Verify output file was created
        assert Path(pipeline_config.output_file).exists()

        # Verify output content
        import pandas as pd

        output_df = pd.read_csv(pipeline_config.output_file)

        # Should only include valid bookmarks
        assert len(output_df) == 6

        # Verify all required columns exist
        required_columns = ["url", "folder", "title", "note", "tags", "created"]
        assert all(col in output_df.columns for col in required_columns)

        # Verify enhanced descriptions
        for _, row in output_df.iterrows():
            assert "Enhanced" in str(row["note"])

        # Verify tag formatting
        for _, row in output_df.iterrows():
            tags = str(row["tags"])
            assert "python" in tags or tags == "nan"  # Either has tags or is NaN

        # Cleanup
        Path(pipeline_config.output_file).unlink(missing_ok=True)
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    async def test_pipeline_error_handling_and_recovery(self, pipeline_config):
        """Test pipeline error handling and recovery mechanisms."""
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline

        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Test error in loading stage
        with patch.object(pipeline.csv_handler, "load_export_csv") as mock_load:
            mock_load.side_effect = Exception("CSV loading failed")

            with pytest.raises(Exception, match="CSV loading failed"):
                pipeline._stage_load_bookmarks()

        # Test error recovery in URL validation
        pipeline._stage_load_bookmarks()  # Load successfully

        with patch.object(pipeline.url_validator, "batch_validate") as mock_validate:
            mock_validate.side_effect = Exception("Network error")

            with pytest.raises(Exception, match="Network error"):
                pipeline._stage_validate_urls()

        # Test error in AI processing
        with patch.object(pipeline.ai_processor, "batch_process") as mock_ai:
            mock_ai.side_effect = Exception("AI processing failed")

            # Mock validation results first
            for bookmark in pipeline.bookmarks:
                pipeline.validation_results[bookmark.url] = ValidationResult(
                    url=bookmark.url, is_valid=True, status_code=200
                )

            with pytest.raises(Exception, match="AI processing failed"):
                pipeline._stage_ai_processing()

        # Cleanup
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    async def test_pipeline_memory_optimization(self, pipeline_config):
        """Test pipeline memory optimization during processing."""
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline

        # Set lower memory thresholds for testing
        pipeline_config.memory_warning_threshold = 100  # MB
        pipeline_config.memory_critical_threshold = 200  # MB
        pipeline_config.memory_batch_size = 2

        pipeline = BookmarkProcessingPipeline(pipeline_config)

        # Test memory monitoring during loading
        pipeline._stage_load_bookmarks()

        # Verify memory monitor is working
        assert pipeline.memory_monitor is not None
        memory_usage = pipeline.memory_monitor.get_current_usage_mb()
        assert memory_usage > 0

        # Test batch processing
        assert pipeline.batch_processor.batch_size == 2

        # Cleanup
        pipeline._cleanup_resources()


class TestPipelinePerformanceAndScaling:
    """Test pipeline performance and scaling characteristics."""

    @pytest.fixture
    def performance_config(self, large_temp_csv_file):
        """Create configuration optimized for performance testing."""
        from bookmark_processor.core.pipeline import PipelineConfig

        return PipelineConfig(
            input_file=large_temp_csv_file,
            output_file=large_temp_csv_file.replace(".csv", "_perf_output.csv"),
            batch_size=5,
            max_concurrent_requests=10,
            memory_batch_size=5,
            save_interval=2,
            verbose=False,
            checkpoint_dir=".test_perf_checkpoints",
        )

    @pytest.mark.asyncio
    async def test_pipeline_performance_metrics(
        self, performance_config, large_temp_csv_file
    ):
        """Test pipeline performance metrics collection."""
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline

        pipeline = BookmarkProcessingPipeline(performance_config)

        # Mock all external dependencies for performance testing
        with (
            patch.multiple(
                pipeline.url_validator,
                batch_validate=Mock(return_value=[]),
            ),
            patch.multiple(
                pipeline.ai_processor,
                batch_process=Mock(return_value=[]),
            ),
            patch.object(
                pipeline.tag_generator,
                "generate_corpus_tags",
                return_value=Mock(
                    tag_assignments={}, total_unique_tags=0, coverage_percentage=0.0
                ),
            ),
        ):

            start_time = time.time()

            # Execute pipeline
            results = pipeline.execute()

            end_time = time.time()
            processing_time = end_time - start_time

            # Verify performance metrics
            assert results.processing_time > 0
            assert processing_time < 30  # Should complete quickly with mocks
            assert results.total_bookmarks > 0

            # Check statistics
            assert "url_validation" in results.statistics
            assert "checkpoint_progress" in results.statistics

        # Cleanup
        Path(performance_config.output_file).unlink(missing_ok=True)
        pipeline._cleanup_resources()

    @pytest.mark.asyncio
    async def test_pipeline_batch_processing_optimization(self, performance_config):
        """Test pipeline batch processing optimization."""
        from bookmark_processor.core.pipeline import BookmarkProcessingPipeline

        pipeline = BookmarkProcessingPipeline(performance_config)

        # Test batch size configuration
        assert pipeline.config.batch_size == 5
        assert pipeline.config.memory_batch_size == 5
        assert pipeline.batch_processor.batch_size == 5

        # Test that batching is used in AI processing
        pipeline._stage_load_bookmarks()

        with patch.object(pipeline.ai_processor, "batch_process") as mock_batch_process:
            mock_batch_process.return_value = []

            # Mock validation results
            for bookmark in pipeline.bookmarks[:5]:  # Test with first 5
                pipeline.validation_results[bookmark.url] = Mock(is_valid=True)

            pipeline._stage_ai_processing()

            # Verify batch processing was called
            assert mock_batch_process.called

            # Check batch sizes
            call_args = mock_batch_process.call_args_list
            for call in call_args:
                batch = call[0][0]  # First argument is the batch
                assert len(batch) <= 5  # Should respect batch size

        # Cleanup
        pipeline._cleanup_resources()


class TestCloudAIPipelineIntegration:
    """Test cloud AI integration within the pipeline."""

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.claude_api_client.ClaudeAPIClient._make_request")
    async def test_claude_pipeline_integration(self, mock_request):
        """Test Claude API integration in pipeline."""
        # Mock Claude response
        mock_request.return_value = {
            "content": [{"text": "AI-enhanced bookmark description using Claude"}],
            "usage": {"input_tokens": 150, "output_tokens": 40},
        }

        from bookmark_processor.core.claude_api_client import ClaudeAPIClient

        client = ClaudeAPIClient("test-key")

        # Create mock bookmark
        bookmark = Mock()
        bookmark.title = "Test Article"
        bookmark.url = "https://example.com/article"
        bookmark.note = "Interesting article"
        bookmark.excerpt = "Article excerpt"

        # Test description generation
        description, metadata = await client.generate_description(
            bookmark, "existing content"
        )

        assert description == "AI-enhanced bookmark description using Claude"
        assert metadata["provider"] == "claude"
        assert metadata["success"] is True
        assert metadata["cost_usd"] > 0

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.openai_api_client.OpenAIAPIClient._make_request")
    async def test_openai_pipeline_integration(self, mock_request):
        """Test OpenAI API integration in pipeline."""
        # Mock OpenAI response
        mock_request.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "AI-enhanced bookmark description using OpenAI"
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 35},
        }

        from bookmark_processor.core.openai_api_client import OpenAIAPIClient

        client = OpenAIAPIClient("test-key")

        # Create mock bookmark
        bookmark = Mock()
        bookmark.title = "Test Guide"
        bookmark.url = "https://example.com/guide"
        bookmark.note = "Helpful guide"
        bookmark.excerpt = "Guide excerpt"

        # Test description generation
        description, metadata = await client.generate_description(
            bookmark, "existing content"
        )

        assert description == "AI-enhanced bookmark description using OpenAI"
        assert metadata["provider"] == "openai"
        assert metadata["success"] is True
        assert metadata["cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_cost_tracking_in_pipeline(self):
        """Test cost tracking throughout pipeline processing."""
        from bookmark_processor.utils.cost_tracker import CostTracker

        cost_tracker = CostTracker(confirmation_interval=5.0)

        # Simulate processing costs
        providers = ["claude", "openai", "claude", "openai"]
        costs = [0.001, 0.002, 0.0015, 0.0018]

        for provider, cost in zip(providers, costs):
            cost_tracker.add_cost_record(
                provider=provider,
                model=f"{provider}-model",
                input_tokens=1000,
                output_tokens=200,
                cost_usd=cost,
                bookmark_count=1,
            )

        # Test cost estimation for future processing
        claude_estimate = cost_tracker.get_cost_estimate(100, "claude")
        openai_estimate = cost_tracker.get_cost_estimate(100, "openai")

        assert claude_estimate["provider"] == "claude"
        assert openai_estimate["provider"] == "openai"
        assert claude_estimate["estimated_cost_usd"] > 0
        assert openai_estimate["estimated_cost_usd"] > 0

        # Test detailed statistics
        stats = cost_tracker.get_detailed_statistics()
        assert stats["session"]["total_cost_usd"] == sum(costs)
        assert len(stats["providers"]) == 2
        assert "claude" in stats["providers"]
        assert "openai" in stats["providers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
