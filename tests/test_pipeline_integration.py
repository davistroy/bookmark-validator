"""
Pipeline Integration Tests

Tests the complete bookmark processing pipeline with cloud AI integration,
including CSV handling, URL validation, content analysis, and AI processing.
"""

import asyncio
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from io import StringIO

from bookmark_processor.core.pipeline import BookmarkProcessor
from bookmark_processor.core.csv_handler import CSVHandler
from bookmark_processor.config.configuration import Configuration


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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_data)
            return f.name
    
    def test_csv_format_validation(self, sample_csv_data):
        """Test CSV format validation for raindrop.io format."""
        csv_handler = CSVHandler()
        
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
        csv_handler = CSVHandler()
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
    @patch('bookmark_processor.core.url_validator.URLValidator.validate_url')
    async def test_url_validation_stage(self, mock_validate, temp_csv_file, mock_config):
        """Test URL validation stage of pipeline."""
        # Mock URL validation responses
        mock_validate.side_effect = [
            (True, {"status_code": 200, "final_url": "https://docs.python.org/tutorial"}),
            (True, {"status_code": 200, "final_url": "https://arxiv.org/abs/12345"}),
            (True, {"status_code": 200, "final_url": "https://developer.mozilla.org/js"}),
            (False, {"status_code": 404, "error": "Not found"}),
        ]
        
        # Create processor
        processor = BookmarkProcessor(mock_config)
        
        # Load bookmarks
        csv_handler = CSVHandler()
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
        with patch.object(analyzer, 'analyze_url') as mock_analyze:
            mock_analyze.return_value = {
                "title": "Extracted Title",
                "description": "Extracted description",
                "keywords": ["python", "programming"],
                "content_type": "documentation",
                "word_count": 1500,
            }
            
            content_data = await analyzer.analyze_url("https://docs.python.org/tutorial")
            
            assert content_data["title"] == "Extracted Title"
            assert content_data["content_type"] == "documentation"
            assert "python" in content_data["keywords"]
    
    @pytest.mark.asyncio
    @patch('bookmark_processor.core.ai_factory.AIManager.generate_description')
    async def test_ai_description_generation(self, mock_generate, temp_csv_file, mock_config):
        """Test AI description generation stage."""
        # Mock AI description generation
        mock_generate.return_value = (
            "Enhanced Python tutorial for beginners and advanced developers",
            {
                "provider": "local",
                "success": True,
                "processing_time": 0.5,
                "confidence": 0.9
            }
        )
        
        # Load bookmarks
        csv_handler = CSVHandler()
        bookmarks = csv_handler.load_bookmarks(temp_csv_file)
        
        # Test description generation
        from bookmark_processor.core.ai_factory import AIManager
        ai_manager = AIManager("local", mock_config)
        
        description, metadata = await mock_generate(bookmarks[0], bookmarks[0].note)
        
        assert description == "Enhanced Python tutorial for beginners and advanced developers"
        assert metadata["provider"] == "local"
        assert metadata["success"] is True
    
    @pytest.mark.asyncio
    async def test_tag_generation_stage(self, temp_csv_file):
        """Test tag generation stage."""
        from bookmark_processor.core.tag_generator import TagGenerator
        
        # Load bookmarks
        csv_handler = CSVHandler()
        bookmarks = csv_handler.load_bookmarks(temp_csv_file)
        
        # Mock tag generation
        tag_generator = TagGenerator()
        
        with patch.object(tag_generator, 'generate_tags') as mock_generate_tags:
            mock_generate_tags.return_value = ["python", "tutorial", "programming", "documentation"]
            
            tags = tag_generator.generate_tags(
                "Python Tutorial",
                "Enhanced Python tutorial for beginners",
                "https://docs.python.org/tutorial"
            )
            
            assert "python" in tags
            assert "tutorial" in tags
            assert len(tags) <= 4
    
    def test_output_csv_format(self, temp_csv_file):
        """Test output CSV format matches raindrop.io import requirements."""
        csv_handler = CSVHandler()
        
        # Create sample processed bookmarks
        processed_bookmarks = [
            {
                "url": "https://docs.python.org/tutorial",
                "folder": "Programming/Python",
                "title": "Python Tutorial",
                "note": "Enhanced Python tutorial for beginners and advanced developers",
                "tags": ["python", "tutorial", "programming", "documentation"],
                "created": "2024-01-01T00:00:00Z"
            },
            {
                "url": "https://arxiv.org/abs/12345",
                "folder": "Research/AI",
                "title": "AI Research",
                "note": "Comprehensive AI research covering latest developments",
                "tags": ["ai", "research", "machine-learning"],
                "created": "2024-01-02T00:00:00Z"
            }
        ]
        
        # Create output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
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
                "https://arxiv.org/abs/12345"
            ],
            "failed_urls": [
                "https://invalid-url-that-does-not-exist.example"
            ],
            "progress": {
                "stage": "generating_descriptions",
                "overall_progress": 50.0,
                "items_processed": 2,
                "items_total": 4
            }
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
    @patch('bookmark_processor.core.url_validator.URLValidator.validate_url')
    @patch('bookmark_processor.core.ai_factory.AIManager.generate_description')
    async def test_complete_pipeline_workflow(self, mock_ai_generate, mock_url_validate, 
                                            temp_csv_file, mock_config):
        """Test complete end-to-end pipeline workflow."""
        # Mock URL validation
        mock_url_validate.side_effect = [
            (True, {"status_code": 200, "final_url": "https://docs.python.org/tutorial"}),
            (True, {"status_code": 200, "final_url": "https://arxiv.org/abs/12345"}),
            (True, {"status_code": 200, "final_url": "https://developer.mozilla.org/js"}),
            (False, {"status_code": 404, "error": "Not found"}),
        ]
        
        # Mock AI description generation
        mock_ai_generate.side_effect = [
            ("Enhanced Python tutorial for beginners", {"provider": "local", "success": True}),
            ("Comprehensive AI research paper", {"provider": "local", "success": True}),
            ("Complete JavaScript development guide", {"provider": "local", "success": True}),
        ]
        
        # Create processor
        processor = BookmarkProcessor(mock_config)
        
        # Create output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name
        
        # Process bookmarks (mock the full pipeline)
        csv_handler = CSVHandler()
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
                description, ai_metadata = await mock_ai_generate(bookmark, bookmark.note)
                
                # Create processed bookmark
                processed_bookmark = {
                    "url": bookmark.url,
                    "folder": bookmark.folder,
                    "title": bookmark.title,
                    "note": description,
                    "tags": bookmark.tags,  # Would normally be AI-generated
                    "created": bookmark.created
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
            
            description, metadata = await error_handler.handle_bookmark_processing_error(
                error, bookmark, "existing content"
            )
            
            assert description is not None
            assert metadata["provider"] == "fallback"
            assert metadata["success"] is True
    
    @pytest.mark.asyncio
    async def test_memory_and_performance_monitoring(self, temp_csv_file):
        """Test memory usage and performance monitoring during processing."""
        from bookmark_processor.utils.progress_tracker import ProgressTracker, ProcessingStage
        
        progress_tracker = ProgressTracker(
            total_items=4,
            verbose=False,
            show_progress_bar=False
        )
        
        # Simulate processing stages
        stages = [
            ProcessingStage.LOADING_DATA,
            ProcessingStage.VALIDATING_URLS,
            ProcessingStage.GENERATING_DESCRIPTIONS,
            ProcessingStage.SAVING_RESULTS
        ]
        
        for stage in stages:
            progress_tracker.start_stage(stage, 4)
            
            # Simulate processing items
            for i in range(4):
                progress_tracker.update_progress(items_delta=1)
                
                # Check snapshot
                snapshot = progress_tracker.get_snapshot()
                assert snapshot.current_stage == stage
                assert snapshot.items_processed == (i + 1)
                assert snapshot.memory_usage_mb >= 0
        
        # Get final performance report
        report = progress_tracker.get_performance_report()
        assert report["total_items_processed"] == 16  # 4 items × 4 stages
        assert report["overall_success_rate"] >= 0
        assert "stage_summary" in report
        
        progress_tracker.complete()


class TestCloudAIPipelineIntegration:
    """Test cloud AI integration within the pipeline."""
    
    @pytest.mark.asyncio
    @patch('bookmark_processor.core.claude_api_client.ClaudeAPIClient._make_request')
    async def test_claude_pipeline_integration(self, mock_request):
        """Test Claude API integration in pipeline."""
        # Mock Claude response
        mock_request.return_value = {
            "content": [{"text": "AI-enhanced bookmark description using Claude"}],
            "usage": {"input_tokens": 150, "output_tokens": 40}
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
        description, metadata = await client.generate_description(bookmark, "existing content")
        
        assert description == "AI-enhanced bookmark description using Claude"
        assert metadata["provider"] == "claude"
        assert metadata["success"] is True
        assert metadata["cost_usd"] > 0
    
    @pytest.mark.asyncio
    @patch('bookmark_processor.core.openai_api_client.OpenAIAPIClient._make_request')
    async def test_openai_pipeline_integration(self, mock_request):
        """Test OpenAI API integration in pipeline."""
        # Mock OpenAI response
        mock_request.return_value = {
            "choices": [{
                "message": {"content": "AI-enhanced bookmark description using OpenAI"}
            }],
            "usage": {"prompt_tokens": 120, "completion_tokens": 35}
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
        description, metadata = await client.generate_description(bookmark, "existing content")
        
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
                bookmark_count=1
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