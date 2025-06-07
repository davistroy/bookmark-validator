"""
Integration Tests for Cloud AI Integration

This module tests the complete cloud AI integration including Claude API,
OpenAI API, batch processing, cost tracking, and error handling.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.ai_factory import AIFactory, AIManager
from bookmark_processor.core.batch_processor import BatchProcessor
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.cost_tracker import CostTracker
from bookmark_processor.utils.error_handler import ErrorHandler
from bookmark_processor.utils.progress_tracker import ProgressTracker


@dataclass
class MockBookmark:
    """Mock bookmark for testing."""

    title: str = "Test Title"
    url: str = "https://example.com"
    note: str = "Test note"
    excerpt: str = "Test excerpt"


class TestCloudAIIntegration:
    """Integration tests for cloud AI functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with API keys."""
        config = Mock(spec=Configuration)
        config.get_api_key.return_value = "test-api-key"
        config.has_api_key.return_value = True
        config.validate_ai_configuration.return_value = (True, None)
        config.get.side_effect = lambda section, key, fallback=None: {
            ("ai", "claude_rpm"): "50",
            ("ai", "openai_rpm"): "60",
            ("ai", "claude_batch_size"): "10",
            ("ai", "openai_batch_size"): "20",
            ("ai", "confirmation_interval"): "10.0",
        }.get((section, key), fallback)
        return config

    @pytest.fixture
    def sample_bookmarks(self):
        """Sample bookmarks for testing."""
        return [
            MockBookmark(
                title="Python Tutorial",
                url="https://docs.python.org/tutorial",
                note="Learn Python programming basics",
                excerpt="Official Python tutorial",
            ),
            MockBookmark(
                title="AI Research Paper",
                url="https://arxiv.org/abs/12345",
                note="Latest developments in AI",
                excerpt="Academic research on AI",
            ),
            MockBookmark(
                title="JavaScript Guide",
                url="https://developer.mozilla.org/js",
                note="Web development reference",
                excerpt="MDN JavaScript documentation",
            ),
        ]

    @pytest.mark.asyncio
    async def test_ai_factory_creation(self, mock_config):
        """Test AI factory can create different providers."""
        factory = AIFactory(mock_config)

        # Test creating local AI
        local_ai = factory.create_ai_client("local")
        assert local_ai is not None

        # Test creating Claude AI (should work with mock config)
        claude_ai = factory.create_ai_client("claude")
        assert claude_ai is not None

        # Test creating OpenAI (should work with mock config)
        openai_ai = factory.create_ai_client("openai")
        assert openai_ai is not None

    @pytest.mark.asyncio
    async def test_ai_manager_fallback(self, mock_config):
        """Test AI manager fallback functionality."""
        # Create AI manager with primary and fallback
        ai_manager = AIManager(
            primary_provider="claude",
            config=mock_config,
            enable_fallback=True,
            fallback_provider="local",
        )

        # Mock the clients to simulate failure
        with patch.object(ai_manager, "_initialize_clients"):
            await ai_manager.__aenter__()

            # Mock primary client failure
            ai_manager.primary_client = Mock()
            ai_manager.primary_client.generate_description = AsyncMock(
                side_effect=Exception("API failure")
            )

            # Mock fallback client success
            ai_manager.fallback_client = Mock()
            ai_manager.fallback_client.generate_description = AsyncMock(
                return_value=(
                    "Fallback description",
                    {"provider": "local", "success": True},
                )
            )

            bookmark = MockBookmark()
            description, metadata = await ai_manager.generate_description(bookmark)

            assert description == "Fallback description"
            assert metadata["provider"] == "local"
            assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_cost_tracking_integration(self):
        """Test cost tracking with batch processing."""
        cost_tracker = CostTracker(confirmation_interval=5.0)

        # Add some cost records
        cost_tracker.add_cost_record(
            provider="claude",
            model="claude-3-haiku-20240307",
            input_tokens=1000,
            output_tokens=150,
            cost_usd=0.001,
            bookmark_count=1,
        )

        cost_tracker.add_cost_record(
            provider="openai",
            model="gpt-3.5-turbo",
            input_tokens=800,
            output_tokens=120,
            cost_usd=0.002,
            bookmark_count=1,
        )

        # Test cost estimation
        estimate = cost_tracker.get_cost_estimate(100, "claude")
        assert estimate["provider"] == "claude"
        assert estimate["bookmark_count"] == 100
        assert estimate["estimated_cost_usd"] > 0

        # Test detailed statistics
        stats = cost_tracker.get_detailed_statistics()
        assert stats["session"]["total_cost_usd"] == 0.003
        assert "claude" in stats["providers"]
        assert "openai" in stats["providers"]

    @pytest.mark.asyncio
    async def test_error_handling_categories(self):
        """Test error handling categorization and retry logic."""
        error_handler = ErrorHandler()

        # Test different error categories
        network_error = Exception("Connection timeout")
        network_details = error_handler.categorize_error(network_error)
        assert network_details.category.value == "network"
        assert network_details.is_recoverable is True

        auth_error = Exception("401 Unauthorized")
        auth_details = error_handler.categorize_error(auth_error)
        assert auth_details.category.value == "api_auth"
        assert auth_details.is_recoverable is False

        rate_limit_error = Exception("429 Rate limit exceeded")
        rate_limit_details = error_handler.categorize_error(rate_limit_error)
        assert rate_limit_details.category.value == "api_limit"
        assert rate_limit_details.is_recoverable is True

    @pytest.mark.asyncio
    async def test_progress_tracking_stages(self):
        """Test progress tracking through different stages."""
        from bookmark_processor.utils.progress_tracker import ProcessingStage

        progress_tracker = ProgressTracker(
            total_items=100, verbose=False, show_progress_bar=False
        )

        # Test stage progression
        progress_tracker.start_stage(ProcessingStage.LOADING_DATA, 100)
        assert progress_tracker.current_stage == ProcessingStage.LOADING_DATA

        # Update progress
        progress_tracker.update_progress(items_delta=10)
        snapshot = progress_tracker.get_snapshot()
        assert snapshot.items_processed == 10
        assert snapshot.overall_progress > 0

        # Move to next stage
        progress_tracker.start_stage(ProcessingStage.GENERATING_DESCRIPTIONS, 100)
        progress_tracker.update_progress(items_delta=50)

        # Test stage summary
        summary = progress_tracker.get_stage_summary()
        assert "loading_data" in summary
        assert "generating_descriptions" in summary
        assert summary["generating_descriptions"]["items_processed"] == 50

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.claude_api_client.ClaudeAPIClient._make_request")
    async def test_batch_processing_with_claude(
        self, mock_request, mock_config, sample_bookmarks
    ):
        """Test batch processing with Claude API."""
        # Mock Claude API response
        mock_request.return_value = {
            "content": [
                {"text": "Test description 1\nTest description 2\nTest description 3"}
            ],
            "usage": {"input_tokens": 500, "output_tokens": 150},
        }

        # Create AI manager with Claude
        ai_manager = AIManager("claude", mock_config)

        # Create batch processor
        cost_tracker = CostTracker()
        batch_processor = BatchProcessor(
            ai_manager=ai_manager, cost_tracker=cost_tracker, verbose=False
        )

        # Mock the AI manager initialization
        with patch.object(ai_manager, "_initialize_clients"):
            await ai_manager.__aenter__()

            # Process bookmarks
            results, stats = await batch_processor.process_bookmarks(sample_bookmarks)

            assert len(results) == len(sample_bookmarks)
            assert stats["processed_count"] >= 0
            assert stats["provider"] == "claude"

    @pytest.mark.asyncio
    @patch("bookmark_processor.core.openai_api_client.OpenAIAPIClient._make_request")
    async def test_batch_processing_with_openai(
        self, mock_request, mock_config, sample_bookmarks
    ):
        """Test batch processing with OpenAI API."""
        # Mock OpenAI API response
        mock_request.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "1. Test description 1\n2. Test description 2\n3. Test description 3"
                    }
                }
            ],
            "usage": {"prompt_tokens": 400, "completion_tokens": 120},
        }

        # Create AI manager with OpenAI
        ai_manager = AIManager("openai", mock_config)

        # Create batch processor
        cost_tracker = CostTracker()
        batch_processor = BatchProcessor(
            ai_manager=ai_manager, cost_tracker=cost_tracker, verbose=False
        )

        # Mock the AI manager initialization
        with patch.object(ai_manager, "_initialize_clients"):
            await ai_manager.__aenter__()

            # Process bookmarks
            results, stats = await batch_processor.process_bookmarks(sample_bookmarks)

            assert len(results) == len(sample_bookmarks)
            assert stats["processed_count"] >= 0
            assert stats["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_cost_confirmation_workflow(self):
        """Test cost confirmation workflow."""
        cost_tracker = CostTracker(
            confirmation_interval=1.0
        )  # Low threshold for testing

        # Add cost to trigger confirmation
        cost_tracker.add_cost_record(
            provider="claude",
            model="claude-3-haiku-20240307",
            input_tokens=5000,
            output_tokens=1000,
            cost_usd=1.5,  # Above threshold
            bookmark_count=10,
        )

        # Test that confirmation is needed
        assert cost_tracker.should_confirm() is True

        # Test confirmation prompt generation
        prompt = cost_tracker.get_confirmation_prompt()
        assert "Cost Update" in prompt
        assert "$1.50" in prompt
        assert "claude" in prompt

    @pytest.mark.asyncio
    async def test_end_to_end_cloud_ai_workflow(self, mock_config, sample_bookmarks):
        """Test complete end-to-end workflow with cloud AI."""
        # Mock API responses
        claude_response = {
            "content": [{"text": "AI-generated description"}],
            "usage": {"input_tokens": 200, "output_tokens": 50},
        }

        with patch(
            "bookmark_processor.core.claude_api_client.ClaudeAPIClient._make_request"
        ) as mock_claude:
            mock_claude.return_value = claude_response

            # Create complete workflow
            ai_manager = AIManager("claude", mock_config, enable_fallback=True)
            cost_tracker = CostTracker(confirmation_interval=100.0)  # High threshold
            batch_processor = BatchProcessor(ai_manager, cost_tracker, verbose=False)

            # Mock initialization
            with patch.object(ai_manager, "_initialize_clients"):
                await ai_manager.__aenter__()

                # Process single bookmark
                bookmark = sample_bookmarks[0]
                description, metadata = await ai_manager.generate_description(bookmark)

                assert description is not None
                assert metadata["success"] is True
                assert metadata["provider"] == "claude"

                # Test usage statistics
                stats = ai_manager.get_usage_statistics()
                assert "provider" in stats
                assert "error_handling" in stats
                assert "health_status" in stats

    def test_prompt_optimization(self):
        """Test optimized prompts for different AI services."""
        from bookmark_processor.core.claude_api_client import ClaudeAPIClient
        from bookmark_processor.core.openai_api_client import OpenAIAPIClient

        bookmark = MockBookmark()

        # Test Claude prompt optimization
        claude_client = ClaudeAPIClient("test-key")
        claude_prompt = claude_client._create_bookmark_prompt(
            bookmark, "existing content"
        )

        # Claude prompts should be concise and structured
        assert len(claude_prompt) < 500  # Optimized for token efficiency
        assert "Focus on:" in claude_prompt
        assert "Description:" in claude_prompt

        # Test OpenAI prompt optimization
        openai_client = OpenAIAPIClient("test-key")
        openai_messages = openai_client._create_bookmark_prompt(
            bookmark, "existing content"
        )

        # OpenAI should use system/user message structure
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[1]["role"] == "user"
        assert len(openai_messages[0]["content"]) < 200  # Concise system message

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, mock_config):
        """Test various error recovery scenarios."""
        ai_manager = AIManager("claude", mock_config, enable_fallback=True)

        # Mock clients
        with patch.object(ai_manager, "_initialize_clients"):
            await ai_manager.__aenter__()

            # Test network timeout recovery
            ai_manager.primary_client = Mock()
            ai_manager.primary_client.generate_description = AsyncMock(
                side_effect=Exception("Connection timeout")
            )

            ai_manager.fallback_client = Mock()
            ai_manager.fallback_client.generate_description = AsyncMock(
                return_value=(
                    "Fallback description",
                    {"provider": "local", "success": True},
                )
            )

            bookmark = MockBookmark()
            description, metadata = await ai_manager.generate_description(bookmark)

            # Should fall back gracefully
            assert description == "Fallback description"
            assert metadata["provider"] == "local"

            # Test error statistics
            error_stats = ai_manager.get_error_statistics()
            assert error_stats["total_errors"] > 0

    def test_configuration_validation(self):
        """Test configuration validation for cloud AI."""
        # Test missing API key
        config = Mock(spec=Configuration)
        config.get_api_key.return_value = None
        config.validate_ai_configuration.return_value = (False, "Missing API key")

        factory = AIFactory(config)

        # Should handle missing API keys gracefully
        with pytest.raises(Exception):
            factory.create_ai_client("claude")

        # Test valid configuration
        config.get_api_key.return_value = "valid-key"
        config.validate_ai_configuration.return_value = (True, None)
        client = factory.create_ai_client("claude")
        assert client is not None


class TestPerformanceAndScaling:
    """Test performance and scaling aspects of cloud AI integration."""

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, mock_config):
        """Test handling of concurrent requests with rate limiting."""
        from bookmark_processor.utils.rate_limiter import RateLimiter

        # Create rate limiter with low limit for testing
        rate_limiter = RateLimiter(requests_per_minute=10, window_size_minutes=1)

        # Test concurrent acquisitions
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(rate_limiter.acquire())
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed for small number
        assert all(results)

    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self):
        """Test memory usage tracking during processing."""
        progress_tracker = ProgressTracker(
            total_items=100, verbose=False, show_progress_bar=False
        )

        # Simulate processing
        for i in range(10):
            progress_tracker.update_progress(items_delta=1)

        snapshot = progress_tracker.get_snapshot()

        # Memory usage should be tracked (may be 0 if psutil not available)
        assert snapshot.memory_usage_mb >= 0
        assert snapshot.health_status in ["healthy", "stable", "degraded", "critical"]

    def test_batch_size_optimization(self):
        """Test batch size optimization for different providers."""
        from bookmark_processor.core.batch_processor import BatchProcessor

        # Mock AI manager
        ai_manager = Mock()
        ai_manager.get_current_provider.return_value = "claude"

        batch_processor = BatchProcessor(ai_manager, verbose=False)

        # Test provider-specific batch sizes
        claude_batch_size = batch_processor.get_batch_size("claude")
        openai_batch_size = batch_processor.get_batch_size("openai")
        local_batch_size = batch_processor.get_batch_size("local")

        assert claude_batch_size == 10  # Conservative for Claude
        assert openai_batch_size == 20  # Larger for OpenAI
        assert local_batch_size == 50  # Largest for local processing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
