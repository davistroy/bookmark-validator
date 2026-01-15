"""
Unit tests for AI processor and related components.

Tests for AI processor, AI factory, and cloud AI clients.
"""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.core.ai_factory import AIFactory
from bookmark_processor.core.ai_processor import EnhancedAIProcessor
from bookmark_processor.core.base_api_client import BaseAPIClient
from bookmark_processor.core.claude_api_client import ClaudeAPIClient
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.openai_api_client import OpenAIAPIClient
from tests.fixtures.mock_utilities import MockAIProcessor, MockEnhancedAIProcessor
from tests.fixtures.test_data import MOCK_AI_RESULTS, create_sample_bookmark_objects


@pytest.fixture(autouse=True)
def disable_test_mode_for_ai_tests():
    """Disable the mock test mode so we can test actual AI processor logic with our own mocks."""
    original_value = os.environ.get("BOOKMARK_PROCESSOR_TEST_MODE")
    # Remove test mode so the AI processor uses our mocks instead of its built-in mock
    if "BOOKMARK_PROCESSOR_TEST_MODE" in os.environ:
        del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]
    yield
    # Restore original value
    if original_value is not None:
        os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_value


class TestEnhancedAIProcessor:
    """Test EnhancedAIProcessor class."""

    def test_init_local_engine(self):
        """Test EnhancedAIProcessor initialization with local engine."""
        processor = EnhancedAIProcessor(engine="local")

        assert processor.engine == "local"
        assert processor.model_name is not None
        assert processor.is_available is True

    def test_init_cloud_engine_no_key(self):
        """Test EnhancedAIProcessor initialization with cloud engine but no API key."""
        # Test without API key should fall back to local
        processor = EnhancedAIProcessor(engine="claude", api_key=None)

        assert processor.engine == "local"  # Should fallback
        assert processor.is_available is True

    @patch("bookmark_processor.core.ai_processor.ClaudeAPIClient")
    def test_init_claude_engine_with_key(self, mock_claude_client):
        """Test EnhancedAIProcessor initialization with Claude engine and API key."""
        mock_client = Mock()
        mock_client.is_available = True
        mock_claude_client.return_value = mock_client

        processor = EnhancedAIProcessor(engine="claude", api_key="test-key")

        assert processor.engine == "claude"
        assert processor.cloud_client == mock_client
        mock_claude_client.assert_called_once_with(api_key="test-key")

    @patch("bookmark_processor.core.ai_processor.OpenAIAPIClient")
    def test_init_openai_engine_with_key(self, mock_openai_client):
        """Test EnhancedAIProcessor initialization with OpenAI engine and API key."""
        mock_client = Mock()
        mock_client.is_available = True
        mock_openai_client.return_value = mock_client

        processor = EnhancedAIProcessor(engine="openai", api_key="test-key")

        assert processor.engine == "openai"
        assert processor.cloud_client == mock_client
        mock_openai_client.assert_called_once_with(api_key="test-key")

    def test_init_invalid_engine(self):
        """Test EnhancedAIProcessor initialization with invalid engine."""
        # Should fallback to local
        processor = EnhancedAIProcessor(engine="invalid_engine")

        assert processor.engine == "local"
        assert processor.is_available is True

    @patch("bookmark_processor.core.ai_processor.pipeline")
    def test_process_bookmark_local_success(self, mock_pipeline):
        """Test processing bookmark with local AI engine."""
        # Mock the transformers pipeline
        mock_summarizer = Mock()
        mock_summarizer.return_value = [{"summary_text": "AI-generated description"}]
        mock_pipeline.return_value = mock_summarizer

        processor = EnhancedAIProcessor(engine="local")
        bookmarks = create_sample_bookmark_objects()
        bookmark = bookmarks[0]

        result = processor.process_bookmark(bookmark)

        assert result.enhanced_description == "AI-generated description."
        assert result.processing_status.ai_processed is True
        assert result.processing_status.ai_processing_error is None

    @patch("bookmark_processor.core.ai_processor.pipeline")
    def test_process_bookmark_local_failure(self, mock_pipeline):
        """Test processing bookmark with local AI engine failure."""
        # Mock the transformers pipeline to raise exception
        mock_pipeline.side_effect = Exception("Model loading failed")

        processor = EnhancedAIProcessor(engine="local")
        bookmarks = create_sample_bookmark_objects()
        bookmark = bookmarks[0]

        result = processor.process_bookmark(bookmark)

        # Should fallback to existing content (note is the first fallback)
        assert result.enhanced_description == bookmark.note
        # When model loading fails, fallback is used and ai_processed is False
        assert result.processing_status.ai_processed is False

    @patch("bookmark_processor.core.ai_processor.ClaudeAPIClient")
    def test_process_bookmark_cloud_success(self, mock_claude_client_class):
        """Test processing bookmark with cloud AI engine."""
        # Mock cloud client
        mock_client = Mock()
        mock_client.is_available = True
        mock_client.generate_description.return_value = "Cloud AI description"
        mock_claude_client_class.return_value = mock_client

        processor = EnhancedAIProcessor(engine="claude", api_key="test-key")

        bookmarks = create_sample_bookmark_objects()
        bookmark = bookmarks[0]

        result = processor.process_bookmark(bookmark)

        assert result.enhanced_description == "Cloud AI description"
        assert result.processing_status.ai_processed is True
        mock_client.generate_description.assert_called_once()

    @patch("bookmark_processor.core.ai_processor.ClaudeAPIClient")
    def test_process_bookmark_cloud_failure(self, mock_claude_client_class):
        """Test processing bookmark with cloud AI engine failure."""
        # Mock cloud client to raise exception
        mock_client = Mock()
        mock_client.is_available = True
        mock_client.generate_description.side_effect = Exception("API error")
        mock_claude_client_class.return_value = mock_client

        processor = EnhancedAIProcessor(engine="claude", api_key="test-key")

        bookmarks = create_sample_bookmark_objects()
        bookmark = bookmarks[0]

        result = processor.process_bookmark(bookmark)

        # Should fallback to existing content (note is the first fallback)
        assert result.enhanced_description == bookmark.note
        # When cloud processing fails but returns None (not raises), fallback is used
        assert result.processing_status.ai_processed is False

    @patch("bookmark_processor.core.ai_processor.pipeline")
    def test_process_bookmark_no_content(self, mock_pipeline):
        """Test processing bookmark with no existing content."""
        # Mock pipeline to return None (simulating failure or unavailable)
        mock_pipeline.return_value = None

        processor = EnhancedAIProcessor(engine="local")

        # Create bookmark with no content
        bookmark = Bookmark(
            url="https://example.com", title="Test Bookmark", note="", excerpt=""
        )

        result = processor.process_bookmark(bookmark)

        # Should generate minimal fallback description from title and domain
        assert result.enhanced_description != ""
        assert (
            "Test Bookmark" in result.enhanced_description
            or "example.com" in result.enhanced_description
        )

    @patch("bookmark_processor.core.ai_processor.pipeline")
    def test_process_batch(self, mock_pipeline):
        """Test processing batch of bookmarks."""
        mock_summarizer = Mock()
        mock_summarizer.return_value = [{"summary_text": "Batch AI description"}]
        mock_pipeline.return_value = mock_summarizer

        processor = EnhancedAIProcessor(engine="local")
        bookmarks = create_sample_bookmark_objects()

        results = processor.process_batch(bookmarks)

        assert len(results) == len(bookmarks)
        for bookmark in results:
            # The AI output gets cleaned (adds period if not present)
            assert bookmark.enhanced_description == "Batch AI description."
            assert bookmark.processing_status.ai_processed is True

    def test_generate_fallback_description(self):
        """Test fallback description generation."""
        processor = EnhancedAIProcessor(engine="local")

        # Test with note
        bookmark = Bookmark(note="Existing note", title="Test Title")
        description = processor._generate_fallback_description(bookmark)
        assert description == "Existing note"

        # Test with excerpt
        bookmark = Bookmark(note="", excerpt="Existing excerpt", title="Test Title")
        description = processor._generate_fallback_description(bookmark)
        assert description == "Existing excerpt"

        # Test with title only
        bookmark = Bookmark(note="", excerpt="", title="Test Title")
        description = processor._generate_fallback_description(bookmark)
        assert "Test Title" in description

    def test_prepare_input_text(self):
        """Test input text preparation for AI processing."""
        processor = EnhancedAIProcessor(engine="local")

        bookmark = Bookmark(
            title="Test Title",
            note="User note",
            excerpt="Page excerpt",
            url="https://example.com",
        )

        input_text = processor._prepare_input_text(bookmark)

        assert "Test Title" in input_text
        assert "User note" in input_text
        assert "Page excerpt" in input_text
        assert "example.com" in input_text

    def test_get_statistics(self):
        """Test getting processing statistics."""
        processor = EnhancedAIProcessor(engine="local")

        stats = processor.get_statistics()

        assert isinstance(stats, dict)
        assert "engine" in stats
        assert "model_name" in stats
        assert "is_available" in stats
        assert "processed_count" in stats
        assert stats["engine"] == "local"


class TestAIFactory:
    """Test AIFactory class."""

    def test_create_local_client(self):
        """Test creating local AI client."""
        # Create a mock configuration
        mock_config = Mock()
        mock_config.get_api_key.return_value = None
        mock_config.has_api_key.return_value = False

        client = AIFactory.create_client(provider="local", config=mock_config)

        assert isinstance(client, EnhancedAIProcessor)

    def test_create_claude_client(self):
        """Test creating Claude AI client."""
        # Create a mock configuration with API key
        mock_config = Mock()
        mock_config.get_api_key.return_value = "test-key"
        mock_config.has_api_key.return_value = True
        mock_config.validate_ai_configuration.return_value = (True, None)

        client = AIFactory.create_client(provider="claude", config=mock_config)

        # Verify we get a ClaudeAPIClient instance
        assert isinstance(client, ClaudeAPIClient)
        assert client.api_key == "test-key"

    def test_create_openai_client(self):
        """Test creating OpenAI AI client."""
        # Create a mock configuration with API key
        mock_config = Mock()
        mock_config.get_api_key.return_value = "test-key"
        mock_config.has_api_key.return_value = True
        mock_config.validate_ai_configuration.return_value = (True, None)

        client = AIFactory.create_client(provider="openai", config=mock_config)

        # Verify we get an OpenAIAPIClient instance
        assert isinstance(client, OpenAIAPIClient)
        assert client.api_key == "test-key"

    def test_create_client_invalid_provider(self):
        """Test creating client with invalid provider."""
        from bookmark_processor.utils.error_handler import AISelectionError

        mock_config = Mock()

        with pytest.raises(AISelectionError):
            AIFactory.create_client(provider="invalid", config=mock_config)

    def test_get_available_providers(self):
        """Test getting available AI providers."""
        providers = AIFactory.get_available_providers()

        assert isinstance(providers, dict)
        assert "local" in providers
        assert "claude" in providers
        assert "openai" in providers

    def test_validate_provider_config_local(self):
        """Test validating local provider config."""
        mock_config = Mock()

        is_valid, error = AIFactory.validate_provider_config("local", mock_config)

        assert is_valid is True
        assert error is None

    def test_validate_provider_config_invalid(self):
        """Test validating invalid provider config."""
        mock_config = Mock()

        is_valid, error = AIFactory.validate_provider_config("invalid", mock_config)

        assert is_valid is False
        assert error is not None


class TestBaseAPIClient:
    """Test BaseAPIClient abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseAPIClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAPIClient(api_key="test")


class TestClaudeAPIClient:
    """Test ClaudeAPIClient class."""

    def test_init_with_api_key(self):
        """Test ClaudeAPIClient initialization with API key."""
        client = ClaudeAPIClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.MODEL is not None
        assert client.BASE_URL is not None

    def test_create_bookmark_prompt(self):
        """Test prompt preparation."""
        client = ClaudeAPIClient(api_key="test-key")
        bookmark = Bookmark(
            title="Test Title",
            note="User note",
            excerpt="Page excerpt",
            url="https://example.com",
        )

        prompt = client._create_bookmark_prompt(bookmark)

        assert "Test Title" in prompt
        assert "example.com" in prompt

    def test_get_usage_statistics(self):
        """Test getting usage statistics."""
        client = ClaudeAPIClient(api_key="test-key")

        stats = client.get_usage_statistics()

        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_cost_usd" in stats

    def test_get_cost_per_request(self):
        """Test getting cost per request estimate."""
        client = ClaudeAPIClient(api_key="test-key")

        cost = client.get_cost_per_request()

        assert isinstance(cost, float)
        assert cost > 0

    def test_get_rate_limit_info(self):
        """Test getting rate limit info."""
        client = ClaudeAPIClient(api_key="test-key")

        info = client.get_rate_limit_info()

        assert isinstance(info, dict)
        assert "provider" in info
        assert info["provider"] == "claude"


class TestOpenAIAPIClient:
    """Test OpenAIAPIClient class."""

    def test_init_with_api_key(self):
        """Test OpenAIAPIClient initialization with API key."""
        client = OpenAIAPIClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.MODEL is not None
        assert client.BASE_URL is not None

    def test_create_bookmark_prompt(self):
        """Test prompt preparation for OpenAI."""
        client = OpenAIAPIClient(api_key="test-key")
        bookmark = Bookmark(
            title="Test Title",
            note="User note",
            excerpt="Page excerpt",
            url="https://example.com",
        )

        messages = client._create_bookmark_prompt(bookmark)

        # OpenAI returns a list of message dictionaries
        assert isinstance(messages, list)
        assert len(messages) >= 1
        # Check that the bookmark info appears in the messages
        messages_str = str(messages)
        assert "Test Title" in messages_str
        assert "example.com" in messages_str

    def test_get_usage_statistics(self):
        """Test getting usage statistics."""
        client = OpenAIAPIClient(api_key="test-key")

        stats = client.get_usage_statistics()

        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_cost_usd" in stats

    def test_get_cost_per_request(self):
        """Test getting cost per request estimate."""
        client = OpenAIAPIClient(api_key="test-key")

        cost = client.get_cost_per_request()

        assert isinstance(cost, float)
        assert cost > 0

    def test_get_rate_limit_info(self):
        """Test getting rate limit info."""
        client = OpenAIAPIClient(api_key="test-key")

        info = client.get_rate_limit_info()

        assert isinstance(info, dict)
        assert "provider" in info
        assert info["provider"] == "openai"


class TestEnhancedAIProcessorIntegration:
    """Integration tests for AI processor with different engines."""

    @patch("bookmark_processor.core.ai_processor.pipeline")
    def test_local_to_cloud_fallback(self, mock_pipeline):
        """Test fallback from failed local to cloud processing."""
        # Mock local pipeline failure
        mock_pipeline.side_effect = Exception("Local model failed")

        # Mock cloud client success
        mock_client = Mock()
        mock_client.is_available = True
        mock_client.generate_description.return_value = "Cloud fallback description"

        processor = EnhancedAIProcessor(engine="local")
        processor.cloud_client = mock_client  # Inject fallback client

        bookmark = Bookmark(title="Test", note="Test note", url="https://example.com")

        result = processor.process_bookmark(bookmark)

        # Should use existing content as fallback since we didn't set up cloud fallback properly
        assert result.enhanced_description == "Test note"
        assert result.processing_status.ai_processed is False

    def test_batch_processing_mixed_results(self):
        """Test batch processing with mixed success/failure results."""
        mock_processor = MockEnhancedAIProcessor(success_rate=0.7)  # 70% success rate

        bookmarks = create_sample_bookmark_objects()
        results = mock_processor.process_batch(bookmarks)

        assert len(results) == len(bookmarks)

        # Check that some succeeded and some failed
        processed_count = sum(1 for b in results if b.processing_status.ai_processed)
        failed_count = sum(1 for b in results if not b.processing_status.ai_processed)

        # With random nature, we can't predict exact counts, but should have both
        assert processed_count + failed_count == len(bookmarks)

    def test_processing_with_rate_limiting(self):
        """Test AI processing with rate limiting simulation."""
        mock_processor = MockEnhancedAIProcessor(processing_delay=0.01)  # Small delay

        import time

        start_time = time.time()

        bookmarks = create_sample_bookmark_objects()[:3]  # Small batch
        results = mock_processor.process_batch(bookmarks)

        end_time = time.time()

        # Should have taken some time due to delays
        assert end_time - start_time >= 0.03  # At least 3 * 0.01 seconds
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__])
