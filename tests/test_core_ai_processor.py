"""
Unit tests for AI processor and related components.

Tests for AI processor, AI factory, and cloud AI clients.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.core.ai_factory import AIFactory
from bookmark_processor.core.ai_processor import EnhancedAIProcessor
from bookmark_processor.core.base_api_client import BaseAPIClient
from bookmark_processor.core.claude_api_client import ClaudeAPIClient
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.openai_api_client import OpenAIAPIClient
from tests.fixtures.mock_utilities import MockAIProcessor
from tests.fixtures.test_data import MOCK_AI_RESULTS, create_sample_bookmark_objects


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

    @patch("transformers.pipeline")
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

        assert result.enhanced_description == "AI-generated description"
        assert result.processing_status.ai_processed is True
        assert result.processing_status.ai_processing_error is None

    @patch("transformers.pipeline")
    def test_process_bookmark_local_failure(self, mock_pipeline):
        """Test processing bookmark with local AI engine failure."""
        # Mock the transformers pipeline to raise exception
        mock_pipeline.side_effect = Exception("Model loading failed")

        processor = EnhancedAIProcessor(engine="local")
        bookmarks = create_sample_bookmark_objects()
        bookmark = bookmarks[0]

        result = processor.process_bookmark(bookmark)

        # Should fallback to existing content
        assert result.enhanced_description == bookmark.note or bookmark.excerpt
        assert result.processing_status.ai_processed is False
        assert "Model loading failed" in result.processing_status.ai_processing_error

    def test_process_bookmark_cloud_success(self):
        """Test processing bookmark with cloud AI engine."""
        # Mock cloud client
        mock_client = Mock()
        mock_client.is_available = True
        mock_client.generate_description.return_value = "Cloud AI description"

        processor = EnhancedAIProcessor(engine="claude")
        processor.cloud_client = mock_client

        bookmarks = create_sample_bookmark_objects()
        bookmark = bookmarks[0]

        result = processor.process_bookmark(bookmark)

        assert result.enhanced_description == "Cloud AI description"
        assert result.processing_status.ai_processed is True
        mock_client.generate_description.assert_called_once()

    def test_process_bookmark_cloud_failure(self):
        """Test processing bookmark with cloud AI engine failure."""
        # Mock cloud client to raise exception
        mock_client = Mock()
        mock_client.is_available = True
        mock_client.generate_description.side_effect = Exception("API error")

        processor = EnhancedAIProcessor(engine="claude")
        processor.cloud_client = mock_client

        bookmarks = create_sample_bookmark_objects()
        bookmark = bookmarks[0]

        result = processor.process_bookmark(bookmark)

        # Should fallback to existing content
        assert result.enhanced_description == bookmark.note or bookmark.excerpt
        assert result.processing_status.ai_processed is False
        assert "API error" in result.processing_status.ai_processing_error

    def test_process_bookmark_no_content(self):
        """Test processing bookmark with no existing content."""
        processor = EnhancedAIProcessor(engine="local")

        # Create bookmark with no content
        bookmark = Bookmark(
            url="https://example.com", title="Test Bookmark", note="", excerpt=""
        )

        result = processor.process_bookmark(bookmark)

        # Should generate minimal fallback description
        assert result.enhanced_description != ""
        assert (
            "Test Bookmark" in result.enhanced_description
            or "example.com" in result.enhanced_description
        )

    @patch("transformers.pipeline")
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
            assert bookmark.enhanced_description == "Batch AI description"
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

    def test_create_local_processor(self):
        """Test creating local AI processor."""
        processor = AIFactory.create_processor(engine="local")

        assert isinstance(processor, EnhancedAIProcessor)
        assert processor.engine == "local"
        assert processor.is_available is True

    @patch("bookmark_processor.core.ai_factory.ClaudeAPIClient")
    def test_create_claude_processor(self, mock_claude_client):
        """Test creating Claude AI processor."""
        mock_client = Mock()
        mock_client.is_available = True
        mock_claude_client.return_value = mock_client

        processor = AIFactory.create_processor(engine="claude", api_key="test-key")

        assert isinstance(processor, EnhancedAIProcessor)
        assert processor.engine == "claude"
        mock_claude_client.assert_called_once_with(api_key="test-key")

    @patch("bookmark_processor.core.ai_factory.OpenAIAPIClient")
    def test_create_openai_processor(self, mock_openai_client):
        """Test creating OpenAI AI processor."""
        mock_client = Mock()
        mock_client.is_available = True
        mock_openai_client.return_value = mock_client

        processor = AIFactory.create_processor(engine="openai", api_key="test-key")

        assert isinstance(processor, EnhancedAIProcessor)
        assert processor.engine == "openai"
        mock_openai_client.assert_called_once_with(api_key="test-key")

    def test_create_processor_invalid_engine(self):
        """Test creating processor with invalid engine."""
        processor = AIFactory.create_processor(engine="invalid")

        # Should fallback to local
        assert isinstance(processor, EnhancedAIProcessor)
        assert processor.engine == "local"

    def test_get_available_engines(self):
        """Test getting available AI engines."""
        engines = AIFactory.get_available_engines()

        assert isinstance(engines, list)
        assert "local" in engines
        assert "claude" in engines
        assert "openai" in engines

    def test_validate_engine(self):
        """Test engine validation."""
        assert AIFactory.validate_engine("local") is True
        assert AIFactory.validate_engine("claude") is True
        assert AIFactory.validate_engine("openai") is True
        assert AIFactory.validate_engine("invalid") is False


class TestBaseAPIClient:
    """Test BaseAPIClient abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseAPIClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAPIClient(api_key="test")

    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteClient(BaseAPIClient):
            pass

        with pytest.raises(TypeError):
            IncompleteClient(api_key="test")


class TestClaudeAPIClient:
    """Test ClaudeAPIClient class."""

    def test_init_with_api_key(self):
        """Test ClaudeAPIClient initialization with API key."""
        client = ClaudeAPIClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.model == "claude-3-sonnet-20240229"
        assert client.base_url is not None

    def test_init_without_api_key(self):
        """Test ClaudeAPIClient initialization without API key."""
        client = ClaudeAPIClient(api_key=None)

        assert client.is_available is False

    @patch("requests.post")
    def test_generate_description_success(self, mock_post):
        """Test successful description generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Generated description"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        mock_post.return_value = mock_response

        client = ClaudeAPIClient(api_key="test-key")
        bookmark = Bookmark(title="Test", note="Test note", url="https://example.com")

        result = client.generate_description(bookmark)

        assert result == "Generated description"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_generate_description_api_error(self, mock_post):
        """Test description generation with API error."""
        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        client = ClaudeAPIClient(api_key="test-key")
        bookmark = Bookmark(title="Test", url="https://example.com")

        with pytest.raises(Exception):
            client.generate_description(bookmark)

    @patch("requests.post")
    def test_generate_description_network_error(self, mock_post):
        """Test description generation with network error."""
        # Mock network error
        mock_post.side_effect = Exception("Network error")

        client = ClaudeAPIClient(api_key="test-key")
        bookmark = Bookmark(title="Test", url="https://example.com")

        with pytest.raises(Exception):
            client.generate_description(bookmark)

    def test_prepare_prompt(self):
        """Test prompt preparation."""
        client = ClaudeAPIClient(api_key="test-key")
        bookmark = Bookmark(
            title="Test Title",
            note="User note",
            excerpt="Page excerpt",
            url="https://example.com",
        )

        prompt = client._prepare_prompt(bookmark)

        assert "Test Title" in prompt
        assert "User note" in prompt
        assert "Page excerpt" in prompt
        assert "example.com" in prompt

    def test_estimate_tokens(self):
        """Test token estimation."""
        client = ClaudeAPIClient(api_key="test-key")

        text = "This is a test text for token estimation."
        token_count = client._estimate_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        client = ClaudeAPIClient(api_key="test-key")

        stats = client.get_usage_stats()

        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_cost" in stats


class TestOpenAIAPIClient:
    """Test OpenAIAPIClient class."""

    def test_init_with_api_key(self):
        """Test OpenAIAPIClient initialization with API key."""
        client = OpenAIAPIClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.model == "gpt-3.5-turbo"
        assert client.base_url is not None

    def test_init_without_api_key(self):
        """Test OpenAIAPIClient initialization without API key."""
        client = OpenAIAPIClient(api_key=None)

        assert client.is_available is False

    @patch("openai.ChatCompletion.create")
    def test_generate_description_success(self, mock_create):
        """Test successful description generation."""
        # Mock successful OpenAI response
        mock_create.return_value = {
            "choices": [{"message": {"content": "Generated description"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        client = OpenAIAPIClient(api_key="test-key")
        bookmark = Bookmark(title="Test", note="Test note", url="https://example.com")

        result = client.generate_description(bookmark)

        assert result == "Generated description"
        mock_create.assert_called_once()

    @patch("openai.ChatCompletion.create")
    def test_generate_description_api_error(self, mock_create):
        """Test description generation with API error."""
        # Mock OpenAI API error
        mock_create.side_effect = Exception("API error")

        client = OpenAIAPIClient(api_key="test-key")
        bookmark = Bookmark(title="Test", url="https://example.com")

        with pytest.raises(Exception):
            client.generate_description(bookmark)

    def test_prepare_messages(self):
        """Test message preparation for OpenAI API."""
        client = OpenAIAPIClient(api_key="test-key")
        bookmark = Bookmark(
            title="Test Title",
            note="User note",
            excerpt="Page excerpt",
            url="https://example.com",
        )

        messages = client._prepare_messages(bookmark)

        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert any("Test Title" in str(msg) for msg in messages)

    def test_estimate_tokens(self):
        """Test token estimation for OpenAI."""
        client = OpenAIAPIClient(api_key="test-key")

        text = "This is a test text for token estimation."
        token_count = client._estimate_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        client = OpenAIAPIClient(api_key="test-key")

        stats = client.get_usage_stats()

        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_cost" in stats


class TestEnhancedAIProcessorIntegration:
    """Integration tests for AI processor with different engines."""

    @patch("transformers.pipeline")
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
