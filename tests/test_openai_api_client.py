"""
Comprehensive Tests for OpenAIAPIClient

This module provides comprehensive unit tests for the OpenAIAPIClient class,
covering chat completions, token counting, error handling, batch processing,
and retry logic.
"""

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from bookmark_processor.core.openai_api_client import OpenAIAPIClient
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.structured_output import BookmarkEnhancement


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def openai_client():
    """Create an OpenAIAPIClient for testing."""
    return OpenAIAPIClient(api_key="sk-test-openai-key-12345", timeout=30)


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        id="test1",
        title="Machine Learning Tutorial",
        url="https://ml-tutorial.com/basics",
        folder="AI/MachineLearning",
        tags=["ml", "tutorial", "ai"],
        note="Introduction to machine learning concepts",
        excerpt="Learn ML fundamentals step by step",
    )


@pytest.fixture
def sample_bookmark_minimal():
    """Create a minimal bookmark for testing."""
    return Bookmark(
        id="test2",
        title="",
        url="https://example.com",
        folder="",
        tags=[],
        note="",
        excerpt="",
    )


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter."""
    limiter = AsyncMock()
    limiter.acquire.return_value = True
    limiter.get_status.return_value = {
        "requests_in_window": 10,
        "requests_per_minute": 60,
        "utilization_percent": 16.7,
    }
    return limiter


# ============================================================================
# Initialization Tests
# ============================================================================


class TestOpenAIAPIClientInitialization:
    """Tests for OpenAIAPIClient initialization."""

    def test_initialization_defaults(self, openai_client):
        """Test initialization with default parameters."""
        assert openai_client.api_key == "sk-test-openai-key-12345"
        assert openai_client.timeout == 30
        assert openai_client.total_input_tokens == 0
        assert openai_client.total_output_tokens == 0
        assert openai_client.request_count == 0

    def test_initialization_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = OpenAIAPIClient(api_key="test-key", timeout=60)
        assert client.timeout == 60

    def test_api_configuration_constants(self):
        """Test API configuration constants."""
        assert OpenAIAPIClient.BASE_URL == "https://api.openai.com/v1/chat/completions"
        assert "gpt" in OpenAIAPIClient.MODEL.lower()

    def test_pricing_constants(self):
        """Test pricing constants are defined."""
        assert OpenAIAPIClient.COST_PER_1K_INPUT_TOKENS > 0
        assert OpenAIAPIClient.COST_PER_1K_OUTPUT_TOKENS > 0

    def test_rate_limiter_initialization(self, openai_client):
        """Test rate limiter is initialized."""
        assert openai_client.rate_limiter is not None


# ============================================================================
# Authentication Header Tests
# ============================================================================


class TestAuthHeaders:
    """Tests for authentication headers."""

    def test_get_auth_headers(self, openai_client):
        """Test auth headers contain bearer token."""
        headers = openai_client._get_auth_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test-openai-key-12345"


# ============================================================================
# Message Creation Tests
# ============================================================================


class TestMessageCreation:
    """Tests for chat message creation."""

    def test_create_bookmark_prompt_structured(self, openai_client, sample_bookmark):
        """Test message creation with structured output."""
        messages = openai_client._create_bookmark_prompt(
            sample_bookmark, "Existing content", structured=True
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "JSON" in messages[0]["content"] or "json" in messages[0]["content"].lower()

    def test_create_bookmark_prompt_unstructured(self, openai_client, sample_bookmark):
        """Test message creation without structured output."""
        messages = openai_client._create_bookmark_prompt(
            sample_bookmark, "Existing content", structured=False
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # System message should not mention JSON
        assert "structured" not in messages[0]["content"].lower()

    def test_create_bookmark_prompt_uses_existing_content(
        self, openai_client, sample_bookmark
    ):
        """Test that existing content is included in prompt."""
        messages = openai_client._create_bookmark_prompt(
            sample_bookmark, "Custom existing content"
        )

        user_content = messages[1]["content"]
        assert "Custom existing content" in user_content

    def test_create_bookmark_prompt_fallback_to_note(self, openai_client):
        """Test fallback to note when no existing content."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="https://example.com",
            note="This is the note content",
            excerpt="",
        )
        messages = openai_client._create_bookmark_prompt(bookmark, None)
        assert "This is the note content" in messages[1]["content"]

    def test_create_bookmark_prompt_fallback_to_excerpt(self, openai_client):
        """Test fallback to excerpt when no note."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="https://example.com",
            note="",
            excerpt="This is the excerpt content",
        )
        messages = openai_client._create_bookmark_prompt(bookmark, None)
        assert "This is the excerpt content" in messages[1]["content"]

    def test_create_batch_messages(self, openai_client, sample_bookmark):
        """Test batch message creation."""
        bookmarks = [sample_bookmark, sample_bookmark]
        messages = openai_client._create_batch_messages(bookmarks)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Bookmarks:" in messages[1]["content"]
        assert "Descriptions:" in messages[1]["content"]

    def test_create_batch_messages_truncates_content(self, openai_client):
        """Test that batch messages truncate long content."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="https://example.com",
            note="A" * 200,  # Long note
        )
        messages = openai_client._create_batch_messages([bookmark])
        # Content should be truncated
        user_content = messages[1]["content"]
        assert "..." in user_content

    def test_create_batch_messages_with_existing_content(
        self, openai_client, sample_bookmark
    ):
        """Test batch messages with provided existing content."""
        bookmarks = [sample_bookmark]
        existing_content = ["Custom content for batch"]
        messages = openai_client._create_batch_messages(bookmarks, existing_content)

        assert "Custom content for batch" in messages[1]["content"]


# ============================================================================
# Generate Description Tests
# ============================================================================


class TestGenerateDescription:
    """Tests for description generation."""

    @pytest.mark.asyncio
    async def test_generate_description_success_structured(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test successful description generation with structured output."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "description": "Comprehensive ML tutorial for beginners",
                                "tags": ["machine-learning", "tutorial", "ai"],
                                "category": "tutorial",
                                "confidence": 0.9,
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 150, "completion_tokens": 50},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await openai_client.generate_description(
                sample_bookmark
            )

        assert description == "Comprehensive ML tutorial for beginners"
        assert metadata["provider"] == "openai"
        assert metadata["success"] is True
        assert metadata["input_tokens"] == 150
        assert metadata["output_tokens"] == 50
        assert metadata["structured_output"] is True
        assert metadata["tags"] == ["machine-learning", "tutorial", "ai"]
        assert metadata["category"] == "tutorial"

    @pytest.mark.asyncio
    async def test_generate_description_text_fallback(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test fallback to text parsing when JSON fails."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [
                {"message": {"content": "A great machine learning resource for beginners"}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await openai_client.generate_description(
                sample_bookmark
            )

        assert description is not None
        assert len(description) > 0
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_generate_description_empty_response(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test handling of empty response."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 0}}

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(Exception, match="OpenAI API error"):
                await openai_client.generate_description(sample_bookmark)

    @pytest.mark.asyncio
    async def test_generate_description_rate_limit_timeout(
        self, openai_client, sample_bookmark
    ):
        """Test handling of rate limit timeout."""
        mock_limiter = AsyncMock()
        mock_limiter.acquire.return_value = False
        openai_client.rate_limiter = mock_limiter

        with pytest.raises(Exception, match="Rate limit timeout"):
            await openai_client.generate_description(sample_bookmark)

    @pytest.mark.asyncio
    async def test_generate_description_api_error(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test handling of API errors."""
        openai_client.rate_limiter = mock_rate_limiter

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = Exception("API connection failed")

            with pytest.raises(Exception, match="OpenAI API error"):
                await openai_client.generate_description(sample_bookmark)

    @pytest.mark.asyncio
    async def test_generate_description_without_structured_output(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test generation without structured output."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [{"message": {"content": "Simple text description"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 15},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await openai_client.generate_description(
                sample_bookmark, use_structured_output=False
            )

        # Verify JSON response format was not requested
        call_args = mock_request.call_args
        request_data = call_args[1]["data"]
        assert "response_format" not in request_data

    @pytest.mark.asyncio
    async def test_generate_description_token_tracking(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test that tokens are tracked correctly."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [{"message": {"content": "Test description"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            await openai_client.generate_description(sample_bookmark)

        assert openai_client.total_input_tokens == 200
        assert openai_client.total_output_tokens == 40
        assert openai_client.request_count == 1


# ============================================================================
# Generate Descriptions Batch Tests
# ============================================================================


class TestGenerateDescriptionsBatch:
    """Tests for batch description generation."""

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_success(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test successful batch description generation."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "1. First description here\n2. Second description here"
                    }
                }
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": 60},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            bookmarks = [sample_bookmark, sample_bookmark]
            results = await openai_client.generate_descriptions_batch(bookmarks)

        assert len(results) == 2
        for desc, meta in results:
            assert meta["success"] is True
            assert meta["batch_size"] == 2

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_large_batch(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test batch generation with more than batch_size bookmarks."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [
                {"message": {"content": "\n".join([f"{i+1}. Desc {i}" for i in range(20)])}}
            ],
            "usage": {"prompt_tokens": 500, "completion_tokens": 200},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            # Create 25 bookmarks (more than batch_size of 20)
            bookmarks = [sample_bookmark] * 25
            results = await openai_client.generate_descriptions_batch(bookmarks)

        assert len(results) == 25
        # Should have made 2 API calls
        assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_with_error(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test batch generation handles errors gracefully."""
        openai_client.rate_limiter = mock_rate_limiter

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = Exception("Batch processing failed")

            bookmarks = [sample_bookmark, sample_bookmark]
            results = await openai_client.generate_descriptions_batch(bookmarks)

        assert len(results) == 2
        for desc, meta in results:
            assert desc == ""
            assert meta["success"] is False

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_with_existing_content(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test batch generation with existing content."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [{"message": {"content": "1. Enhanced description"}}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 30},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            bookmarks = [sample_bookmark]
            existing_content = ["Custom existing content"]
            results = await openai_client.generate_descriptions_batch(
                bookmarks, existing_content
            )

        assert len(results) == 1


# ============================================================================
# Parse Batch Response Tests
# ============================================================================


class TestParseBatchResponse:
    """Tests for batch response parsing."""

    def test_parse_batch_response_numbered_list(self, openai_client):
        """Test parsing numbered list response."""
        content = "1. First description\n2. Second description\n3. Third description"
        descriptions = openai_client._parse_batch_response(content, 3)

        assert len(descriptions) == 3
        assert descriptions[0] == "First description"
        assert descriptions[1] == "Second description"
        assert descriptions[2] == "Third description"

    def test_parse_batch_response_with_periods(self, openai_client):
        """Test parsing response with period-prefixed numbers."""
        content = "1. Description one.\n2. Description two.\n3. Description three."
        descriptions = openai_client._parse_batch_response(content, 3)

        assert len(descriptions) == 3

    def test_parse_batch_response_insufficient_results(self, openai_client):
        """Test parsing when response has fewer items than expected."""
        content = "1. Only one description"
        descriptions = openai_client._parse_batch_response(content, 3)

        assert len(descriptions) == 3
        assert descriptions[0] == "Only one description"
        assert descriptions[1] == ""  # Padded with empty
        assert descriptions[2] == ""

    def test_parse_batch_response_excess_results(self, openai_client):
        """Test parsing when response has more items than expected."""
        content = "1. First\n2. Second\n3. Third\n4. Fourth\n5. Fifth"
        descriptions = openai_client._parse_batch_response(content, 3)

        assert len(descriptions) == 3  # Should truncate

    def test_parse_batch_response_empty_content(self, openai_client):
        """Test parsing empty content."""
        descriptions = openai_client._parse_batch_response("", 3)

        assert len(descriptions) == 3
        assert all(d == "" for d in descriptions)

    def test_parse_batch_response_no_numbers(self, openai_client):
        """Test parsing response without numbers."""
        content = "Just plain text without numbers"
        descriptions = openai_client._parse_batch_response(content, 2)

        assert len(descriptions) == 2
        assert all(d == "" for d in descriptions)


# ============================================================================
# Cost Calculation Tests
# ============================================================================


class TestCostCalculation:
    """Tests for cost calculation methods."""

    def test_get_cost_per_request(self, openai_client):
        """Test estimated cost per request calculation."""
        cost = openai_client.get_cost_per_request()

        assert cost > 0
        assert cost < 0.1  # Should be less than 10 cents

    def test_get_usage_statistics_initial(self, openai_client):
        """Test initial usage statistics."""
        stats = openai_client.get_usage_statistics()

        assert stats["provider"] == "openai"
        assert stats["total_requests"] == 0
        assert stats["total_input_tokens"] == 0
        assert stats["total_output_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0

    def test_get_usage_statistics_after_requests(self, openai_client):
        """Test usage statistics after simulated requests."""
        openai_client.total_input_tokens = 2000
        openai_client.total_output_tokens = 400
        openai_client.request_count = 10

        stats = openai_client.get_usage_statistics()

        assert stats["total_requests"] == 10
        assert stats["total_input_tokens"] == 2000
        assert stats["total_output_tokens"] == 400
        assert stats["total_cost_usd"] > 0
        assert stats["avg_input_tokens_per_request"] == 200
        assert stats["avg_output_tokens_per_request"] == 40

    def test_get_usage_statistics_averages_with_zero_requests(self, openai_client):
        """Test average calculation with zero requests."""
        stats = openai_client.get_usage_statistics()

        assert stats["avg_input_tokens_per_request"] == 0
        assert stats["avg_output_tokens_per_request"] == 0
        assert stats["avg_cost_per_request"] == 0


# ============================================================================
# Rate Limit Info Tests
# ============================================================================


class TestRateLimitInfo:
    """Tests for rate limit information."""

    def test_get_rate_limit_info(self, openai_client, mock_rate_limiter):
        """Test rate limit info retrieval."""
        openai_client.rate_limiter = mock_rate_limiter

        info = openai_client.get_rate_limit_info()

        assert info["provider"] == "openai"
        assert info["requests_per_minute"] == 60
        assert info["burst_size"] == 20
        assert "model" in info
        assert "status" in info


# ============================================================================
# Mock Response Tests
# ============================================================================


class TestMockResponse:
    """Tests for mock response generation."""

    def test_get_mock_response(self, openai_client):
        """Test OpenAI-specific mock response."""
        response = openai_client._get_mock_response(
            "POST", OpenAIAPIClient.BASE_URL, {"model": "test"}
        )

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        assert "usage" in response
        assert "prompt_tokens" in response["usage"]
        assert "completion_tokens" in response["usage"]
        assert response["model"] == openai_client.MODEL
        assert response["test_mode"] is True


# ============================================================================
# Reset Statistics Tests
# ============================================================================


class TestResetStatistics:
    """Tests for statistics reset."""

    def test_reset_statistics(self, openai_client):
        """Test resetting usage statistics."""
        # Set some values
        openai_client.total_input_tokens = 2000
        openai_client.total_output_tokens = 400
        openai_client.request_count = 15

        # Reset
        openai_client.reset_statistics()

        # Verify reset
        assert openai_client.total_input_tokens == 0
        assert openai_client.total_output_tokens == 0
        assert openai_client.request_count == 0


# ============================================================================
# JSON Response Parsing Tests
# ============================================================================


class TestJSONResponseParsing:
    """Tests for JSON response parsing."""

    @pytest.mark.asyncio
    async def test_parse_valid_json_response(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test parsing of valid JSON response."""
        openai_client.rate_limiter = mock_rate_limiter

        json_content = {
            "description": "Excellent ML learning resource",
            "tags": ["ml", "learning"],
            "category": "tutorial",
            "confidence": 0.85,
        }

        mock_response = {
            "choices": [{"message": {"content": json.dumps(json_content)}}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 50},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await openai_client.generate_description(
                sample_bookmark
            )

        assert description == "Excellent ML learning resource"
        assert metadata["tags"] == ["ml", "learning"]
        assert metadata["category"] == "tutorial"
        assert metadata["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_parse_invalid_json_falls_back(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test that invalid JSON falls back to text parsing."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [{"message": {"content": "Not valid JSON {broken"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await openai_client.generate_description(
                sample_bookmark
            )

        # Should fall back to text parsing
        assert description is not None
        assert len(description) > 0
        assert metadata["success"] is True


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_create_bookmark_prompt_with_none_values(self, openai_client):
        """Test message creation when bookmark has None values."""
        bookmark = Bookmark(
            id=None,
            title=None,
            url=None,
            folder=None,
            tags=[],
            note=None,
            excerpt=None,
        )

        messages = openai_client._create_bookmark_prompt(bookmark, None)
        assert len(messages) == 2
        assert "Untitled" in messages[1]["content"]

    def test_create_batch_messages_invalid_url(self, openai_client):
        """Test batch messages with invalid URL."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="not-a-valid-url",
            note="Test note",
        )

        messages = openai_client._create_batch_messages([bookmark])
        user_content = messages[1]["content"]
        assert "unknown" in user_content

    @pytest.mark.asyncio
    async def test_generate_description_with_special_characters(
        self, openai_client, mock_rate_limiter
    ):
        """Test description generation with special characters."""
        openai_client.rate_limiter = mock_rate_limiter

        bookmark = Bookmark(
            id="test",
            title="Test with <script> and 'quotes' and \"double quotes\"",
            url="https://example.com?param=value&other=<test>",
            note="Content with newlines\nand\ttabs",
        )

        mock_response = {
            "choices": [{"message": {"content": "Safe description"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 15},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await openai_client.generate_description(bookmark)

        assert description is not None
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_batch_rate_limit_per_batch(
        self, openai_client, sample_bookmark, mock_rate_limiter
    ):
        """Test that rate limiter is called per batch."""
        openai_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [
                {"message": {"content": "\n".join([f"{i+1}. Desc {i}" for i in range(20)])}}
            ],
            "usage": {"prompt_tokens": 500, "completion_tokens": 200},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            # 25 bookmarks = 2 batches
            bookmarks = [sample_bookmark] * 25
            await openai_client.generate_descriptions_batch(bookmarks)

        # Rate limiter should be called for each batch
        assert mock_rate_limiter.acquire.call_count == 2


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestIntegration:
    """Integration-like tests for OpenAIAPIClient."""

    @pytest.mark.asyncio
    async def test_full_flow_with_context_manager(self, sample_bookmark, mock_rate_limiter):
        """Test full flow using async context manager."""
        client = OpenAIAPIClient(api_key="test-key")
        client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "description": "Test description",
                                "tags": ["test"],
                                "category": "article",
                                "confidence": 0.9,
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }

        async with client:
            with patch.object(
                client, "_make_request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response

                description, metadata = await client.generate_description(sample_bookmark)

        assert description == "Test description"
        assert metadata["provider"] == "openai"
        assert client.request_count == 1

    @pytest.mark.asyncio
    async def test_cost_tracking_across_multiple_requests(
        self, sample_bookmark, mock_rate_limiter
    ):
        """Test cost tracking across multiple requests."""
        client = OpenAIAPIClient(api_key="test-key")
        client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [{"message": {"content": "Description"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25},
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            # Make multiple requests
            for _ in range(5):
                await client.generate_description(sample_bookmark)

        assert client.request_count == 5
        assert client.total_input_tokens == 500
        assert client.total_output_tokens == 125

        stats = client.get_usage_statistics()
        assert stats["total_requests"] == 5
        assert stats["total_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_batch_processing_token_accumulation(
        self, sample_bookmark, mock_rate_limiter
    ):
        """Test token accumulation in batch processing."""
        client = OpenAIAPIClient(api_key="test-key")
        client.rate_limiter = mock_rate_limiter

        mock_response = {
            "choices": [
                {"message": {"content": "1. First desc\n2. Second desc\n3. Third desc"}}
            ],
            "usage": {"prompt_tokens": 300, "completion_tokens": 75},
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            bookmarks = [sample_bookmark] * 3
            await client.generate_descriptions_batch(bookmarks)

        assert client.request_count == 1
        assert client.total_input_tokens == 300
        assert client.total_output_tokens == 75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
