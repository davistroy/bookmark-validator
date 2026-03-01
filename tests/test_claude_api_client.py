"""
Comprehensive Tests for ClaudeAPIClient

This module provides comprehensive unit tests for the ClaudeAPIClient class,
covering message formatting, response parsing, error handling, token tracking,
cost calculation, and retry logic.
"""

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from bookmark_processor.core.claude_api_client import ClaudeAPIClient
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.structured_output import BookmarkEnhancement


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def claude_client():
    """Create a ClaudeAPIClient for testing."""
    return ClaudeAPIClient(api_key="sk-ant-api03-test-key-12345", timeout=30)


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        id="test1",
        title="Python Documentation",
        url="https://docs.python.org/3/",
        folder="Programming/Python",
        tags=["python", "documentation"],
        note="Official Python documentation",
        excerpt="Python is a programming language",
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
        "requests_in_window": 5,
        "requests_per_minute": 50,
        "utilization_percent": 10.0,
    }
    return limiter


# ============================================================================
# Initialization Tests
# ============================================================================


class TestClaudeAPIClientInitialization:
    """Tests for ClaudeAPIClient initialization."""

    def test_initialization_defaults(self, claude_client):
        """Test initialization with default parameters."""
        assert claude_client.api_key == "sk-ant-api03-test-key-12345"
        assert claude_client.timeout == 30
        assert claude_client.total_input_tokens == 0
        assert claude_client.total_output_tokens == 0
        assert claude_client.request_count == 0

    def test_initialization_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = ClaudeAPIClient(api_key="test-key", timeout=60)
        assert client.timeout == 60

    def test_api_configuration_constants(self):
        """Test API configuration constants."""
        assert ClaudeAPIClient.BASE_URL == "https://api.anthropic.com/v1/messages"
        assert ClaudeAPIClient.API_VERSION == "2023-06-01"
        assert "claude" in ClaudeAPIClient.MODEL.lower()

    def test_pricing_constants(self):
        """Test pricing constants are defined."""
        assert ClaudeAPIClient.COST_PER_1K_INPUT_TOKENS > 0
        assert ClaudeAPIClient.COST_PER_1K_OUTPUT_TOKENS > 0

    def test_rate_limiter_initialization(self, claude_client):
        """Test rate limiter is initialized."""
        assert claude_client.rate_limiter is not None


# ============================================================================
# Authentication Header Tests
# ============================================================================


class TestAuthHeaders:
    """Tests for authentication headers."""

    def test_get_auth_headers(self, claude_client):
        """Test auth headers contain API key."""
        headers = claude_client._get_auth_headers()

        assert "x-api-key" in headers
        assert headers["x-api-key"] == "sk-ant-api03-test-key-12345"
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == ClaudeAPIClient.API_VERSION


# ============================================================================
# Prompt Creation Tests
# ============================================================================


class TestPromptCreation:
    """Tests for bookmark prompt creation."""

    def test_create_bookmark_prompt_full(self, claude_client, sample_bookmark):
        """Test prompt creation with full bookmark data."""
        prompt = claude_client._create_bookmark_prompt(
            sample_bookmark, "Existing content here"
        )

        assert "Python Documentation" in prompt
        assert "docs.python.org" in prompt
        assert "Existing content here" in prompt
        assert "Requirements:" in prompt

    def test_create_bookmark_prompt_minimal(self, claude_client, sample_bookmark_minimal):
        """Test prompt creation with minimal bookmark data."""
        prompt = claude_client._create_bookmark_prompt(sample_bookmark_minimal)

        assert "Untitled" in prompt
        assert "example.com" in prompt

    def test_create_bookmark_prompt_uses_note_as_fallback(self, claude_client):
        """Test prompt uses note when no existing content provided."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="https://example.com",
            note="This is the note",
            excerpt="",
        )
        prompt = claude_client._create_bookmark_prompt(bookmark)
        assert "This is the note" in prompt

    def test_create_bookmark_prompt_uses_excerpt_as_fallback(self, claude_client):
        """Test prompt uses excerpt when no note available."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="https://example.com",
            note="",
            excerpt="This is the excerpt",
        )
        prompt = claude_client._create_bookmark_prompt(bookmark)
        assert "This is the excerpt" in prompt

    def test_create_batch_prompt(self, claude_client, sample_bookmark):
        """Test batch prompt creation."""
        bookmarks = [sample_bookmark, sample_bookmark]
        prompt = claude_client._create_batch_prompt(bookmarks)

        assert "1." in prompt
        assert "2." in prompt
        assert "Descriptions:" in prompt
        assert "100-150 chars" in prompt

    def test_create_batch_prompt_truncates_content(self, claude_client):
        """Test that batch prompt truncates long content."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="https://example.com",
            note="A" * 200,  # Long note
        )
        prompt = claude_client._create_batch_prompt([bookmark])
        # Content should be truncated with ellipsis
        assert "..." in prompt

    def test_create_batch_prompt_with_existing_content(self, claude_client, sample_bookmark):
        """Test batch prompt with existing content provided."""
        bookmarks = [sample_bookmark]
        existing_content = ["Custom existing content"]
        prompt = claude_client._create_batch_prompt(bookmarks, existing_content)

        assert "Custom existing content" in prompt


# ============================================================================
# Generate Description Tests
# ============================================================================


class TestGenerateDescription:
    """Tests for description generation."""

    @pytest.mark.asyncio
    async def test_generate_description_success(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test successful description generation."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [
                {
                    "type": "tool_use",
                    "input": {
                        "description": "Official Python language documentation with tutorials",
                        "tags": ["python", "programming", "documentation"],
                        "category": "documentation",
                        "confidence": 0.9,
                    },
                }
            ],
            "usage": {"input_tokens": 150, "output_tokens": 50},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await claude_client.generate_description(
                sample_bookmark
            )

        assert "Python" in description
        assert metadata["provider"] == "claude"
        assert metadata["success"] is True
        assert metadata["input_tokens"] == 150
        assert metadata["output_tokens"] == 50
        assert metadata["structured_output"] is True

    @pytest.mark.asyncio
    async def test_generate_description_text_fallback(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test fallback to text parsing when tool use fails."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": "A comprehensive Python documentation resource for developers",
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await claude_client.generate_description(
                sample_bookmark
            )

        assert description is not None
        assert len(description) > 0
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_generate_description_empty_response(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test handling of empty response."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {"content": [], "usage": {"input_tokens": 100, "output_tokens": 0}}

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(Exception, match="Claude API error"):
                await claude_client.generate_description(sample_bookmark)

    @pytest.mark.asyncio
    async def test_generate_description_rate_limit_timeout(
        self, claude_client, sample_bookmark
    ):
        """Test handling of rate limit timeout."""
        mock_limiter = AsyncMock()
        mock_limiter.acquire.return_value = False
        claude_client.rate_limiter = mock_limiter

        with pytest.raises(Exception, match="Rate limit timeout"):
            await claude_client.generate_description(sample_bookmark)

    @pytest.mark.asyncio
    async def test_generate_description_api_error(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test handling of API errors."""
        claude_client.rate_limiter = mock_rate_limiter

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = Exception("API connection failed")

            with pytest.raises(Exception, match="Claude API error"):
                await claude_client.generate_description(sample_bookmark)

    @pytest.mark.asyncio
    async def test_generate_description_without_structured_output(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test generation without structured output."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [{"type": "text", "text": "Simple text description"}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await claude_client.generate_description(
                sample_bookmark, use_structured_output=False
            )

        assert description is not None
        # Verify tool was not requested
        call_args = mock_request.call_args
        request_data = call_args[1]["data"]
        assert "tools" not in request_data

    @pytest.mark.asyncio
    async def test_generate_description_token_tracking(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test that tokens are tracked correctly."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [{"type": "text", "text": "Test description"}],
            "usage": {"input_tokens": 200, "output_tokens": 50},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            await claude_client.generate_description(sample_bookmark)

        assert claude_client.total_input_tokens == 200
        assert claude_client.total_output_tokens == 50
        assert claude_client.request_count == 1


# ============================================================================
# Generate Descriptions Batch Tests
# ============================================================================


class TestGenerateDescriptionsBatch:
    """Tests for batch description generation."""

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_success(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test successful batch description generation."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [{"type": "text", "text": "Test description"}],
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            bookmarks = [sample_bookmark, sample_bookmark]
            results = await claude_client.generate_descriptions_batch(bookmarks)

        assert len(results) == 2
        for desc, meta in results:
            assert meta["success"] is True

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_with_errors(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test batch generation handles individual errors."""
        claude_client.rate_limiter = mock_rate_limiter

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Second bookmark failed")
            return {
                "content": [{"type": "text", "text": "Test description"}],
                "usage": {"input_tokens": 100, "output_tokens": 30},
            }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = side_effect

            bookmarks = [sample_bookmark, sample_bookmark, sample_bookmark]
            results = await claude_client.generate_descriptions_batch(bookmarks)

        assert len(results) == 3
        # First should succeed, second should fail, third should succeed
        assert results[0][1]["success"] is True
        assert results[1][1]["success"] is False
        assert results[2][1]["success"] is True

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_with_existing_content(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test batch generation with existing content."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [{"type": "text", "text": "Enhanced description"}],
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            bookmarks = [sample_bookmark]
            existing_content = ["Custom content"]
            results = await claude_client.generate_descriptions_batch(
                bookmarks, existing_content
            )

        assert len(results) == 1


# ============================================================================
# Cost Calculation Tests
# ============================================================================


class TestCostCalculation:
    """Tests for cost calculation methods."""

    def test_get_cost_per_request(self, claude_client):
        """Test estimated cost per request calculation."""
        cost = claude_client.get_cost_per_request()

        # Should be a small positive value
        assert cost > 0
        assert cost < 0.1  # Should be less than 10 cents per request

    def test_get_usage_statistics_initial(self, claude_client):
        """Test initial usage statistics."""
        stats = claude_client.get_usage_statistics()

        assert stats["provider"] == "claude"
        assert stats["total_requests"] == 0
        assert stats["total_input_tokens"] == 0
        assert stats["total_output_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0

    def test_get_usage_statistics_after_requests(self, claude_client):
        """Test usage statistics after simulated requests."""
        claude_client.total_input_tokens = 1000
        claude_client.total_output_tokens = 200
        claude_client.request_count = 5

        stats = claude_client.get_usage_statistics()

        assert stats["total_requests"] == 5
        assert stats["total_input_tokens"] == 1000
        assert stats["total_output_tokens"] == 200
        assert stats["total_cost_usd"] > 0
        assert stats["avg_input_tokens_per_request"] == 200
        assert stats["avg_output_tokens_per_request"] == 40

    def test_get_usage_statistics_averages_with_zero_requests(self, claude_client):
        """Test average calculation with zero requests."""
        stats = claude_client.get_usage_statistics()

        # Should not divide by zero
        assert stats["avg_input_tokens_per_request"] == 0
        assert stats["avg_output_tokens_per_request"] == 0
        assert stats["avg_cost_per_request"] == 0


# ============================================================================
# Rate Limit Info Tests
# ============================================================================


class TestRateLimitInfo:
    """Tests for rate limit information."""

    def test_get_rate_limit_info(self, claude_client, mock_rate_limiter):
        """Test rate limit info retrieval."""
        claude_client.rate_limiter = mock_rate_limiter

        info = claude_client.get_rate_limit_info()

        assert info["provider"] == "claude"
        assert info["requests_per_minute"] == 50
        assert info["burst_size"] == 10
        assert "model" in info
        assert "status" in info


# ============================================================================
# Mock Response Tests
# ============================================================================


class TestMockResponse:
    """Tests for mock response generation."""

    def test_get_mock_response(self, claude_client):
        """Test Claude-specific mock response."""
        response = claude_client._get_mock_response(
            "POST", ClaudeAPIClient.BASE_URL, {"model": "test"}
        )

        assert "content" in response
        assert len(response["content"]) > 0
        assert "text" in response["content"][0]
        assert "usage" in response
        assert response["model"] == claude_client.MODEL
        assert response["test_mode"] is True


# ============================================================================
# Reset Statistics Tests
# ============================================================================


class TestResetStatistics:
    """Tests for statistics reset."""

    def test_reset_statistics(self, claude_client):
        """Test resetting usage statistics."""
        # Set some values
        claude_client.total_input_tokens = 1000
        claude_client.total_output_tokens = 200
        claude_client.request_count = 10

        # Reset
        claude_client.reset_statistics()

        # Verify reset
        assert claude_client.total_input_tokens == 0
        assert claude_client.total_output_tokens == 0
        assert claude_client.request_count == 0


# ============================================================================
# Tool Use Response Parsing Tests
# ============================================================================


class TestToolUseResponseParsing:
    """Tests for parsing tool use responses."""

    @pytest.mark.asyncio
    async def test_parse_tool_use_response(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test parsing of tool use response format."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "enhance_bookmark",
                    "input": {
                        "description": "Comprehensive Python documentation",
                        "tags": ["python", "docs"],
                        "category": "documentation",
                        "confidence": 0.95,
                    },
                }
            ],
            "usage": {"input_tokens": 150, "output_tokens": 50},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await claude_client.generate_description(
                sample_bookmark
            )

        assert description == "Comprehensive Python documentation"
        assert metadata["tags"] == ["python", "docs"]
        assert metadata["category"] == "documentation"
        assert metadata["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_parse_malformed_tool_response(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test handling of malformed tool response."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [
                {
                    "type": "tool_use",
                    "input": {
                        # Missing required fields
                        "tags": ["test"],
                    },
                },
                {
                    "type": "text",
                    "text": "Fallback text description for the bookmark",
                },
            ],
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await claude_client.generate_description(
                sample_bookmark
            )

        # Should fall back to text parsing
        assert description is not None
        assert len(description) > 0


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_create_bookmark_prompt_with_none_values(self, claude_client):
        """Test prompt creation when bookmark has None values."""
        bookmark = Bookmark(
            id=None,
            title=None,
            url=None,
            folder=None,
            tags=[],
            note=None,
            excerpt=None,
        )

        # Should not raise, should use defaults
        prompt = claude_client._create_bookmark_prompt(bookmark)
        assert "Untitled" in prompt
        assert "No URL" in prompt or "unknown" in prompt

    def test_create_batch_prompt_invalid_url(self, claude_client):
        """Test batch prompt with invalid URL."""
        bookmark = Bookmark(
            id="test",
            title="Test",
            url="not-a-valid-url",
            note="Test note",
        )

        prompt = claude_client._create_batch_prompt([bookmark])
        assert "unknown" in prompt

    @pytest.mark.asyncio
    async def test_generate_description_with_special_characters(
        self, claude_client, mock_rate_limiter
    ):
        """Test description generation with special characters in content."""
        claude_client.rate_limiter = mock_rate_limiter

        bookmark = Bookmark(
            id="test",
            title="Test with <script> and 'quotes' and \"double quotes\"",
            url="https://example.com?param=value&other=<test>",
            note="Content with newlines\nand\ttabs",
        )

        mock_response = {
            "content": [{"type": "text", "text": "Safe description"}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await claude_client.generate_description(bookmark)

        assert description is not None
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_multiple_content_blocks_in_response(
        self, claude_client, sample_bookmark, mock_rate_limiter
    ):
        """Test handling of multiple content blocks in response."""
        claude_client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [
                {"type": "text", "text": "Preamble text"},
                {
                    "type": "tool_use",
                    "input": {
                        "description": "Main description",
                        "tags": ["tag1"],
                        "category": "article",
                        "confidence": 0.8,
                    },
                },
                {"type": "text", "text": "Postamble text"},
            ],
            "usage": {"input_tokens": 200, "output_tokens": 80},
        }

        with patch.object(
            claude_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            description, metadata = await claude_client.generate_description(
                sample_bookmark
            )

        # Should use the tool_use block
        assert description == "Main description"


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestIntegration:
    """Integration-like tests for ClaudeAPIClient."""

    @pytest.mark.asyncio
    async def test_full_flow_with_context_manager(self, sample_bookmark, mock_rate_limiter):
        """Test full flow using async context manager."""
        client = ClaudeAPIClient(api_key="test-key")
        client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [
                {
                    "type": "tool_use",
                    "input": {
                        "description": "Test description",
                        "tags": ["test"],
                        "category": "article",
                        "confidence": 0.9,
                    },
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }

        async with client:
            with patch.object(
                client, "_make_request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response

                description, metadata = await client.generate_description(sample_bookmark)

        assert description == "Test description"
        assert metadata["provider"] == "claude"
        assert client.request_count == 1

    @pytest.mark.asyncio
    async def test_cost_tracking_across_multiple_requests(
        self, sample_bookmark, mock_rate_limiter
    ):
        """Test cost tracking across multiple requests."""
        client = ClaudeAPIClient(api_key="test-key")
        client.rate_limiter = mock_rate_limiter

        mock_response = {
            "content": [{"type": "text", "text": "Description"}],
            "usage": {"input_tokens": 100, "output_tokens": 25},
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
