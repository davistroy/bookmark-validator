"""
Comprehensive Tests for BaseAPIClient

This module provides comprehensive unit tests for the BaseAPIClient class,
covering initialization, headers, rate limiting, retry logic, error handling,
and mock response generation.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from bookmark_processor.core.base_api_client import BaseAPIClient
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.error_handler import (
    APIClientError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)


# ============================================================================
# Concrete Implementation for Testing
# ============================================================================


class ConcreteAPIClient(BaseAPIClient):
    """Concrete implementation of BaseAPIClient for testing."""

    async def generate_description(
        self,
        bookmark: Bookmark,
        existing_content: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate an enhanced description for a bookmark."""
        return "Test description", {"provider": "test", "success": True}

    async def generate_descriptions_batch(
        self,
        bookmarks: List[Bookmark],
        existing_content: Optional[List[str]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate enhanced descriptions for a batch of bookmarks."""
        return [
            (f"Test description {i}", {"provider": "test", "success": True})
            for i in range(len(bookmarks))
        ]

    def get_cost_per_request(self) -> float:
        """Get the estimated cost per request."""
        return 0.001

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information."""
        return {"requests_per_minute": 60, "burst_size": 10}


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def api_client():
    """Create a ConcreteAPIClient for testing."""
    return ConcreteAPIClient(
        api_key="test-api-key-12345",
        timeout=30,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
    )


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        id="test1",
        title="Test Bookmark",
        url="https://example.com",
        folder="Test",
        tags=["test", "example"],
        note="Test note",
        excerpt="Test excerpt",
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestBaseAPIClientInitialization:
    """Tests for BaseAPIClient initialization."""

    def test_initialization_with_default_params(self):
        """Test initialization with default parameters."""
        client = ConcreteAPIClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.timeout == 30  # default
        assert client.max_retries == 3  # default
        assert client.base_delay == 1.0  # default
        assert client.max_delay == 60.0  # default
        assert client.request_count == 0
        assert client.error_count == 0
        assert client.retry_count == 0
        assert client._client is None

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        client = ConcreteAPIClient(
            api_key="custom-key",
            timeout=60,
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
        )

        assert client.api_key == "custom-key"
        assert client.timeout == 60
        assert client.max_retries == 5
        assert client.base_delay == 2.0
        assert client.max_delay == 120.0

    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        client = ConcreteAPIClient(api_key="test-key")
        assert client.logger is not None
        assert client.logger.name == "ConcreteAPIClient"


# ============================================================================
# Async Context Manager Tests
# ============================================================================


class TestAsyncContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_client(self, api_client):
        """Test that async context manager initializes HTTP client."""
        assert api_client._client is None

        async with api_client:
            assert api_client._client is not None
            assert isinstance(api_client._client, httpx.AsyncClient)

        # Client should be cleaned up after exiting context
        assert api_client._client is None

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self, api_client):
        """Test that async context manager returns self."""
        async with api_client as client:
            assert client is api_client

    @pytest.mark.asyncio
    async def test_initialize_client_configuration(self, api_client):
        """Test HTTP client configuration during initialization."""
        async with api_client:
            client = api_client._client
            assert client.timeout.connect == 30
            assert client.timeout.read == 30
            assert client.timeout.write == 30

    @pytest.mark.asyncio
    async def test_cleanup_client_closes_connection(self, api_client):
        """Test that cleanup properly closes HTTP client."""
        async with api_client:
            assert api_client._client is not None

        # After exit, client should be None
        assert api_client._client is None

    @pytest.mark.asyncio
    async def test_manual_initialize_and_cleanup(self, api_client):
        """Test manual initialization and cleanup."""
        await api_client._initialize_client()
        assert api_client._client is not None

        await api_client._cleanup_client()
        assert api_client._client is None


# ============================================================================
# Headers Tests
# ============================================================================


class TestHeaders:
    """Tests for header generation."""

    def test_common_headers(self, api_client):
        """Test common headers generation."""
        headers = api_client._get_common_headers()

        assert headers["User-Agent"] == "BookmarkProcessor/1.0"
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    def test_auth_headers_default(self, api_client):
        """Test default auth headers (empty)."""
        headers = api_client._get_auth_headers()
        assert headers == {}


# ============================================================================
# Retry Delay Calculation Tests
# ============================================================================


class TestRetryDelayCalculation:
    """Tests for retry delay calculation."""

    @pytest.mark.asyncio
    async def test_retry_delay_first_attempt(self, api_client):
        """Test retry delay for first attempt."""
        delay = api_client._calculate_retry_delay(0)
        # Base delay with possible jitter
        assert 0.1 <= delay <= api_client.base_delay * 1.1

    @pytest.mark.asyncio
    async def test_retry_delay_exponential_backoff(self, api_client):
        """Test exponential backoff in retry delays."""
        delay_0 = api_client._calculate_retry_delay(0)
        delay_1 = api_client._calculate_retry_delay(1)
        delay_2 = api_client._calculate_retry_delay(2)

        # Each delay should roughly double (with jitter variance)
        assert delay_1 >= delay_0 * 0.8  # Allow for jitter
        assert delay_2 >= delay_1 * 0.8

    @pytest.mark.asyncio
    async def test_retry_delay_respects_max_delay(self, api_client):
        """Test that retry delay respects maximum delay."""
        # Large attempt number should still respect max_delay
        delay = api_client._calculate_retry_delay(20)
        assert delay <= api_client.max_delay * 1.1  # Allow small jitter

    @pytest.mark.asyncio
    async def test_retry_delay_minimum_value(self, api_client):
        """Test that retry delay has minimum value."""
        delay = api_client._calculate_retry_delay(0)
        assert delay >= 0.1


# ============================================================================
# Should Retry Logic Tests
# ============================================================================


class TestShouldRetry:
    """Tests for retry decision logic."""

    def test_should_retry_max_retries_exceeded(self, api_client):
        """Test that retry is rejected when max retries exceeded."""
        api_client.max_retries = 3
        exception = httpx.HTTPStatusError(
            "Server Error",
            request=Mock(),
            response=Mock(status_code=500),
        )
        assert api_client._should_retry(exception, 3) is False
        assert api_client._should_retry(exception, 4) is False

    def test_should_retry_rate_limit_429(self, api_client):
        """Test retry on 429 rate limit status."""
        response = Mock()
        response.status_code = 429
        exception = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=Mock(),
            response=response,
        )
        assert api_client._should_retry(exception, 0) is True

    def test_should_retry_server_errors(self, api_client):
        """Test retry on server error status codes."""
        for status_code in [500, 502, 503, 504]:
            response = Mock()
            response.status_code = status_code
            exception = httpx.HTTPStatusError(
                f"Server Error {status_code}",
                request=Mock(),
                response=response,
            )
            assert api_client._should_retry(exception, 0) is True

    def test_should_retry_408_timeout(self, api_client):
        """Test retry on 408 timeout status."""
        response = Mock()
        response.status_code = 408
        exception = httpx.HTTPStatusError(
            "Request Timeout",
            request=Mock(),
            response=response,
        )
        assert api_client._should_retry(exception, 0) is True

    def test_should_retry_423_locked(self, api_client):
        """Test retry on 423 locked status."""
        response = Mock()
        response.status_code = 423
        exception = httpx.HTTPStatusError(
            "Locked",
            request=Mock(),
            response=response,
        )
        assert api_client._should_retry(exception, 0) is True

    def test_should_not_retry_client_errors(self, api_client):
        """Test no retry on client error status codes."""
        for status_code in [400, 401, 403, 404, 422]:
            response = Mock()
            response.status_code = status_code
            exception = httpx.HTTPStatusError(
                f"Client Error {status_code}",
                request=Mock(),
                response=response,
            )
            assert api_client._should_retry(exception, 0) is False

    def test_should_retry_connect_error(self, api_client):
        """Test retry on connection errors."""
        exception = httpx.ConnectError("Connection refused")
        assert api_client._should_retry(exception, 0) is True

    def test_should_retry_read_timeout(self, api_client):
        """Test retry on read timeout."""
        exception = httpx.ReadTimeout("Read timed out")
        assert api_client._should_retry(exception, 0) is True

    def test_should_retry_write_timeout(self, api_client):
        """Test retry on write timeout."""
        exception = httpx.WriteTimeout("Write timed out")
        assert api_client._should_retry(exception, 0) is True

    def test_should_not_retry_generic_exception(self, api_client):
        """Test no retry on generic exceptions."""
        exception = ValueError("Invalid value")
        assert api_client._should_retry(exception, 0) is False


# ============================================================================
# Error Message Sanitization Tests
# ============================================================================


class TestErrorMessageSanitization:
    """Tests for error message sanitization."""

    def test_sanitize_error_with_api_key(self, api_client):
        """Test that API key is masked in error messages."""
        message = "Error with key: test-api-key-12345 failed"
        sanitized = api_client._sanitize_error_message(message)
        assert "test-api-key-12345" not in sanitized
        assert "test-a" in sanitized  # Partial key visible

    def test_sanitize_error_without_api_key(self, api_client):
        """Test sanitization of message without API key."""
        message = "Generic error occurred"
        sanitized = api_client._sanitize_error_message(message)
        assert sanitized == message


# ============================================================================
# Mock Response Tests
# ============================================================================


class TestMockResponse:
    """Tests for mock response generation."""

    def test_get_mock_response_default(self, api_client):
        """Test default mock response generation."""
        response = api_client._get_mock_response("POST", "https://api.example.com")

        assert "content" in response
        assert len(response["content"]) > 0
        assert "text" in response["content"][0]
        assert "usage" in response
        assert "input_tokens" in response["usage"]
        assert "output_tokens" in response["usage"]
        assert response["test_mode"] is True

    def test_get_mock_response_with_data(self, api_client):
        """Test mock response with request data."""
        request_data = {"model": "test-model", "prompt": "test"}
        response = api_client._get_mock_response(
            "POST", "https://api.example.com", request_data
        )

        assert "content" in response
        assert response["test_mode"] is True


# ============================================================================
# Make Request Tests
# ============================================================================


class TestMakeRequest:
    """Tests for HTTP request functionality."""

    @pytest.mark.asyncio
    async def test_make_request_test_mode(self, api_client, monkeypatch):
        """Test that test mode returns mock response."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "true")

        async with api_client:
            response = await api_client._make_request(
                "POST", "https://api.example.com", {"data": "test"}
            )

        assert response["test_mode"] is True
        assert "content" in response

    @pytest.mark.asyncio
    async def test_make_request_client_not_initialized(self, api_client, monkeypatch):
        """Test error when client not initialized."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        with pytest.raises(APIClientError, match="Client not initialized"):
            await api_client._make_request("POST", "https://api.example.com")

    @pytest.mark.asyncio
    async def test_make_request_successful(self, api_client, monkeypatch):
        """Test successful HTTP request."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response
                response = await api_client._make_request(
                    "POST", "https://api.example.com", {"data": "test"}
                )

        assert response == {"result": "success"}
        assert api_client.request_count == 1

    @pytest.mark.asyncio
    async def test_make_request_401_unauthorized(self, api_client, monkeypatch):
        """Test handling of 401 unauthorized response."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        mock_response = Mock()
        mock_response.status_code = 401

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response

                with pytest.raises(AuthenticationError):
                    await api_client._make_request("POST", "https://api.example.com")

    @pytest.mark.asyncio
    async def test_make_request_429_rate_limit(self, api_client, monkeypatch):
        """Test handling of 429 rate limit response."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        mock_response = Mock()
        mock_response.status_code = 429

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response

                with pytest.raises(RateLimitError):
                    await api_client._make_request("POST", "https://api.example.com")

        assert api_client.retry_count >= 1

    @pytest.mark.asyncio
    async def test_make_request_500_server_error(self, api_client, monkeypatch):
        """Test handling of 500 server error response."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        mock_response = Mock()
        mock_response.status_code = 500

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response

                with pytest.raises(ServiceUnavailableError):
                    await api_client._make_request("POST", "https://api.example.com")

    @pytest.mark.asyncio
    async def test_make_request_invalid_json(self, api_client, monkeypatch):
        """Test handling of invalid JSON response."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response

                with pytest.raises(APIClientError, match="Invalid JSON response"):
                    await api_client._make_request("POST", "https://api.example.com")

    @pytest.mark.asyncio
    async def test_make_request_with_retries(self, api_client, monkeypatch):
        """Test request with retry logic."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        # First two calls fail with 500, third succeeds
        fail_response = Mock()
        fail_response.status_code = 500

        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"result": "success"}

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.HTTPStatusError(
                    "Server Error",
                    request=Mock(),
                    response=fail_response,
                )
            return success_response

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.side_effect = side_effect

                # Reduce delays for faster testing
                api_client.base_delay = 0.01
                api_client.max_delay = 0.1

                response = await api_client._make_request(
                    "POST", "https://api.example.com"
                )

        assert response == {"result": "success"}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_make_request_headers_merged(self, api_client, monkeypatch):
        """Test that headers are properly merged."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        captured_headers = {}

        async def capture_request(method, url, json, headers):
            nonlocal captured_headers
            captured_headers = headers
            return mock_response

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.side_effect = capture_request
                await api_client._make_request(
                    "POST",
                    "https://api.example.com",
                    {"data": "test"},
                    headers={"X-Custom-Header": "custom-value"},
                )

        assert captured_headers["User-Agent"] == "BookmarkProcessor/1.0"
        assert captured_headers["X-Custom-Header"] == "custom-value"


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for client statistics."""

    def test_get_statistics_initial(self, api_client):
        """Test initial statistics values."""
        stats = api_client.get_statistics()

        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["retry_count"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_statistics_after_requests(self, api_client):
        """Test statistics after simulated requests."""
        api_client.request_count = 10
        api_client.error_count = 2
        api_client.retry_count = 3

        stats = api_client.get_statistics()

        assert stats["request_count"] == 10
        assert stats["error_count"] == 2
        assert stats["retry_count"] == 3
        assert stats["success_rate"] == 80.0  # (10-2)/10 * 100

    def test_get_statistics_success_rate_no_requests(self, api_client):
        """Test success rate calculation with no requests."""
        api_client.request_count = 0
        stats = api_client.get_statistics()
        assert stats["success_rate"] == 0.0


# ============================================================================
# Abstract Method Tests
# ============================================================================


class TestAbstractMethods:
    """Tests for abstract method implementations."""

    @pytest.mark.asyncio
    async def test_generate_description(self, api_client, sample_bookmark):
        """Test generate_description implementation."""
        description, metadata = await api_client.generate_description(sample_bookmark)
        assert description == "Test description"
        assert metadata["provider"] == "test"
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch(self, api_client, sample_bookmark):
        """Test generate_descriptions_batch implementation."""
        bookmarks = [sample_bookmark, sample_bookmark, sample_bookmark]
        results = await api_client.generate_descriptions_batch(bookmarks)

        assert len(results) == 3
        for i, (desc, meta) in enumerate(results):
            assert f"Test description {i}" in desc
            assert meta["provider"] == "test"

    def test_get_cost_per_request(self, api_client):
        """Test get_cost_per_request implementation."""
        cost = api_client.get_cost_per_request()
        assert cost == 0.001

    def test_get_rate_limit_info(self, api_client):
        """Test get_rate_limit_info implementation."""
        info = api_client.get_rate_limit_info()
        assert info["requests_per_minute"] == 60
        assert info["burst_size"] == 10


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_multiple_context_manager_entries(self, api_client):
        """Test multiple entries into context manager."""
        async with api_client:
            assert api_client._client is not None
            first_client = api_client._client

        # After exit, client should be None
        assert api_client._client is None

        # Re-entering should create new client
        async with api_client:
            assert api_client._client is not None
            assert api_client._client is not first_client

    @pytest.mark.asyncio
    async def test_cleanup_with_no_client(self, api_client):
        """Test cleanup when client is None."""
        assert api_client._client is None
        # Should not raise any errors
        await api_client._cleanup_client()
        assert api_client._client is None

    @pytest.mark.asyncio
    async def test_retry_delay_with_zero_base_delay(self):
        """Test retry delay calculation with zero base delay."""
        client = ConcreteAPIClient(api_key="test", base_delay=0.0)
        delay = client._calculate_retry_delay(0)
        assert delay >= 0.1  # Should still have minimum delay

    @pytest.mark.asyncio
    async def test_make_request_exhausts_retries(self, api_client, monkeypatch):
        """Test that request exhausts all retries before failing."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        fail_response = Mock()
        fail_response.status_code = 500

        api_client.max_retries = 2
        api_client.base_delay = 0.01
        api_client.max_delay = 0.1

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.side_effect = httpx.HTTPStatusError(
                    "Server Error",
                    request=Mock(),
                    response=fail_response,
                )

                with pytest.raises(ServiceUnavailableError):
                    await api_client._make_request("POST", "https://api.example.com")

        # Should have tried max_retries + 1 times
        assert mock_request.call_count == api_client.max_retries + 1

    @pytest.mark.asyncio
    async def test_re_raise_custom_exceptions(self, api_client, monkeypatch):
        """Test that custom exceptions are re-raised properly."""
        monkeypatch.setenv("BOOKMARK_PROCESSOR_TEST_MODE", "false")

        api_client.max_retries = 0

        async with api_client:
            with patch.object(
                api_client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.side_effect = RateLimitError("Rate limit exceeded")

                with pytest.raises(RateLimitError):
                    await api_client._make_request("POST", "https://api.example.com")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
