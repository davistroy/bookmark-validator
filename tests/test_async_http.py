"""
Tests for Async HTTP Utilities

Tests the modern async HTTP client using httpx with rate limiting
and structured concurrency.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bookmark_processor.utils.async_http import (
    AsyncHTTPClient,
    AsyncRateLimiter,
    HTTPResponse,
    RateLimitConfig,
    validate_urls_batch,
)


class TestHTTPResponse:
    """Tests for HTTPResponse dataclass."""

    def test_success_response(self):
        """Test creating a successful response."""
        response = HTTPResponse(
            url="https://example.com",
            status_code=200,
            headers={"content-type": "text/html"},
            is_success=True,
        )
        assert response.is_success
        assert response.status_code == 200
        assert response.error is None

    def test_error_response(self):
        """Test creating an error response."""
        response = HTTPResponse(
            url="https://example.com",
            status_code=0,
            headers={},
            is_success=False,
            error="Connection refused",
        )
        assert not response.is_success
        assert response.error == "Connection refused"


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_second == 10.0
        assert config.burst_size == 20
        assert "github.com" in config.domain_limits

    def test_custom_config(self):
        """Test custom configuration."""
        config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
            domain_limits={"custom.com": 1.0},
        )
        assert config.requests_per_second == 5.0
        assert config.burst_size == 10


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test acquiring rate limit permission."""
        limiter = AsyncRateLimiter(RateLimitConfig(per_domain_limit=0.01))
        # First acquire should be immediate
        await limiter.acquire("https://example.com/page1")
        # Verify domain tracking
        assert limiter._get_domain("https://example.com/page1") == "example.com"

    def test_domain_extraction(self):
        """Test domain extraction from URLs."""
        limiter = AsyncRateLimiter()
        assert limiter._get_domain("https://www.example.com/path") == "example.com"
        assert limiter._get_domain("https://api.github.com/repos") == "api.github.com"
        assert limiter._get_domain("invalid-url") == "unknown"

    def test_domain_limit_lookup(self):
        """Test domain-specific limit lookup."""
        config = RateLimitConfig(
            domain_limits={"github.com": 1.5, "google.com": 2.0}
        )
        limiter = AsyncRateLimiter(config)
        assert limiter._get_domain_limit("github.com") == 1.5
        assert limiter._get_domain_limit("api.github.com") == 1.5
        assert limiter._get_domain_limit("example.com") == config.per_domain_limit


class TestAsyncHTTPClient:
    """Tests for AsyncHTTPClient."""

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as context manager."""
        async with AsyncHTTPClient() as client:
            assert client._client is not None
        # Client should be closed after exiting context

    @pytest.mark.asyncio
    async def test_client_not_initialized_error(self):
        """Test error when client not initialized."""
        client = AsyncHTTPClient()
        with pytest.raises(RuntimeError):
            await client.get("https://example.com")

    @pytest.mark.asyncio
    async def test_get_request_mock(self):
        """Test GET request with mocked response."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_response.url = "https://example.com"
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            async with AsyncHTTPClient() as client:
                client._client = mock_client
                response = await client.get("https://example.com")

            assert response.is_success
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_urls_empty_list(self):
        """Test validating empty URL list."""
        async with AsyncHTTPClient() as client:
            results = await client.validate_urls([])
            assert results == []

    @pytest.mark.asyncio
    async def test_validate_urls_with_progress(self):
        """Test URL validation with progress callback."""
        progress_updates = []

        def progress_callback(completed, total):
            progress_updates.append((completed, total))

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.url = "https://example.com"
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            async with AsyncHTTPClient() as client:
                client._client = mock_client
                urls = ["https://example1.com", "https://example2.com"]
                results = await client.validate_urls(
                    urls,
                    progress_callback=progress_callback,
                )

            assert len(results) == 2
            # Progress should be tracked
            assert len(progress_updates) >= 1


class TestValidateUrlsBatch:
    """Tests for validate_urls_batch convenience function."""

    @pytest.mark.asyncio
    async def test_batch_validation_mock(self):
        """Test batch validation with mocked client."""
        with patch("bookmark_processor.utils.async_http.AsyncHTTPClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.validate_urls = AsyncMock(return_value=[
                HTTPResponse(
                    url="https://example.com",
                    status_code=200,
                    headers={},
                    is_success=True,
                )
            ])
            mock_class.return_value = mock_client

            results = await validate_urls_batch(["https://example.com"])
            assert len(results) == 1
            assert results[0].is_success


class TestClientHeaders:
    """Tests for client headers."""

    def test_default_headers_present(self):
        """Test that default headers are set."""
        assert "User-Agent" in AsyncHTTPClient.DEFAULT_HEADERS
        assert "Accept" in AsyncHTTPClient.DEFAULT_HEADERS
        assert "Accept-Language" in AsyncHTTPClient.DEFAULT_HEADERS

    def test_user_agent_looks_like_browser(self):
        """Test that User-Agent looks like a real browser."""
        ua = AsyncHTTPClient.DEFAULT_HEADERS["User-Agent"]
        assert "Mozilla" in ua
        assert "Chrome" in ua or "Firefox" in ua
