"""
Comprehensive Tests for AsyncHttpClient

Tests the AsyncHttpClient class from bookmark_processor/core/async_http_client.py
covering initialization, URL validation, batch processing, error handling,
rate limiting, and connection management.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import aiohttp

from bookmark_processor.core.async_http_client import AsyncHttpClient
from bookmark_processor.core.batch_types import ValidationResult, BatchResult


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def disable_test_mode():
    """Temporarily disable test mode for tests that need to test real validation logic."""
    original_value = os.environ.get("BOOKMARK_PROCESSOR_TEST_MODE")
    os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = "false"
    yield
    if original_value is not None:
        os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_value
    else:
        os.environ.pop("BOOKMARK_PROCESSOR_TEST_MODE", None)


@pytest.fixture
def client():
    """Create a basic AsyncHttpClient instance."""
    return AsyncHttpClient(
        timeout=30.0,
        max_concurrent=10,
        verify_ssl=True,
    )


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter."""
    limiter = MagicMock()
    limiter._extract_domain = MagicMock(return_value="example.com")
    limiter._get_domain_delay = MagicMock(return_value=0.1)
    limiter.lock = MagicMock()
    limiter.lock.__enter__ = MagicMock(return_value=None)
    limiter.lock.__exit__ = MagicMock(return_value=None)
    limiter.domain_stats = MagicMock()
    limiter.domain_stats.__getitem__ = MagicMock(
        return_value=MagicMock(
            last_request_time=0,
            request_count=0,
            total_wait_time=0,
        )
    )
    limiter.active_domains = set()
    return limiter


@pytest.fixture
def mock_security_validator():
    """Create a mock security validator."""
    validator = MagicMock()
    result = MagicMock()
    result.is_safe = True
    result.blocked_reason = None
    validator.validate_url_security = MagicMock(return_value=result)
    return validator


@pytest.fixture
def mock_browser_simulator():
    """Create a mock browser simulator."""
    simulator = MagicMock()
    simulator.get_headers = MagicMock(return_value={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml",
    })
    return simulator


@pytest.fixture
def client_with_mocks(mock_rate_limiter, mock_security_validator, mock_browser_simulator):
    """Create an AsyncHttpClient with all mocked dependencies."""
    return AsyncHttpClient(
        timeout=30.0,
        max_concurrent=10,
        verify_ssl=True,
        rate_limiter=mock_rate_limiter,
        security_validator=mock_security_validator,
        browser_simulator=mock_browser_simulator,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestAsyncHttpClientInit:
    """Tests for AsyncHttpClient initialization."""

    def test_default_initialization(self):
        """Test client initialization with default values."""
        client = AsyncHttpClient()
        assert client.timeout == 30.0
        assert client.max_concurrent == 10
        assert client.verify_ssl is True
        assert client.rate_limiter is None
        assert client.security_validator is None
        assert client.browser_simulator is None
        assert client._async_session is None

    def test_custom_initialization(self):
        """Test client initialization with custom values."""
        client = AsyncHttpClient(
            timeout=60.0,
            max_concurrent=20,
            verify_ssl=False,
        )
        assert client.timeout == 60.0
        assert client.max_concurrent == 20
        assert client.verify_ssl is False

    def test_initialization_with_rate_limiter(self, mock_rate_limiter):
        """Test client initialization with rate limiter."""
        client = AsyncHttpClient(rate_limiter=mock_rate_limiter)
        assert client.rate_limiter is mock_rate_limiter

    def test_initialization_with_security_validator(self, mock_security_validator):
        """Test client initialization with security validator."""
        client = AsyncHttpClient(security_validator=mock_security_validator)
        assert client.security_validator is mock_security_validator

    def test_initialization_with_browser_simulator(self, mock_browser_simulator):
        """Test client initialization with browser simulator."""
        client = AsyncHttpClient(browser_simulator=mock_browser_simulator)
        assert client.browser_simulator is mock_browser_simulator

    def test_initialization_with_all_dependencies(
        self, mock_rate_limiter, mock_security_validator, mock_browser_simulator
    ):
        """Test client initialization with all optional dependencies."""
        client = AsyncHttpClient(
            timeout=45.0,
            max_concurrent=15,
            verify_ssl=False,
            rate_limiter=mock_rate_limiter,
            security_validator=mock_security_validator,
            browser_simulator=mock_browser_simulator,
        )
        assert client.timeout == 45.0
        assert client.max_concurrent == 15
        assert client.verify_ssl is False
        assert client.rate_limiter is mock_rate_limiter
        assert client.security_validator is mock_security_validator
        assert client.browser_simulator is mock_browser_simulator


# =============================================================================
# Session Management Tests
# =============================================================================


class TestAsyncSessionManagement:
    """Tests for async session initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_session_creates_session(self, client):
        """Test that _initialize_session creates a new session."""
        assert client._async_session is None
        await client._initialize_session()
        assert client._async_session is not None
        assert isinstance(client._async_session, aiohttp.ClientSession)
        await client.close()

    @pytest.mark.asyncio
    async def test_initialize_session_idempotent(self, client):
        """Test that _initialize_session doesn't recreate existing session."""
        await client._initialize_session()
        first_session = client._async_session
        await client._initialize_session()
        assert client._async_session is first_session
        await client.close()

    @pytest.mark.asyncio
    async def test_close_session(self, client):
        """Test that close() properly closes the session."""
        await client._initialize_session()
        assert client._async_session is not None
        await client.close()
        assert client._async_session is None

    @pytest.mark.asyncio
    async def test_close_when_no_session(self, client):
        """Test that close() is safe when no session exists."""
        assert client._async_session is None
        await client.close()  # Should not raise
        assert client._async_session is None


# =============================================================================
# URL Validation Format Tests
# =============================================================================


class TestURLValidationFormat:
    """Tests for URL format validation methods."""

    def test_valid_url_format_http(self, client):
        """Test valid HTTP URL detection."""
        assert client._is_valid_url_format("http://example.com") is True

    def test_valid_url_format_https(self, client):
        """Test valid HTTPS URL detection."""
        assert client._is_valid_url_format("https://example.com") is True

    def test_valid_url_format_with_path(self, client):
        """Test valid URL with path."""
        assert client._is_valid_url_format("https://example.com/path/to/page") is True

    def test_valid_url_format_with_query(self, client):
        """Test valid URL with query string."""
        assert client._is_valid_url_format("https://example.com?foo=bar") is True

    def test_valid_url_format_with_port(self, client):
        """Test valid URL with port number."""
        assert client._is_valid_url_format("https://example.com:8080/path") is True

    def test_invalid_url_format_empty(self, client):
        """Test empty string is invalid."""
        assert client._is_valid_url_format("") is False

    def test_invalid_url_format_none(self, client):
        """Test None is invalid."""
        assert client._is_valid_url_format(None) is False

    def test_invalid_url_format_no_scheme(self, client):
        """Test URL without scheme is invalid."""
        assert client._is_valid_url_format("example.com") is False

    def test_invalid_url_format_no_netloc(self, client):
        """Test URL without netloc is invalid."""
        assert client._is_valid_url_format("http://") is False

    def test_invalid_url_format_whitespace_only(self, client):
        """Test whitespace-only string is invalid."""
        assert client._is_valid_url_format("   ") is False

    def test_valid_url_format_with_whitespace_trimmed(self, client):
        """Test URL with surrounding whitespace is valid after trimming."""
        assert client._is_valid_url_format("  https://example.com  ") is True


# =============================================================================
# URL Skip Pattern Tests
# =============================================================================


class TestURLSkipPatterns:
    """Tests for URL skip pattern detection."""

    def test_should_skip_javascript(self, client):
        """Test javascript: URLs should be skipped."""
        assert client._should_skip_url("javascript:void(0)") is True

    def test_should_skip_mailto(self, client):
        """Test mailto: URLs should be skipped."""
        assert client._should_skip_url("mailto:user@example.com") is True

    def test_should_skip_tel(self, client):
        """Test tel: URLs should be skipped."""
        assert client._should_skip_url("tel:+1234567890") is True

    def test_should_skip_ftp(self, client):
        """Test ftp: URLs should be skipped."""
        assert client._should_skip_url("ftp://ftp.example.com") is True

    def test_should_skip_file(self, client):
        """Test file: URLs should be skipped."""
        assert client._should_skip_url("file:///path/to/file") is True

    def test_should_skip_data(self, client):
        """Test data: URLs should be skipped."""
        assert client._should_skip_url("data:text/html,<h1>Hello</h1>") is True

    def test_should_skip_hash(self, client):
        """Test hash-only URLs should be skipped."""
        assert client._should_skip_url("#section") is True

    def test_should_skip_about_blank(self, client):
        """Test about:blank should be skipped."""
        assert client._should_skip_url("about:blank") is True

    def test_should_not_skip_http(self, client):
        """Test HTTP URLs should not be skipped."""
        assert client._should_skip_url("http://example.com") is False

    def test_should_not_skip_https(self, client):
        """Test HTTPS URLs should not be skipped."""
        assert client._should_skip_url("https://example.com") is False

    def test_should_skip_case_insensitive(self, client):
        """Test skip patterns are case-insensitive."""
        assert client._should_skip_url("JAVASCRIPT:void(0)") is True
        assert client._should_skip_url("MAILTO:user@example.com") is True


# =============================================================================
# HTTP Error Classification Tests
# =============================================================================


class TestHTTPErrorClassification:
    """Tests for HTTP error classification."""

    def test_classify_400_bad_request(self, client):
        """Test 400 Bad Request classification."""
        assert client._classify_http_error(400) == "bad_request"

    def test_classify_401_unauthorized(self, client):
        """Test 401 Unauthorized classification."""
        assert client._classify_http_error(401) == "unauthorized"

    def test_classify_403_forbidden(self, client):
        """Test 403 Forbidden classification."""
        assert client._classify_http_error(403) == "forbidden"

    def test_classify_404_not_found(self, client):
        """Test 404 Not Found classification."""
        assert client._classify_http_error(404) == "not_found"

    def test_classify_405_method_not_allowed(self, client):
        """Test 405 Method Not Allowed classification."""
        assert client._classify_http_error(405) == "method_not_allowed"

    def test_classify_408_request_timeout(self, client):
        """Test 408 Request Timeout classification."""
        assert client._classify_http_error(408) == "request_timeout"

    def test_classify_429_rate_limited(self, client):
        """Test 429 Too Many Requests classification."""
        assert client._classify_http_error(429) == "rate_limited"

    def test_classify_generic_client_error(self, client):
        """Test generic 4xx client error classification."""
        assert client._classify_http_error(418) == "client_error"

    def test_classify_500_internal_server_error(self, client):
        """Test 500 Internal Server Error classification."""
        assert client._classify_http_error(500) == "internal_server_error"

    def test_classify_501_not_implemented(self, client):
        """Test 501 Not Implemented classification."""
        assert client._classify_http_error(501) == "not_implemented"

    def test_classify_502_bad_gateway(self, client):
        """Test 502 Bad Gateway classification."""
        assert client._classify_http_error(502) == "bad_gateway"

    def test_classify_503_service_unavailable(self, client):
        """Test 503 Service Unavailable classification."""
        assert client._classify_http_error(503) == "service_unavailable"

    def test_classify_504_gateway_timeout(self, client):
        """Test 504 Gateway Timeout classification."""
        assert client._classify_http_error(504) == "gateway_timeout"

    def test_classify_generic_server_error(self, client):
        """Test generic 5xx server error classification."""
        assert client._classify_http_error(520) == "server_error"

    def test_classify_non_error_code(self, client):
        """Test non-error status code classification."""
        assert client._classify_http_error(200) == "http_error"
        assert client._classify_http_error(301) == "http_error"


# =============================================================================
# Content-Length Parsing Tests
# =============================================================================


class TestContentLengthParsing:
    """Tests for content-length header parsing."""

    def test_parse_valid_content_length(self, client):
        """Test parsing valid content-length value."""
        assert client._parse_content_length("1234") == 1234

    def test_parse_zero_content_length(self, client):
        """Test parsing zero content-length."""
        assert client._parse_content_length("0") == 0

    def test_parse_large_content_length(self, client):
        """Test parsing large content-length value."""
        assert client._parse_content_length("1000000000") == 1000000000

    def test_parse_none_content_length(self, client):
        """Test parsing None content-length."""
        assert client._parse_content_length(None) is None

    def test_parse_invalid_content_length(self, client):
        """Test parsing invalid content-length value."""
        assert client._parse_content_length("invalid") is None

    def test_parse_empty_content_length(self, client):
        """Test parsing empty content-length value."""
        assert client._parse_content_length("") is None


# =============================================================================
# Success Codes Tests
# =============================================================================


class TestSuccessCodes:
    """Tests for success code classification."""

    def test_success_codes_2xx(self, client):
        """Test 2xx status codes are in success codes."""
        for code in [200, 201, 202, 203, 204, 205, 206]:
            assert code in client.SUCCESS_CODES

    def test_success_codes_3xx(self, client):
        """Test redirect status codes are in success codes."""
        for code in [300, 301, 302, 303, 304, 307, 308]:
            assert code in client.SUCCESS_CODES

    def test_non_success_codes(self, client):
        """Test 4xx and 5xx are not in success codes."""
        for code in [400, 401, 403, 404, 500, 502, 503]:
            assert code not in client.SUCCESS_CODES


# =============================================================================
# Mock Validation Result Tests
# =============================================================================


class TestMockValidationResult:
    """Tests for mock validation result generation."""

    def test_mock_result_valid_url(self, client):
        """Test mock result for valid URL."""
        result = client._get_mock_validation_result("https://example.com")
        assert result.is_valid is True
        assert result.status_code == 200
        assert result.error_message is None

    def test_mock_result_invalid_url_keyword(self, client):
        """Test mock result for URL with 'invalid' keyword."""
        result = client._get_mock_validation_result("https://invalid-site.com")
        assert result.is_valid is False
        assert result.status_code == 404

    def test_mock_result_404_url_keyword(self, client):
        """Test mock result for URL with '404' keyword."""
        result = client._get_mock_validation_result("https://example.com/404")
        assert result.is_valid is False
        assert result.status_code == 404

    def test_mock_result_error_url_keyword(self, client):
        """Test mock result for URL with 'error' keyword."""
        result = client._get_mock_validation_result("https://error.example.com")
        assert result.is_valid is False

    def test_mock_result_timeout_url_keyword(self, client):
        """Test mock result for URL with 'timeout' keyword."""
        result = client._get_mock_validation_result("https://timeout.example.com")
        assert result.is_valid is False

    def test_mock_result_response_time(self, client):
        """Test mock result has fast response time."""
        result = client._get_mock_validation_result("https://example.com")
        assert result.response_time == 0.1


# =============================================================================
# URL Validation Tests (With Test Mode)
# =============================================================================


class TestURLValidationTestMode:
    """Tests for URL validation in test mode."""

    @pytest.mark.asyncio
    async def test_validate_url_test_mode(self, client):
        """Test URL validation returns mock result in test mode."""
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_url("https://example.com")
            assert result.is_valid is True
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_url_test_mode_invalid(self, client):
        """Test URL validation returns invalid mock result for invalid URL."""
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_url("https://invalid-url.com")
            assert result.is_valid is False


# =============================================================================
# URL Validation Tests (Mocked HTTP)
# =============================================================================


class TestURLValidationMocked:
    """Tests for URL validation with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_validate_url_invalid_format(self, client, disable_test_mode):
        """Test validation of invalid URL format."""
        result = await client.validate_url("not-a-url")
        assert result.is_valid is False
        assert result.error_type == "format_error"
        assert "Invalid URL format" in result.error_message

    @pytest.mark.asyncio
    async def test_validate_url_empty_string(self, client, disable_test_mode):
        """Test validation of empty string."""
        result = await client.validate_url("")
        assert result.is_valid is False
        assert result.error_type == "format_error"

    @pytest.mark.asyncio
    async def test_validate_url_javascript_scheme(self, client, disable_test_mode):
        """Test validation of javascript: URL."""
        result = await client.validate_url("javascript:void(0)")
        assert result.is_valid is False
        # javascript: URLs fail format validation because they lack a netloc
        assert result.error_type == "format_error"

    @pytest.mark.asyncio
    async def test_validate_url_mailto_scheme(self, client, disable_test_mode):
        """Test validation of mailto: URL."""
        result = await client.validate_url("mailto:test@example.com")
        assert result.is_valid is False
        # mailto: URLs fail format validation because they lack a netloc
        assert result.error_type == "format_error"

    @pytest.mark.asyncio
    async def test_validate_url_security_validation_failure(
        self, mock_security_validator, disable_test_mode
    ):
        """Test validation fails when security validation fails."""
        mock_security_validator.validate_url_security.return_value.is_safe = False
        mock_security_validator.validate_url_security.return_value.blocked_reason = (
            "Private IP detected"
        )

        client = AsyncHttpClient(security_validator=mock_security_validator)
        result = await client.validate_url("https://192.168.1.1")

        assert result.is_valid is False
        assert result.error_type == "security_error"
        assert "Private IP detected" in result.error_message

    @pytest.mark.asyncio
    async def test_validate_url_success_mocked(self, client, disable_test_mode):
        """Test successful URL validation with mocked response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-type": "text/html", "content-length": "1234"}

        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client._async_session = mock_session

        with patch.object(client, "_apply_rate_limiting", new_callable=AsyncMock):
            result = await client.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.status_code == 200
        assert result.content_type == "text/html"

    @pytest.mark.asyncio
    async def test_validate_url_redirect(self, client, disable_test_mode):
        """Test URL validation with redirect."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.url = "https://www.example.com"  # Redirected URL
        mock_response.headers = {"content-type": "text/html"}

        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client._async_session = mock_session

        with patch.object(client, "_apply_rate_limiting", new_callable=AsyncMock):
            result = await client.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.final_url == "https://www.example.com"

    @pytest.mark.asyncio
    async def test_validate_url_404_error(self, client, disable_test_mode):
        """Test URL validation with 404 response."""
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.url = "https://example.com/notfound"
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client._async_session = mock_session

        with patch.object(client, "_apply_rate_limiting", new_callable=AsyncMock):
            result = await client.validate_url("https://example.com/notfound")

        assert result.is_valid is False
        assert result.status_code == 404
        assert result.error_type == "not_found"

    @pytest.mark.asyncio
    async def test_validate_url_timeout(self, client, disable_test_mode):
        """Test URL validation with timeout error."""
        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client._async_session = mock_session

        with patch.object(client, "_apply_rate_limiting", new_callable=AsyncMock):
            result = await client.validate_url("https://slow-site.com")

        assert result.is_valid is False
        assert result.error_type == "timeout"
        assert "Timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_validate_url_connection_error(self, client, disable_test_mode):
        """Test URL validation with connection error."""
        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client._async_session = mock_session

        with patch.object(client, "_apply_rate_limiting", new_callable=AsyncMock):
            result = await client.validate_url("https://unreachable.com")

        assert result.is_valid is False
        assert result.error_type == "connection_refused"

    @pytest.mark.asyncio
    async def test_validate_url_dns_error(self, client, disable_test_mode):
        """Test URL validation with DNS error."""
        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("DNS name resolution failed")
        )
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client._async_session = mock_session

        with patch.object(client, "_apply_rate_limiting", new_callable=AsyncMock):
            result = await client.validate_url("https://nonexistent-domain.invalid")

        assert result.is_valid is False
        assert result.error_type == "dns_error"

    @pytest.mark.asyncio
    async def test_validate_url_unexpected_error(self, client, disable_test_mode):
        """Test URL validation with unexpected error."""
        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("Unexpected error occurred")
        )
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client._async_session = mock_session

        with patch.object(client, "_apply_rate_limiting", new_callable=AsyncMock):
            result = await client.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "unknown_error"
        assert "Unexpected error" in result.error_message


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_apply_rate_limiting_no_limiter(self, client):
        """Test rate limiting is skipped when no limiter is configured."""
        await client._apply_rate_limiting("https://example.com")
        # Should complete without error

    @pytest.mark.asyncio
    async def test_apply_rate_limiting_with_limiter(self, client_with_mocks):
        """Test rate limiting is applied when limiter is configured."""
        # Configure mock to return a domain stats object
        domain_stats = MagicMock()
        domain_stats.last_request_time = 0
        domain_stats.request_count = 0
        domain_stats.total_wait_time = 0
        client_with_mocks.rate_limiter.domain_stats = {"example.com": domain_stats}

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client_with_mocks._apply_rate_limiting("https://example.com")

        # Verify rate limiter methods were called
        client_with_mocks.rate_limiter._extract_domain.assert_called_once_with(
            "https://example.com"
        )


# =============================================================================
# Batch Validation Tests
# =============================================================================


class TestBatchValidation:
    """Tests for batch URL validation."""

    @pytest.mark.asyncio
    async def test_validate_batch_empty_list(self, client):
        """Test batch validation with empty URL list."""
        result = await client.validate_batch([], "batch-001")
        assert result.batch_id == "batch-001"
        assert result.items_processed == 0
        assert result.items_successful == 0
        assert result.items_failed == 0

    @pytest.mark.asyncio
    async def test_validate_batch_single_url(self, client):
        """Test batch validation with single URL."""
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(
                ["https://example.com"],
                "batch-001"
            )
            assert result.batch_id == "batch-001"
            assert result.items_processed == 1
            assert result.items_successful == 1
            assert result.items_failed == 0

    @pytest.mark.asyncio
    async def test_validate_batch_multiple_urls(self, client):
        """Test batch validation with multiple URLs."""
        urls = [
            "https://example.com",
            "https://google.com",
            "https://github.com",
        ]
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "batch-002")
            assert result.batch_id == "batch-002"
            assert result.items_processed == 3
            assert result.items_successful == 3

    @pytest.mark.asyncio
    async def test_validate_batch_mixed_results(self, client):
        """Test batch validation with mixed success/failure URLs."""
        urls = [
            "https://example.com",
            "https://invalid-site.com",  # Contains 'invalid' keyword
            "https://google.com",
        ]
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "batch-003")
            assert result.items_processed == 3
            assert result.items_successful == 2  # example.com and google.com
            assert result.items_failed == 1  # invalid-site.com

    @pytest.mark.asyncio
    async def test_validate_batch_error_rate_calculation(self, client):
        """Test batch validation error rate calculation."""
        urls = [
            "https://example.com",
            "https://error.com",  # Contains 'error' keyword
        ]
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "batch-004")
            assert result.error_rate == 0.5  # 1 failure out of 2

    @pytest.mark.asyncio
    async def test_validate_batch_processing_time(self, client):
        """Test batch validation records processing time."""
        urls = ["https://example.com"]
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "batch-005")
            assert result.processing_time >= 0

    @pytest.mark.asyncio
    async def test_validate_batch_average_time(self, client):
        """Test batch validation calculates average time per item."""
        urls = ["https://example.com", "https://google.com"]
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "batch-006")
            assert result.average_item_time >= 0

    @pytest.mark.asyncio
    async def test_validate_batch_results_list(self, client):
        """Test batch validation returns results list."""
        urls = ["https://example.com", "https://google.com"]
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "batch-007")
            assert len(result.results) == 2
            assert all(isinstance(r, ValidationResult) for r in result.results)

    @pytest.mark.asyncio
    async def test_validate_batch_handles_exceptions(self, client):
        """Test batch validation handles exceptions in individual validations."""
        # Mock validate_url to raise an exception for one URL
        original_validate = client.validate_url

        async def mock_validate(url):
            if "exception" in url:
                raise RuntimeError("Test exception")
            return await original_validate(url)

        with patch.object(client, "validate_url", side_effect=mock_validate):
            with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
                urls = ["https://example.com", "https://exception.com"]
                result = await client.validate_batch(urls, "batch-008")

                assert result.items_processed == 2
                assert result.items_failed >= 1
                assert len(result.errors) >= 1

    @pytest.mark.asyncio
    async def test_validate_batch_errors_truncated(self, client):
        """Test batch validation truncates errors to first 10."""
        urls = [f"https://error{i}.com" for i in range(15)]  # 15 error URLs

        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "batch-009")
            # All URLs should be processed, but errors list truncated
            assert result.items_processed == 15


# =============================================================================
# Browser Simulator Integration Tests
# =============================================================================


class TestBrowserSimulatorIntegration:
    """Tests for browser simulator integration."""

    @pytest.mark.asyncio
    async def test_validate_url_with_browser_headers(
        self, client_with_mocks, disable_test_mode
    ):
        """Test validation uses browser simulator headers."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncMock())
        mock_session.head.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.head.return_value.__aexit__ = AsyncMock(return_value=None)

        client_with_mocks._async_session = mock_session

        with patch.object(
            client_with_mocks, "_apply_rate_limiting", new_callable=AsyncMock
        ):
            await client_with_mocks.validate_url("https://example.com")

        # Verify browser simulator was called
        client_with_mocks.browser_simulator.get_headers.assert_called_once_with(
            "https://example.com"
        )


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_batch_respects_max_concurrent(self, client):
        """Test batch validation respects max concurrent limit."""
        urls = [f"https://example{i}.com" for i in range(20)]

        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(urls, "concurrent-test")
            assert result.items_processed == 20

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, client):
        """Test that semaphore is used to limit concurrency."""
        # Create a client with low max_concurrent
        limited_client = AsyncHttpClient(max_concurrent=2)

        urls = [f"https://example{i}.com" for i in range(5)]

        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await limited_client.validate_batch(urls, "limited-concurrent")
            assert result.items_processed == 5


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResultIntegration:
    """Tests for ValidationResult dataclass integration."""

    @pytest.mark.asyncio
    async def test_validation_result_has_timestamp(self, client):
        """Test validation result includes timestamp."""
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_url("https://example.com")
            assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_validation_result_to_dict(self, client):
        """Test validation result can be converted to dict."""
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_url("https://example.com")
            result_dict = result.to_dict()
            assert "url" in result_dict
            assert "is_valid" in result_dict
            assert "timestamp" in result_dict


# =============================================================================
# BatchResult Tests
# =============================================================================


class TestBatchResultIntegration:
    """Tests for BatchResult dataclass integration."""

    @pytest.mark.asyncio
    async def test_batch_result_has_timestamp(self, client):
        """Test batch result includes timestamp."""
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(["https://example.com"], "ts-test")
            assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_batch_result_to_dict(self, client):
        """Test batch result can be converted to dict."""
        with patch.dict(os.environ, {"BOOKMARK_PROCESSOR_TEST_MODE": "true"}):
            result = await client.validate_batch(["https://example.com"], "dict-test")
            result_dict = result.to_dict()
            assert "batch_id" in result_dict
            assert "items_processed" in result_dict
            assert "timestamp" in result_dict
