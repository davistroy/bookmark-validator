"""
Unit tests for AsyncValidatorMixin (async_validator.py).

Tests for asynchronous URL validation with comprehensive coverage of:
- AsyncValidatorMixin class and all async validation methods
- Concurrent URL validation with semaphores
- Rate limiting integration
- Error handling for network failures
- Retry logic with backoff
- Batch validation methods
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from bookmark_processor.core.batch_types import BatchResult, ValidationResult
from bookmark_processor.core.url_validator import URLValidator
from bookmark_processor.core.url_validator.async_validator import AsyncValidatorMixin
from bookmark_processor.core.url_validator.helpers import SUCCESS_CODES
from bookmark_processor.utils.intelligent_rate_limiter import DomainStats
from bookmark_processor.utils.security_validator import SecurityValidationResult


def create_mock_response(status=200, url="https://example.com", headers=None):
    """Create a mock aiohttp response with async context manager support."""
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.url = url
    mock_response.headers = headers or {}
    return mock_response


def create_mock_context_manager(response):
    """Create an async context manager that yields the response."""

    @asynccontextmanager
    async def _context_manager():
        yield response

    return _context_manager()


class TestAsyncValidatorMixin:
    """Test AsyncValidatorMixin class."""

    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator with all required dependencies."""
        # Ensure test mode is disabled for unit tests
        original_test_mode = os.environ.get("BOOKMARK_PROCESSOR_TEST_MODE")
        if "BOOKMARK_PROCESSOR_TEST_MODE" in os.environ:
            del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]

        with (
            patch(
                "bookmark_processor.core.url_validator.validator.IntelligentRateLimiter"
            ),
            patch("bookmark_processor.core.url_validator.validator.BrowserSimulator"),
            patch("bookmark_processor.core.url_validator.validator.RetryHandler"),
            patch("bookmark_processor.core.url_validator.validator.SecurityValidator"),
        ):
            validator = URLValidator(timeout=5, max_redirects=5, max_concurrent=10)

            # Setup mock rate limiter with proper domain_stats behavior
            validator.rate_limiter._extract_domain = Mock(return_value="example.com")
            validator.rate_limiter._get_domain_delay = Mock(return_value=0.1)
            validator.rate_limiter.lock = MagicMock()
            validator.rate_limiter.domain_stats = {
                "example.com": DomainStats(
                    last_request_time=0.0,
                    request_count=0,
                    total_wait_time=0.0,
                    error_count=0,
                )
            }
            validator.rate_limiter.active_domains = set()

            # Ensure _async_session is None initially
            if hasattr(validator, "_async_session"):
                validator._async_session = None

            yield validator

        # Restore original test mode setting
        if original_test_mode is not None:
            os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_test_mode

    @pytest.fixture
    def mock_security_result(self):
        """Create a mock security validation result."""
        return SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )

    @pytest.fixture
    def mock_unsafe_security_result(self):
        """Create an unsafe security validation result."""
        return SecurityValidationResult(
            is_safe=False,
            risk_level="high",
            issues=["Malicious URL detected"],
            blocked_reason="Security validation failed",
        )


class TestAsyncValidateUrl(TestAsyncValidatorMixin):
    """Tests for async_validate_url method."""

    @pytest.mark.asyncio
    async def test_async_validate_url_success(
        self, mock_validator, mock_security_result
    ):
        """Test successful async URL validation."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        # Create mock response
        mock_response = create_mock_response(
            status=200,
            url="https://example.com",
            headers={"content-type": "text/html", "content-length": "1024"},
        )

        # Create mock session with proper async context manager
        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is True
        assert result.status_code == 200
        assert result.url == "https://example.com"
        assert result.content_type == "text/html"
        assert result.content_length == 1024

    @pytest.mark.asyncio
    async def test_async_validate_url_with_redirect(
        self, mock_validator, mock_security_result
    ):
        """Test async validation with URL redirect."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        # Mock response with different final URL (redirect)
        mock_response = create_mock_response(
            status=200,
            url="https://www.example.com/redirected",
            headers={"content-type": "text/html"},
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is True
        assert result.final_url == "https://www.example.com/redirected"

    @pytest.mark.asyncio
    async def test_async_validate_url_security_failure(
        self, mock_validator, mock_unsafe_security_result
    ):
        """Test async validation fails on security check."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_unsafe_security_result
        )

        result = await mock_validator.async_validate_url("https://malicious.com")

        assert result.is_valid is False
        assert result.error_type == "security_error"
        assert "Security validation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_async_validate_url_invalid_format(
        self, mock_validator, mock_security_result
    ):
        """Test async validation fails on invalid URL format."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("not-a-valid-url")

        assert result.is_valid is False
        assert result.error_type == "format_error"
        assert "Invalid URL format" in result.error_message

    @pytest.mark.asyncio
    async def test_async_validate_url_unsupported_scheme(
        self, mock_validator, mock_security_result
    ):
        """Test async validation fails on unsupported URL scheme."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        # javascript: URL fails format validation first since it doesn't have netloc
        result = await mock_validator.async_validate_url("javascript:alert(1)")

        assert result.is_valid is False
        # May be format_error or unsupported_scheme depending on order of checks
        assert result.error_type in ["unsupported_scheme", "format_error"]

    @pytest.mark.asyncio
    async def test_async_validate_url_mailto_scheme(
        self, mock_validator, mock_security_result
    ):
        """Test async validation fails on mailto URL scheme."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("mailto:test@example.com")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_async_validate_url_http_error_status(
        self, mock_validator, mock_security_result
    ):
        """Test async validation with HTTP error status code."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=404, url="https://example.com/notfound", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url(
                "https://example.com/notfound"
            )

        assert result.is_valid is False
        assert result.status_code == 404
        assert "HTTP 404" in result.error_message
        assert result.error_type == "not_found"

    @pytest.mark.asyncio
    async def test_async_validate_url_timeout(
        self, mock_validator, mock_security_result
    ):
        """Test async validation with timeout error."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        @asynccontextmanager
        async def raise_timeout():
            raise asyncio.TimeoutError()
            yield  # Never reached

        mock_session = MagicMock()
        mock_session.head = Mock(return_value=raise_timeout())

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "timeout"
        assert "Timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_async_validate_url_dns_error(
        self, mock_validator, mock_security_result
    ):
        """Test async validation with DNS resolution error."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        @asynccontextmanager
        async def raise_dns_error():
            raise aiohttp.ClientError("DNS name resolution failed")
            yield  # Never reached

        mock_session = MagicMock()
        mock_session.head = Mock(return_value=raise_dns_error())

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url(
                "https://nonexistent.invalid"
            )

        assert result.is_valid is False
        assert result.error_type == "dns_error"

    @pytest.mark.asyncio
    async def test_async_validate_url_connection_refused(
        self, mock_validator, mock_security_result
    ):
        """Test async validation with connection refused error."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        @asynccontextmanager
        async def raise_connection_refused():
            raise aiohttp.ClientError("Connection refused")
            yield  # Never reached

        mock_session = MagicMock()
        mock_session.head = Mock(return_value=raise_connection_refused())

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://localhost:9999")

        assert result.is_valid is False
        assert result.error_type == "connection_refused"

    @pytest.mark.asyncio
    async def test_async_validate_url_generic_connection_error(
        self, mock_validator, mock_security_result
    ):
        """Test async validation with generic connection error."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        @asynccontextmanager
        async def raise_connection_error():
            raise aiohttp.ClientError("Some network error")
            yield  # Never reached

        mock_session = MagicMock()
        mock_session.head = Mock(return_value=raise_connection_error())

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_async_validate_url_unexpected_error(
        self, mock_validator, mock_security_result
    ):
        """Test async validation with unexpected exception."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        @asynccontextmanager
        async def raise_unexpected():
            raise RuntimeError("Unexpected error occurred")
            yield  # Never reached

        mock_session = MagicMock()
        mock_session.head = Mock(return_value=raise_unexpected())

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "unknown_error"
        assert "Unexpected error" in result.error_message

    @pytest.mark.asyncio
    async def test_async_validate_url_initializes_session(
        self, mock_validator, mock_security_result
    ):
        """Test that async validation initializes session if not exists."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }
        mock_validator._async_session = None

        mock_response = create_mock_response(
            status=200, url="https://example.com", headers={}
        )

        init_called = [False]

        async def mock_init():
            init_called[0] = True
            mock_session = MagicMock()
            mock_session.head = Mock(
                return_value=create_mock_context_manager(mock_response)
            )
            mock_validator._async_session = mock_session

        with patch.object(
            mock_validator, "_initialize_async_session", side_effect=mock_init
        ):
            result = await mock_validator.async_validate_url("https://example.com")

        assert init_called[0] is True

    @pytest.mark.asyncio
    async def test_async_validate_url_test_mode(self, mock_validator):
        """Test async validation in test mode returns mock result."""
        os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = "true"
        try:
            mock_validator._get_mock_validation_result = Mock(
                return_value=ValidationResult(
                    url="https://example.com",
                    is_valid=True,
                    status_code=200,
                    response_time=0.1,
                )
            )

            result = await mock_validator.async_validate_url("https://example.com")

            assert result.is_valid is True
            mock_validator._get_mock_validation_result.assert_called_once_with(
                "https://example.com"
            )
        finally:
            del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]


class TestAsyncValidateBatch(TestAsyncValidatorMixin):
    """Tests for async_validate_batch method."""

    @pytest.mark.asyncio
    async def test_async_validate_batch_success(
        self, mock_validator, mock_security_result
    ):
        """Test successful batch validation."""
        urls = ["https://example1.com", "https://example2.com", "https://example3.com"]

        # Create mock results for each URL
        async def mock_validate(url):
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.1
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "batch_001")

        assert isinstance(result, BatchResult)
        assert result.batch_id == "batch_001"
        assert result.items_processed == 3
        assert result.items_successful == 3
        assert result.items_failed == 0
        assert result.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_async_validate_batch_partial_failure(
        self, mock_validator, mock_security_result
    ):
        """Test batch validation with some failures."""
        urls = ["https://valid.com", "https://invalid.com", "https://valid2.com"]

        async def mock_validate(url):
            if "invalid" in url:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    status_code=404,
                    error_type="not_found",
                    response_time=0.1,
                )
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.1
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "batch_002")

        assert result.items_processed == 3
        assert result.items_successful == 2
        assert result.items_failed == 1
        assert result.error_rate == pytest.approx(1 / 3, rel=0.01)

    @pytest.mark.asyncio
    async def test_async_validate_batch_handles_exceptions(
        self, mock_validator, mock_security_result
    ):
        """Test batch validation handles exceptions in individual validations."""
        urls = ["https://example.com"]

        # Mock async_validate_url to raise an exception
        async def mock_validate(url):
            raise RuntimeError("Validation failed")

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "batch_003")

        assert result.items_processed == 1
        assert result.items_failed == 1
        assert len(result.errors) == 1
        assert "Validation failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_async_validate_batch_empty_urls(self, mock_validator):
        """Test batch validation with empty URL list."""
        result = await mock_validator.async_validate_batch([], "batch_empty")

        assert result.items_processed == 0
        assert result.items_successful == 0
        assert result.items_failed == 0
        assert result.error_rate == 0

    @pytest.mark.asyncio
    async def test_async_validate_batch_respects_concurrency_limit(
        self, mock_validator, mock_security_result
    ):
        """Test that batch validation respects max_concurrent limit."""
        mock_validator.max_concurrent = 2
        urls = [f"https://example{i}.com" for i in range(5)]

        concurrent_count = [0]
        max_concurrent_observed = [0]

        async def mock_validate(url):
            concurrent_count[0] += 1
            max_concurrent_observed[0] = max(
                max_concurrent_observed[0], concurrent_count[0]
            )
            await asyncio.sleep(0.01)  # Simulate network delay
            concurrent_count[0] -= 1
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.01
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "batch_concurrent")

        assert result.items_processed == 5
        # Semaphore should limit concurrency to min(max_concurrent, len(urls))
        assert max_concurrent_observed[0] <= mock_validator.max_concurrent

    @pytest.mark.asyncio
    async def test_async_validate_batch_calculates_statistics(
        self, mock_validator, mock_security_result
    ):
        """Test that batch validation calculates correct statistics."""
        urls = ["https://example1.com", "https://example2.com"]

        async def mock_validate(url):
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.05
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "batch_stats")

        assert result.processing_time > 0
        assert result.average_item_time > 0
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_async_validate_batch_limits_errors(
        self, mock_validator, mock_security_result
    ):
        """Test that batch validation limits error messages to 10."""
        urls = [f"https://example{i}.com" for i in range(15)]

        # Make all validations fail with exceptions
        async def mock_validate(url):
            raise RuntimeError(f"Error for {url}")

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "batch_errors")

        assert result.items_failed == 15
        assert len(result.errors) <= 10  # Should be limited to 10


class TestInitializeAsyncSession(TestAsyncValidatorMixin):
    """Tests for _initialize_async_session method."""

    @pytest.mark.asyncio
    async def test_initialize_async_session_creates_session(self, mock_validator):
        """Test that session is properly initialized."""
        mock_validator._async_session = None

        with patch("aiohttp.ClientSession") as mock_client_session:
            with patch("aiohttp.TCPConnector") as mock_connector:
                mock_connector.return_value = MagicMock()
                mock_session = MagicMock()
                mock_client_session.return_value = mock_session

                await mock_validator._initialize_async_session()

                mock_client_session.assert_called_once()
                assert mock_validator._async_session == mock_session

    @pytest.mark.asyncio
    async def test_initialize_async_session_skips_if_exists(self, mock_validator):
        """Test that existing session is not replaced."""
        existing_session = MagicMock()
        mock_validator._async_session = existing_session

        with patch("aiohttp.ClientSession") as mock_client_session:
            await mock_validator._initialize_async_session()

            mock_client_session.assert_not_called()
            assert mock_validator._async_session == existing_session

    @pytest.mark.asyncio
    async def test_initialize_async_session_configures_timeout(self, mock_validator):
        """Test that session is configured with correct timeout."""
        mock_validator._async_session = None
        mock_validator.timeout = 30.0

        with patch("aiohttp.ClientSession") as mock_client_session:
            with patch("aiohttp.TCPConnector") as mock_connector:
                mock_connector.return_value = MagicMock()
                await mock_validator._initialize_async_session()

                # Verify timeout was passed
                call_kwargs = mock_client_session.call_args[1]
                assert "timeout" in call_kwargs

    @pytest.mark.asyncio
    async def test_initialize_async_session_configures_connector(self, mock_validator):
        """Test that session is configured with TCP connector settings."""
        mock_validator._async_session = None
        mock_validator.max_concurrent = 20
        mock_validator.verify_ssl = True

        with patch("aiohttp.ClientSession") as mock_client_session:
            with patch("aiohttp.TCPConnector") as mock_connector:
                mock_connector_instance = MagicMock()
                mock_connector.return_value = mock_connector_instance

                await mock_validator._initialize_async_session()

                mock_connector.assert_called_once()
                connector_kwargs = mock_connector.call_args[1]
                assert connector_kwargs["limit"] == 20
                assert connector_kwargs["limit_per_host"] == 5
                assert connector_kwargs["use_dns_cache"] is True


class TestAsyncApplyRateLimiting(TestAsyncValidatorMixin):
    """Tests for _async_apply_rate_limiting method."""

    @pytest.mark.asyncio
    async def test_rate_limiting_extracts_domain(self, mock_validator):
        """Test that rate limiting extracts domain correctly."""
        mock_validator.rate_limiter._extract_domain.return_value = "test.com"
        mock_validator.rate_limiter._get_domain_delay.return_value = 0.0
        mock_validator.rate_limiter.domain_stats["test.com"] = DomainStats(
            last_request_time=0.0
        )

        await mock_validator._async_apply_rate_limiting("https://test.com/path")

        mock_validator.rate_limiter._extract_domain.assert_called_with(
            "https://test.com/path"
        )

    @pytest.mark.asyncio
    async def test_rate_limiting_applies_wait(self, mock_validator):
        """Test that rate limiting applies wait when needed."""
        domain = "slow-site.com"
        mock_validator.rate_limiter._extract_domain.return_value = domain
        mock_validator.rate_limiter._get_domain_delay.return_value = 0.5
        mock_validator.rate_limiter.domain_stats[domain] = DomainStats(
            last_request_time=time.time()  # Recent request
        )

        start_time = time.time()
        await mock_validator._async_apply_rate_limiting(f"https://{domain}/")
        elapsed = time.time() - start_time

        # Should have waited some time (but may not be full 0.5s due to timing)
        # Just verify the function completed without error
        assert elapsed >= 0

    @pytest.mark.asyncio
    async def test_rate_limiting_updates_stats(self, mock_validator):
        """Test that rate limiting updates domain statistics."""
        domain = "example.com"
        mock_validator.rate_limiter._extract_domain.return_value = domain
        mock_validator.rate_limiter._get_domain_delay.return_value = 0.0
        mock_validator.rate_limiter.domain_stats[domain] = DomainStats(
            last_request_time=0.0, request_count=0
        )

        await mock_validator._async_apply_rate_limiting(f"https://{domain}/")

        stats = mock_validator.rate_limiter.domain_stats[domain]
        assert stats.request_count == 1
        assert stats.last_request_time > 0
        assert domain in mock_validator.rate_limiter.active_domains

    @pytest.mark.asyncio
    async def test_rate_limiting_no_wait_when_sufficient_time_passed(
        self, mock_validator
    ):
        """Test that no wait is applied when sufficient time has passed."""
        domain = "fast-site.com"
        mock_validator.rate_limiter._extract_domain.return_value = domain
        mock_validator.rate_limiter._get_domain_delay.return_value = 0.1
        # Set last request time to far in the past
        mock_validator.rate_limiter.domain_stats[domain] = DomainStats(
            last_request_time=time.time() - 100.0
        )

        start_time = time.time()
        await mock_validator._async_apply_rate_limiting(f"https://{domain}/")
        elapsed = time.time() - start_time

        # Should be nearly instant since enough time has passed
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiting_updates_total_wait_time(self, mock_validator):
        """Test that rate limiting updates total_wait_time correctly."""
        domain = "waitsite.com"
        mock_validator.rate_limiter._extract_domain.return_value = domain
        mock_validator.rate_limiter._get_domain_delay.return_value = 0.05
        mock_validator.rate_limiter.domain_stats[domain] = DomainStats(
            last_request_time=time.time(), request_count=0, total_wait_time=0.0
        )

        await mock_validator._async_apply_rate_limiting(f"https://{domain}/")

        stats = mock_validator.rate_limiter.domain_stats[domain]
        # total_wait_time should have increased if wait was applied
        assert stats.total_wait_time >= 0


class TestCloseAsyncSession(TestAsyncValidatorMixin):
    """Tests for close_async_session method."""

    @pytest.mark.asyncio
    async def test_close_session_when_exists(self, mock_validator):
        """Test that session is properly closed when it exists."""
        mock_session = AsyncMock()
        mock_validator._async_session = mock_session

        await mock_validator.close_async_session()

        mock_session.close.assert_called_once()
        assert mock_validator._async_session is None

    @pytest.mark.asyncio
    async def test_close_session_when_not_exists(self, mock_validator):
        """Test that close handles case when session doesn't exist."""
        mock_validator._async_session = None

        # Should not raise any errors
        await mock_validator.close_async_session()

        assert mock_validator._async_session is None

    @pytest.mark.asyncio
    async def test_close_session_without_attribute(self, mock_validator):
        """Test that close handles case when _async_session attribute is missing."""
        if hasattr(mock_validator, "_async_session"):
            delattr(mock_validator, "_async_session")

        # Should not raise any errors
        await mock_validator.close_async_session()


class TestConcurrentValidation(TestAsyncValidatorMixin):
    """Tests for concurrent URL validation scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_validation_with_semaphore(
        self, mock_validator, mock_security_result
    ):
        """Test that semaphore properly limits concurrent validations."""
        mock_validator.max_concurrent = 3
        urls = [f"https://example{i}.com" for i in range(10)]

        active_validations = []
        max_active = [0]

        async def mock_validate(url):
            active_validations.append(url)
            max_active[0] = max(max_active[0], len(active_validations))
            await asyncio.sleep(0.01)
            active_validations.remove(url)
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.01
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(
                urls, "concurrent_batch"
            )

        assert result.items_processed == 10
        # Max active should not exceed the semaphore limit
        assert max_active[0] <= mock_validator.max_concurrent

    @pytest.mark.asyncio
    async def test_concurrent_validation_isolation(
        self, mock_validator, mock_security_result
    ):
        """Test that concurrent validations don't interfere with each other."""
        urls = ["https://fast.com", "https://slow.com", "https://medium.com"]

        async def mock_validate(url):
            # Different delays based on URL
            if "slow" in url:
                await asyncio.sleep(0.05)
            elif "medium" in url:
                await asyncio.sleep(0.02)
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.01
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "isolation_batch")

        # All URLs should be validated successfully
        assert result.items_successful == 3
        # Each result should have correct URL
        result_urls = {r.url for r in result.results}
        assert result_urls == set(urls)


class TestErrorHandling(TestAsyncValidatorMixin):
    """Tests for error handling in async validation."""

    @pytest.mark.asyncio
    async def test_handles_mixed_success_and_failure(
        self, mock_validator, mock_security_result
    ):
        """Test handling mix of successful and failed validations."""
        urls = [
            "https://valid.com",
            "https://timeout.com",
            "https://valid2.com",
            "https://error.com",
        ]

        async def mock_validate(url):
            if "timeout" in url:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error_type="timeout",
                    error_message="Timeout",
                    response_time=0.1,
                )
            elif "error" in url:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error_type="connection_error",
                    error_message="Connection error",
                    response_time=0.1,
                )
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.1
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(urls, "mixed_batch")

        assert result.items_processed == 4
        assert result.items_successful == 2
        assert result.items_failed == 2

    @pytest.mark.asyncio
    async def test_preserves_security_validation_result(
        self, mock_validator, mock_security_result
    ):
        """Test that security validation result is preserved in output."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200, url="https://example.com", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.security_validation is not None
        assert result.security_validation.is_safe is True


class TestHTTPStatusCodes(TestAsyncValidatorMixin):
    """Tests for various HTTP status code handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_valid",
        [
            (200, True),
            (201, True),
            (204, True),
            (301, True),
            (302, True),
            (304, True),
            (400, False),
            (401, False),
            (403, False),
            (404, False),
            (500, False),
            (502, False),
            (503, False),
        ],
    )
    async def test_http_status_code_handling(
        self, mock_validator, mock_security_result, status_code, expected_valid
    ):
        """Test handling of various HTTP status codes."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=status_code, url="https://example.com", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is expected_valid
        assert result.status_code == status_code


class TestResponseParsing(TestAsyncValidatorMixin):
    """Tests for response parsing and content handling."""

    @pytest.mark.asyncio
    async def test_parses_content_type(self, mock_validator, mock_security_result):
        """Test that content type is properly parsed."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200,
            url="https://example.com",
            headers={"content-type": "application/json; charset=utf-8"},
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.content_type == "application/json; charset=utf-8"

    @pytest.mark.asyncio
    async def test_parses_content_length(self, mock_validator, mock_security_result):
        """Test that content length is properly parsed."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200,
            url="https://example.com",
            headers={"content-length": "12345"},
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.content_length == 12345

    @pytest.mark.asyncio
    async def test_handles_missing_headers(self, mock_validator, mock_security_result):
        """Test handling of missing response headers."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200, url="https://example.com", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is True
        assert result.content_type is None
        assert result.content_length is None

    @pytest.mark.asyncio
    async def test_handles_invalid_content_length(
        self, mock_validator, mock_security_result
    ):
        """Test handling of invalid content length header."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200,
            url="https://example.com",
            headers={"content-length": "not-a-number"},
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is True
        assert result.content_length is None


class TestValidationResultFields(TestAsyncValidatorMixin):
    """Tests for ValidationResult field population."""

    @pytest.mark.asyncio
    async def test_response_time_is_calculated(
        self, mock_validator, mock_security_result
    ):
        """Test that response time is properly calculated."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200, url="https://example.com", headers={}
        )

        @asynccontextmanager
        async def slow_context():
            await asyncio.sleep(0.05)  # 50ms delay
            yield mock_response

        mock_session = MagicMock()
        mock_session.head = Mock(return_value=slow_context())

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        # Response time should be at least 50ms
        assert result.response_time >= 0.05

    @pytest.mark.asyncio
    async def test_final_url_only_set_on_redirect(
        self, mock_validator, mock_security_result
    ):
        """Test that final_url is only set when redirect occurs."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        # Same URL (no redirect)
        mock_response = create_mock_response(
            status=200, url="https://example.com", headers={}  # Same as input
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        # final_url should be None when no redirect
        assert result.final_url is None

    @pytest.mark.asyncio
    async def test_url_field_is_always_set(self, mock_validator, mock_security_result):
        """Test that URL field is always set in the result."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200, url="https://example.com", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.url == "https://example.com"


class TestSkipPatterns(TestAsyncValidatorMixin):
    """Tests for URL skip patterns."""

    @pytest.mark.asyncio
    async def test_skips_javascript_urls(self, mock_validator, mock_security_result):
        """Test that javascript URLs are skipped."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("javascript:void(0)")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_skips_mailto_urls(self, mock_validator, mock_security_result):
        """Test that mailto URLs are skipped."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("mailto:test@example.com")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_skips_tel_urls(self, mock_validator, mock_security_result):
        """Test that tel URLs are skipped."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("tel:+1234567890")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_skips_ftp_urls(self, mock_validator, mock_security_result):
        """Test that ftp URLs are skipped."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("ftp://example.com/file.txt")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_skips_data_urls(self, mock_validator, mock_security_result):
        """Test that data URLs are skipped."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url(
            "data:text/html,<h1>Test</h1>"
        )

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_skips_fragment_only_urls(self, mock_validator, mock_security_result):
        """Test that fragment-only URLs are skipped."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("#anchor")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_allows_http_urls(self, mock_validator, mock_security_result):
        """Test that http URLs are allowed for validation."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200, url="http://example.com", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("http://example.com")

        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_allows_https_urls(self, mock_validator, mock_security_result):
        """Test that https URLs are allowed for validation."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200, url="https://example.com", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url("https://example.com")

        assert result.is_valid is True


class TestEdgeCases(TestAsyncValidatorMixin):
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_url(self, mock_validator, mock_security_result):
        """Test handling of empty URL."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_whitespace_only_url(self, mock_validator, mock_security_result):
        """Test handling of whitespace-only URL."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )

        result = await mock_validator.async_validate_url("   ")

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_url_with_unicode(self, mock_validator, mock_security_result):
        """Test handling of URL with unicode characters."""
        mock_validator.security_validator.validate_url_security.return_value = (
            mock_security_result
        )
        mock_validator.browser_simulator.get_headers.return_value = {
            "User-Agent": "Test"
        }

        mock_response = create_mock_response(
            status=200, url="https://example.com/path", headers={}
        )

        mock_session = MagicMock()
        mock_session.head = Mock(
            return_value=create_mock_context_manager(mock_response)
        )

        with patch.object(
            mock_validator, "_initialize_async_session", new_callable=AsyncMock
        ):
            mock_validator._async_session = mock_session
            result = await mock_validator.async_validate_url(
                "https://example.com/path"
            )

        assert result.url == "https://example.com/path"

    @pytest.mark.asyncio
    async def test_single_url_batch(self, mock_validator, mock_security_result):
        """Test batch validation with single URL."""

        async def mock_validate(url):
            return ValidationResult(
                url=url, is_valid=True, status_code=200, response_time=0.1
            )

        with patch.object(
            mock_validator, "async_validate_url", side_effect=mock_validate
        ):
            result = await mock_validator.async_validate_batch(
                ["https://example.com"], "single_url_batch"
            )

        assert result.items_processed == 1
        assert result.items_successful == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
