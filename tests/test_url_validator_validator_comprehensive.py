"""
Comprehensive unit tests for URL validator module (validator.py).

Tests the URLValidator class with focus on:
1. All public methods
2. URL validation and normalization
3. Batch validation methods
4. Error handling paths
5. Edge cases

All network calls are mocked - no actual HTTP requests are made.
"""

import os
import threading
import time
from concurrent.futures import Future
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
import requests
from requests.exceptions import ConnectionError, SSLError, Timeout

from bookmark_processor.core.url_validator import URLValidator
from bookmark_processor.core.batch_types import ValidationResult, ValidationStats
from bookmark_processor.utils.security_validator import SecurityValidationResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_security_safe():
    """Create a safe security validation result."""
    return SecurityValidationResult(
        is_safe=True,
        risk_level="low",
        issues=[],
        blocked_reason=None,
    )


@pytest.fixture
def mock_security_unsafe():
    """Create an unsafe security validation result."""
    return SecurityValidationResult(
        is_safe=False,
        risk_level="high",
        issues=["Suspicious URL pattern detected"],
        blocked_reason="Security validation failed",
    )


@pytest.fixture
def mock_security_critical():
    """Create a critical security validation result."""
    return SecurityValidationResult(
        is_safe=False,
        risk_level="critical",
        issues=["Private IP address detected"],
        blocked_reason="Private IP not allowed",
    )


@pytest.fixture
def validator():
    """Create a URLValidator instance with mocked dependencies."""
    # Ensure test mode is disabled for unit tests
    original_test_mode = os.environ.get("BOOKMARK_PROCESSOR_TEST_MODE")
    if "BOOKMARK_PROCESSOR_TEST_MODE" in os.environ:
        del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]

    with (
        patch("bookmark_processor.core.url_validator.validator.IntelligentRateLimiter"),
        patch("bookmark_processor.core.url_validator.validator.BrowserSimulator"),
        patch("bookmark_processor.core.url_validator.validator.RetryHandler"),
        patch("bookmark_processor.core.url_validator.validator.SecurityValidator"),
    ):
        v = URLValidator(
            timeout=5.0,
            max_redirects=5,
            max_concurrent=5,
            user_agent_rotation=True,
            verify_ssl=True,
        )
        yield v

    # Restore original test mode setting
    if original_test_mode is not None:
        os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_test_mode


@pytest.fixture
def validator_no_ssl():
    """Create a URLValidator instance with SSL verification disabled."""
    original_test_mode = os.environ.get("BOOKMARK_PROCESSOR_TEST_MODE")
    if "BOOKMARK_PROCESSOR_TEST_MODE" in os.environ:
        del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]

    with (
        patch("bookmark_processor.core.url_validator.validator.IntelligentRateLimiter"),
        patch("bookmark_processor.core.url_validator.validator.BrowserSimulator"),
        patch("bookmark_processor.core.url_validator.validator.RetryHandler"),
        patch("bookmark_processor.core.url_validator.validator.SecurityValidator"),
    ):
        v = URLValidator(
            timeout=5.0,
            max_redirects=5,
            max_concurrent=5,
            verify_ssl=False,
        )
        yield v

    if original_test_mode is not None:
        os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_test_mode


# =============================================================================
# Test URLValidator Initialization
# =============================================================================


class TestURLValidatorInitialization:
    """Tests for URLValidator initialization."""

    def test_initialization_with_defaults(self, validator):
        """Test validator initializes with expected default values."""
        assert validator.timeout == 5.0
        assert validator.max_redirects == 5
        assert validator.max_concurrent == 5
        assert validator.verify_ssl is True
        assert validator.session is not None
        assert validator.lock is not None
        assert isinstance(validator.stats, ValidationStats)

    def test_initialization_with_custom_values(self):
        """Test validator initializes with custom values."""
        original_test_mode = os.environ.get("BOOKMARK_PROCESSOR_TEST_MODE")
        if "BOOKMARK_PROCESSOR_TEST_MODE" in os.environ:
            del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]

        with (
            patch("bookmark_processor.core.url_validator.validator.IntelligentRateLimiter"),
            patch("bookmark_processor.core.url_validator.validator.BrowserSimulator"),
            patch("bookmark_processor.core.url_validator.validator.RetryHandler"),
            patch("bookmark_processor.core.url_validator.validator.SecurityValidator"),
        ):
            v = URLValidator(
                timeout=60.0,
                max_redirects=10,
                max_concurrent=20,
                user_agent_rotation=False,
                verify_ssl=False,
            )
            assert v.timeout == 60.0
            assert v.max_redirects == 10
            assert v.max_concurrent == 20
            assert v.verify_ssl is False

        if original_test_mode is not None:
            os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_test_mode

    def test_initialization_with_custom_rate_limiter(self):
        """Test validator initializes with a custom rate limiter."""
        original_test_mode = os.environ.get("BOOKMARK_PROCESSOR_TEST_MODE")
        if "BOOKMARK_PROCESSOR_TEST_MODE" in os.environ:
            del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]

        mock_rate_limiter = MagicMock()

        with (
            patch("bookmark_processor.core.url_validator.validator.IntelligentRateLimiter"),
            patch("bookmark_processor.core.url_validator.validator.BrowserSimulator"),
            patch("bookmark_processor.core.url_validator.validator.RetryHandler"),
            patch("bookmark_processor.core.url_validator.validator.SecurityValidator"),
        ):
            v = URLValidator(
                timeout=5.0,
                rate_limiter=mock_rate_limiter,
            )
            assert v.rate_limiter == mock_rate_limiter

        if original_test_mode is not None:
            os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_test_mode


# =============================================================================
# Test validate_url Method
# =============================================================================


class TestValidateURL:
    """Tests for the validate_url method."""

    def test_validate_url_success(self, validator, mock_security_safe):
        """Test successful URL validation."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {
            "content-type": "text/html; charset=utf-8",
            "content-length": "12345",
        }

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.status_code == 200
        assert result.content_type == "text/html; charset=utf-8"
        assert result.content_length == 12345
        assert result.final_url is None  # No redirect
        assert result.error_message is None

    def test_validate_url_with_redirect(self, validator, mock_security_safe):
        """Test URL validation with redirect."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://www.example.com/new-path"
        mock_response.headers = {"content-type": "text/html"}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.final_url == "https://www.example.com/new-path"

    def test_validate_url_invalid_format(self, validator, mock_security_safe):
        """Test URL validation with invalid format."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("not-a-valid-url")

        assert result.is_valid is False
        assert result.error_type == "format_error"
        assert "Invalid URL format" in result.error_message

    def test_validate_url_empty_string(self, validator, mock_security_safe):
        """Test URL validation with empty string."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("")

        assert result.is_valid is False
        assert result.error_type == "format_error"

    def test_validate_url_unsupported_scheme_javascript(self, validator, mock_security_safe):
        """Test URL validation with javascript: scheme.

        Note: javascript: URLs without a netloc fail format validation first,
        so we get format_error instead of unsupported_scheme.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("javascript:void(0)")

        assert result.is_valid is False
        # javascript:void(0) has no netloc so fails format validation first
        assert result.error_type == "format_error"

    def test_validate_url_unsupported_scheme_mailto(self, validator, mock_security_safe):
        """Test URL validation with mailto: scheme.

        Note: mailto: URLs without netloc fail format validation first.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("mailto:test@example.com")

        assert result.is_valid is False
        # mailto:user@domain has no netloc so fails format validation
        assert result.error_type == "format_error"

    def test_validate_url_unsupported_scheme_ftp(self, validator, mock_security_safe):
        """Test URL validation with ftp: scheme."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("ftp://ftp.example.com/file.txt")

        assert result.is_valid is False
        # ftp:// has netloc, so passes format check but should be skipped
        assert result.error_type == "unsupported_scheme"

    def test_validate_url_unsupported_scheme_tel(self, validator, mock_security_safe):
        """Test URL validation with tel: scheme.

        Note: tel: URLs without netloc fail format validation first.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("tel:+1234567890")

        assert result.is_valid is False
        # tel:number has no netloc so fails format validation
        assert result.error_type == "format_error"

    def test_validate_url_unsupported_scheme_fragment_only(self, validator, mock_security_safe):
        """Test URL validation with fragment-only URL.

        Note: Fragment-only URLs have no scheme/netloc so fail format validation.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("#section1")

        assert result.is_valid is False
        # #section has no scheme/netloc so fails format validation
        assert result.error_type == "format_error"

    def test_validate_url_unsupported_scheme_file(self, validator, mock_security_safe):
        """Test URL validation with file: scheme.

        Note: file:/// URLs have scheme but no netloc (the path starts at the third /),
        so they fail format validation first.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("file:///path/to/file.txt")

        assert result.is_valid is False
        # file:/// has no netloc, so fails format validation
        assert result.error_type == "format_error"

    def test_validate_url_unsupported_scheme_data(self, validator, mock_security_safe):
        """Test URL validation with data: scheme."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        result = validator.validate_url("data:text/plain,Hello")

        assert result.is_valid is False
        # data: URLs have no netloc so fail format validation
        assert result.error_type == "format_error"

    def test_validate_url_security_failure_high_risk(self, validator, mock_security_unsafe):
        """Test URL validation with high-risk security failure."""
        validator.security_validator.validate_url_security.return_value = mock_security_unsafe

        result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "security_error"
        assert "Security validation failed" in result.error_message

    def test_validate_url_security_failure_critical_risk(self, validator, mock_security_critical):
        """Test URL validation with critical security failure."""
        validator.security_validator.validate_url_security.return_value = mock_security_critical

        result = validator.validate_url("https://192.168.1.1/admin")

        assert result.is_valid is False
        assert result.error_type == "security_error"

    def test_validate_url_http_404(self, validator, mock_security_safe):
        """Test URL validation with 404 response."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "https://example.com/not-found"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com/not-found")

        assert result.is_valid is False
        assert result.status_code == 404
        assert "404" in result.error_message
        assert result.error_type == "not_found"

    def test_validate_url_http_403(self, validator, mock_security_safe):
        """Test URL validation with 403 response."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.url = "https://example.com/forbidden"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com/forbidden")

        assert result.is_valid is False
        assert result.status_code == 403
        assert result.error_type == "forbidden"

    def test_validate_url_http_500(self, validator, mock_security_safe):
        """Test URL validation with 500 response."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.status_code == 500
        assert result.error_type == "internal_server_error"

    def test_validate_url_http_502(self, validator, mock_security_safe):
        """Test URL validation with 502 response."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.status_code == 502
        assert result.error_type == "bad_gateway"

    def test_validate_url_http_503(self, validator, mock_security_safe):
        """Test URL validation with 503 response."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.status_code == 503
        assert result.error_type == "service_unavailable"

    def test_validate_url_http_429_rate_limited(self, validator, mock_security_safe):
        """Test URL validation with 429 rate limit response."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.status_code == 429
        assert result.error_type == "rate_limited"

    def test_validate_url_timeout(self, validator, mock_security_safe):
        """Test URL validation with timeout error."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        with patch.object(
            validator.session, "head", side_effect=Timeout("Request timed out")
        ):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "timeout"
        assert "Timeout" in result.error_message
        assert str(validator.timeout) in result.error_message

    def test_validate_url_connection_error(self, validator, mock_security_safe):
        """Test URL validation with connection error."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        with patch.object(
            validator.session, "head", side_effect=ConnectionError("Connection refused")
        ):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "connection_refused"
        assert "refused" in result.error_message.lower()

    def test_validate_url_dns_error(self, validator, mock_security_safe):
        """Test URL validation with DNS resolution error."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        with patch.object(
            validator.session, "head", side_effect=ConnectionError("DNS resolution failed: Name resolution error")
        ):
            result = validator.validate_url("https://nonexistent-domain.example.com")

        assert result.is_valid is False
        assert result.error_type == "dns_error"

    def test_validate_url_ssl_error_with_fallback_success(self, validator, mock_security_safe):
        """Test URL validation with SSL error that succeeds on fallback."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        # First call raises SSL error, second succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-type": "text/html"}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SSLError("Certificate verification failed")
            return mock_response

        with patch.object(validator.session, "head", side_effect=side_effect):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.error_message == "SSL certificate warning"
        assert result.error_type == "ssl_warning"

    def test_validate_url_ssl_error_with_fallback_failure(self, validator, mock_security_safe):
        """Test URL validation with SSL error that fails on fallback too."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        # Both calls raise SSL error
        def side_effect(*args, **kwargs):
            raise SSLError("Certificate verification failed")

        with patch.object(validator.session, "head", side_effect=side_effect):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "ssl_error"
        assert "SSL Error" in result.error_message

    def test_validate_url_ssl_error_fallback_returns_error_status(self, validator, mock_security_safe):
        """Test URL validation with SSL error where fallback returns an error status."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        # First call raises SSL error, second returns 404
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SSLError("Certificate verification failed")
            return mock_response

        with patch.object(validator.session, "head", side_effect=side_effect):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.status_code == 404
        assert result.error_type == "not_found"

    def test_validate_url_ssl_error_no_verify(self, validator_no_ssl, mock_security_safe):
        """Test URL validation with SSL error when verify_ssl is False."""
        validator_no_ssl.security_validator.validate_url_security.return_value = mock_security_safe

        with patch.object(
            validator_no_ssl.session, "head", side_effect=SSLError("SSL Error")
        ):
            result = validator_no_ssl.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "ssl_error"

    def test_validate_url_unexpected_error(self, validator, mock_security_safe):
        """Test URL validation with unexpected exception."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        with patch.object(
            validator.session, "head", side_effect=RuntimeError("Unexpected error occurred")
        ):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert result.error_type == "unknown_error"
        assert "Unexpected error" in result.error_message

    def test_validate_url_rate_limiter_called(self, validator, mock_security_safe):
        """Test that rate limiter is called during validation."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            validator.validate_url("https://example.com")

        validator.rate_limiter.wait_if_needed.assert_called_once_with("https://example.com")
        validator.rate_limiter.record_success.assert_called_once_with("https://example.com")

    def test_validate_url_rate_limiter_error_recorded(self, validator, mock_security_safe):
        """Test that rate limiter records errors on failure."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        with patch.object(
            validator.session, "head", side_effect=Timeout("Request timed out")
        ):
            validator.validate_url("https://example.com")

        validator.rate_limiter.record_error.assert_called_once_with("https://example.com")

    def test_validate_url_browser_simulator_headers(self, validator, mock_security_safe):
        """Test that browser simulator headers are used."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml",
        }
        validator.browser_simulator.get_headers.return_value = mock_headers

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response) as mock_head:
            validator.validate_url("https://example.com")

            validator.browser_simulator.get_headers.assert_called_once_with("https://example.com")
            mock_head.assert_called_once()
            call_kwargs = mock_head.call_args[1]
            assert call_kwargs["headers"] == mock_headers

    def test_validate_url_response_time_measured(self, validator, mock_security_safe):
        """Test that response time is measured."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.response_time >= 0

    def test_validate_url_content_length_parsed(self, validator, mock_security_safe):
        """Test that content length is properly parsed."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-length": "9876"}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.content_length == 9876

    def test_validate_url_content_length_invalid(self, validator, mock_security_safe):
        """Test that invalid content length is handled."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-length": "not-a-number"}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.content_length is None


# =============================================================================
# Test validate_url in Test Mode
# =============================================================================


class TestValidateURLTestMode:
    """Tests for validate_url method in test mode."""

    def test_test_mode_valid_url(self):
        """Test mock validation in test mode returns valid result."""
        os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = "true"

        try:
            with (
                patch("bookmark_processor.core.url_validator.validator.IntelligentRateLimiter"),
                patch("bookmark_processor.core.url_validator.validator.BrowserSimulator"),
                patch("bookmark_processor.core.url_validator.validator.RetryHandler"),
                patch("bookmark_processor.core.url_validator.validator.SecurityValidator"),
            ):
                v = URLValidator(timeout=5.0)
                result = v.validate_url("https://example.com")

            assert result.is_valid is True
            assert result.status_code == 200
            assert result.response_time == 0.1
        finally:
            del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]

    def test_test_mode_invalid_url_patterns(self):
        """Test mock validation in test mode returns invalid for specific patterns."""
        os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = "true"

        try:
            with (
                patch("bookmark_processor.core.url_validator.validator.IntelligentRateLimiter"),
                patch("bookmark_processor.core.url_validator.validator.BrowserSimulator"),
                patch("bookmark_processor.core.url_validator.validator.RetryHandler"),
                patch("bookmark_processor.core.url_validator.validator.SecurityValidator"),
            ):
                v = URLValidator(timeout=5.0)

                # Test patterns that should return invalid
                invalid_patterns = ["invalid", "404", "error", "timeout"]
                for pattern in invalid_patterns:
                    result = v.validate_url(f"https://example.com/{pattern}")
                    assert result.is_valid is False, f"Expected invalid for pattern: {pattern}"
                    assert result.status_code == 404
        finally:
            del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]


# =============================================================================
# Test batch_validate Method
# =============================================================================


class TestBatchValidate:
    """Tests for the batch_validate method."""

    def test_batch_validate_success(self, validator, mock_security_safe):
        """Test batch validation with all successful URLs."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com", "https://example3.com"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            results = validator.batch_validate(urls, enable_retries=False)

        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_batch_validate_with_duplicates(self, validator, mock_security_safe):
        """Test batch validation removes duplicates."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = [
            "https://example.com",
            "https://example.com",
            "https://other.com",
            "https://example.com",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            results = validator.batch_validate(urls, enable_retries=False)

        # Should only have 2 unique URLs
        assert len(results) == 2
        urls_in_results = {r.url for r in results}
        assert urls_in_results == {"https://example.com", "https://other.com"}

    def test_batch_validate_with_progress_callback(self, validator, mock_security_safe):
        """Test batch validation with progress callback."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]
        progress_messages = []

        def progress_callback(message):
            progress_messages.append(message)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            validator.batch_validate(urls, progress_callback=progress_callback, enable_retries=False)

        assert len(progress_messages) == 2
        assert all("Validated" in msg for msg in progress_messages)

    def test_batch_validate_with_progress_tracker(self, validator, mock_security_safe):
        """Test batch validation with progress tracker."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]

        mock_tracker = MagicMock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            validator.batch_validate(urls, enable_retries=False, progress_tracker=mock_tracker)

        mock_tracker.start_stage.assert_called_once()
        mock_tracker.set_active_tasks.assert_called()
        assert mock_tracker.update_progress.call_count >= 2

    def test_batch_validate_mixed_results(self, validator, mock_security_safe):
        """Test batch validation with mixed success/failure."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://valid.com", "https://notfound.com", "https://timeout.com"]

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            if "notfound" in url:
                mock_resp = Mock()
                mock_resp.status_code = 404
                mock_resp.url = url
                mock_resp.headers = {}
                return mock_resp
            elif "timeout" in url:
                raise Timeout("Request timed out")
            else:
                mock_resp = Mock()
                mock_resp.status_code = 200
                mock_resp.url = url
                mock_resp.headers = {}
                return mock_resp

        with patch.object(validator.session, "head", side_effect=get_response):
            results = validator.batch_validate(urls, enable_retries=False)

        assert len(results) == 3

        results_by_url = {r.url: r for r in results}
        assert results_by_url["https://valid.com"].is_valid is True
        assert results_by_url["https://notfound.com"].is_valid is False
        assert results_by_url["https://timeout.com"].is_valid is False

    def test_batch_validate_with_retries(self, validator, mock_security_safe):
        """Test batch validation with retry logic enabled."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]

        # Setup retry handler mock
        validator.retry_handler.retry_queue = []
        validator.retry_handler.retry_failed_items.return_value = []

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            results = validator.batch_validate(urls, enable_retries=True)

        assert len(results) == 1

    def test_batch_validate_retry_handler_populated(self, validator, mock_security_safe):
        """Test that retry handler is populated with failed URLs."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://timeout.com"]
        validator.retry_handler.retry_queue = []
        validator.retry_handler.retry_failed_items.return_value = []

        with patch.object(
            validator.session, "head", side_effect=Timeout("Request timed out")
        ):
            validator.batch_validate(urls, enable_retries=True)

        validator.retry_handler.add_failed_url.assert_called()

    def test_batch_validate_exception_handling(self, validator, mock_security_safe):
        """Test batch validation handles exceptions from futures."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]
        validator.retry_handler.retry_queue = []
        validator.retry_handler.retry_failed_items.return_value = []

        # Simulate an exception during future.result()
        with patch.object(
            validator.session, "head", side_effect=RuntimeError("Future exception")
        ):
            results = validator.batch_validate(urls, enable_retries=False)

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].error_type == "unknown_error"

    def test_batch_validate_stats_updated(self, validator, mock_security_safe):
        """Test that validation stats are updated during batch validation."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            validator.batch_validate(urls, enable_retries=False)

        stats = validator.get_validation_statistics()
        assert stats["total_urls"] == 2
        assert stats["valid_urls"] == 2


# =============================================================================
# Test batch_validate_optimized Method
# =============================================================================


class TestBatchValidateOptimized:
    """Tests for the batch_validate_optimized method."""

    def test_batch_validate_optimized_success(self, validator, mock_security_safe):
        """Test optimized batch validation with successful URLs."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            # gc is imported inside the method, so we need to patch it globally
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                results = validator.batch_validate_optimized(urls, batch_size=10)

        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_batch_validate_optimized_with_batching(self, validator, mock_security_safe):
        """Test optimized batch validation processes in batches."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        # Create 5 URLs but use batch_size of 2
        urls = [f"https://example{i}.com" for i in range(5)]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            import gc as real_gc
            with patch.object(real_gc, "collect") as mock_gc_collect:
                results = validator.batch_validate_optimized(urls, batch_size=2)

        assert len(results) == 5
        # GC should be called between batches (3 batches, so at least 2 gc.collect calls)
        assert mock_gc_collect.call_count >= 2

    def test_batch_validate_optimized_with_progress_callback(self, validator, mock_security_safe):
        """Test optimized batch validation with progress callback."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]
        progress_messages = []

        def progress_callback(message):
            progress_messages.append(message)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                validator.batch_validate_optimized(
                    urls, progress_callback=progress_callback, batch_size=10
                )

        assert len(progress_messages) == 2

    def test_batch_validate_optimized_with_progress_tracker(self, validator, mock_security_safe):
        """Test optimized batch validation with progress tracker."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]
        mock_tracker = MagicMock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                validator.batch_validate_optimized(
                    urls, batch_size=10, progress_tracker=mock_tracker
                )

        mock_tracker.start_stage.assert_called_once()
        assert mock_tracker.update_progress.call_count >= 2

    def test_batch_validate_optimized_removes_duplicates(self, validator, mock_security_safe):
        """Test optimized batch validation removes duplicates."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = [
            "https://example.com",
            "https://example.com",
            "https://other.com",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                results = validator.batch_validate_optimized(urls, batch_size=10)

        assert len(results) == 2

    def test_batch_validate_optimized_exception_handling(self, validator, mock_security_safe):
        """Test optimized batch validation handles exceptions.

        Note: validate_url catches RuntimeError and returns it as "Unexpected error",
        so the batch_validate_optimized method doesn't see the exception at the
        executor level. The error message will be "Unexpected error: <message>".
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]

        with patch.object(
            validator.session, "head", side_effect=RuntimeError("Unexpected error")
        ):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                results = validator.batch_validate_optimized(urls, batch_size=10)

        assert len(results) == 1
        assert results[0].is_valid is False
        # The error message comes from validate_url's general exception handler
        assert "Unexpected error" in results[0].error_message

    def test_batch_validate_optimized_custom_max_workers(self, validator, mock_security_safe):
        """Test optimized batch validation with custom max_workers."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                results = validator.batch_validate_optimized(
                    urls, batch_size=10, max_workers=2
                )

        assert len(results) == 2

    def test_batch_validate_optimized_progress_tracker_logs_errors(self, validator, mock_security_safe):
        """Test optimized batch validation logs errors to progress tracker.

        Note: validate_url catches RuntimeError and returns a ValidationResult with
        is_valid=False. The exception is not raised to the executor level, so
        log_error is not called. Instead, update_progress is called with failed_delta=1.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]
        mock_tracker = MagicMock()

        with patch.object(
            validator.session, "head", side_effect=RuntimeError("Test error")
        ):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                validator.batch_validate_optimized(
                    urls, batch_size=10, progress_tracker=mock_tracker
                )

        # Since validate_url handles the exception internally and returns
        # a ValidationResult, update_progress should be called with failed_delta=1
        mock_tracker.update_progress.assert_called()
        # Check that at least one call included failed_delta > 0
        calls = mock_tracker.update_progress.call_args_list
        failed_updates = [c for c in calls if c.kwargs.get("failed_delta", 0) > 0]
        assert len(failed_updates) >= 1


# =============================================================================
# Test Statistics Methods
# =============================================================================


class TestStatisticsMethods:
    """Tests for statistics-related methods."""

    def test_get_validation_statistics_empty(self, validator):
        """Test getting statistics when no validations performed."""
        stats = validator.get_validation_statistics()

        assert stats["total_urls"] == 0
        assert stats["valid_urls"] == 0
        assert stats["invalid_urls"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["average_response_time"] == 0.0

    def test_get_validation_statistics_after_validations(self, validator, mock_security_safe):
        """Test getting statistics after batch validation."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://valid1.com", "https://valid2.com", "https://invalid.com"]

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            if "invalid" in url:
                mock_resp = Mock()
                mock_resp.status_code = 404
                mock_resp.url = url
                mock_resp.headers = {}
                return mock_resp
            else:
                mock_resp = Mock()
                mock_resp.status_code = 200
                mock_resp.url = url
                mock_resp.headers = {}
                return mock_resp

        with patch.object(validator.session, "head", side_effect=get_response):
            validator.batch_validate(urls, enable_retries=False)

        stats = validator.get_validation_statistics()

        assert stats["total_urls"] == 3
        assert stats["valid_urls"] == 2
        assert stats["invalid_urls"] == 1
        assert "success_rate" in stats
        assert "error_distribution" in stats
        assert "rate_limiter_stats" in stats

    def test_get_validation_statistics_with_retry_stats(self, validator):
        """Test getting statistics includes retry stats."""
        validator.retry_handler.get_retry_statistics.return_value = {
            "total_items": 5,
            "successful_retries": 2,
        }

        stats = validator.get_validation_statistics()

        assert "retry_stats" in stats

    def test_reset_statistics(self, validator, mock_security_safe):
        """Test resetting validation statistics."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        # Perform some validations first
        urls = ["https://example.com"]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            validator.batch_validate(urls, enable_retries=False)

        # Verify stats are populated
        stats_before = validator.get_validation_statistics()
        assert stats_before["total_urls"] == 1

        # Reset and verify
        validator.reset_statistics()

        stats_after = validator.get_validation_statistics()
        assert stats_after["total_urls"] == 0

        validator.rate_limiter.reset_domain_stats.assert_called_once()
        validator.retry_handler.clear_completed.assert_called_once()


# =============================================================================
# Test Session and Cleanup
# =============================================================================


class TestSessionAndCleanup:
    """Tests for session creation and cleanup."""

    def test_create_session(self, validator):
        """Test session is properly created with adapters."""
        assert validator.session is not None
        assert hasattr(validator.session, "adapters")

    def test_close_validator(self, validator):
        """Test validator cleanup."""
        # Should not raise any exception
        validator.close()

    def test_close_validator_without_session(self, validator):
        """Test validator cleanup when session doesn't exist."""
        # Remove session attribute
        del validator.session

        # Should not raise any exception
        validator.close()


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of the URLValidator."""

    def test_concurrent_validation(self, validator, mock_security_safe):
        """Test concurrent URL validation is thread-safe."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = [f"https://example{i}.com" for i in range(20)]
        results = []
        lock = threading.Lock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def validate_url(url):
            mock_response.url = url
            with patch.object(validator.session, "head", return_value=mock_response):
                result = validator.validate_url(url)
                with lock:
                    results.append(result)

        threads = []
        for url in urls:
            thread = threading.Thread(target=validate_url, args=(url,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 20
        assert all(r.is_valid for r in results)

    def test_lock_usage_in_batch_validate(self, validator, mock_security_safe):
        """Test that lock is used during batch validation."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example1.com", "https://example2.com"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            # This should complete without deadlock
            results = validator.batch_validate(urls, enable_retries=False)

        assert len(results) == 2


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_validate_url_with_special_characters(self, validator, mock_security_safe):
        """Test URL validation with special characters in path."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/path%20with%20spaces"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com/path%20with%20spaces")

        assert result.is_valid is True

    def test_validate_url_with_query_parameters(self, validator, mock_security_safe):
        """Test URL validation with query parameters."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/search?q=test&page=1"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com/search?q=test&page=1")

        assert result.is_valid is True

    def test_validate_url_with_fragments(self, validator, mock_security_safe):
        """Test URL validation with URL fragments."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/page#section1"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com/page#section1")

        assert result.is_valid is True

    def test_validate_url_with_port(self, validator, mock_security_safe):
        """Test URL validation with explicit port number."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com:8443/api"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com:8443/api")

        assert result.is_valid is True

    def test_validate_url_with_username_password(self, validator, mock_security_safe):
        """Test URL validation with username/password in URL."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://user:pass@example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://user:pass@example.com")

        # This may or may not be valid depending on security validation
        # The test verifies the code handles it without crashing

    def test_validate_url_very_long_url(self, validator, mock_security_safe):
        """Test URL validation with a very long URL."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        long_path = "a" * 500
        url = f"https://example.com/{long_path}"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = url
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url(url)

        assert result is not None

    def test_batch_validate_empty_list(self, validator):
        """Test batch validation with empty URL list.

        Note: The current implementation has a division by zero bug when results
        are empty. This test documents the expected behavior and can be updated
        when the bug is fixed.
        """
        # The batch_validate function has a bug where it divides by len(results)
        # when results is empty, causing ZeroDivisionError.
        # This test documents the current behavior.
        with pytest.raises(ZeroDivisionError):
            validator.batch_validate([], enable_retries=False)

    def test_batch_validate_single_url(self, validator, mock_security_safe):
        """Test batch validation with single URL."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            results = validator.batch_validate(
                ["https://example.com"], enable_retries=False
            )

        assert len(results) == 1

    def test_validate_url_http_scheme(self, validator, mock_security_safe):
        """Test URL validation with HTTP (non-HTTPS) scheme."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "http://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("http://example.com")

        assert result.is_valid is True

    def test_validate_url_success_codes(self, validator, mock_security_safe):
        """Test various success HTTP status codes."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        success_codes = [200, 201, 202, 203, 204, 205, 206, 301, 302, 303, 304, 307, 308]

        for code in success_codes:
            mock_response = Mock()
            mock_response.status_code = code
            mock_response.url = "https://example.com"
            mock_response.headers = {}

            with patch.object(validator.session, "head", return_value=mock_response):
                result = validator.validate_url("https://example.com")

            assert result.is_valid is True, f"Expected status {code} to be valid"


# =============================================================================
# Test ValidationStats Integration
# =============================================================================


class TestValidationStatsIntegration:
    """Tests for ValidationStats integration with URLValidator."""

    def test_stats_updated_for_valid_url(self, validator, mock_security_safe):
        """Test that stats are updated for valid URLs."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            validator.batch_validate(["https://example.com"], enable_retries=False)

        assert validator.stats.total_urls == 1
        assert validator.stats.valid_urls == 1
        assert validator.stats.invalid_urls == 0

    def test_stats_updated_for_invalid_url(self, validator, mock_security_safe):
        """Test that stats are updated for invalid URLs."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "https://example.com/notfound"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            validator.batch_validate(["https://example.com/notfound"], enable_retries=False)

        assert validator.stats.total_urls == 1
        assert validator.stats.valid_urls == 0
        assert validator.stats.invalid_urls == 1

    def test_stats_error_distribution(self, validator, mock_security_safe):
        """Test that error distribution is tracked."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://notfound.com", "https://forbidden.com", "https://timeout.com"]

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get("url")
            if "notfound" in url:
                mock_resp = Mock()
                mock_resp.status_code = 404
                mock_resp.url = url
                mock_resp.headers = {}
                return mock_resp
            elif "forbidden" in url:
                mock_resp = Mock()
                mock_resp.status_code = 403
                mock_resp.url = url
                mock_resp.headers = {}
                return mock_resp
            else:
                raise Timeout("Request timed out")

        with patch.object(validator.session, "head", side_effect=get_response):
            validator.batch_validate(urls, enable_retries=False)

        assert "not_found" in validator.stats.error_distribution
        assert "forbidden" in validator.stats.error_distribution
        assert "timeout" in validator.stats.error_distribution

    def test_stats_redirected_urls_counted(self, validator, mock_security_safe):
        """Test that redirected URLs are counted."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://www.example.com"  # Different from input
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            validator.batch_validate(["https://example.com"], enable_retries=False)

        assert validator.stats.redirected_urls == 1


# =============================================================================
# Test Retry Path in Batch Validate
# =============================================================================


class TestBatchValidateRetryPath:
    """Tests for retry handling in batch_validate."""

    def test_batch_validate_with_retry_queue_populated(self, validator, mock_security_safe):
        """Test batch validation when retry handler has URLs to retry."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]

        # Setup retry handler with populated queue and return successful retry results
        mock_retry_result = ValidationResult(
            url="https://example.com",
            is_valid=True,
            status_code=200,
        )

        # Make retry_queue look populated (will be checked in the if condition)
        validator.retry_handler.retry_queue = ["https://example.com"]
        validator.retry_handler.retry_failed_items.return_value = [mock_retry_result]

        mock_response = Mock()
        mock_response.status_code = 500  # First attempt fails
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            results = validator.batch_validate(urls, enable_retries=True)

        # The retry_failed_items should have been called
        validator.retry_handler.retry_failed_items.assert_called()

    def test_batch_validate_retry_with_progress_tracker(self, validator, mock_security_safe):
        """Test batch validation retry path with progress tracker."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]
        mock_tracker = MagicMock()

        mock_retry_result = ValidationResult(
            url="https://example.com",
            is_valid=True,
            status_code=200,
        )
        validator.retry_handler.retry_queue = ["https://example.com"]
        validator.retry_handler.retry_failed_items.return_value = [mock_retry_result]

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            results = validator.batch_validate(
                urls, enable_retries=True, progress_tracker=mock_tracker
            )

        # Check that progress tracker was used for retry logging
        # The log_error should be called for retry start/completion messages
        assert mock_tracker.log_error.call_count >= 1


class TestBatchValidateExceptionFromFuture:
    """Tests for exception handling when future.result() raises."""

    def test_batch_validate_future_raises_exception(self, validator, mock_security_safe):
        """Test batch validation when future.result() raises an exception.

        This tests the exception path in batch_validate where the future's
        result() call itself raises an exception (lines 365-382).
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe
        validator.retry_handler.retry_queue = []
        validator.retry_handler.retry_failed_items.return_value = []

        urls = ["https://example.com"]

        # To make future.result() raise, we need to patch at a deeper level
        # We'll patch validate_url to raise an exception that propagates
        def raise_in_validate(url):
            raise RuntimeError("Future execution error")

        with patch.object(validator, "validate_url", side_effect=raise_in_validate):
            results = validator.batch_validate(urls, enable_retries=True)

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].error_type == "validation_error"
        assert "Future execution error" in results[0].error_message

        # With enable_retries=True, should add to retry handler
        validator.retry_handler.add_failed_url.assert_called()

    def test_batch_validate_future_exception_with_progress_tracker(
        self, validator, mock_security_safe
    ):
        """Test batch validation exception with progress tracker."""
        validator.security_validator.validate_url_security.return_value = mock_security_safe
        validator.retry_handler.retry_queue = []
        validator.retry_handler.retry_failed_items.return_value = []

        urls = ["https://example.com"]
        mock_tracker = MagicMock()

        def raise_in_validate(url):
            raise RuntimeError("Future execution error")

        with patch.object(validator, "validate_url", side_effect=raise_in_validate):
            results = validator.batch_validate(
                urls, enable_retries=False, progress_tracker=mock_tracker
            )

        # Progress tracker should update with failure
        mock_tracker.update_progress.assert_called()


class TestBatchValidateOptimizedExceptionPath:
    """Tests for exception path in batch_validate_optimized."""

    def test_batch_validate_optimized_future_raises(self, validator, mock_security_safe):
        """Test optimized batch validation when future.result() raises.

        This tests lines 520-541 where the exception is caught at the
        executor level.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]

        def raise_in_validate(url):
            raise RuntimeError("Future execution error")

        with patch.object(validator, "validate_url", side_effect=raise_in_validate):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                results = validator.batch_validate_optimized(urls, batch_size=10)

        assert len(results) == 1
        assert results[0].is_valid is False
        assert "Validation error" in results[0].error_message

    def test_batch_validate_optimized_future_raises_with_tracker(
        self, validator, mock_security_safe
    ):
        """Test optimized batch when future raises with progress tracker.

        This specifically tests lines 534-541.
        """
        validator.security_validator.validate_url_security.return_value = mock_security_safe

        urls = ["https://example.com"]
        mock_tracker = MagicMock()

        def raise_in_validate(url):
            raise RuntimeError("Future execution error")

        with patch.object(validator, "validate_url", side_effect=raise_in_validate):
            import gc as real_gc
            with patch.object(real_gc, "collect"):
                validator.batch_validate_optimized(
                    urls, batch_size=10, progress_tracker=mock_tracker
                )

        # Progress tracker should update and log error
        mock_tracker.update_progress.assert_called()
        mock_tracker.log_error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
