"""
Unit tests for URL validator module.

Tests the URLValidator for validating bookmark URLs with retry logic.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

from bookmark_processor.core.url_validator import (
    URLValidator,
    ValidationResult,
)
from bookmark_processor.core.url_validator.helpers import (
    is_valid_url_format,
    should_skip_url,
)
from bookmark_processor.utils.browser_simulator import BrowserSimulator
from bookmark_processor.utils.intelligent_rate_limiter import IntelligentRateLimiter
from bookmark_processor.utils.retry_handler import RetryHandler


class TestValidationResult:
    """Test ValidationResult class."""

    def test_successful_result(self):
        """Test creating a successful validation result."""
        result = ValidationResult(
            url="https://example.com",
            is_valid=True,
            final_url="https://example.com",
            status_code=200,
            response_time=0.5,
        )

        assert result.url == "https://example.com"
        assert result.is_valid is True
        assert result.final_url == "https://example.com"
        assert result.status_code == 200
        assert result.response_time == 0.5
        assert result.error_message is None
        assert result.error_type is None

    def test_failed_result(self):
        """Test creating a failed validation result."""
        result = ValidationResult(
            url="https://example.com",
            is_valid=False,
            error_message="Connection timeout",
            error_type="timeout",
        )

        assert result.url == "https://example.com"
        assert result.is_valid is False
        assert result.error_message == "Connection timeout"
        assert result.error_type == "timeout"
        assert result.status_code is None
        assert result.final_url is None

    def test_result_with_redirects(self):
        """Test result with redirect chain."""
        result = ValidationResult(
            url="https://example.com",
            is_valid=True,
            final_url="https://www.example.com",
            status_code=200,
        )

        # final_url being different from url indicates a redirect occurred
        assert result.final_url == "https://www.example.com"
        assert result.url == "https://example.com"

    def test_string_representation(self):
        """Test string representation of ValidationResult."""
        result = ValidationResult(
            url="https://example.com", is_valid=True, status_code=200
        )

        # Test to_dict method which provides string representation
        result_dict = result.to_dict()
        assert result_dict["url"] == "https://example.com"
        assert result_dict["status_code"] == 200
        assert result_dict["is_valid"] is True


class TestURLValidator:
    """Test URLValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a URLValidator instance with mocked dependencies."""
        import os
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
            validator = URLValidator(timeout=5, max_redirects=5, max_concurrent=10)
            yield validator

        # Restore original test mode setting
        if original_test_mode is not None:
            os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = original_test_mode

    def test_initialization(self, validator):
        """Test URLValidator initialization."""
        assert validator.timeout == 5
        assert validator.max_redirects == 5
        assert validator.max_concurrent == 10
        assert validator.session is not None
        assert hasattr(validator, "rate_limiter")
        assert hasattr(validator, "browser_simulator")
        assert hasattr(validator, "retry_handler")

    def test_validate_single_url_success(self, validator):
        """Test successful validation of a single URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-type": "text/html", "content-length": "1024"}

        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.status_code == 200
        # If redirect occurred, final_url would be set
        assert result.final_url is None or result.final_url == "https://example.com"

    def test_validate_single_url_with_redirect(self, validator):
        """Test validation with redirects."""
        final_resp = Mock()
        final_resp.status_code = 200
        final_resp.url = "https://www.example.com"  # Redirected URL (different from input)
        final_resp.headers = {"content-type": "text/html"}

        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        with patch.object(validator.session, "head", return_value=final_resp):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.status_code == 200
        # final_url should be set when redirect occurs
        assert result.final_url == "https://www.example.com"

    def test_validate_single_url_timeout(self, validator):
        """Test validation with timeout error."""
        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        with patch.object(
            validator.session, "head", side_effect=Timeout("Request timeout")
        ):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert "timeout" in result.error_message.lower()
        assert result.error_type == "timeout"
        assert result.status_code is None

    def test_validate_single_url_connection_error(self, validator):
        """Test validation with connection error."""
        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        with patch.object(
            validator.session, "head", side_effect=ConnectionError("Connection failed")
        ):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert "connection" in result.error_message.lower()
        assert result.error_type in ["connection_error", "dns_error", "connection_refused"]

    def test_validate_single_url_invalid_status(self, validator):
        """Test validation with invalid status code."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "https://example.com/notfound"
        mock_response.headers = {}

        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com/notfound")

        assert result.is_valid is False
        assert result.status_code == 404
        assert "404" in result.error_message

    def test_validate_single_url_with_retries(self, validator):
        """Test validation with retry logic handled by retry_handler."""
        # The retry logic is now handled by RetryHandler separately
        # This test validates that failures are recorded for retry
        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with patch.object(validator.session, "head", return_value=mock_response):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is True
        assert result.status_code == 200

    def test_validate_single_url_max_retries_exceeded(self, validator):
        """Test validation failure."""
        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        with patch.object(
            validator.session, "head", side_effect=Timeout("Persistent timeout")
        ):
            result = validator.validate_url("https://example.com")

        assert result.is_valid is False
        assert "timeout" in result.error_message.lower()

    def test_validate_single_url_rate_limiting(self, validator):
        """Test that rate limiting is applied."""
        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        with (
            patch.object(validator.session, "head", return_value=mock_response),
            patch.object(validator.rate_limiter, "wait_if_needed") as mock_wait,
        ):
            result = validator.validate_url("https://example.com")

            # Rate limiter should have been called
            mock_wait.assert_called_once()

    def test_validate_single_url_browser_simulation(self, validator):
        """Test that browser headers are applied."""
        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {}

        mock_headers = {"User-Agent": "Mozilla/5.0 Test Browser"}
        validator.browser_simulator.get_headers.return_value = mock_headers

        with patch.object(
            validator.session, "head", return_value=mock_response
        ) as mock_head:
            result = validator.validate_url("https://example.com")

            # Should have called head with browser headers
            mock_head.assert_called_once()
            call_kwargs = mock_head.call_args[1]
            assert "headers" in call_kwargs

    def test_batch_validate(self, validator):
        """Test batch validation of multiple URLs."""
        urls = ["https://example.com", "https://test.com", "https://invalid.com"]

        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        # Create a mapping of URL to response
        url_to_response = {}
        for i, url in enumerate(urls):
            mock_resp = Mock()
            mock_resp.status_code = 200 if i < 2 else 404
            mock_resp.url = url
            mock_resp.headers = {}
            url_to_response[url] = mock_resp

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get('url')
            return url_to_response.get(url, url_to_response[urls[0]])

        with patch.object(validator.session, "head", side_effect=get_response):
            results = validator.batch_validate(urls, enable_retries=False)

        assert len(results) == 3

        # Check results by URL (order may vary due to concurrent processing)
        results_by_url = {r.url: r for r in results}
        assert results_by_url["https://example.com"].is_valid is True
        assert results_by_url["https://test.com"].is_valid is True
        assert results_by_url["https://invalid.com"].is_valid is False

    def test_batch_validate_with_progress_callback(self, validator):
        """Test batch validation with progress callback."""
        urls = ["https://example.com", "https://test.com"]

        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        progress_calls = []

        def progress_callback(message):
            progress_calls.append(message)

        with patch.object(validator.session, "head", return_value=mock_response):
            results = validator.batch_validate(
                urls, progress_callback=progress_callback, enable_retries=False
            )

        # Progress callback should have been called for each URL
        assert len(progress_calls) >= 2

    def test_batch_validate_without_retries(self, validator):
        """Test batch validation with retries disabled."""
        urls = ["https://example.com"]

        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        with patch.object(validator.session, "head", side_effect=Timeout("Timeout")):
            results = validator.batch_validate(urls, enable_retries=False)

        assert len(results) == 1
        assert results[0].is_valid is False

    def test_is_valid_url_scheme(self, validator):
        """Test URL format validation."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://www.example.com/path?query=1",
        ]

        invalid_urls = [
            "",
            "not-a-url",
            "javascript:void(0)",
            "mailto:test@example.com",
        ]

        for url in valid_urls:
            assert (
                is_valid_url_format(url) is True
            ), f"Should be valid: {url}"

        for url in invalid_urls:
            assert (
                is_valid_url_format(url) is False
            ), f"Should be invalid: {url}"

    def test_normalize_url(self, validator):
        """Test URL skip patterns."""
        # URLs that should be skipped
        skip_urls = [
            "javascript:void(0)",
            "mailto:test@example.com",
            "ftp://example.com",
            "tel:+1234567890",
            "#anchor",
        ]

        # URLs that should not be skipped
        valid_urls = [
            "https://example.com",
            "http://example.com",
        ]

        for url in skip_urls:
            assert (
                should_skip_url(url) is True
            ), f"Should be skipped: {url}"

        for url in valid_urls:
            assert (
                should_skip_url(url) is False
            ), f"Should not be skipped: {url}"

    def test_get_validation_statistics(self, validator):
        """Test getting validation statistics."""
        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        # Perform batch validation to populate stats (stats are only updated in batch operations)
        urls = [f"https://example{i}.com" for i in range(5)]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        def get_response(*args, **kwargs):
            url = args[0] if args else kwargs.get('url')
            mock_response.url = url
            return mock_response

        with patch.object(validator.session, "head", side_effect=get_response):
            validator.batch_validate(urls, enable_retries=False)

        stats = validator.get_validation_statistics()

        assert stats["total_urls"] == 5
        assert stats["valid_urls"] == 5
        assert "success_rate" in stats
        assert "average_response_time" in stats

    def test_get_validation_statistics_empty(self, validator):
        """Test getting statistics when no validations performed."""
        stats = validator.get_validation_statistics()

        assert stats["total_urls"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["average_response_time"] == 0.0

    def test_session_configuration(self, validator):
        """Test that session is properly configured."""
        session = validator.session

        # Should have appropriate timeout
        assert validator.timeout is not None

        # Should have retry configuration via adapters
        assert hasattr(session, "adapters")

    def test_cleanup(self, validator):
        """Test cleanup method."""
        validator.close()
        # Should not raise any errors

    def test_concurrent_validation_safety(self, validator):
        """Test that validator is thread-safe for concurrent use."""
        import threading

        # Mock security validator
        from bookmark_processor.utils.security_validator import SecurityValidationResult
        mock_security_result = SecurityValidationResult(
            is_safe=True, risk_level="low", issues=[], blocked_reason=None
        )
        validator.security_validator.validate_url_security.return_value = mock_security_result

        urls = [f"https://example{i}.com" for i in range(10)]
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

        assert len(results) == len(urls)
        assert all(r.is_valid for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
