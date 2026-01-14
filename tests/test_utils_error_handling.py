"""
Unit tests for error handling and retry utilities.

Tests for error handler, retry handler, and related utilities.
"""

import asyncio
import time
from typing import Any, Callable
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.error_handler import (
    AIProcessingError,
    BookmarkProcessorError,
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    NetworkError,
    RetryStrategy,
    ValidationError,
)
from bookmark_processor.utils.retry_handler import (
    ErrorType,
    ExponentialBackoff,
    FixedBackoff,
    LinearBackoff,
    RetryConfig,
    RetryHandler,
)


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_bookmark_processing_error(self):
        """Test BookmarkProcessorError exception."""
        error = BookmarkProcessorError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, BookmarkProcessorError)

    def test_network_error(self):
        """Test NetworkError exception."""
        error = NetworkError("Network failed")
        assert str(error) == "Network failed"
        assert isinstance(error, BookmarkProcessorError)

    def test_ai_processing_error(self):
        """Test AIProcessingError exception."""
        error = AIProcessingError("AI failed")
        assert str(error) == "AI failed"
        assert isinstance(error, BookmarkProcessorError)


class TestErrorHandler:
    """Test ErrorHandler class."""

    def test_init_default(self):
        """Test ErrorHandler initialization with defaults."""
        handler = ErrorHandler()

        assert handler.enable_fallback is True
        assert len(handler.error_counts) == 0
        assert len(handler.recent_errors) == 0

    def test_init_custom(self):
        """Test ErrorHandler initialization with custom values."""
        handler = ErrorHandler(enable_fallback=False)

        assert handler.enable_fallback is False

    def test_categorize_error_timeout(self):
        """Test error categorization for timeout errors."""
        handler = ErrorHandler()

        error = asyncio.TimeoutError("Operation timed out")
        details = handler.categorize_error(error)

        assert details.category == ErrorCategory.NETWORK
        assert details.severity == ErrorSeverity.MEDIUM
        assert details.is_recoverable is True

    def test_categorize_error_rate_limit(self):
        """Test error categorization for rate limit errors."""
        handler = ErrorHandler()

        error = Exception("Rate limit exceeded")
        details = handler.categorize_error(error)

        assert details.category == ErrorCategory.API_LIMIT
        assert details.severity == ErrorSeverity.HIGH

    def test_categorize_error_validation(self):
        """Test error categorization for validation errors."""
        handler = ErrorHandler()

        error = ValidationError("Invalid data")
        details = handler.categorize_error(error)

        assert details.category == ErrorCategory.VALIDATION
        assert details.severity == ErrorSeverity.LOW
        assert details.is_recoverable is False

    @pytest.mark.asyncio
    async def test_handle_with_retry_success(self):
        """Test handle_with_retry with successful operation."""
        handler = ErrorHandler()

        async def successful_operation():
            return "success"

        result = await handler.handle_with_retry(successful_operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_handle_with_retry_with_retry(self):
        """Test handle_with_retry with retry."""
        handler = ErrorHandler()
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NetworkError("Connection failed")
            return "success"

        result = await handler.handle_with_retry(flaky_operation)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_handle_bookmark_processing_error_fallback(self):
        """Test handle_bookmark_processing_error with fallback."""
        handler = ErrorHandler(enable_fallback=True)

        bookmark = Bookmark(
            url="https://example.com",
            title="Test Bookmark",
            note="Test note",
            folder="Test",
        )

        error = NetworkError("Connection failed")
        description, metadata = await handler.handle_bookmark_processing_error(
            error, bookmark
        )

        assert isinstance(description, str)
        assert len(description) > 0
        assert metadata["provider"] == "fallback"
        assert metadata["success"] is True

    def test_get_error_statistics(self):
        """Test error statistics generation."""
        handler = ErrorHandler()

        # Categorize some errors to populate statistics
        errors = [
            NetworkError("Network error 1"),
            NetworkError("Network error 2"),
            ValidationError("Validation error"),
        ]

        for error in errors:
            handler.categorize_error(error)
            handler._track_error(handler.categorize_error(error))

        stats = handler.get_error_statistics()

        assert isinstance(stats, dict)
        assert stats["total_errors"] > 0
        assert "error_counts_by_category" in stats
        assert "error_rates" in stats
        assert "recent_errors" in stats

    def test_reset_statistics(self):
        """Test resetting error statistics."""
        handler = ErrorHandler()

        # Add some errors
        error = NetworkError("Test error")
        handler._track_error(handler.categorize_error(error))

        assert len(handler.recent_errors) > 0

        handler.reset_statistics()
        assert len(handler.error_counts) == 0
        assert len(handler.recent_errors) == 0

    def test_get_health_status_healthy(self):
        """Test health status when no errors."""
        handler = ErrorHandler()

        status = handler.get_health_status()

        assert status["status"] == "healthy"
        assert status["recent_error_count"] == 0

    def test_get_health_status_with_errors(self):
        """Test health status with errors."""
        handler = ErrorHandler()

        # Add some errors
        for _ in range(5):
            error = NetworkError("Test error")
            handler._track_error(handler.categorize_error(error))

        status = handler.get_health_status()

        assert status["status"] in ["healthy", "stable", "concerning"]
        assert status["recent_error_count"] == 5


class TestRetryStrategy:
    """Test RetryStrategy class from error_handler."""

    def test_init_default(self):
        """Test RetryStrategy initialization with defaults."""
        strategy = RetryStrategy()

        assert strategy.max_attempts == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 60.0
        assert strategy.exponential_backoff is True
        assert strategy.jitter is True

    def test_init_custom(self):
        """Test RetryStrategy initialization with custom values."""
        strategy = RetryStrategy(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_backoff=False,
            jitter=False,
        )

        assert strategy.max_attempts == 5
        assert strategy.base_delay == 2.0
        assert strategy.max_delay == 120.0
        assert strategy.exponential_backoff is False
        assert strategy.jitter is False

    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        strategy = RetryStrategy(base_delay=1.0, exponential_backoff=True, jitter=False)

        assert strategy.get_delay(0) == 1.0  # 1.0 * 2^0
        assert strategy.get_delay(1) == 2.0  # 1.0 * 2^1
        assert strategy.get_delay(2) == 4.0  # 1.0 * 2^2

    def test_get_delay_linear(self):
        """Test linear backoff delay calculation."""
        strategy = RetryStrategy(base_delay=2.0, exponential_backoff=False, jitter=False)

        # Linear backoff always returns base_delay
        assert strategy.get_delay(0) == 2.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 2.0

    def test_should_retry_max_attempts(self):
        """Test should_retry with max attempts."""
        strategy = RetryStrategy(max_attempts=3)
        handler = ErrorHandler()
        error = handler.categorize_error(NetworkError("Test"))

        assert strategy.should_retry(0, error) is True
        assert strategy.should_retry(2, error) is True
        assert strategy.should_retry(3, error) is False


class TestRetryConfig:
    """Test RetryConfig class from retry_handler."""

    def test_init_default(self):
        """Test RetryConfig initialization with defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True

    def test_init_custom(self):
        """Test RetryConfig initialization with custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_multiplier=3.0,
            jitter=False,
        )

        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_multiplier == 3.0
        assert config.jitter is False


class TestBackoffStrategies:
    """Test backoff strategy classes."""

    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        backoff = ExponentialBackoff(base_delay=1.0, multiplier=2.0, max_delay=10.0, jitter=False)

        # Test delay calculation for different attempts
        # Implementation uses attempt directly as exponent
        assert backoff.get_delay(0) == 1.0  # base_delay * multiplier^0
        assert backoff.get_delay(1) == 2.0  # base_delay * multiplier^1
        assert backoff.get_delay(2) == 4.0  # base_delay * multiplier^2
        assert backoff.get_delay(3) == 8.0  # base_delay * multiplier^3
        assert backoff.get_delay(4) == 10.0  # Capped at max_delay
        assert backoff.get_delay(10) == 10.0  # Still capped

    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        backoff = LinearBackoff(base_delay=1.0, increment=0.5, max_delay=5.0)

        # Test delay calculation for different attempts
        # Implementation uses attempt directly
        assert backoff.get_delay(0) == 1.0  # base_delay + increment * 0
        assert backoff.get_delay(1) == 1.5  # base_delay + increment * 1
        assert backoff.get_delay(2) == 2.0  # base_delay + increment * 2
        assert backoff.get_delay(3) == 2.5  # base_delay + increment * 3
        assert backoff.get_delay(10) == 5.0  # Capped at max_delay

    def test_fixed_backoff(self):
        """Test fixed backoff strategy."""
        backoff = FixedBackoff(delay=2.5)

        # Should always return the same delay
        for attempt in range(0, 10):
            assert backoff.get_delay(attempt) == 2.5


class TestRetryHandler:
    """Test RetryHandler class for URL validation retries."""

    def test_init_default(self):
        """Test RetryHandler initialization with defaults."""
        handler = RetryHandler()

        assert handler.default_max_retries == 3
        assert handler.default_base_delay == 2.0
        assert len(handler.retry_queue) == 0
        assert len(handler.completed_retries) == 0

    def test_init_custom(self):
        """Test RetryHandler initialization with custom values."""
        handler = RetryHandler(default_max_retries=5, default_base_delay=1.0)

        assert handler.default_max_retries == 5
        assert handler.default_base_delay == 1.0

    def test_classify_error_timeout(self):
        """Test error classification for timeout."""
        handler = RetryHandler()

        import requests
        error = requests.exceptions.Timeout("Request timed out")
        error_type = handler._classify_error(error)

        assert error_type == ErrorType.TIMEOUT

    def test_classify_error_connection(self):
        """Test error classification for connection error."""
        handler = RetryHandler()

        import requests
        error = requests.exceptions.ConnectionError("Connection refused")
        error_type = handler._classify_error(error)

        assert error_type == ErrorType.CONNECTION_ERROR

    def test_add_failed_url(self):
        """Test adding a failed URL to retry queue."""
        handler = RetryHandler()

        import requests
        error = requests.exceptions.Timeout("Timeout")
        handler.add_failed_url("https://example.com", error)

        assert len(handler.retry_queue) == 1
        assert handler.retry_queue[0].url == "https://example.com"
        assert handler.retry_queue[0].error_type == ErrorType.TIMEOUT

    def test_add_failed_url_non_retryable(self):
        """Test adding a non-retryable error."""
        handler = RetryHandler()

        # Client errors have max_retries = 0 in the strategy
        import requests
        response = Mock()
        response.status_code = 404
        error = requests.exceptions.HTTPError("Not found")
        error.response = response

        handler.add_failed_url("https://example.com", error)

        # Should not be added to queue if max_retries is 0
        assert len(handler.retry_queue) == 0

    def test_get_retry_statistics_empty(self):
        """Test getting statistics when no retries."""
        handler = RetryHandler()

        stats = handler.get_retry_statistics()

        assert stats["total_items"] == 0

    def test_get_retry_statistics_with_items(self):
        """Test getting statistics with retry items."""
        handler = RetryHandler()

        import requests
        error = requests.exceptions.Timeout("Timeout")
        handler.add_failed_url("https://example1.com", error)
        handler.add_failed_url("https://example2.com", error)

        stats = handler.get_retry_statistics()

        assert stats["total_items"] == 2
        assert stats["remaining_in_queue"] == 2
        assert "error_type_distribution" in stats

    def test_clear_completed(self):
        """Test clearing completed retries."""
        handler = RetryHandler()

        import requests
        error = requests.exceptions.Timeout("Timeout")
        handler.add_failed_url("https://example.com", error)

        # Simulate completion
        item = handler.retry_queue[0]
        handler.completed_retries.append(item)

        assert len(handler.completed_retries) == 1

        handler.clear_completed()
        assert len(handler.completed_retries) == 0


if __name__ == "__main__":
    pytest.main([__file__])
