"""
Unit tests for error handling and retry utilities.

Tests for error handler, retry handler, and related utilities.
"""

import time
from typing import Any, Callable
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.utils.error_handler import (
    AIProcessingError,
    BookmarkProcessingError,
    ErrorHandler,
    NetworkError,
    ValidationError,
)
from bookmark_processor.utils.retry_handler import (
    ExponentialBackoff,
    FixedBackoff,
    LinearBackoff,
    RetryConfig,
    RetryHandler,
)


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_bookmark_processing_error(self):
        """Test BookmarkProcessingError exception."""
        error = BookmarkProcessingError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Validation failed", field="url")
        assert str(error) == "Validation failed"
        assert error.field == "url"
        assert isinstance(error, BookmarkProcessingError)

    def test_network_error(self):
        """Test NetworkError exception."""
        error = NetworkError(
            "Network failed", url="https://example.com", status_code=404
        )
        assert str(error) == "Network failed"
        assert error.url == "https://example.com"
        assert error.status_code == 404
        assert isinstance(error, BookmarkProcessingError)

    def test_ai_processing_error(self):
        """Test AIProcessingError exception."""
        error = AIProcessingError(
            "AI failed", model="test-model", input_text="test input"
        )
        assert str(error) == "AI failed"
        assert error.model == "test-model"
        assert error.input_text == "test input"
        assert isinstance(error, BookmarkProcessingError)


class TestErrorHandler:
    """Test ErrorHandler class."""

    def test_init_default(self):
        """Test ErrorHandler initialization with defaults."""
        handler = ErrorHandler()

        assert handler.max_errors == 100
        assert handler.error_threshold == 0.2
        assert len(handler.errors) == 0
        assert handler.total_operations == 0

    def test_init_custom(self):
        """Test ErrorHandler initialization with custom values."""
        handler = ErrorHandler(max_errors=50, error_threshold=0.1)

        assert handler.max_errors == 50
        assert handler.error_threshold == 0.1

    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler()

        try:
            raise ValueError("Test error")
        except Exception as e:
            handler.handle_error(e, context="test_operation")

        assert len(handler.errors) == 1
        error_record = handler.errors[0]
        assert error_record["error_type"] == "ValueError"
        assert error_record["message"] == "Test error"
        assert error_record["context"] == "test_operation"
        assert "timestamp" in error_record
        assert "traceback" in error_record

    def test_handle_error_with_url(self):
        """Test error handling with URL context."""
        handler = ErrorHandler()

        try:
            raise NetworkError("Connection failed")
        except Exception as e:
            handler.handle_error(e, context="url_validation", url="https://example.com")

        error_record = handler.errors[0]
        assert error_record["url"] == "https://example.com"
        assert error_record["context"] == "url_validation"

    def test_handle_error_with_bookmark_id(self):
        """Test error handling with bookmark ID context."""
        handler = ErrorHandler()

        try:
            raise ValidationError("Invalid bookmark")
        except Exception as e:
            handler.handle_error(e, bookmark_id="123")

        error_record = handler.errors[0]
        assert error_record["bookmark_id"] == "123"

    def test_record_operation(self):
        """Test operation recording."""
        handler = ErrorHandler()

        assert handler.total_operations == 0

        handler.record_operation()
        assert handler.total_operations == 1

        handler.record_operation(count=5)
        assert handler.total_operations == 6

    def test_get_error_rate(self):
        """Test error rate calculation."""
        handler = ErrorHandler()

        # No operations yet
        assert handler.get_error_rate() == 0.0

        # Record operations and errors
        handler.record_operation(count=10)

        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                handler.handle_error(e)

        # Should be 3/10 = 0.3
        assert handler.get_error_rate() == 0.3

    def test_should_continue_below_threshold(self):
        """Test should_continue when below error threshold."""
        handler = ErrorHandler(error_threshold=0.5)

        # Record operations with errors below threshold
        handler.record_operation(count=10)
        for i in range(4):  # 40% error rate
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                handler.handle_error(e)

        assert handler.should_continue() is True

    def test_should_continue_above_threshold(self):
        """Test should_continue when above error threshold."""
        handler = ErrorHandler(error_threshold=0.3)

        # Record operations with errors above threshold
        handler.record_operation(count=10)
        for i in range(5):  # 50% error rate
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                handler.handle_error(e)

        assert handler.should_continue() is False

    def test_should_continue_max_errors(self):
        """Test should_continue when max errors reached."""
        handler = ErrorHandler(max_errors=3)

        # Record errors up to max
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                handler.handle_error(e)

        assert handler.should_continue() is True

        # One more error should trigger failure
        try:
            raise ValueError("Final error")
        except Exception as e:
            handler.handle_error(e)

        assert handler.should_continue() is False

    def test_get_error_summary(self):
        """Test error summary generation."""
        handler = ErrorHandler()

        # Record different types of errors
        errors = [
            ValueError("Error 1"),
            ValueError("Error 2"),
            NetworkError("Network failed"),
            AIProcessingError("AI failed"),
        ]

        for error in errors:
            try:
                raise error
            except Exception as e:
                handler.handle_error(e)

        summary = handler.get_error_summary()

        assert isinstance(summary, dict)
        assert summary["total_errors"] == 4
        assert "error_types" in summary
        assert summary["error_types"]["ValueError"] == 2
        assert summary["error_types"]["NetworkError"] == 1
        assert summary["error_types"]["AIProcessingError"] == 1

    def test_get_errors_by_type(self):
        """Test filtering errors by type."""
        handler = ErrorHandler()

        # Record different types of errors
        try:
            raise ValueError("Value error")
        except Exception as e:
            handler.handle_error(e)

        try:
            raise NetworkError("Network error")
        except Exception as e:
            handler.handle_error(e)

        value_errors = handler.get_errors_by_type("ValueError")
        assert len(value_errors) == 1
        assert value_errors[0]["message"] == "Value error"

        network_errors = handler.get_errors_by_type("NetworkError")
        assert len(network_errors) == 1
        assert network_errors[0]["message"] == "Network error"

        # Non-existent type
        missing_errors = handler.get_errors_by_type("MissingError")
        assert len(missing_errors) == 0

    def test_clear_errors(self):
        """Test clearing error history."""
        handler = ErrorHandler()

        # Record some errors
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                handler.handle_error(e)

        assert len(handler.errors) == 3

        handler.clear_errors()
        assert len(handler.errors) == 0
        # Operations count should remain
        assert handler.total_operations == 0


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_init_default(self):
        """Test RetryConfig initialization with defaults."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_strategy == "exponential"
        assert config.retry_on_status_codes == [408, 429, 500, 502, 503, 504]

    def test_init_custom(self):
        """Test RetryConfig initialization with custom values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_strategy="linear",
            retry_on_status_codes=[503, 504],
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_strategy == "linear"
        assert config.retry_on_status_codes == [503, 504]


class TestBackoffStrategies:
    """Test backoff strategy classes."""

    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        backoff = ExponentialBackoff(base_delay=1.0, multiplier=2.0, max_delay=10.0)

        # Test delay calculation for different attempts
        assert backoff.get_delay(1) == 1.0  # base_delay * multiplier^0
        assert backoff.get_delay(2) == 2.0  # base_delay * multiplier^1
        assert backoff.get_delay(3) == 4.0  # base_delay * multiplier^2
        assert backoff.get_delay(4) == 8.0  # base_delay * multiplier^3
        assert backoff.get_delay(5) == 10.0  # Capped at max_delay
        assert backoff.get_delay(10) == 10.0  # Still capped

    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        backoff = LinearBackoff(base_delay=1.0, increment=0.5, max_delay=5.0)

        # Test delay calculation for different attempts
        assert backoff.get_delay(1) == 1.0  # base_delay + increment * 0
        assert backoff.get_delay(2) == 1.5  # base_delay + increment * 1
        assert backoff.get_delay(3) == 2.0  # base_delay + increment * 2
        assert backoff.get_delay(4) == 2.5  # base_delay + increment * 3
        assert backoff.get_delay(10) == 5.0  # Capped at max_delay

    def test_fixed_backoff(self):
        """Test fixed backoff strategy."""
        backoff = FixedBackoff(delay=2.5)

        # Should always return the same delay
        for attempt in range(1, 10):
            assert backoff.get_delay(attempt) == 2.5


class TestRetryHandler:
    """Test RetryHandler class."""

    def test_init_default(self):
        """Test RetryHandler initialization with defaults."""
        handler = RetryHandler()

        assert isinstance(handler.config, RetryConfig)
        assert handler.config.max_attempts == 3
        assert handler.total_attempts == 0
        assert handler.successful_retries == 0

    def test_init_custom_config(self):
        """Test RetryHandler initialization with custom config."""
        config = RetryConfig(max_attempts=5, base_delay=2.0)
        handler = RetryHandler(config)

        assert handler.config.max_attempts == 5
        assert handler.config.base_delay == 2.0

    def test_should_retry_success(self):
        """Test should_retry decision for successful operation."""
        handler = RetryHandler()

        # Successful operations should not retry
        assert handler.should_retry(None, attempt=1) is False
        assert (
            handler.should_retry(ValueError("test"), attempt=1) is False
        )  # Non-retryable error

    def test_should_retry_retryable_error(self):
        """Test should_retry decision for retryable errors."""
        config = RetryConfig(retry_on_status_codes=[503, 504])
        handler = RetryHandler(config)

        # Network errors with retryable status codes
        retryable_error = NetworkError("Service unavailable", status_code=503)
        assert handler.should_retry(retryable_error, attempt=1) is True
        assert handler.should_retry(retryable_error, attempt=2) is True
        assert (
            handler.should_retry(retryable_error, attempt=3) is False
        )  # Max attempts reached

    def test_should_retry_non_retryable_error(self):
        """Test should_retry decision for non-retryable errors."""
        handler = RetryHandler()

        # Network errors with non-retryable status codes
        non_retryable_error = NetworkError("Not found", status_code=404)
        assert handler.should_retry(non_retryable_error, attempt=1) is False

        # Validation errors are generally not retryable
        validation_error = ValidationError("Invalid data")
        assert handler.should_retry(validation_error, attempt=1) is False

    def test_get_delay_exponential(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(backoff_strategy="exponential", base_delay=1.0)
        handler = RetryHandler(config)

        assert handler.get_delay(1) == 1.0
        assert handler.get_delay(2) == 2.0
        assert handler.get_delay(3) == 4.0

    def test_get_delay_linear(self):
        """Test delay calculation with linear backoff."""
        config = RetryConfig(backoff_strategy="linear", base_delay=1.0)
        handler = RetryHandler(config)

        assert handler.get_delay(1) == 1.0
        assert handler.get_delay(2) == 1.5
        assert handler.get_delay(3) == 2.0

    def test_get_delay_fixed(self):
        """Test delay calculation with fixed backoff."""
        config = RetryConfig(backoff_strategy="fixed", base_delay=2.0)
        handler = RetryHandler(config)

        assert handler.get_delay(1) == 2.0
        assert handler.get_delay(2) == 2.0
        assert handler.get_delay(3) == 2.0

    def test_execute_success_no_retry(self):
        """Test execute with successful operation (no retry needed)."""
        handler = RetryHandler()

        # Mock successful operation
        def successful_operation():
            return "success"

        result = handler.execute(successful_operation)

        assert result == "success"
        assert handler.total_attempts == 1
        assert handler.successful_retries == 0

    def test_execute_success_after_retry(self):
        """Test execute with success after retries."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)  # Fast retry for testing
        handler = RetryHandler(config)

        # Mock operation that fails twice then succeeds
        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise NetworkError("Service unavailable", status_code=503)
            return "success"

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = handler.execute(flaky_operation)

        assert result == "success"
        assert handler.total_attempts == 3
        assert handler.successful_retries == 1

    def test_execute_max_attempts_exceeded(self):
        """Test execute when max attempts are exceeded."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        handler = RetryHandler(config)

        # Mock operation that always fails with retryable error
        def always_fails():
            raise NetworkError("Service unavailable", status_code=503)

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(NetworkError):
                handler.execute(always_fails)

        assert handler.total_attempts == 2
        assert handler.successful_retries == 0

    def test_execute_non_retryable_error(self):
        """Test execute with non-retryable error."""
        handler = RetryHandler()

        # Mock operation that fails with non-retryable error
        def non_retryable_failure():
            raise NetworkError("Not found", status_code=404)

        with pytest.raises(NetworkError):
            handler.execute(non_retryable_failure)

        assert handler.total_attempts == 1  # Should not retry
        assert handler.successful_retries == 0

    def test_execute_with_args_kwargs(self):
        """Test execute with function arguments."""
        handler = RetryHandler()

        def operation_with_args(x, y, z=None):
            return f"{x}-{y}-{z}"

        result = handler.execute(operation_with_args, 1, 2, z=3)
        assert result == "1-2-3"

    def test_get_statistics(self):
        """Test getting retry statistics."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        handler = RetryHandler(config)

        # Perform some operations
        handler.execute(lambda: "success")  # Immediate success

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NetworkError("Service unavailable", status_code=503)
            return "success"

        with patch("time.sleep"):
            handler.execute(flaky_operation)  # Success after 1 retry

        stats = handler.get_statistics()

        assert isinstance(stats, dict)
        assert stats["total_operations"] == 2
        assert stats["total_attempts"] == 3  # 1 + 2 attempts
        assert stats["successful_retries"] == 1
        assert stats["average_attempts"] == 1.5  # 3/2

    def test_reset_statistics(self):
        """Test resetting retry statistics."""
        handler = RetryHandler()

        # Perform an operation
        handler.execute(lambda: "success")

        assert handler.total_attempts > 0

        # Reset and verify
        handler.reset_statistics()

        assert handler.total_attempts == 0
        assert handler.successful_retries == 0


if __name__ == "__main__":
    pytest.main([__file__])
