"""
Retry Handler Module

Provides intelligent retry logic with exponential backoff for failed URL validations.
Handles different types of errors with appropriate retry strategies.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests


class ErrorType(Enum):
    """Categories of errors for retry logic"""

    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    DNS_ERROR = "dns_error"
    SSL_ERROR = "ssl_error"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    UNKNOWN = "unknown"


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""

    timestamp: datetime
    error_type: ErrorType
    error_message: str
    response_code: Optional[int] = None
    duration: float = 0.0


@dataclass
class RetryableItem:
    """Item that can be retried with attempt history"""

    url: str
    original_error: Exception
    error_type: ErrorType
    attempts: List[RetryAttempt] = field(default_factory=list)
    max_retries: int = 3
    last_attempt: Optional[datetime] = None
    should_retry: bool = True

    def add_attempt(
        self,
        error_type: ErrorType,
        error_message: str,
        response_code: Optional[int] = None,
        duration: float = 0.0,
    ) -> None:
        """Add a retry attempt record"""
        attempt = RetryAttempt(
            timestamp=datetime.now(),
            error_type=error_type,
            error_message=error_message,
            response_code=response_code,
            duration=duration,
        )
        self.attempts.append(attempt)
        self.last_attempt = attempt.timestamp

        # Update retry eligibility
        if len(self.attempts) >= self.max_retries:
            self.should_retry = False

    @property
    def attempt_count(self) -> int:
        """Get number of retry attempts made"""
        return len(self.attempts)

    @property
    def next_retry_delay(self) -> float:
        """Calculate delay before next retry attempt"""
        if not self.should_retry:
            return 0.0

        # Exponential backoff with jitter
        base_delay = 2.0**self.attempt_count
        jitter = random.uniform(0.5, 1.5)
        return min(base_delay * jitter, 60.0)  # Cap at 60 seconds


class RetryHandler:
    """Handle URL validation retries with intelligent backoff"""

    # Error classification patterns
    ERROR_PATTERNS = {
        ErrorType.NETWORK_ERROR: ["network", "unreachable", "no route", "host down"],
        ErrorType.TIMEOUT: ["timeout", "timed out", "time out"],
        ErrorType.DNS_ERROR: ["dns", "name resolution", "hostname", "nodename"],
        ErrorType.SSL_ERROR: ["ssl", "certificate", "handshake", "tls"],
        ErrorType.CONNECTION_ERROR: ["connection", "refused", "reset", "broken pipe"],
        ErrorType.RATE_LIMITED: ["rate limit", "too many requests", "429"],
    }

    # Retry strategies per error type
    RETRY_STRATEGIES = {
        ErrorType.NETWORK_ERROR: {"max_retries": 2, "base_delay": 5.0},
        ErrorType.TIMEOUT: {"max_retries": 3, "base_delay": 3.0},
        ErrorType.DNS_ERROR: {"max_retries": 2, "base_delay": 2.0},
        ErrorType.SSL_ERROR: {"max_retries": 1, "base_delay": 1.0},
        ErrorType.CONNECTION_ERROR: {"max_retries": 3, "base_delay": 2.0},
        ErrorType.RATE_LIMITED: {"max_retries": 2, "base_delay": 10.0},
        ErrorType.SERVER_ERROR: {"max_retries": 2, "base_delay": 5.0},
        ErrorType.CLIENT_ERROR: {"max_retries": 0, "base_delay": 0.0},
        ErrorType.UNKNOWN: {"max_retries": 1, "base_delay": 3.0},
    }

    def __init__(self, default_max_retries: int = 3, default_base_delay: float = 2.0):
        """
        Initialize retry handler.

        Args:
            default_max_retries: Default maximum retry attempts
            default_base_delay: Default base delay for exponential backoff
        """
        self.default_max_retries = default_max_retries
        self.default_base_delay = default_base_delay
        self.retry_queue: List[RetryableItem] = []
        self.completed_retries: List[RetryableItem] = []

        logging.info(
            f"Initialized retry handler with max_retries={default_max_retries}"
        )

    def add_failed_url(
        self, url: str, error: Exception, custom_max_retries: Optional[int] = None
    ) -> None:
        """
        Add a failed URL to the retry queue.

        Args:
            url: The URL that failed
            error: The exception that occurred
            custom_max_retries: Override default max retries
        """
        error_type = self._classify_error(error)
        strategy = self.RETRY_STRATEGIES.get(error_type, {})

        max_retries = custom_max_retries or strategy.get(
            "max_retries", self.default_max_retries
        )

        if max_retries > 0:
            item = RetryableItem(
                url=url,
                original_error=error,
                error_type=error_type,
                max_retries=max_retries,
            )
            self.retry_queue.append(item)

            logging.debug(
                f"Added {url} to retry queue (error: {error_type.value}, "
                f"max_retries: {max_retries})"
            )
        else:
            logging.debug(
                f"Not retrying {url} (error: {error_type.value}, no retries allowed)"
            )

    def retry_failed_items(
        self,
        validator_func: Callable[[str], Any],
        progress_callback: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Retry all failed items using the provided validator function.

        Args:
            validator_func: Function to validate URLs (should handle exceptions)
            progress_callback: Optional callback for progress updates

        Returns:
            List of successful validation results
        """
        successful_results = []

        while self.retry_queue:
            # Get items ready for retry
            ready_items = self._get_ready_items()

            if not ready_items:
                # Wait for items to become ready
                min_wait = min(item.next_retry_delay for item in self.retry_queue)
                if min_wait > 0:
                    logging.info(f"Waiting {min_wait:.1f}s before next retry batch")
                    time.sleep(min(min_wait, 5.0))  # Don't wait more than 5s at a time
                continue

            logging.info(f"Retrying {len(ready_items)} failed URLs")

            for item in ready_items:
                self.retry_queue.remove(item)

                try:
                    start_time = time.time()
                    result = validator_func(item.url)
                    duration = time.time() - start_time

                    # Check if validation was successful
                    if self._is_validation_successful(result):
                        successful_results.append(result)
                        self.completed_retries.append(item)

                        logging.debug(f"Retry successful for {item.url}")

                        if progress_callback:
                            progress_callback(f"Retry successful: {item.url}")
                    else:
                        # Validation failed, add attempt and re-queue if appropriate
                        self._handle_retry_failure(item, result, duration)

                except Exception as e:
                    # Exception during retry
                    duration = time.time() - start_time
                    self._handle_retry_exception(item, e, duration)

        logging.info(
            f"Retry process completed. {len(successful_results)} URLs recovered."
        )
        return successful_results

    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get statistics about retry operations"""
        total_items = len(self.retry_queue) + len(self.completed_retries)

        if total_items == 0:
            return {"total_items": 0}

        error_type_counts = {}
        total_attempts = 0
        successful_retries = 0

        all_items = self.retry_queue + self.completed_retries

        for item in all_items:
            error_type_counts[item.error_type.value] = (
                error_type_counts.get(item.error_type.value, 0) + 1
            )
            total_attempts += item.attempt_count

            # Check if item was eventually successful
            if item in self.completed_retries:
                successful_retries += 1

        return {
            "total_items": total_items,
            "successful_retries": successful_retries,
            "remaining_in_queue": len(self.retry_queue),
            "total_attempts": total_attempts,
            "error_type_distribution": error_type_counts,
            "success_rate": (
                successful_retries / total_items if total_items > 0 else 0.0
            ),
        }

    def clear_completed(self) -> None:
        """Clear completed retry items to free memory"""
        self.completed_retries.clear()

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for retry strategy"""
        error_message = str(error).lower()

        # Check HTTP response codes first
        if hasattr(error, "response") and error.response:
            status_code = error.response.status_code
            if 500 <= status_code < 600:
                return ErrorType.SERVER_ERROR
            elif status_code == 429:
                return ErrorType.RATE_LIMITED
            elif 400 <= status_code < 500:
                return ErrorType.CLIENT_ERROR

        # Check exception types
        if isinstance(error, requests.exceptions.Timeout):
            return ErrorType.TIMEOUT
        elif isinstance(error, requests.exceptions.ConnectionError):
            if "ssl" in error_message or "certificate" in error_message:
                return ErrorType.SSL_ERROR
            elif "dns" in error_message or "name resolution" in error_message:
                return ErrorType.DNS_ERROR
            else:
                return ErrorType.CONNECTION_ERROR
        elif isinstance(error, requests.exceptions.HTTPError):
            return ErrorType.HTTP_ERROR

        # Check error message patterns
        for error_type, patterns in self.ERROR_PATTERNS.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type

        return ErrorType.UNKNOWN

    def _get_ready_items(self) -> List[RetryableItem]:
        """Get items that are ready for retry (delay period has passed)"""
        ready_items = []
        current_time = datetime.now()

        for item in self.retry_queue:
            if not item.should_retry:
                continue

            if item.last_attempt is None:
                # First retry attempt
                ready_items.append(item)
            else:
                # Check if enough time has passed
                time_since_last = (current_time - item.last_attempt).total_seconds()
                if time_since_last >= item.next_retry_delay:
                    ready_items.append(item)

        return ready_items

    def _is_validation_successful(self, result: Any) -> bool:
        """Determine if validation result indicates success"""
        # This depends on the validator function's return format
        # Assume it returns an object with is_valid or success attribute
        if hasattr(result, "is_valid"):
            return result.is_valid
        elif hasattr(result, "success"):
            return result.success
        elif isinstance(result, bool):
            return result
        elif isinstance(result, dict):
            return result.get("success", False) or result.get("is_valid", False)

        # Default: assume non-None result is success
        return result is not None

    def _handle_retry_failure(
        self, item: RetryableItem, result: Any, duration: float
    ) -> None:
        """Handle a failed retry attempt"""
        error_message = "Validation failed"
        response_code = None

        # Extract error details from result
        if hasattr(result, "error_message"):
            error_message = result.error_message
        if hasattr(result, "status_code"):
            response_code = result.status_code

        item.add_attempt(
            error_type=item.error_type,
            error_message=error_message,
            response_code=response_code,
            duration=duration,
        )

        if item.should_retry:
            self.retry_queue.append(item)
            logging.debug(
                f"Re-queued {item.url} for retry "
                f"(attempt {item.attempt_count}/{item.max_retries})"
            )
        else:
            logging.debug(
                f"Giving up on {item.url} after {item.attempt_count} attempts"
            )

    def _handle_retry_exception(
        self, item: RetryableItem, error: Exception, duration: float
    ) -> None:
        """Handle an exception during retry attempt"""
        error_type = self._classify_error(error)

        item.add_attempt(
            error_type=error_type, error_message=str(error), duration=duration
        )

        if item.should_retry:
            self.retry_queue.append(item)
            logging.debug(f"Exception during retry of {item.url}: {error}")
        else:
            logging.debug(f"Giving up on {item.url} after exception: {error}")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


class ExponentialBackoff:
    """Exponential backoff strategy."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = min(self.base_delay * (self.multiplier**attempt), self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay


class LinearBackoff:
    """Linear backoff strategy."""

    def __init__(
        self, base_delay: float = 1.0, increment: float = 1.0, max_delay: float = 60.0
    ):
        self.base_delay = base_delay
        self.increment = increment
        self.max_delay = max_delay

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        return min(self.base_delay + (self.increment * attempt), self.max_delay)


class FixedBackoff:
    """Fixed backoff strategy."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        return self.delay
