"""
Comprehensive Error Handling and Fallback System

This module provides robust error handling, graceful degradation, and fallback
strategies for the bookmark processing pipeline with cloud AI integration.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from bookmark_processor.core.data_models import Bookmark


# ============================================================================
# Unified Exception Hierarchy for Bookmark Processor
# ============================================================================
# All custom exceptions for the bookmark processor project are defined here.
# Import these exceptions from bookmark_processor.utils.error_handler
# ============================================================================


class BookmarkProcessorError(Exception):
    """Base exception for all bookmark processor errors."""

    pass


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(BookmarkProcessorError):
    """General validation errors."""

    pass


class URLValidationError(ValidationError):
    """URL-specific validation errors."""

    pass


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(BookmarkProcessorError):
    """Configuration-related errors."""

    pass


# ============================================================================
# Network Errors
# ============================================================================


class NetworkError(BookmarkProcessorError):
    """Network/HTTP related errors."""

    pass


# ============================================================================
# API Errors
# ============================================================================


class APIError(BookmarkProcessorError):
    """Base class for API-related errors."""

    pass


class APIClientError(APIError):
    """API client errors."""

    pass


class RateLimitError(APIClientError):
    """Rate limit exceeded errors."""

    pass


class AuthenticationError(APIClientError):
    """Authentication/authorization errors."""

    pass


class ServiceUnavailableError(APIClientError):
    """Service unavailable errors."""

    pass


# ============================================================================
# AI Processing Errors
# ============================================================================


class AIProcessingError(BookmarkProcessorError):
    """AI processing errors."""

    pass


class AISelectionError(AIProcessingError):
    """AI engine selection errors."""

    pass


# ============================================================================
# Data Errors
# ============================================================================


class DataError(BookmarkProcessorError):
    """Base class for data-related errors."""

    pass


class CSVError(DataError):
    """CSV file handling errors."""

    pass


class CSVStructureError(CSVError):
    """CSV structure errors."""

    pass


class CSVParsingError(CSVError):
    """CSV parsing errors."""

    pass


class CSVValidationError(CSVError):
    """CSV validation errors."""

    pass


class CSVEncodingError(CSVError):
    """CSV encoding errors."""

    pass


class CSVFormatError(CSVError):
    """CSV format errors."""

    pass


# ============================================================================
# Import/Export Errors
# ============================================================================


class ImportError(DataError):
    """Import-related errors (renamed to avoid conflict with builtin)."""

    pass


class UnsupportedFormatError(ImportError):
    """Unsupported format errors."""

    pass


class BookmarkImportError(ImportError):
    """Bookmark import errors."""

    pass


# ============================================================================
# HTML/Chrome Errors
# ============================================================================


class HTMLError(DataError):
    """HTML processing errors."""

    pass


class ChromeHTMLError(HTMLError):
    """Chrome HTML errors."""

    pass


class ChromeHTMLStructureError(ChromeHTMLError):
    """Chrome HTML structure errors."""

    pass


class ChromeHTMLGeneratorError(HTMLError):
    """Chrome HTML generator errors."""

    pass


# ============================================================================
# Processing Errors
# ============================================================================


class ProcessingError(BookmarkProcessorError):
    """General processing errors."""

    pass


class BatchProcessingError(ProcessingError):
    """Batch processing errors."""

    pass


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""

    LOW = "low"  # Non-critical, processing can continue normally
    MEDIUM = "medium"  # Concerning, may affect quality but processing continues
    HIGH = "high"  # Serious, immediate attention needed
    CRITICAL = "critical"  # Fatal, processing should stop


class ErrorCategory(Enum):
    """Categories of errors for proper handling strategies."""

    NETWORK = "network"  # Network connectivity issues
    API_AUTH = "api_auth"  # API authentication/authorization failures
    API_LIMIT = "api_limit"  # Rate limiting or quota exceeded
    API_ERROR = "api_error"  # General API errors
    VALIDATION = "validation"  # Data validation errors
    PROCESSING = "processing"  # Processing logic errors
    CONFIGURATION = "configuration"  # Configuration/setup errors
    SYSTEM = "system"  # System resource errors
    DATA = "data"  # Data-related errors
    UNKNOWN = "unknown"  # Unknown or unclassified errors


@dataclass
class ErrorDetails:
    """Detailed error information for tracking and analysis."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: float = None
    retry_count: int = 0
    is_recoverable: bool = True

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RetryStrategy:
    """Configurable retry strategy for different error types."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.exponential_backoff:
            delay = self.base_delay * (2**attempt)
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

        return delay

    def should_retry(self, attempt: int, error: ErrorDetails) -> bool:
        """Determine if we should retry based on attempt count and error type."""
        if attempt >= self.max_attempts:
            return False

        if not error.is_recoverable:
            return False

        # Don't retry authentication errors
        if error.category == ErrorCategory.API_AUTH:
            return False

        # Don't retry validation errors
        if error.category == ErrorCategory.VALIDATION:
            return False

        return True


class FallbackStrategy:
    """Defines fallback behavior for different failure scenarios."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def create_basic_description(
        self, bookmark: Bookmark
    ) -> Tuple[str, Dict[str, Any]]:
        """Create a basic description when all AI processing fails."""
        title = getattr(bookmark, "title", "") or "Untitled"
        url = getattr(bookmark, "url", "") or "No URL"
        existing_note = getattr(bookmark, "note", "") or ""
        existing_excerpt = getattr(bookmark, "excerpt", "") or ""

        # Use existing content if available
        if existing_note and existing_note.strip():
            description = existing_note.strip()
        elif existing_excerpt and existing_excerpt.strip():
            description = existing_excerpt.strip()
        else:
            # Create minimal description from title
            if title and title.lower() != "untitled":
                description = f"Bookmark for {title}"
            else:
                # Extract domain from URL as last resort
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(url).netloc
                    if domain:
                        description = f"Bookmark from {domain}"
                    else:
                        description = "Saved bookmark"
                except Exception:
                    description = "Saved bookmark"

        # Limit description length
        if len(description) > 150:
            description = description[:147] + "..."

        metadata = {
            "provider": "fallback",
            "method": "basic_description",
            "success": True,
            "fallback_reason": "AI processing unavailable",
        }

        return description, metadata

    async def handle_api_fallback(
        self,
        primary_error: ErrorDetails,
        bookmark: Bookmark,
        existing_content: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle fallback when API processing fails."""
        self.logger.warning(f"API fallback triggered: {primary_error.message}")

        # Try to use existing content first
        existing_note = getattr(bookmark, "note", "") or ""
        existing_excerpt = getattr(bookmark, "excerpt", "") or ""

        if existing_content and existing_content.strip():
            description = existing_content.strip()
            method = "existing_provided_content"
        elif existing_note and existing_note.strip():
            description = existing_note.strip()
            method = "existing_note"
        elif existing_excerpt and existing_excerpt.strip():
            description = existing_excerpt.strip()
            method = "existing_excerpt"
        else:
            # Fall back to basic description
            return await self.create_basic_description(bookmark)

        # Ensure reasonable length
        if len(description) > 150:
            description = description[:147] + "..."

        metadata = {
            "provider": "fallback",
            "method": method,
            "success": True,
            "fallback_reason": f"API failed: {primary_error.category.value}",
            "primary_error": primary_error.message,
        }

        return description, metadata


class ErrorHandler:
    """Comprehensive error handler with retry logic and fallback strategies."""

    # Default retry strategies for different error categories
    DEFAULT_RETRY_STRATEGIES = {
        ErrorCategory.NETWORK: RetryStrategy(max_attempts=3, base_delay=2.0),
        ErrorCategory.API_LIMIT: RetryStrategy(
            max_attempts=5, base_delay=5.0, max_delay=120.0
        ),
        ErrorCategory.API_ERROR: RetryStrategy(max_attempts=2, base_delay=1.0),
        ErrorCategory.PROCESSING: RetryStrategy(max_attempts=2, base_delay=0.5),
        ErrorCategory.SYSTEM: RetryStrategy(max_attempts=3, base_delay=1.0),
        # Non-retryable by default
        ErrorCategory.API_AUTH: RetryStrategy(max_attempts=1),
        ErrorCategory.VALIDATION: RetryStrategy(max_attempts=1),
        ErrorCategory.CONFIGURATION: RetryStrategy(max_attempts=1),
    }

    def __init__(self, enable_fallback: bool = True):
        self.enable_fallback = enable_fallback
        self.fallback_strategy = FallbackStrategy()
        self.error_counts = {}
        self.recent_errors = []
        self.logger = logging.getLogger(__name__)

    def categorize_error(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorDetails:
        """Categorize an exception into structured error details."""
        error_msg = str(exception)
        error_msg_lower = error_msg.lower()

        # Determine category based on exception type and message
        if isinstance(exception, asyncio.TimeoutError) or "timeout" in error_msg_lower:
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
        elif (
            "rate limit" in error_msg_lower
            or "quota" in error_msg_lower
            or "429" in error_msg_lower
        ):
            category = ErrorCategory.API_LIMIT
            severity = ErrorSeverity.HIGH
        elif (
            "unauthorized" in error_msg_lower
            or "403" in error_msg_lower
            or "401" in error_msg_lower
        ):
            category = ErrorCategory.API_AUTH
            severity = ErrorSeverity.CRITICAL
            is_recoverable = False
        elif "api key" in error_msg_lower or "authentication" in error_msg_lower:
            category = ErrorCategory.API_AUTH
            severity = ErrorSeverity.CRITICAL
            is_recoverable = False
        elif (
            "network" in error_msg_lower
            or "connection" in error_msg_lower
            or "dns" in error_msg_lower
        ):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
        elif (
            "500" in error_msg_lower
            or "502" in error_msg_lower
            or "503" in error_msg_lower
        ):
            category = ErrorCategory.API_ERROR
            severity = ErrorSeverity.HIGH
        elif "validation" in error_msg_lower or "invalid" in error_msg_lower:
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.LOW
            is_recoverable = False
        elif "memory" in error_msg_lower or "resource" in error_msg_lower:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.HIGH
        else:
            category = ErrorCategory.PROCESSING
            severity = ErrorSeverity.MEDIUM

        # Override recoverable flag if set above
        is_recoverable = locals().get("is_recoverable", True)

        return ErrorDetails(
            category=category,
            severity=severity,
            message=error_msg,
            original_exception=exception,
            context=context,
            is_recoverable=is_recoverable,
        )

    async def handle_with_retry(
        self,
        operation,
        operation_args: Tuple = (),
        operation_kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        custom_retry_strategy: Optional[RetryStrategy] = None,
    ) -> Any:
        """Execute an operation with retry logic based on error categorization."""
        if operation_kwargs is None:
            operation_kwargs = {}

        attempt = 0
        last_error = None

        while True:
            try:
                result = await operation(*operation_args, **operation_kwargs)

                # Log successful retry if this wasn't the first attempt
                if attempt > 0:
                    self.logger.info(f"Operation succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                attempt += 1
                error_details = self.categorize_error(e, context)
                error_details.retry_count = attempt
                last_error = error_details

                # Track error for statistics
                self._track_error(error_details)

                # Determine retry strategy
                retry_strategy = (
                    custom_retry_strategy
                    or self.DEFAULT_RETRY_STRATEGIES.get(
                        error_details.category, RetryStrategy(max_attempts=1)
                    )
                )

                # Check if we should retry
                if not retry_strategy.should_retry(attempt, error_details):
                    self.logger.error(
                        f"Operation failed after {attempt} attempts: "
                        f"{error_details.message}"
                    )
                    break

                # Calculate delay and wait
                delay = retry_strategy.get_delay(attempt - 1)
                self.logger.warning(
                    f"Attempt {attempt} failed "
                    f"({error_details.category.value}): {error_details.message}. "
                    f"Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

        # All retries exhausted, raise the last error
        raise last_error.original_exception

    async def handle_bookmark_processing_error(
        self,
        error: Exception,
        bookmark: Bookmark,
        existing_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle errors during bookmark processing with appropriate fallback."""
        error_details = self.categorize_error(error, context)
        self._track_error(error_details)

        self.logger.error(
            f"Bookmark processing error "
            f"({error_details.category.value}): {error_details.message}"
        )

        if not self.enable_fallback:
            raise error

        # Apply fallback strategy
        try:
            return await self.fallback_strategy.handle_api_fallback(
                error_details, bookmark, existing_content
            )
        except Exception as fallback_error:
            self.logger.error(f"Fallback strategy failed: {fallback_error}")

            # Last resort: basic description
            return await self.fallback_strategy.create_basic_description(bookmark)

    def _track_error(self, error_details: ErrorDetails) -> None:
        """Track error for statistics and monitoring."""
        # Count errors by category
        category_key = error_details.category.value
        if category_key not in self.error_counts:
            self.error_counts[category_key] = 0
        self.error_counts[category_key] += 1

        # Keep recent errors (limit to last 100)
        self.recent_errors.append(error_details)
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        # Calculate error rates by category
        total_errors = sum(self.error_counts.values())
        error_rates = {}

        for category, count in self.error_counts.items():
            error_rates[category] = {
                "count": count,
                "percentage": (count / max(total_errors, 1)) * 100,
            }

        # Recent error analysis (last 10 errors)
        recent_analysis = []
        for error in self.recent_errors[-10:]:
            recent_analysis.append(
                {
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "timestamp": error.timestamp,
                    "retry_count": error.retry_count,
                    "recoverable": error.is_recoverable,
                }
            )

        return {
            "total_errors": total_errors,
            "error_counts_by_category": self.error_counts,
            "error_rates": error_rates,
            "recent_errors": recent_analysis,
            "fallback_enabled": self.enable_fallback,
            "high_severity_count": sum(
                1
                for e in self.recent_errors
                if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            ),
        }

    def reset_statistics(self) -> None:
        """Reset error tracking statistics."""
        self.error_counts.clear()
        self.recent_errors.clear()
        self.logger.info("Error handler statistics reset")

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status based on recent errors."""
        recent_count = len(self.recent_errors)

        if recent_count == 0:
            status = "healthy"
            message = "No recent errors"
        else:
            # Check for critical errors in last 10
            recent_critical = sum(
                1
                for e in self.recent_errors[-10:]
                if e.severity == ErrorSeverity.CRITICAL
            )

            recent_high = sum(
                1 for e in self.recent_errors[-10:] if e.severity == ErrorSeverity.HIGH
            )

            if recent_critical > 0:
                status = "critical"
                message = f"{recent_critical} critical errors in recent activity"
            elif recent_high > 3:
                status = "degraded"
                message = f"{recent_high} high-severity errors in recent activity"
            elif recent_count > 20:
                status = "concerning"
                message = f"High error rate: {recent_count} recent errors"
            else:
                status = "stable"
                message = f"{recent_count} minor errors, system stable"

        return {
            "status": status,
            "message": message,
            "recent_error_count": recent_count,
            "fallback_available": self.enable_fallback,
        }


# Global error handler instance
_global_error_handler = None


def get_error_handler(enable_fallback: bool = True) -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler

    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(enable_fallback)

    return _global_error_handler


def reset_error_handler() -> None:
    """Reset the global error handler (useful for testing)."""
    global _global_error_handler
    _global_error_handler = None
