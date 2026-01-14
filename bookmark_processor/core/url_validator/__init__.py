"""
URL Validator Package

This package provides URL validation functionality with support for:
- Synchronous and asynchronous validation
- Batch processing with intelligent optimization
- Rate limiting and retry logic
- Security validation

The package is organized into:
- validator: Core synchronous URL validation
- async_validator: Asynchronous validation methods
- batch_interface: Batch processing interface implementation
- helpers: Utility functions for URL validation

Usage:
    from bookmark_processor.core.url_validator import URLValidator

    validator = URLValidator(timeout=30, max_concurrent=10)
    result = validator.validate_url("https://example.com")
"""

from __future__ import annotations

from typing import Type

# Import the base validator
from .validator import URLValidator as _BaseURLValidator

# Import mixins
from .async_validator import AsyncValidatorMixin
from .batch_interface import BatchProcessorMixin

# Import helpers (for external use if needed)
from .helpers import (
    SKIP_PATTERNS,
    SUCCESS_CODES,
    classify_http_error,
    is_valid_url_format,
    parse_content_length,
    should_skip_url,
)


# Create the complete URLValidator class by combining base and mixins
class URLValidator(
    _BaseURLValidator,
    AsyncValidatorMixin,
    BatchProcessorMixin,
):
    """
    Complete URL validator with sync/async validation and batch processing.

    This class combines:
    - Base synchronous validation from URLValidator
    - Asynchronous validation from AsyncValidatorMixin
    - Batch processing interface from BatchProcessorMixin

    Example:
        >>> validator = URLValidator(timeout=30, max_concurrent=10)
        >>> result = validator.validate_url("https://example.com")
        >>> print(f"Valid: {result.is_valid}")
    """

    pass


# Re-export EnhancedBatchProcessor for backward compatibility
from ..batch_validator import EnhancedBatchProcessor

# Re-export AsyncHttpClient for backward compatibility
from ..async_http_client import AsyncHttpClient

# Re-export batch types
from ..batch_types import (
    BatchConfig,
    BatchProcessorInterface,
    BatchResult,
    CostBreakdown,
    ProgressUpdate,
    ValidationResult,
    ValidationStats,
)

# Re-export error types
from ...utils.error_handler import URLValidationError, ValidationError

__all__ = [
    # Main class
    "URLValidator",
    # Batch processing
    "EnhancedBatchProcessor",
    "AsyncHttpClient",
    # Data types
    "ValidationResult",
    "ValidationStats",
    "BatchConfig",
    "BatchResult",
    "BatchProcessorInterface",
    "CostBreakdown",
    "ProgressUpdate",
    # Error types
    "ValidationError",
    "URLValidationError",
    # Helper functions
    "is_valid_url_format",
    "should_skip_url",
    "classify_http_error",
    "parse_content_length",
    "SKIP_PATTERNS",
    "SUCCESS_CODES",
]
