"""
URL Validation Module (Backward Compatibility Facade)

This module maintains backward compatibility by re-exporting all
classes and functions from the url_validator package.

The actual implementation has been decomposed into:
- url_validator/validator.py - Core synchronous validation
- url_validator/async_validator.py - Asynchronous validation
- url_validator/batch_interface.py - Batch processing interface
- url_validator/helpers.py - Utility functions

All imports from this module will continue to work as before.
"""

from __future__ import annotations

# Re-export everything from the package
from .url_validator import (
    AsyncHttpClient,
    BatchConfig,
    BatchProcessorInterface,
    BatchResult,
    CostBreakdown,
    EnhancedBatchProcessor,
    ProgressUpdate,
    SUCCESS_CODES,
    SKIP_PATTERNS,
    URLValidationError,
    URLValidator,
    ValidationError,
    ValidationResult,
    ValidationStats,
    classify_http_error,
    is_valid_url_format,
    parse_content_length,
    should_skip_url,
)

__all__ = [
    "ValidationError",
    "ValidationResult",
    "ValidationStats",
    "BatchConfig",
    "CostBreakdown",
    "ProgressUpdate",
    "BatchResult",
    "BatchProcessorInterface",
    "URLValidator",
    "EnhancedBatchProcessor",
    "AsyncHttpClient",
    "URLValidationError",
    "is_valid_url_format",
    "should_skip_url",
    "classify_http_error",
    "parse_content_length",
    "SKIP_PATTERNS",
    "SUCCESS_CODES",
]
