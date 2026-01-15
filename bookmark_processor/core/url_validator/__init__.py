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


# Lazy import of EnhancedBatchProcessor to avoid circular import
def _get_enhanced_batch_processor():
    """Lazy import of EnhancedBatchProcessor."""
    # Import at runtime to avoid circular dependency
    import importlib.util
    import sys
    import os

    # Get the path to batch_validator.py
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    batch_validator_path = os.path.join(core_dir, 'batch_validator.py')

    # Check if already loaded
    module_name = 'bookmark_processor.core._batch_validator'
    if module_name in sys.modules:
        return sys.modules[module_name].EnhancedBatchProcessor

    # Load from file path with proper package info for relative imports
    spec = importlib.util.spec_from_file_location(
        module_name,
        batch_validator_path,
        submodule_search_locations=[]
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        # Set up module's package info so relative imports work
        module.__package__ = 'bookmark_processor.core'
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.EnhancedBatchProcessor

    raise ImportError("Could not load EnhancedBatchProcessor")


# Module-level __getattr__ for lazy loading
_lazy_imports = {
    "EnhancedBatchProcessor": _get_enhanced_batch_processor,
}


def __getattr__(name):
    if name in _lazy_imports:
        return _lazy_imports[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
