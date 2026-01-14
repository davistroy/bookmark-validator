"""
Input validation utilities for the Bookmark Processor.

This module provides backward-compatible validation functions for command-line
arguments and other user inputs. It re-exports functionality from the consolidated
validators package.

DEPRECATED: This module is maintained for backward compatibility.
New code should import directly from bookmark_processor.utils.validators
"""

import warnings

# Re-export exception class
from .validators.base import ValidationError

# Re-export all CLI validation functions
from .validators.cli_functions import (
    sanitize_input,
    validate_ai_engine,
    validate_auto_detection_mode,
    validate_batch_size,
    validate_bookmark_data,
    validate_config_file,
    validate_conflicting_arguments,
    validate_csv_structure,
    validate_input_file,
    validate_max_retries,
    validate_output_file,
)

# Re-export URL validation function
from .validators.url import validate_url_format

__all__ = [
    "ValidationError",
    "validate_input_file",
    "validate_auto_detection_mode",
    "validate_output_file",
    "validate_config_file",
    "validate_batch_size",
    "validate_max_retries",
    "validate_conflicting_arguments",
    "validate_ai_engine",
    "validate_csv_structure",
    "validate_bookmark_data",
    "validate_url_format",
    "sanitize_input",
]

# Issue deprecation warning
warnings.warn(
    "The validation module is deprecated. "
    "Use bookmark_processor.utils.validators instead.",
    DeprecationWarning,
    stacklevel=2,
)
