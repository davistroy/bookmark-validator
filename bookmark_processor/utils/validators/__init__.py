"""
Validators Package

Consolidated validation modules for the Bookmark Processor.
This package provides all validation functionality in an organized structure.
"""

# Base classes
from .base import (
    CompositeValidator,
    ValidationError,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    Validator,
)

# Primitive validators
from .primitives import (
    DateTimeValidator,
    ListValidator,
    NumberValidator,
    StringValidator,
    create_datetime_validator,
    create_folder_validator,
    create_optional_string,
    create_required_string,
    create_tags_validator,
    create_title_validator,
)

# Security validators
from .security import SecurityValidationResult, SecurityValidator

# URL validators
from .url import BookmarkURLValidator, URLValidator, create_url_validator, validate_url_format

__all__ = [
    # Base classes
    "Validator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationError",
    "CompositeValidator",
    # Primitive validators
    "StringValidator",
    "NumberValidator",
    "DateTimeValidator",
    "ListValidator",
    # Primitive validator factories
    "create_required_string",
    "create_optional_string",
    "create_datetime_validator",
    "create_folder_validator",
    "create_title_validator",
    "create_tags_validator",
    # Security
    "SecurityValidator",
    "SecurityValidationResult",
    # URL validators
    "URLValidator",
    "BookmarkURLValidator",
    "create_url_validator",
    "validate_url_format",
]
