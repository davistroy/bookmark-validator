"""
Input Data Validation Framework

This module provides a comprehensive validation framework for all user-provided data
and imported bookmarks. It includes base classes, interfaces, and specialized validators
for different data types.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from urllib.parse import urlparse

from bookmark_processor.utils.security_validator import (
    SecurityValidationResult,
    SecurityValidator,
)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""

    severity: ValidationSeverity
    message: str
    field_name: Optional[str] = None
    suggested_value: Optional[Any] = None
    error_code: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        if self.field_name:
            prefix += f" {self.field_name}:"
        return f"{prefix} {self.message}"


@dataclass
class ValidationResult:
    """Result of validation with detailed feedback"""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        field_name: Optional[str] = None,
        suggested_value: Optional[Any] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Add a validation issue"""
        self.issues.append(
            ValidationIssue(
                severity=severity,
                message=message,
                field_name=field_name,
                suggested_value=suggested_value,
                error_code=error_code,
            )
        )

        # Set validity based on severity
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False

    def add_info(self, message: str, field_name: Optional[str] = None) -> None:
        """Add an info-level issue"""
        self.add_issue(ValidationSeverity.INFO, message, field_name)

    def add_warning(self, message: str, field_name: Optional[str] = None) -> None:
        """Add a warning-level issue"""
        self.add_issue(ValidationSeverity.WARNING, message, field_name)

    def add_error(self, message: str, field_name: Optional[str] = None) -> None:
        """Add an error-level issue"""
        self.add_issue(ValidationSeverity.ERROR, message, field_name)

    def add_critical(self, message: str, field_name: Optional[str] = None) -> None:
        """Add a critical-level issue"""
        self.add_issue(ValidationSeverity.CRITICAL, message, field_name)

    def has_errors(self) -> bool:
        """Check if there are any error or critical issues"""
        return any(
            issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for issue in self.issues
        )

    def has_warnings(self) -> bool:
        """Check if there are any warning issues"""
        return any(
            issue.severity == ValidationSeverity.WARNING for issue in self.issues
        )

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error and critical issues"""
        return [
            issue
            for issue in self.issues
            if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        ]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning issues"""
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one"""
        result = ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            issues=self.issues + other.issues,
            sanitized_value=(
                other.sanitized_value
                if other.sanitized_value is not None
                else self.sanitized_value
            ),
            metadata={**self.metadata, **other.metadata},
        )
        return result


class Validator(ABC):
    """Abstract base class for all validators"""

    def __init__(
        self,
        field_name: Optional[str] = None,
        required: bool = False,
        allow_none: bool = True,
    ):
        """
        Initialize validator

        Args:
            field_name: Name of the field being validated (for error messages)
            required: Whether the field is required (cannot be None/empty)
            allow_none: Whether None values are allowed
        """
        self.field_name = field_name
        self.required = required
        self.allow_none = allow_none
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a value

        Args:
            value: Value to validate

        Returns:
            ValidationResult with validation details
        """
        pass

    def _check_required_and_none(self, value: Any) -> Optional[ValidationResult]:
        """
        Check required and None value constraints

        Args:
            value: Value to check

        Returns:
            ValidationResult if validation fails, None if checks pass
        """
        result = ValidationResult(is_valid=True)

        # Check None values
        if value is None:
            if not self.allow_none:
                result.add_error("None values are not allowed", self.field_name)
                return result
            if self.required:
                result.add_error("Field is required but received None", self.field_name)
                return result
            # None is allowed and field is not required
            result.sanitized_value = None
            return result

        # Check empty values for required fields
        if self.required:
            if isinstance(value, str) and not value.strip():
                result.add_error(
                    "Field is required but received empty string", self.field_name
                )
                return result
            if isinstance(value, (list, dict)) and len(value) == 0:
                result.add_error(
                    "Field is required but received empty collection", self.field_name
                )
                return result

        return None  # Checks passed

    def _sanitize_string(self, value: str) -> str:
        """
        Sanitize string value

        Args:
            value: String to sanitize

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)

        # Remove null bytes and control characters
        sanitized = value.replace("\x00", "").replace("\r", "")

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        return sanitized


class StringValidator(Validator):
    """Validator for string values"""

    def __init__(
        self,
        field_name: Optional[str] = None,
        required: bool = False,
        allow_none: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[str]] = None,
        strip_whitespace: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize string validator

        Args:
            field_name: Name of the field being validated
            required: Whether the field is required
            allow_none: Whether None values are allowed
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern the string must match
            allowed_values: List of allowed string values
            strip_whitespace: Whether to strip leading/trailing whitespace
            normalize_whitespace: Whether to normalize internal whitespace
        """
        super().__init__(field_name, required, allow_none)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_values = set(allowed_values) if allowed_values else None
        self.strip_whitespace = strip_whitespace
        self.normalize_whitespace = normalize_whitespace

    def validate(self, value: Any) -> ValidationResult:
        """Validate string value"""
        # Check basic requirements
        basic_check = self._check_required_and_none(value)
        if basic_check:
            return basic_check

        result = ValidationResult(is_valid=True)

        if value is None:
            result.sanitized_value = None
            return result

        # Convert to string
        if not isinstance(value, str):
            result.add_info(
                f"Converting {type(value).__name__} to string", self.field_name
            )
            value = str(value)

        # Sanitize the string
        if self.normalize_whitespace:
            value = self._sanitize_string(value)
        elif self.strip_whitespace:
            value = value.strip()

        result.sanitized_value = value

        # Check length constraints
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(
                f"String too short: {len(value)} < {self.min_length}", self.field_name
            )

        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(
                f"String too long: {len(value)} > {self.max_length}", self.field_name
            )

        # Check pattern
        if self.pattern and not self.pattern.match(value):
            result.add_error(f"String does not match required pattern", self.field_name)

        # Check allowed values
        if self.allowed_values and value not in self.allowed_values:
            result.add_error(
                f"String '{value}' not in allowed values: {list(self.allowed_values)}",
                self.field_name,
            )

        return result


class NumberValidator(Validator):
    """Validator for numeric values"""

    def __init__(
        self,
        field_name: Optional[str] = None,
        required: bool = False,
        allow_none: bool = True,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        integer_only: bool = False,
        positive_only: bool = False,
    ):
        """
        Initialize number validator

        Args:
            field_name: Name of the field being validated
            required: Whether the field is required
            allow_none: Whether None values are allowed
            min_value: Minimum numeric value
            max_value: Maximum numeric value
            integer_only: Whether only integers are allowed
            positive_only: Whether only positive numbers are allowed
        """
        super().__init__(field_name, required, allow_none)
        self.min_value = min_value
        self.max_value = max_value
        self.integer_only = integer_only
        self.positive_only = positive_only

    def validate(self, value: Any) -> ValidationResult:
        """Validate numeric value"""
        # Check basic requirements
        basic_check = self._check_required_and_none(value)
        if basic_check:
            return basic_check

        result = ValidationResult(is_valid=True)

        if value is None:
            result.sanitized_value = None
            return result

        # Try to convert to number
        numeric_value = None

        if isinstance(value, (int, float)):
            numeric_value = value
        elif isinstance(value, str):
            try:
                # Try integer first if integer_only is True
                if self.integer_only:
                    numeric_value = int(value)
                else:
                    # Try float
                    numeric_value = float(value)
                    # Convert to int if it's a whole number and integer_only is True
                    if self.integer_only and numeric_value.is_integer():
                        numeric_value = int(numeric_value)
            except ValueError:
                result.add_error(f"Cannot convert '{value}' to number", self.field_name)
                return result
        else:
            result.add_error(
                f"Cannot convert {type(value).__name__} to number", self.field_name
            )
            return result

        result.sanitized_value = numeric_value

        # Check integer constraint
        if self.integer_only and not isinstance(numeric_value, int):
            result.add_error("Only integer values are allowed", self.field_name)

        # Check positive constraint
        if self.positive_only and numeric_value <= 0:
            result.add_error("Only positive values are allowed", self.field_name)

        # Check range constraints
        if self.min_value is not None and numeric_value < self.min_value:
            result.add_error(
                f"Value {numeric_value} is below minimum {self.min_value}",
                self.field_name,
            )

        if self.max_value is not None and numeric_value > self.max_value:
            result.add_error(
                f"Value {numeric_value} is above maximum {self.max_value}",
                self.field_name,
            )

        return result


class DateTimeValidator(Validator):
    """Validator for datetime values"""

    def __init__(
        self,
        field_name: Optional[str] = None,
        required: bool = False,
        allow_none: bool = True,
        formats: Optional[List[str]] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
    ):
        """
        Initialize datetime validator

        Args:
            field_name: Name of the field being validated
            required: Whether the field is required
            allow_none: Whether None values are allowed
            formats: List of acceptable datetime format strings
            min_date: Minimum allowed date
            max_date: Maximum allowed date
        """
        super().__init__(field_name, required, allow_none)
        self.formats = formats or [
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format with Z
            "%Y-%m-%dT%H:%M:%S",  # ISO format
            "%Y-%m-%d %H:%M:%S",  # Standard format
            "%Y-%m-%d",  # Date only
            "%m/%d/%Y",  # US format
            "%d/%m/%Y",  # European format
        ]
        self.min_date = min_date
        self.max_date = max_date

    def validate(self, value: Any) -> ValidationResult:
        """Validate datetime value"""
        # Check basic requirements
        basic_check = self._check_required_and_none(value)
        if basic_check:
            return basic_check

        result = ValidationResult(is_valid=True)

        if value is None:
            result.sanitized_value = None
            return result

        # If already a datetime object
        if isinstance(value, datetime):
            result.sanitized_value = value
        else:
            # Try to parse string datetime
            if not isinstance(value, str):
                value = str(value)

            value = value.strip()
            parsed_datetime = None

            for fmt in self.formats:
                try:
                    parsed_datetime = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue

            if parsed_datetime is None:
                result.add_error(
                    f"Cannot parse datetime '{value}' with any known format",
                    self.field_name,
                )
                return result

            result.sanitized_value = parsed_datetime

        # Check date range constraints
        if self.min_date and result.sanitized_value < self.min_date:
            result.add_error(
                f"Date {result.sanitized_value} is before minimum {self.min_date}",
                self.field_name,
            )

        if self.max_date and result.sanitized_value > self.max_date:
            result.add_error(
                f"Date {result.sanitized_value} is after maximum {self.max_date}",
                self.field_name,
            )

        return result


class URLValidator(Validator):
    """Validator for URL values with security checks"""

    def __init__(
        self,
        field_name: Optional[str] = None,
        required: bool = False,
        allow_none: bool = True,
        allowed_schemes: Optional[List[str]] = None,
        security_check: bool = True,
        normalize_url: bool = True,
    ):
        """
        Initialize URL validator

        Args:
            field_name: Name of the field being validated
            required: Whether the field is required
            allow_none: Whether None values are allowed
            allowed_schemes: List of allowed URL schemes
            security_check: Whether to perform security validation
            normalize_url: Whether to normalize the URL
        """
        super().__init__(field_name, required, allow_none)
        self.allowed_schemes = (
            set(allowed_schemes) if allowed_schemes else {"http", "https"}
        )
        self.security_check = security_check
        self.normalize_url = normalize_url
        self.security_validator = SecurityValidator() if security_check else None

    def validate(self, value: Any) -> ValidationResult:
        """Validate URL value"""
        # Check basic requirements
        basic_check = self._check_required_and_none(value)
        if basic_check:
            return basic_check

        result = ValidationResult(is_valid=True)

        if value is None:
            result.sanitized_value = None
            return result

        # Convert to string and clean
        if not isinstance(value, str):
            value = str(value)

        value = value.strip()
        result.sanitized_value = value

        if not value:
            if self.required:
                result.add_error("URL cannot be empty", self.field_name)
            return result

        # Basic URL format validation
        try:
            parsed = urlparse(value)
        except Exception as e:
            result.add_error(f"Invalid URL format: {e}", self.field_name)
            return result

        # Check scheme
        if not parsed.scheme:
            result.add_warning("URL missing scheme, assuming https://", self.field_name)
            value = f"https://{value}"
            result.sanitized_value = value
            try:
                parsed = urlparse(value)
            except Exception as e:
                result.add_error(
                    f"Invalid URL after adding scheme: {e}", self.field_name
                )
                return result

        if parsed.scheme.lower() not in self.allowed_schemes:
            result.add_error(
                f"URL scheme '{parsed.scheme}' not allowed. Allowed: {list(self.allowed_schemes)}",
                self.field_name,
            )

        # Check netloc (domain)
        if not parsed.netloc:
            result.add_error("URL missing domain/hostname", self.field_name)

        # Security validation
        if self.security_check and self.security_validator:
            security_result = self.security_validator.validate_url_security(value)
            if not security_result.is_safe:
                result.add_critical(
                    f"Security validation failed: {security_result.blocked_reason}",
                    self.field_name,
                )
                result.metadata["security_issues"] = security_result.issues
            elif security_result.issues:
                result.add_warning(
                    f"Security concerns: {', '.join(security_result.issues)}",
                    self.field_name,
                )
                result.metadata["security_issues"] = security_result.issues

        # URL normalization
        if self.normalize_url and result.is_valid:
            try:
                # Basic normalization: lowercase domain, remove trailing slash
                normalized = f"{parsed.scheme}://{parsed.netloc.lower()}"
                if parsed.path and parsed.path != "/":
                    normalized += parsed.path.rstrip("/")
                elif not parsed.path:
                    normalized += "/"

                if parsed.query:
                    normalized += f"?{parsed.query}"
                if parsed.fragment:
                    normalized += f"#{parsed.fragment}"

                result.sanitized_value = normalized

            except Exception as e:
                result.add_warning(f"URL normalization failed: {e}", self.field_name)

        return result


class ListValidator(Validator):
    """Validator for list values"""

    def __init__(
        self,
        field_name: Optional[str] = None,
        required: bool = False,
        allow_none: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        item_validator: Optional[Validator] = None,
        unique_items: bool = False,
    ):
        """
        Initialize list validator

        Args:
            field_name: Name of the field being validated
            required: Whether the field is required
            allow_none: Whether None values are allowed
            min_length: Minimum list length
            max_length: Maximum list length
            item_validator: Validator for individual list items
            unique_items: Whether list items must be unique
        """
        super().__init__(field_name, required, allow_none)
        self.min_length = min_length
        self.max_length = max_length
        self.item_validator = item_validator
        self.unique_items = unique_items

    def validate(self, value: Any) -> ValidationResult:
        """Validate list value"""
        # Check basic requirements
        basic_check = self._check_required_and_none(value)
        if basic_check:
            return basic_check

        result = ValidationResult(is_valid=True)

        if value is None:
            result.sanitized_value = None
            return result

        # Convert to list if needed
        if not isinstance(value, list):
            if isinstance(value, str):
                # Try to parse as comma-separated values
                items = [item.strip() for item in value.split(",") if item.strip()]
                result.add_info(
                    "Converted comma-separated string to list", self.field_name
                )
            elif hasattr(value, "__iter__"):
                items = list(value)
                result.add_info(
                    f"Converted {type(value).__name__} to list", self.field_name
                )
            else:
                items = [value]
                result.add_info(
                    f"Wrapped single {type(value).__name__} in list", self.field_name
                )
        else:
            items = value[:]  # Copy the list

        # Check length constraints
        if self.min_length is not None and len(items) < self.min_length:
            result.add_error(
                f"List too short: {len(items)} < {self.min_length}", self.field_name
            )

        if self.max_length is not None and len(items) > self.max_length:
            result.add_error(
                f"List too long: {len(items)} > {self.max_length}", self.field_name
            )

        # Validate individual items
        validated_items = []
        if self.item_validator:
            for i, item in enumerate(items):
                item_result = self.item_validator.validate(item)
                if item_result.has_errors():
                    for issue in item_result.get_errors():
                        result.add_error(f"Item {i}: {issue.message}", self.field_name)
                else:
                    validated_items.append(item_result.sanitized_value)

                # Add warnings from item validation
                for warning in item_result.get_warnings():
                    result.add_warning(f"Item {i}: {warning.message}", self.field_name)
        else:
            validated_items = items

        # Check uniqueness
        if self.unique_items:
            original_length = len(validated_items)
            validated_items = list(
                dict.fromkeys(validated_items)
            )  # Preserve order while removing duplicates
            if len(validated_items) < original_length:
                duplicates_removed = original_length - len(validated_items)
                result.add_warning(
                    f"Removed {duplicates_removed} duplicate items", self.field_name
                )

        result.sanitized_value = validated_items
        return result


class CompositeValidator(Validator):
    """Validator that applies multiple validators in sequence"""

    def __init__(
        self,
        validators: List[Validator],
        field_name: Optional[str] = None,
        stop_on_error: bool = False,
    ):
        """
        Initialize composite validator

        Args:
            validators: List of validators to apply
            field_name: Name of the field being validated
            stop_on_error: Whether to stop validation on first error
        """
        super().__init__(
            field_name, False, True
        )  # Let individual validators handle requirements
        self.validators = validators
        self.stop_on_error = stop_on_error

    def validate(self, value: Any) -> ValidationResult:
        """Apply all validators in sequence"""
        result = ValidationResult(is_valid=True, sanitized_value=value)

        current_value = value

        for validator in self.validators:
            validator_result = validator.validate(current_value)
            result = result.merge(validator_result)

            # Use sanitized value from this validator for next validator
            if validator_result.sanitized_value is not None:
                current_value = validator_result.sanitized_value
                result.sanitized_value = current_value

            # Stop on error if requested
            if self.stop_on_error and validator_result.has_errors():
                break

        return result


# Factory functions for common validators
def create_required_string(
    field_name: str, max_length: Optional[int] = None
) -> StringValidator:
    """Create a required string validator"""
    return StringValidator(
        field_name=field_name, required=True, allow_none=False, max_length=max_length
    )


def create_optional_string(
    field_name: str, max_length: Optional[int] = None
) -> StringValidator:
    """Create an optional string validator"""
    return StringValidator(
        field_name=field_name, required=False, allow_none=True, max_length=max_length
    )


def create_url_validator(field_name: str, required: bool = True) -> URLValidator:
    """Create a URL validator with security checks"""
    return URLValidator(
        field_name=field_name,
        required=required,
        allow_none=not required,
        security_check=True,
    )


def create_tag_list_validator(field_name: str) -> ListValidator:
    """Create a validator for tag lists"""
    tag_validator = StringValidator(
        field_name="tag", required=True, allow_none=False, min_length=1, max_length=50
    )
    return ListValidator(
        field_name=field_name,
        required=False,
        allow_none=True,
        max_length=20,  # Reasonable max number of tags
        item_validator=tag_validator,
        unique_items=True,
    )


def create_datetime_validator(
    field_name: str, required: bool = False
) -> DateTimeValidator:
    """Create a datetime validator"""
    return DateTimeValidator(
        field_name=field_name, required=required, allow_none=not required
    )


def create_folder_validator(field_name: str) -> StringValidator:
    """Create a validator for folder paths"""
    return StringValidator(
        field_name=field_name,
        required=False,
        allow_none=True,
        max_length=500,
        pattern=r'^[^<>:"|?*\x00-\x1f]*$',  # Avoid filesystem-invalid characters
    )


def create_title_validator(field_name: str, required: bool = True) -> StringValidator:
    """Create a validator for bookmark titles."""
    return StringValidator(
        field_name=field_name,
        required=required,
        allow_none=not required,
        max_length=1000,
        min_length=1 if required else 0,
        pattern=r"^.+$",  # Allow any non-empty content
    )


def create_tags_validator(field_name: str) -> StringValidator:
    """Create a validator for bookmark tags."""
    return StringValidator(
        field_name=field_name,
        required=False,
        allow_none=True,
        max_length=2000,
        pattern=r"^[^<>]*$",  # Avoid HTML-like content
    )


class InputValidator:
    """General-purpose input validator that combines multiple validation strategies."""

    def __init__(self):
        self.url_validator = create_url_validator("url", required=True)
        self.title_validator = create_title_validator("title", required=True)
        self.tags_validator = create_tags_validator("tags")
        self.folder_validator = create_folder_validator("folder")

    def validate_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate input data using appropriate validators.

        Args:
            input_data: Dictionary containing input data to validate

        Returns:
            ValidationResult containing all validation issues
        """
        issues = []

        # Validate URL if present
        if "url" in input_data:
            url_result = self.url_validator.validate(input_data["url"])
            issues.extend(url_result.issues)

        # Validate title if present
        if "title" in input_data:
            title_result = self.title_validator.validate(input_data["title"])
            issues.extend(title_result.issues)

        # Validate tags if present
        if "tags" in input_data:
            tags_result = self.tags_validator.validate(input_data["tags"])
            issues.extend(tags_result.issues)

        # Validate folder if present
        if "folder" in input_data:
            folder_result = self.folder_validator.validate(input_data["folder"])
            issues.extend(folder_result.issues)

        return ValidationResult(issues=issues)

    def is_valid(self, input_data: Dict[str, Any]) -> bool:
        """Check if input data is valid."""
        result = self.validate_input(input_data)
        return result.is_valid
