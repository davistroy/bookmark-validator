"""
Base Validation Classes

This module provides the foundation for all validators in the system,
including base classes, result structures, and common utilities.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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

    is_valid: bool = True
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


# Import unified ValidationError from error_handler
from ..error_handler import ValidationError


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
