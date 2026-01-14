"""
Primitive Validators

This module provides validators for primitive types: strings, numbers,
dates, and lists.
"""

import re
from datetime import datetime
from typing import Any, List, Optional, Union

from .base import ValidationResult, Validator


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
            result.add_error("String does not match required pattern", self.field_name)

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
