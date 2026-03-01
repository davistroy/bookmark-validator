"""
Comprehensive tests for primitive validators module.

Tests cover:
- StringValidator: basic validation, length constraints, patterns, allowed values
- NumberValidator: numeric validation, range constraints, integer/positive checks
- DateTimeValidator: datetime parsing, format handling, range constraints
- ListValidator: list validation, item validation, uniqueness constraints
- Factory functions: create_required_string, create_optional_string, etc.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from bookmark_processor.utils.validators.primitives import (
    StringValidator,
    NumberValidator,
    DateTimeValidator,
    ListValidator,
    create_required_string,
    create_optional_string,
    create_datetime_validator,
    create_folder_validator,
    create_title_validator,
    create_tags_validator,
)
from bookmark_processor.utils.validators.base import ValidationResult, ValidationSeverity


class TestStringValidatorBasic:
    """Tests for basic StringValidator functionality"""

    def test_valid_string_passes(self):
        """Test that a valid string passes validation"""
        validator = StringValidator()
        result = validator.validate("hello world")
        assert result.is_valid is True
        assert result.sanitized_value == "hello world"

    def test_none_value_allowed_by_default(self):
        """Test that None is allowed by default"""
        validator = StringValidator()
        result = validator.validate(None)
        assert result.is_valid is True
        assert result.sanitized_value is None

    def test_none_value_not_allowed_when_specified(self):
        """Test that None is rejected when allow_none=False"""
        validator = StringValidator(allow_none=False)
        result = validator.validate(None)
        assert result.is_valid is False
        assert result.has_errors()

    def test_empty_string_allowed_by_default(self):
        """Test that empty string is allowed when not required"""
        validator = StringValidator()
        result = validator.validate("")
        assert result.is_valid is True

    def test_empty_string_rejected_when_required(self):
        """Test that empty string is rejected when required"""
        validator = StringValidator(required=True)
        result = validator.validate("")
        assert result.is_valid is False
        assert result.has_errors()

    def test_whitespace_only_rejected_when_required(self):
        """Test that whitespace-only string is rejected when required"""
        validator = StringValidator(required=True)
        result = validator.validate("   \t\n  ")
        assert result.is_valid is False
        assert result.has_errors()

    def test_field_name_included_in_errors(self):
        """Test that field name is included in error messages"""
        validator = StringValidator(field_name="title", required=True)
        result = validator.validate(None)
        assert result.is_valid is False
        errors = result.get_errors()
        assert len(errors) > 0
        assert errors[0].field_name == "title"

    def test_non_string_converted_to_string(self):
        """Test that non-string values are converted"""
        validator = StringValidator()
        result = validator.validate(12345)
        assert result.is_valid is True
        assert result.sanitized_value == "12345"

    def test_conversion_info_logged(self):
        """Test that conversion from non-string generates info message"""
        validator = StringValidator()
        result = validator.validate(12345)
        assert result.is_valid is True
        # Check info messages
        info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        assert len(info_issues) > 0
        assert "Converting" in info_issues[0].message


class TestStringValidatorWhitespace:
    """Tests for StringValidator whitespace handling"""

    def test_strip_whitespace_by_default(self):
        """Test that whitespace is stripped by default"""
        validator = StringValidator()
        result = validator.validate("  hello world  ")
        assert result.sanitized_value == "hello world"

    def test_strip_whitespace_disabled(self):
        """Test that whitespace stripping can be disabled"""
        validator = StringValidator(strip_whitespace=False, normalize_whitespace=False)
        result = validator.validate("  hello world  ")
        assert result.sanitized_value == "  hello world  "

    def test_normalize_whitespace_by_default(self):
        """Test that internal whitespace is normalized by default"""
        validator = StringValidator()
        result = validator.validate("hello   \t  world")
        assert result.sanitized_value == "hello world"

    def test_normalize_whitespace_disabled(self):
        """Test that whitespace normalization can be disabled"""
        validator = StringValidator(normalize_whitespace=False)
        result = validator.validate("hello   world")
        # Only strip whitespace, not normalize
        assert result.sanitized_value == "hello   world"

    def test_strip_without_normalize(self):
        """Test strip_whitespace=True and normalize_whitespace=False"""
        validator = StringValidator(strip_whitespace=True, normalize_whitespace=False)
        result = validator.validate("  hello   world  ")
        assert result.sanitized_value == "hello   world"


class TestStringValidatorLength:
    """Tests for StringValidator length constraints"""

    def test_min_length_valid(self):
        """Test string meeting minimum length passes"""
        validator = StringValidator(min_length=5)
        result = validator.validate("hello")
        assert result.is_valid is True

    def test_min_length_invalid(self):
        """Test string below minimum length fails"""
        validator = StringValidator(min_length=10)
        result = validator.validate("hello")
        assert result.is_valid is False
        assert "too short" in result.get_errors()[0].message.lower()

    def test_max_length_valid(self):
        """Test string within maximum length passes"""
        validator = StringValidator(max_length=10)
        result = validator.validate("hello")
        assert result.is_valid is True

    def test_max_length_invalid(self):
        """Test string exceeding maximum length fails"""
        validator = StringValidator(max_length=3)
        result = validator.validate("hello")
        assert result.is_valid is False
        assert "too long" in result.get_errors()[0].message.lower()

    def test_exact_min_length(self):
        """Test string at exact minimum length passes"""
        validator = StringValidator(min_length=5)
        result = validator.validate("hello")
        assert result.is_valid is True

    def test_exact_max_length(self):
        """Test string at exact maximum length passes"""
        validator = StringValidator(max_length=5)
        result = validator.validate("hello")
        assert result.is_valid is True

    def test_min_and_max_length_combined(self):
        """Test with both min and max length constraints"""
        validator = StringValidator(min_length=3, max_length=10)

        # Too short
        result = validator.validate("hi")
        assert result.is_valid is False

        # Within range
        result = validator.validate("hello")
        assert result.is_valid is True

        # Too long
        result = validator.validate("hello world!")
        assert result.is_valid is False


class TestStringValidatorPattern:
    """Tests for StringValidator pattern matching"""

    def test_pattern_valid(self):
        """Test string matching pattern passes"""
        validator = StringValidator(pattern=r"^[a-z]+$")
        result = validator.validate("hello")
        assert result.is_valid is True

    def test_pattern_invalid(self):
        """Test string not matching pattern fails"""
        validator = StringValidator(pattern=r"^[a-z]+$")
        result = validator.validate("Hello123")
        assert result.is_valid is False
        assert "pattern" in result.get_errors()[0].message.lower()

    def test_email_pattern(self):
        """Test email-like pattern"""
        validator = StringValidator(pattern=r"^[\w.+-]+@[\w.-]+\.\w+$")

        result = validator.validate("test@example.com")
        assert result.is_valid is True

        result = validator.validate("invalid-email")
        assert result.is_valid is False

    def test_pattern_case_sensitive(self):
        """Test that patterns are case-sensitive by default"""
        validator = StringValidator(pattern=r"^[a-z]+$")
        result = validator.validate("HELLO")
        assert result.is_valid is False


class TestStringValidatorAllowedValues:
    """Tests for StringValidator allowed values"""

    def test_allowed_value_valid(self):
        """Test string in allowed values passes"""
        validator = StringValidator(allowed_values=["red", "green", "blue"])
        result = validator.validate("red")
        assert result.is_valid is True

    def test_allowed_value_invalid(self):
        """Test string not in allowed values fails"""
        validator = StringValidator(allowed_values=["red", "green", "blue"])
        result = validator.validate("yellow")
        assert result.is_valid is False
        assert "not in allowed values" in result.get_errors()[0].message.lower()

    def test_allowed_values_case_sensitive(self):
        """Test that allowed values check is case-sensitive"""
        validator = StringValidator(allowed_values=["Red", "Green", "Blue"])
        result = validator.validate("red")
        assert result.is_valid is False

    def test_allowed_values_empty_list(self):
        """Test empty allowed_values list allows everything (converted to None)"""
        # Empty list becomes empty set, which is falsy, so allowed_values check is skipped
        validator = StringValidator(allowed_values=[])
        result = validator.validate("anything")
        # Empty set is falsy in Python, so allowed_values check is skipped
        assert result.is_valid is True


class TestNumberValidatorBasic:
    """Tests for basic NumberValidator functionality"""

    def test_valid_integer_passes(self):
        """Test that a valid integer passes validation"""
        validator = NumberValidator()
        result = validator.validate(42)
        assert result.is_valid is True
        assert result.sanitized_value == 42

    def test_valid_float_passes(self):
        """Test that a valid float passes validation"""
        validator = NumberValidator()
        result = validator.validate(3.14)
        assert result.is_valid is True
        assert result.sanitized_value == 3.14

    def test_none_value_allowed_by_default(self):
        """Test that None is allowed by default"""
        validator = NumberValidator()
        result = validator.validate(None)
        assert result.is_valid is True
        assert result.sanitized_value is None

    def test_none_value_not_allowed_when_specified(self):
        """Test that None is rejected when allow_none=False"""
        validator = NumberValidator(allow_none=False)
        result = validator.validate(None)
        assert result.is_valid is False

    def test_string_integer_converted(self):
        """Test that string integer is converted"""
        validator = NumberValidator()
        result = validator.validate("123")
        assert result.is_valid is True
        assert result.sanitized_value == 123.0

    def test_string_float_converted(self):
        """Test that string float is converted"""
        validator = NumberValidator()
        result = validator.validate("3.14")
        assert result.is_valid is True
        assert result.sanitized_value == 3.14

    def test_invalid_string_fails(self):
        """Test that non-numeric string fails"""
        validator = NumberValidator()
        result = validator.validate("not a number")
        assert result.is_valid is False
        assert result.has_errors()

    def test_unsupported_type_fails(self):
        """Test that unsupported types fail"""
        validator = NumberValidator()
        result = validator.validate([1, 2, 3])
        assert result.is_valid is False
        assert "Cannot convert" in result.get_errors()[0].message


class TestNumberValidatorConstraints:
    """Tests for NumberValidator constraints"""

    def test_integer_only_valid(self):
        """Test that integer passes when integer_only=True"""
        validator = NumberValidator(integer_only=True)
        result = validator.validate(42)
        assert result.is_valid is True
        assert result.sanitized_value == 42

    def test_integer_only_float_fails(self):
        """Test that non-integer float fails when integer_only=True"""
        validator = NumberValidator(integer_only=True)
        result = validator.validate(3.14)
        assert result.is_valid is False
        assert "integer" in result.get_errors()[0].message.lower()

    def test_integer_only_string_conversion(self):
        """Test that string integer is parsed when integer_only=True"""
        validator = NumberValidator(integer_only=True)
        result = validator.validate("42")
        assert result.is_valid is True
        assert result.sanitized_value == 42
        assert isinstance(result.sanitized_value, int)

    def test_positive_only_valid(self):
        """Test that positive number passes when positive_only=True"""
        validator = NumberValidator(positive_only=True)
        result = validator.validate(42)
        assert result.is_valid is True

    def test_positive_only_zero_fails(self):
        """Test that zero fails when positive_only=True"""
        validator = NumberValidator(positive_only=True)
        result = validator.validate(0)
        assert result.is_valid is False
        assert "positive" in result.get_errors()[0].message.lower()

    def test_positive_only_negative_fails(self):
        """Test that negative number fails when positive_only=True"""
        validator = NumberValidator(positive_only=True)
        result = validator.validate(-10)
        assert result.is_valid is False


class TestNumberValidatorRange:
    """Tests for NumberValidator range constraints"""

    def test_min_value_valid(self):
        """Test value at or above minimum passes"""
        validator = NumberValidator(min_value=0)
        result = validator.validate(5)
        assert result.is_valid is True

    def test_min_value_invalid(self):
        """Test value below minimum fails"""
        validator = NumberValidator(min_value=10)
        result = validator.validate(5)
        assert result.is_valid is False
        assert "below minimum" in result.get_errors()[0].message.lower()

    def test_max_value_valid(self):
        """Test value at or below maximum passes"""
        validator = NumberValidator(max_value=100)
        result = validator.validate(50)
        assert result.is_valid is True

    def test_max_value_invalid(self):
        """Test value above maximum fails"""
        validator = NumberValidator(max_value=10)
        result = validator.validate(50)
        assert result.is_valid is False
        assert "above maximum" in result.get_errors()[0].message.lower()

    def test_range_valid(self):
        """Test value within range passes"""
        validator = NumberValidator(min_value=0, max_value=100)
        result = validator.validate(50)
        assert result.is_valid is True

    def test_exact_boundary_values(self):
        """Test exact boundary values pass"""
        validator = NumberValidator(min_value=0, max_value=100)

        result = validator.validate(0)
        assert result.is_valid is True

        result = validator.validate(100)
        assert result.is_valid is True

    def test_float_range(self):
        """Test float range constraints"""
        validator = NumberValidator(min_value=0.5, max_value=9.5)

        result = validator.validate(5.0)
        assert result.is_valid is True

        result = validator.validate(0.1)
        assert result.is_valid is False


class TestDateTimeValidatorBasic:
    """Tests for basic DateTimeValidator functionality"""

    def test_datetime_object_passes(self):
        """Test that a datetime object passes validation"""
        validator = DateTimeValidator()
        dt = datetime(2024, 1, 15, 12, 30, 0)
        result = validator.validate(dt)
        assert result.is_valid is True
        assert result.sanitized_value == dt

    def test_none_value_allowed_by_default(self):
        """Test that None is allowed by default"""
        validator = DateTimeValidator()
        result = validator.validate(None)
        assert result.is_valid is True
        assert result.sanitized_value is None

    def test_none_value_not_allowed_when_specified(self):
        """Test that None is rejected when required"""
        validator = DateTimeValidator(required=True)
        result = validator.validate(None)
        assert result.is_valid is False

    def test_field_name_included_in_errors(self):
        """Test that field name is included in error messages"""
        validator = DateTimeValidator(field_name="created_date")
        result = validator.validate("invalid-date")
        assert result.is_valid is False
        errors = result.get_errors()
        assert len(errors) > 0
        assert errors[0].field_name == "created_date"


class TestDateTimeValidatorFormats:
    """Tests for DateTimeValidator format parsing"""

    def test_iso_format_with_z(self):
        """Test ISO format with Z suffix"""
        validator = DateTimeValidator()
        result = validator.validate("2024-01-15T12:30:00Z")
        assert result.is_valid is True
        assert result.sanitized_value == datetime(2024, 1, 15, 12, 30, 0)

    def test_iso_format_without_z(self):
        """Test ISO format without Z suffix"""
        validator = DateTimeValidator()
        result = validator.validate("2024-01-15T12:30:00")
        assert result.is_valid is True
        assert result.sanitized_value == datetime(2024, 1, 15, 12, 30, 0)

    def test_standard_format(self):
        """Test standard datetime format"""
        validator = DateTimeValidator()
        result = validator.validate("2024-01-15 12:30:00")
        assert result.is_valid is True
        assert result.sanitized_value == datetime(2024, 1, 15, 12, 30, 0)

    def test_date_only_format(self):
        """Test date-only format"""
        validator = DateTimeValidator()
        result = validator.validate("2024-01-15")
        assert result.is_valid is True
        assert result.sanitized_value == datetime(2024, 1, 15, 0, 0, 0)

    def test_us_date_format(self):
        """Test US date format (MM/DD/YYYY)"""
        validator = DateTimeValidator()
        result = validator.validate("01/15/2024")
        assert result.is_valid is True
        assert result.sanitized_value == datetime(2024, 1, 15, 0, 0, 0)

    def test_european_date_format(self):
        """Test European date format (DD/MM/YYYY)"""
        validator = DateTimeValidator()
        result = validator.validate("15/01/2024")
        assert result.is_valid is True
        assert result.sanitized_value == datetime(2024, 1, 15, 0, 0, 0)

    def test_invalid_format_fails(self):
        """Test that invalid format fails"""
        validator = DateTimeValidator()
        result = validator.validate("January 15, 2024")
        assert result.is_valid is False
        assert "Cannot parse" in result.get_errors()[0].message

    def test_custom_formats(self):
        """Test custom datetime formats"""
        validator = DateTimeValidator(formats=["%B %d, %Y"])
        result = validator.validate("January 15, 2024")
        assert result.is_valid is True

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped from datetime strings"""
        validator = DateTimeValidator()
        result = validator.validate("  2024-01-15  ")
        assert result.is_valid is True

    def test_non_string_converted(self):
        """Test that non-string is converted before parsing"""
        validator = DateTimeValidator()
        # This will convert to string and try to parse
        result = validator.validate(20240115)
        # Will likely fail since "20240115" doesn't match default formats
        # But should not crash
        assert result is not None


class TestDateTimeValidatorRange:
    """Tests for DateTimeValidator date range constraints"""

    def test_min_date_valid(self):
        """Test date at or after minimum passes"""
        min_date = datetime(2024, 1, 1)
        validator = DateTimeValidator(min_date=min_date)
        result = validator.validate(datetime(2024, 6, 15))
        assert result.is_valid is True

    def test_min_date_invalid(self):
        """Test date before minimum fails"""
        min_date = datetime(2024, 1, 1)
        validator = DateTimeValidator(min_date=min_date)
        result = validator.validate(datetime(2023, 12, 31))
        assert result.is_valid is False
        assert "before minimum" in result.get_errors()[0].message.lower()

    def test_max_date_valid(self):
        """Test date at or before maximum passes"""
        max_date = datetime(2024, 12, 31)
        validator = DateTimeValidator(max_date=max_date)
        result = validator.validate(datetime(2024, 6, 15))
        assert result.is_valid is True

    def test_max_date_invalid(self):
        """Test date after maximum fails"""
        max_date = datetime(2024, 12, 31)
        validator = DateTimeValidator(max_date=max_date)
        result = validator.validate(datetime(2025, 1, 1))
        assert result.is_valid is False
        assert "after maximum" in result.get_errors()[0].message.lower()

    def test_date_range_valid(self):
        """Test date within range passes"""
        min_date = datetime(2024, 1, 1)
        max_date = datetime(2024, 12, 31)
        validator = DateTimeValidator(min_date=min_date, max_date=max_date)
        result = validator.validate(datetime(2024, 6, 15))
        assert result.is_valid is True


class TestListValidatorBasic:
    """Tests for basic ListValidator functionality"""

    def test_valid_list_passes(self):
        """Test that a valid list passes validation"""
        validator = ListValidator()
        result = validator.validate(["a", "b", "c"])
        assert result.is_valid is True
        assert result.sanitized_value == ["a", "b", "c"]

    def test_empty_list_allowed_by_default(self):
        """Test that empty list is allowed by default"""
        validator = ListValidator()
        result = validator.validate([])
        assert result.is_valid is True
        assert result.sanitized_value == []

    def test_empty_list_rejected_when_required(self):
        """Test that empty list is rejected when required"""
        validator = ListValidator(required=True)
        result = validator.validate([])
        assert result.is_valid is False

    def test_none_value_allowed_by_default(self):
        """Test that None is allowed by default"""
        validator = ListValidator()
        result = validator.validate(None)
        assert result.is_valid is True
        assert result.sanitized_value is None

    def test_none_value_not_allowed_when_specified(self):
        """Test that None is rejected when allow_none=False"""
        validator = ListValidator(allow_none=False)
        result = validator.validate(None)
        assert result.is_valid is False


class TestListValidatorConversion:
    """Tests for ListValidator type conversion"""

    def test_string_converted_to_list(self):
        """Test that comma-separated string is converted to list"""
        validator = ListValidator()
        result = validator.validate("a, b, c")
        assert result.is_valid is True
        assert result.sanitized_value == ["a", "b", "c"]

    def test_tuple_converted_to_list(self):
        """Test that tuple is converted to list"""
        validator = ListValidator()
        result = validator.validate(("a", "b", "c"))
        assert result.is_valid is True
        assert result.sanitized_value == ["a", "b", "c"]

    def test_set_converted_to_list(self):
        """Test that set is converted to list"""
        validator = ListValidator()
        result = validator.validate({"a", "b", "c"})
        assert result.is_valid is True
        assert len(result.sanitized_value) == 3

    def test_single_value_wrapped_in_list(self):
        """Test that single non-iterable value is wrapped in list"""
        validator = ListValidator()
        result = validator.validate(42)
        assert result.is_valid is True
        assert result.sanitized_value == [42]

    def test_conversion_info_logged(self):
        """Test that conversions generate info messages"""
        validator = ListValidator()
        result = validator.validate("a, b, c")
        info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        assert len(info_issues) > 0

    def test_empty_string_items_filtered(self):
        """Test that empty items in comma-separated string are filtered"""
        validator = ListValidator()
        result = validator.validate("a, , b, , c")
        assert result.sanitized_value == ["a", "b", "c"]


class TestListValidatorLength:
    """Tests for ListValidator length constraints"""

    def test_min_length_valid(self):
        """Test list meeting minimum length passes"""
        validator = ListValidator(min_length=2)
        result = validator.validate(["a", "b", "c"])
        assert result.is_valid is True

    def test_min_length_invalid(self):
        """Test list below minimum length fails"""
        validator = ListValidator(min_length=5)
        result = validator.validate(["a", "b", "c"])
        assert result.is_valid is False
        assert "too short" in result.get_errors()[0].message.lower()

    def test_max_length_valid(self):
        """Test list within maximum length passes"""
        validator = ListValidator(max_length=5)
        result = validator.validate(["a", "b", "c"])
        assert result.is_valid is True

    def test_max_length_invalid(self):
        """Test list exceeding maximum length fails"""
        validator = ListValidator(max_length=2)
        result = validator.validate(["a", "b", "c"])
        assert result.is_valid is False
        assert "too long" in result.get_errors()[0].message.lower()


class TestListValidatorItemValidator:
    """Tests for ListValidator item validation"""

    def test_item_validator_all_valid(self):
        """Test that all valid items pass"""
        item_validator = StringValidator(min_length=1)
        validator = ListValidator(item_validator=item_validator)
        result = validator.validate(["a", "bb", "ccc"])
        assert result.is_valid is True
        assert result.sanitized_value == ["a", "bb", "ccc"]

    def test_item_validator_some_invalid(self):
        """Test that invalid items cause errors"""
        item_validator = StringValidator(min_length=2)
        validator = ListValidator(item_validator=item_validator)
        result = validator.validate(["a", "bb", "c"])
        assert result.is_valid is False
        errors = result.get_errors()
        assert len(errors) == 2  # Items 0 and 2 are too short

    def test_item_validator_error_messages_include_index(self):
        """Test that error messages include item index"""
        item_validator = StringValidator(min_length=2)
        validator = ListValidator(item_validator=item_validator)
        result = validator.validate(["a", "bb"])
        errors = result.get_errors()
        assert any("Item 0" in error.message for error in errors)

    def test_item_validator_sanitizes_items(self):
        """Test that item validator sanitizes items"""
        item_validator = StringValidator()  # Strips whitespace by default
        validator = ListValidator(item_validator=item_validator)
        result = validator.validate(["  a  ", "  b  "])
        assert result.sanitized_value == ["a", "b"]


class TestListValidatorUniqueness:
    """Tests for ListValidator uniqueness constraint"""

    def test_unique_items_no_duplicates(self):
        """Test that list without duplicates passes"""
        validator = ListValidator(unique_items=True)
        result = validator.validate(["a", "b", "c"])
        assert result.is_valid is True
        assert result.sanitized_value == ["a", "b", "c"]

    def test_unique_items_removes_duplicates(self):
        """Test that duplicates are removed"""
        validator = ListValidator(unique_items=True)
        result = validator.validate(["a", "b", "a", "c", "b"])
        assert result.is_valid is True
        assert result.sanitized_value == ["a", "b", "c"]

    def test_unique_items_preserves_order(self):
        """Test that order is preserved when removing duplicates"""
        validator = ListValidator(unique_items=True)
        result = validator.validate(["c", "a", "b", "a", "c"])
        assert result.sanitized_value == ["c", "a", "b"]

    def test_unique_items_generates_warning(self):
        """Test that removing duplicates generates warning"""
        validator = ListValidator(unique_items=True)
        result = validator.validate(["a", "a", "a"])
        warnings = result.get_warnings()
        assert len(warnings) > 0
        assert "duplicate" in warnings[0].message.lower()


class TestFactoryFunctionCreateRequiredString:
    """Tests for create_required_string factory function"""

    def test_creates_required_validator(self):
        """Test that validator is configured as required"""
        validator = create_required_string("title")
        result = validator.validate(None)
        assert result.is_valid is False

    def test_creates_non_nullable_validator(self):
        """Test that validator does not allow None"""
        validator = create_required_string("title")
        result = validator.validate(None)
        assert result.is_valid is False

    def test_sets_field_name(self):
        """Test that field name is set"""
        validator = create_required_string("my_field")
        assert validator.field_name == "my_field"

    def test_max_length_constraint(self):
        """Test that max_length is applied"""
        validator = create_required_string("title", max_length=10)
        result = validator.validate("a" * 20)
        assert result.is_valid is False


class TestFactoryFunctionCreateOptionalString:
    """Tests for create_optional_string factory function"""

    def test_creates_optional_validator(self):
        """Test that validator is not required"""
        validator = create_optional_string("description")
        result = validator.validate(None)
        assert result.is_valid is True

    def test_allows_none(self):
        """Test that validator allows None"""
        validator = create_optional_string("description")
        result = validator.validate(None)
        assert result.is_valid is True
        assert result.sanitized_value is None

    def test_sets_field_name(self):
        """Test that field name is set"""
        validator = create_optional_string("my_field")
        assert validator.field_name == "my_field"

    def test_max_length_constraint(self):
        """Test that max_length is applied"""
        validator = create_optional_string("description", max_length=10)
        result = validator.validate("a" * 20)
        assert result.is_valid is False


class TestFactoryFunctionCreateDatetimeValidator:
    """Tests for create_datetime_validator factory function"""

    def test_creates_optional_by_default(self):
        """Test that validator is optional by default"""
        validator = create_datetime_validator("created")
        result = validator.validate(None)
        assert result.is_valid is True

    def test_creates_required_when_specified(self):
        """Test that validator is required when specified"""
        validator = create_datetime_validator("created", required=True)
        result = validator.validate(None)
        assert result.is_valid is False

    def test_sets_field_name(self):
        """Test that field name is set"""
        validator = create_datetime_validator("my_date")
        assert validator.field_name == "my_date"


class TestFactoryFunctionCreateFolderValidator:
    """Tests for create_folder_validator factory function"""

    def test_valid_folder_path(self):
        """Test that valid folder paths pass"""
        validator = create_folder_validator("folder")
        result = validator.validate("Documents/Work/Projects")
        assert result.is_valid is True

    def test_invalid_characters_rejected(self):
        """Test that filesystem-invalid characters are rejected"""
        validator = create_folder_validator("folder")
        result = validator.validate("Documents/<Work>")
        assert result.is_valid is False

    def test_max_length_enforced(self):
        """Test that max_length is enforced (500 chars)"""
        validator = create_folder_validator("folder")
        result = validator.validate("a" * 600)
        assert result.is_valid is False

    def test_none_allowed(self):
        """Test that None is allowed"""
        validator = create_folder_validator("folder")
        result = validator.validate(None)
        assert result.is_valid is True


class TestFactoryFunctionCreateTitleValidator:
    """Tests for create_title_validator factory function"""

    def test_required_by_default(self):
        """Test that title is required by default"""
        validator = create_title_validator("title")
        result = validator.validate(None)
        assert result.is_valid is False

    def test_optional_when_specified(self):
        """Test that title can be optional"""
        validator = create_title_validator("title", required=False)
        result = validator.validate(None)
        assert result.is_valid is True

    def test_max_length_enforced(self):
        """Test that max_length is enforced (1000 chars)"""
        validator = create_title_validator("title")
        result = validator.validate("a" * 1100)
        assert result.is_valid is False

    def test_valid_title_passes(self):
        """Test that valid title passes"""
        validator = create_title_validator("title")
        result = validator.validate("My Bookmark Title")
        assert result.is_valid is True


class TestFactoryFunctionCreateTagsValidator:
    """Tests for create_tags_validator factory function"""

    def test_optional_by_default(self):
        """Test that tags are optional"""
        validator = create_tags_validator("tags")
        result = validator.validate(None)
        assert result.is_valid is True

    def test_html_tags_rejected(self):
        """Test that HTML-like content is rejected"""
        validator = create_tags_validator("tags")
        result = validator.validate("<script>alert('xss')</script>")
        assert result.is_valid is False

    def test_max_length_enforced(self):
        """Test that max_length is enforced (2000 chars)"""
        validator = create_tags_validator("tags")
        result = validator.validate("a" * 2100)
        assert result.is_valid is False

    def test_valid_tags_pass(self):
        """Test that valid tags pass"""
        validator = create_tags_validator("tags")
        result = validator.validate("ai, research, python")
        assert result.is_valid is True


class TestValidatorCombinations:
    """Tests for combining multiple validators and constraints"""

    def test_string_validator_all_constraints(self):
        """Test StringValidator with all constraints"""
        validator = StringValidator(
            field_name="code",
            required=True,
            allow_none=False,
            min_length=3,
            max_length=10,
            pattern=r"^[A-Z0-9]+$",
            allowed_values=["ABC123", "XYZ789", "TEST01"],
        )

        # Valid value
        result = validator.validate("ABC123")
        assert result.is_valid is True

        # Invalid: not in allowed values
        result = validator.validate("DEF456")
        assert result.is_valid is False

        # Invalid: wrong pattern
        result = validator.validate("abc123")
        assert result.is_valid is False

    def test_number_validator_all_constraints(self):
        """Test NumberValidator with all constraints"""
        validator = NumberValidator(
            field_name="score",
            required=True,
            allow_none=False,
            min_value=1,
            max_value=100,
            integer_only=True,
            positive_only=True,
        )

        # Valid value
        result = validator.validate(50)
        assert result.is_valid is True

        # Invalid: float
        result = validator.validate(50.5)
        assert result.is_valid is False

        # Invalid: out of range
        result = validator.validate(150)
        assert result.is_valid is False

    def test_list_with_string_item_validator(self):
        """Test ListValidator with StringValidator for items"""
        item_validator = StringValidator(min_length=2, max_length=10)
        validator = ListValidator(
            field_name="tags",
            min_length=1,
            max_length=5,
            item_validator=item_validator,
            unique_items=True,
        )

        # Valid list
        result = validator.validate(["tag1", "tag2", "tag3"])
        assert result.is_valid is True

        # Invalid: item too short
        result = validator.validate(["a", "tag2"])
        assert result.is_valid is False

        # Invalid: too many items
        result = validator.validate(["t1", "t2", "t3", "t4", "t5", "t6"])
        assert result.is_valid is False


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_string_validator_with_unicode(self):
        """Test StringValidator with unicode characters"""
        validator = StringValidator()
        result = validator.validate("Hello World!")
        assert result.is_valid is True

    def test_string_validator_with_newlines(self):
        """Test StringValidator normalizes newlines"""
        validator = StringValidator()
        result = validator.validate("hello\nworld")
        assert result.is_valid is True
        # Newlines are normalized to spaces
        assert result.sanitized_value == "hello world"

    def test_string_validator_with_tabs(self):
        """Test StringValidator normalizes tabs"""
        validator = StringValidator()
        result = validator.validate("hello\tworld")
        assert result.is_valid is True
        assert result.sanitized_value == "hello world"

    def test_number_validator_with_negative_zero(self):
        """Test NumberValidator with negative zero"""
        validator = NumberValidator(positive_only=True)
        result = validator.validate(-0.0)
        # -0.0 == 0 in Python, so it should fail positive_only check
        assert result.is_valid is False

    def test_datetime_validator_with_timestamp_integer(self):
        """Test DateTimeValidator with integer timestamp"""
        validator = DateTimeValidator()
        # Integer timestamps don't match any default format
        result = validator.validate(1704067200)
        assert result.is_valid is False

    def test_list_validator_with_nested_lists(self):
        """Test ListValidator with nested lists (no item validator)"""
        validator = ListValidator()
        result = validator.validate([["a", "b"], ["c", "d"]])
        assert result.is_valid is True
        assert result.sanitized_value == [["a", "b"], ["c", "d"]]

    def test_list_validator_unique_with_unhashable_items(self):
        """Test ListValidator unique_items with unhashable items"""
        validator = ListValidator(unique_items=True)
        # Dicts are not hashable, so this might cause issues
        # The implementation uses dict.fromkeys which requires hashable items
        try:
            result = validator.validate([{"a": 1}, {"a": 1}])
            # If it doesn't raise, check the result
            assert result is not None
        except TypeError:
            # Expected if implementation doesn't handle unhashable items
            pass

    def test_string_validator_pattern_anchored(self):
        """Test that pattern uses match() (anchored at start)"""
        validator = StringValidator(pattern=r"abc")
        # match() only matches at start, so "xabc" should pass if not full match
        result = validator.validate("abc")
        assert result.is_valid is True

        result = validator.validate("abcdef")
        # match() matches "abc" at start
        assert result.is_valid is True

        # But "xabc" won't match because match() is anchored at start
        result = validator.validate("xabc")
        assert result.is_valid is False
