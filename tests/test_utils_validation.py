"""
Unit tests for validation utilities.

Tests for input validation, CSV field validation, and configuration validation.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from bookmark_processor.utils.cli_validators import CLIValidator
# ConfigValidator was removed during Pydantic migration - validation is now handled by Configuration class
from bookmark_processor.utils.csv_field_validators import CSVFieldValidator
from bookmark_processor.utils.input_validator import InputValidator
from bookmark_processor.utils.validation import (
    ValidationError,
    sanitize_input,
    validate_bookmark_data,
    validate_csv_structure,
    validate_url_format,
)
from tests.fixtures.test_data import (
    SAMPLE_RAINDROP_EXPORT_ROWS,
    create_invalid_csv_content,
    create_malformed_bookmark_data,
)


class TestValidationUtilities:
    """Test core validation utilities."""

    def test_validate_csv_structure_valid(self, sample_export_dataframe):
        """Test CSV structure validation with valid data."""
        result = validate_csv_structure(sample_export_dataframe)
        assert result is True

    def test_validate_csv_structure_missing_columns(self):
        """Test CSV structure validation with missing required columns."""
        # Create DataFrame with missing columns
        invalid_df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "title": ["Title 1", "Title 2"],
                "url": ["https://example.com", "https://test.com"],
                # Missing: note, excerpt, folder, tags, created, cover, highlights, favorite
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            validate_csv_structure(invalid_df)

        assert "missing required columns" in str(exc_info.value).lower()

    def test_validate_csv_structure_extra_columns(self, sample_export_dataframe):
        """Test CSV structure validation with extra columns (should be allowed)."""
        # Add extra column
        sample_export_dataframe["extra_column"] = "extra_data"

        # Should still be valid
        result = validate_csv_structure(sample_export_dataframe)
        assert result is True

    def test_validate_csv_structure_empty_dataframe(self):
        """Test CSV structure validation with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValidationError) as exc_info:
            validate_csv_structure(empty_df)

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://example.com", True),
            ("http://example.com", True),
            ("ftp://files.example.com", True),
            (
                "mailto:test@example.com",
                False,
            ),  # Should be invalid for bookmark processing
            ("javascript:void(0)", False),
            ("", False),
            ("not-a-url", False),
            ("   ", False),
            ("https://", False),
            ("://invalid", False),
        ],
    )
    def test_validate_url_format(self, url, expected):
        """Test URL format validation with various inputs."""
        result = validate_url_format(url)
        assert result == expected

    def test_validate_bookmark_data_valid(self):
        """Test bookmark data validation with valid data."""
        valid_data = SAMPLE_RAINDROP_EXPORT_ROWS[0]
        result = validate_bookmark_data(valid_data)
        assert result is True

    def test_validate_bookmark_data_invalid(self):
        """Test bookmark data validation with invalid data."""
        malformed_data = create_malformed_bookmark_data()

        for invalid_row in malformed_data:
            with pytest.raises(ValidationError):
                validate_bookmark_data(invalid_row)

    @pytest.mark.parametrize(
        "input_data,expected",
        [
            ("normal text", "normal text"),
            ("<script>alert('xss')</script>", "alert('xss')"),  # HTML tags removed
            ("text with\nnewlines\r\n", "text with newlines "),  # Normalized whitespace
            ("   spaced   text   ", "spaced text"),  # Trimmed and normalized spaces
            ("", ""),
            (None, ""),
            (123, "123"),  # Number converted to string
        ],
    )
    def test_sanitize_input(self, input_data, expected):
        """Test input sanitization."""
        result = sanitize_input(input_data)
        assert result == expected


class TestInputValidator:
    """Test InputValidator class."""

    def test_init(self):
        """Test InputValidator initialization."""
        validator = InputValidator()
        assert validator is not None

    def test_validate_file_path_valid(self, temp_csv_file):
        """Test file path validation with valid file."""
        validator = InputValidator()

        # Create the file first
        temp_csv_file.write_text("test,data\n1,2")

        result = validator.validate_file_path(str(temp_csv_file))
        assert result is True

    def test_validate_file_path_nonexistent(self, temp_dir):
        """Test file path validation with non-existent file."""
        validator = InputValidator()
        nonexistent_file = temp_dir / "nonexistent.csv"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_file_path(str(nonexistent_file))

        assert "does not exist" in str(exc_info.value).lower()

    def test_validate_file_path_directory(self, temp_dir):
        """Test file path validation with directory instead of file."""
        validator = InputValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_file_path(str(temp_dir))

        assert "not a file" in str(exc_info.value).lower()

    def test_validate_output_path_valid(self, temp_dir):
        """Test output path validation with valid directory."""
        validator = InputValidator()
        output_file = temp_dir / "output.csv"

        result = validator.validate_output_path(str(output_file))
        assert result is True

    def test_validate_output_path_readonly_directory(self, temp_dir):
        """Test output path validation with read-only directory."""
        validator = InputValidator()

        # Make directory read-only (if supported by OS)
        import os

        try:
            temp_dir.chmod(0o444)  # Read-only
            output_file = temp_dir / "output.csv"

            with pytest.raises(ValidationError):
                validator.validate_output_path(str(output_file))
        except (OSError, PermissionError):
            # Skip if permission change not supported
            pytest.skip("Cannot change directory permissions on this system")
        finally:
            # Restore permissions
            try:
                temp_dir.chmod(0o755)
            except (OSError, PermissionError):
                pass

    def test_validate_batch_size(self):
        """Test batch size validation."""
        validator = InputValidator()

        # Valid batch sizes
        assert validator.validate_batch_size(1) is True
        assert validator.validate_batch_size(100) is True
        assert validator.validate_batch_size(1000) is True

        # Invalid batch sizes
        with pytest.raises(ValidationError):
            validator.validate_batch_size(0)

        with pytest.raises(ValidationError):
            validator.validate_batch_size(-1)

        with pytest.raises(ValidationError):
            validator.validate_batch_size(10001)  # Too large

    def test_validate_timeout(self):
        """Test timeout validation."""
        validator = InputValidator()

        # Valid timeouts
        assert validator.validate_timeout(1) is True
        assert validator.validate_timeout(30) is True
        assert validator.validate_timeout(300) is True

        # Invalid timeouts
        with pytest.raises(ValidationError):
            validator.validate_timeout(0)

        with pytest.raises(ValidationError):
            validator.validate_timeout(-1)

        with pytest.raises(ValidationError):
            validator.validate_timeout(3601)  # Too large (> 1 hour)


class TestCSVFieldValidator:
    """Test CSV field validator."""

    def test_init(self):
        """Test CSVFieldValidator initialization."""
        validator = CSVFieldValidator()
        assert validator is not None

    def test_validate_required_columns(self, sample_export_dataframe):
        """Test required columns validation."""
        validator = CSVFieldValidator()

        result = validator.validate_required_columns(sample_export_dataframe)
        assert result is True

    def test_validate_required_columns_missing(self):
        """Test required columns validation with missing columns."""
        validator = CSVFieldValidator()

        # DataFrame with missing columns
        invalid_df = pd.DataFrame(
            {"id": ["1"], "title": ["Test"], "url": ["https://example.com"]}
        )

        missing_columns = validator.validate_required_columns(invalid_df)
        assert isinstance(missing_columns, list)
        assert len(missing_columns) > 0
        assert "note" in missing_columns
        assert "tags" in missing_columns

    def test_validate_data_types(self, sample_export_dataframe):
        """Test data type validation."""
        validator = CSVFieldValidator()

        errors = validator.validate_data_types(sample_export_dataframe)
        assert isinstance(errors, list)
        # Should have no errors for valid data
        assert len(errors) == 0

    def test_validate_data_types_invalid(self):
        """Test data type validation with invalid data."""
        validator = CSVFieldValidator()

        # Create DataFrame with invalid data types
        invalid_df = pd.DataFrame(
            {
                "id": [None, 2, "3"],  # Mixed types, None value
                "title": ["Title 1", "", None],  # None value
                "note": ["Note 1", "Note 2", "Note 3"],
                "excerpt": ["Excerpt 1", "Excerpt 2", "Excerpt 3"],
                "url": ["https://example.com", "invalid-url", ""],  # Invalid URL
                "folder": ["Folder 1", "Folder 2", "Folder 3"],
                "tags": ["tag1, tag2", "", "tag3"],
                "created": ["2024-01-01T00:00:00Z", "invalid-date", ""],  # Invalid date
                "cover": ["", "", ""],
                "highlights": ["", "", ""],
                "favorite": ["true", "invalid-bool", ""],  # Invalid boolean
            }
        )

        errors = validator.validate_data_types(invalid_df)
        assert isinstance(errors, list)
        assert len(errors) > 0

    def test_validate_url_column(self):
        """Test URL column validation."""
        validator = CSVFieldValidator()

        urls = [
            "https://example.com",
            "http://test.com",
            "invalid-url",
            "",
            "ftp://files.example.com",
        ]

        df = pd.DataFrame({"url": urls})
        errors = validator.validate_url_column(df)

        assert isinstance(errors, list)
        assert len(errors) >= 1  # Should have errors for invalid URLs

    def test_validate_date_column(self):
        """Test date column validation."""
        validator = CSVFieldValidator()

        dates = [
            "2024-01-01T00:00:00Z",
            "2024-12-31T23:59:59+00:00",
            "invalid-date",
            "",
            "2024-13-01T00:00:00Z",  # Invalid month
        ]

        df = pd.DataFrame({"created": dates})
        errors = validator.validate_date_column(df)

        assert isinstance(errors, list)
        assert len(errors) >= 2  # Should have errors for invalid dates


# NOTE: ConfigValidator tests commented out - ConfigValidator was removed during Pydantic migration
# Configuration validation is now handled by the Configuration class using Pydantic models
# class TestConfigValidator:
#     """Test configuration validator."""
#     ... (tests removed - validation moved to Configuration class)


class TestCLIValidator:
    """Test CLI validator."""

    def test_init(self):
        """Test CLIValidator initialization."""
        validator = CLIValidator()
        assert validator is not None

    def test_validate_arguments_valid(self, temp_csv_file, temp_dir):
        """Test CLI arguments validation with valid arguments."""
        validator = CLIValidator()

        # Create input file
        temp_csv_file.write_text("id,title,url\n1,Test,https://example.com")
        output_file = temp_dir / "output.csv"

        args = {
            "input": str(temp_csv_file),
            "output": str(output_file),
            "batch_size": 50,
            "timeout": 30,
            "max_retries": 3,
            "verbose": True,
            "resume": False,
        }

        errors = validator.validate_arguments(args)
        assert len(errors) == 0

    def test_validate_arguments_invalid(self, temp_dir):
        """Test CLI arguments validation with invalid arguments."""
        validator = CLIValidator()

        nonexistent_file = temp_dir / "nonexistent.csv"
        readonly_dir = temp_dir / "readonly"

        args = {
            "input": str(nonexistent_file),  # Non-existent file
            "output": str(readonly_dir / "output.csv"),  # Invalid output
            "batch_size": 0,  # Invalid batch size
            "timeout": -1,  # Invalid timeout
            "max_retries": 100,  # Invalid retries
            "verbose": "not_boolean",  # Invalid boolean
            "resume": "not_boolean",  # Invalid boolean
        }

        errors = validator.validate_arguments(args)
        assert len(errors) > 0

    def test_validate_file_compatibility(self, temp_csv_file, temp_dir):
        """Test file compatibility validation."""
        validator = CLIValidator()

        # Create input file
        temp_csv_file.write_text("test,data\n1,2")
        output_file = temp_dir / "output.csv"

        # Same file should cause error
        errors = validator.validate_file_compatibility(
            str(temp_csv_file), str(temp_csv_file)
        )
        assert len(errors) > 0
        assert "same file" in errors[0].lower()

        # Different files should be valid
        errors = validator.validate_file_compatibility(
            str(temp_csv_file), str(output_file)
        )
        assert len(errors) == 0

    def test_validate_ai_engine_selection(self):
        """Test AI engine selection validation."""
        validator = CLIValidator()

        # Valid engines
        for engine in ["local", "claude", "openai"]:
            errors = validator.validate_ai_engine_selection(engine)
            assert len(errors) == 0

        # Invalid engine
        errors = validator.validate_ai_engine_selection("invalid_engine")
        assert len(errors) > 0


class TestIntegratedValidation:
    """Test integrated validation functionality."""

    def test_comprehensive_validation_valid(self, sample_csv_file, temp_dir):
        """Test comprehensive validation with valid inputs."""
        from bookmark_processor.utils.integrated_validation import IntegratedValidator

        validator = IntegratedValidator()
        output_file = temp_dir / "output.csv"

        args = {
            "input": str(sample_csv_file),
            "output": str(output_file),
            "batch_size": 50,
            "timeout": 30,
            "ai_engine": "local",
        }

        errors = validator.validate_all(args)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_comprehensive_validation_invalid(self, temp_dir):
        """Test comprehensive validation with invalid inputs."""
        from bookmark_processor.utils.integrated_validation import IntegratedValidator

        validator = IntegratedValidator()

        args = {
            "input": str(temp_dir / "nonexistent.csv"),
            "output": "",  # Invalid output
            "batch_size": 0,  # Invalid
            "timeout": -1,  # Invalid
            "ai_engine": "invalid",  # Invalid
        }

        errors = validator.validate_all(args)
        assert isinstance(errors, list)
        assert len(errors) > 0

    def test_validation_error_aggregation(self, temp_dir):
        """Test that validation errors are properly aggregated."""
        from bookmark_processor.utils.integrated_validation import IntegratedValidator

        validator = IntegratedValidator()

        # Multiple invalid arguments
        args = {
            "input": "",  # Invalid
            "output": "",  # Invalid
            "batch_size": -1,  # Invalid
            "timeout": 0,  # Invalid
            "max_retries": -1,  # Invalid
            "ai_engine": "invalid",  # Invalid
        }

        errors = validator.validate_all(args)
        assert len(errors) >= 4  # Should have multiple errors


if __name__ == "__main__":
    pytest.main([__file__])
