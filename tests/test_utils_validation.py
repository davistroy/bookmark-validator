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
        from bookmark_processor.core.csv_handler import RaindropCSVHandler

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
            validate_csv_structure(invalid_df, RaindropCSVHandler.EXPORT_COLUMNS)

        assert "missing" in str(exc_info.value).lower()

    def test_validate_csv_structure_extra_columns(self, sample_export_dataframe):
        """Test CSV structure validation with extra columns (should be allowed)."""
        # Add extra column
        sample_export_dataframe["extra_column"] = "extra_data"

        # Should still be valid
        result = validate_csv_structure(sample_export_dataframe)
        assert result is True

    def test_validate_csv_structure_empty_dataframe(self):
        """Test CSV structure validation with empty DataFrame."""
        from bookmark_processor.core.csv_handler import RaindropCSVHandler

        empty_df = pd.DataFrame()

        with pytest.raises(ValidationError) as exc_info:
            validate_csv_structure(empty_df, RaindropCSVHandler.EXPORT_COLUMNS)

        assert "empty" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()

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

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        validator = InputValidator()

        input_data = {
            "url": "https://example.com",
            "title": "Test Bookmark",
            "tags": "test, bookmark",
            "folder": "Test/Folder"
        }

        result = validator.validate_input(input_data)
        assert result.is_valid or not result.has_errors()

    def test_validate_input_invalid_url(self):
        """Test input validation with invalid URL."""
        validator = InputValidator()

        input_data = {
            "url": "javascript:alert('xss')",  # Invalid scheme
            "title": "Test Bookmark"
        }

        result = validator.validate_input(input_data)
        assert result.has_errors() or result.has_warnings()

    def test_validate_input_missing_title(self):
        """Test input validation with missing title."""
        validator = InputValidator()

        input_data = {
            "url": "https://example.com",
            "title": ""
        }

        result = validator.validate_input(input_data)
        assert result.has_errors()

    def test_is_valid_method(self):
        """Test is_valid method."""
        validator = InputValidator()

        # Valid data
        valid_data = {
            "url": "https://example.com",
            "title": "Test"
        }
        assert validator.is_valid(valid_data)

        # Invalid data - missing required URL
        invalid_data = {
            "url": "",  # Empty URL
            "title": "Test"
        }
        assert not validator.is_valid(invalid_data)

        # Invalid data - bad scheme
        invalid_data2 = {
            "url": "javascript:void(0)",
            "title": "Test"
        }
        # This should have errors due to invalid scheme
        result = validator.validate_input(invalid_data2)
        assert result.has_errors() or not result.is_valid


class TestCSVFieldValidator:
    """Test CSV field validator."""

    def test_init(self):
        """Test CSVFieldValidator initialization."""
        validator = CSVFieldValidator()
        assert validator is not None

    def test_validate_row_valid(self):
        """Test row validation with valid data."""
        from bookmark_processor.core.csv_handler import RaindropCSVHandler

        validator = CSVFieldValidator()

        valid_row = {
            "id": "1",
            "title": "Test",
            "note": "Note",
            "excerpt": "Excerpt",
            "url": "https://example.com",
            "folder": "Test",
            "tags": "test, tag",
            "created": "2024-01-01T00:00:00Z",
            "cover": "",
            "highlights": "",
            "favorite": "false"
        }

        result = validator.validate_row(valid_row)
        assert result.is_valid or not result.has_errors()

    def test_validate_row_missing_columns(self):
        """Test row validation with missing columns."""
        validator = CSVFieldValidator()

        # Row with missing columns
        invalid_row = {
            "id": "1",
            "title": "Test",
            "url": "https://example.com"
        }

        result = validator.validate_row(invalid_row)
        assert result.has_errors()

    def test_validate_csv_structure_valid(self, sample_export_dataframe):
        """Test CSV structure validation with valid data."""
        validator = CSVFieldValidator()

        result = validator.validate_csv_structure(sample_export_dataframe)
        assert result.is_valid or not result.has_errors()

    def test_validate_csv_structure_missing_columns(self):
        """Test CSV structure validation with missing columns."""
        validator = CSVFieldValidator()

        # DataFrame with missing columns
        invalid_df = pd.DataFrame(
            {"id": ["1"], "title": ["Test"], "url": ["https://example.com"]}
        )

        result = validator.validate_csv_structure(invalid_df)
        assert result.has_errors()

    def test_validate_csv_structure_empty(self):
        """Test CSV structure validation with empty DataFrame."""
        validator = CSVFieldValidator()

        empty_df = pd.DataFrame()
        result = validator.validate_csv_structure(empty_df)
        assert result.has_errors()


# NOTE: ConfigValidator tests commented out - ConfigValidator was removed during Pydantic migration
# Configuration validation is now handled by the Configuration class using Pydantic models
# class TestConfigValidator:
#     """Test configuration validator."""
#     ... (tests removed - validation moved to Configuration class)


class TestCLIValidator:
    """Test CLI validator."""

    def test_init(self):
        """Test CLIValidator initialization."""
        from bookmark_processor.utils.cli_validators import CLIArgumentValidator
        validator = CLIArgumentValidator()
        assert validator is not None

    def test_validate_all_arguments_valid(self, temp_csv_file, temp_dir):
        """Test CLI arguments validation with valid arguments."""
        from bookmark_processor.utils.cli_validators import CLIArgumentValidator
        import argparse

        validator = CLIArgumentValidator()

        # Create input file
        temp_csv_file.write_text("id,title,url\n1,Test,https://example.com")
        output_file = temp_dir / "output.csv"

        # Create argparse Namespace
        args = argparse.Namespace(
            input=str(temp_csv_file),
            output=str(output_file),
            batch_size=50,
            max_retries=3,
            ai_engine="local",
            duplicate_strategy="highest_quality",
            verbose=True,
            resume=False,
            clear_checkpoints=False,
            no_duplicates=False,
            config=None
        )

        result = validator.validate_all_arguments(args)
        assert result.is_valid or not result.has_errors()

    def test_validate_all_arguments_invalid(self, temp_dir):
        """Test CLI arguments validation with invalid arguments."""
        from bookmark_processor.utils.cli_validators import CLIArgumentValidator
        import argparse

        validator = CLIArgumentValidator()

        nonexistent_file = temp_dir / "nonexistent.csv"

        # Create argparse Namespace with invalid args
        args = argparse.Namespace(
            input=str(nonexistent_file),  # Non-existent file
            output=str(temp_dir / "output.csv"),
            batch_size=0,  # Invalid batch size
            max_retries=100,  # Invalid retries
            ai_engine="invalid",  # Invalid engine
            duplicate_strategy="highest_quality",
            verbose=True,
            resume=False,
            clear_checkpoints=False,
            no_duplicates=False,
            config=None
        )

        result = validator.validate_all_arguments(args)
        assert result.has_errors()

    def test_validate_argument_combinations(self, temp_csv_file, temp_dir):
        """Test argument combination validation."""
        from bookmark_processor.utils.cli_validators import ArgumentCombinationValidator

        validator = ArgumentCombinationValidator()

        # Create input file
        temp_csv_file.write_text("test,data\n1,2")
        output_file = temp_dir / "output.csv"

        # Test conflicting resume and clear_checkpoints
        args_dict = {
            "resume": True,
            "clear_checkpoints": True,
            "input_path": str(temp_csv_file),
            "output_path": str(output_file)
        }

        result = validator.validate(args_dict)
        assert result.has_errors()

        # Test same input and output files
        args_dict = {
            "resume": False,
            "clear_checkpoints": False,
            "input_path": str(temp_csv_file),
            "output_path": str(temp_csv_file)
        }

        result = validator.validate(args_dict)
        assert result.has_errors()


class TestIntegratedValidation:
    """Test integrated validation functionality."""

    def test_validate_bookmark_record_valid(self):
        """Test bookmark record validation with valid data."""
        from bookmark_processor.utils.integrated_validation import IntegratedValidator

        validator = IntegratedValidator()

        valid_bookmark = {
            "id": "1",
            "title": "Test",
            "note": "Note",
            "excerpt": "Excerpt",
            "url": "https://example.com",
            "folder": "Test",
            "tags": "test",
            "created": "2024-01-01T00:00:00Z",
            "cover": "",
            "highlights": "",
            "favorite": False
        }

        result = validator.validate_bookmark_record(valid_bookmark)
        assert result.is_valid or not result.has_errors()

    def test_validate_bookmark_record_invalid(self):
        """Test bookmark record validation with invalid data."""
        from bookmark_processor.utils.integrated_validation import IntegratedValidator

        validator = IntegratedValidator()

        invalid_bookmark = {
            "url": "not-a-url",  # Invalid URL
            "title": "",  # Empty title
        }

        result = validator.validate_bookmark_record(invalid_bookmark)
        assert result.has_errors()

    def test_validate_and_recover_bookmarks(self):
        """Test validation and recovery of multiple bookmarks."""
        from bookmark_processor.utils.integrated_validation import IntegratedValidator

        validator = IntegratedValidator()

        bookmarks = [
            {
                "url": "https://example.com",
                "title": "Valid Bookmark",
                "note": "",
                "excerpt": "",
                "folder": "",
                "tags": "",
                "created": "2024-01-01T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": False
            },
            {
                "url": "invalid-url",
                "title": "Invalid Bookmark",
            }
        ]

        valid_bookmarks, summary = validator.validate_and_recover_bookmarks(bookmarks)

        assert isinstance(valid_bookmarks, list)
        assert isinstance(summary, dict)
        assert "total_processed" in summary
        assert "valid_bookmarks" in summary
        assert "failed_bookmarks" in summary

    def test_get_validation_statistics(self):
        """Test getting validation statistics."""
        from bookmark_processor.utils.integrated_validation import IntegratedValidator

        validator = IntegratedValidator()

        # Perform some validations
        validator.validate_bookmark_record({"url": "https://example.com", "title": "Test"})

        stats = validator.get_validation_statistics()
        assert isinstance(stats, dict)
        assert "total_validations" in stats
        assert stats["total_validations"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
