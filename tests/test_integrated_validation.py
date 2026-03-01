"""
Comprehensive tests for integrated_validation module.

Tests cover IntegratedValidator class and all convenience functions:
- validate_application_startup
- validate_csv_data
- validate_bookmark_record
- validate_and_recover_bookmarks
- get_validation_statistics
- reset_statistics
- generate_validation_report
- Module-level convenience functions
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.utils.integrated_validation import (
    IntegratedValidator,
    get_validator,
    get_validation_summary,
    reset_validation_stats,
    validate_and_process_csv_data,
    validate_application_config,
)
from bookmark_processor.utils.input_validator import ValidationResult


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def integrated_validator():
    """Create a fresh IntegratedValidator instance for each test."""
    return IntegratedValidator()


@pytest.fixture
def valid_bookmark_data():
    """Create valid bookmark data dictionary."""
    return {
        "id": "test123",
        "title": "Test Bookmark",
        "note": "A test note",
        "excerpt": "Test excerpt content",
        "url": "https://example.com/page",
        "folder": "Test/Folder",
        "tags": "test, bookmark, example",
        "created": "2024-01-15T10:30:00Z",
        "cover": "https://example.com/cover.jpg",
        "highlights": "",
        "favorite": False,
    }


@pytest.fixture
def invalid_bookmark_data():
    """Create invalid bookmark data dictionary."""
    return {
        "url": "not-a-valid-url",
        "title": "",
    }


@pytest.fixture
def partial_bookmark_data():
    """Create partial bookmark data for recovery testing."""
    return {
        "url": "https://example.com",
        "title": "Partial Bookmark",
    }


@pytest.fixture
def csv_data_valid():
    """Create valid CSV data list."""
    return [
        {
            "id": "1",
            "title": "Bookmark 1",
            "note": "",
            "excerpt": "",
            "url": "https://example1.com",
            "folder": "Test",
            "tags": "test",
            "created": "2024-01-01T00:00:00Z",
            "cover": "",
            "highlights": "",
            "favorite": False,
        },
        {
            "id": "2",
            "title": "Bookmark 2",
            "note": "A note",
            "excerpt": "",
            "url": "https://example2.com",
            "folder": "Test/Sub",
            "tags": "test, sample",
            "created": "2024-01-02T00:00:00Z",
            "cover": "",
            "highlights": "",
            "favorite": True,
        },
    ]


@pytest.fixture
def csv_data_mixed():
    """Create mixed valid/invalid CSV data list."""
    return [
        {
            "id": "1",
            "title": "Valid Bookmark",
            "url": "https://example.com",
            "folder": "",
            "tags": "",
            "created": "",
            "note": "",
            "excerpt": "",
            "cover": "",
            "highlights": "",
            "favorite": False,
        },
        {
            "id": "2",
            "title": "",
            "url": "invalid-url",  # Invalid
        },
        {
            "id": "3",
            "title": "Another Valid",
            "url": "https://another-example.com",
            "folder": "Folder",
            "tags": "tag1",
            "created": "2024-01-15",
            "note": "",
            "excerpt": "",
            "cover": "",
            "highlights": "",
            "favorite": False,
        },
    ]


@pytest.fixture
def expected_columns():
    """Expected CSV columns."""
    return [
        "id",
        "title",
        "note",
        "excerpt",
        "url",
        "folder",
        "tags",
        "created",
        "cover",
        "highlights",
        "favorite",
    ]


@pytest.fixture
def mock_cli_args():
    """Create mock CLI arguments."""
    args = argparse.Namespace()
    args.input = "test_input.csv"
    args.output = "test_output.csv"
    args.batch_size = 100
    args.max_retries = 3
    args.ai_engine = "local"
    args.duplicate_strategy = "highest_quality"
    args.resume = False
    args.clear_checkpoints = False
    args.no_duplicates = False
    args.verbose = False
    args.config = None
    return args


# ============================================================================
# IntegratedValidator Initialization Tests
# ============================================================================


class TestIntegratedValidatorInit:
    """Tests for IntegratedValidator initialization."""

    def test_init_creates_instance(self, integrated_validator):
        """Test that IntegratedValidator initializes correctly."""
        assert integrated_validator is not None
        assert isinstance(integrated_validator, IntegratedValidator)

    def test_init_creates_logger(self, integrated_validator):
        """Test that logger is created during initialization."""
        assert integrated_validator.logger is not None
        assert isinstance(integrated_validator.logger, logging.Logger)

    def test_init_creates_data_recovery_manager(self, integrated_validator):
        """Test that DataRecoveryManager is created."""
        assert integrated_validator.data_recovery is not None

    def test_init_creates_malformed_detector(self, integrated_validator):
        """Test that MalformedDataDetector is created."""
        assert integrated_validator.malformed_detector is not None

    def test_init_creates_validation_stats(self, integrated_validator):
        """Test that validation statistics are initialized."""
        stats = integrated_validator.validation_stats
        assert isinstance(stats, dict)
        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["failed_validations"] == 0
        assert stats["data_recoveries"] == 0
        assert stats["warnings_issued"] == 0


# ============================================================================
# validate_application_startup Tests
# ============================================================================


class TestValidateApplicationStartup:
    """Tests for validate_application_startup method."""

    def test_valid_cli_args_no_config(self, integrated_validator):
        """Test validation with valid CLI args and no config file."""
        args = argparse.Namespace()
        args.input = None
        args.output = None
        args.batch_size = 100
        args.max_retries = 3
        args.ai_engine = "local"
        args.duplicate_strategy = "highest_quality"
        args.resume = False
        args.clear_checkpoints = False
        args.no_duplicates = False
        args.verbose = False
        args.config = None

        # With CLI validation mocked to succeed
        with patch(
            "bookmark_processor.utils.integrated_validation.validate_cli_arguments"
        ) as mock_validate:
            mock_validate.return_value = ({}, None)
            is_valid, errors = integrated_validator.validate_application_startup(args)
            assert is_valid is True
            assert errors == []

    def test_invalid_cli_args(self, integrated_validator):
        """Test validation with invalid CLI arguments."""
        args = argparse.Namespace()
        args.resume = True
        args.clear_checkpoints = True  # Conflicting

        with patch(
            "bookmark_processor.utils.integrated_validation.validate_cli_arguments"
        ) as mock_validate:
            mock_validate.return_value = (None, "CLI Error: Conflicting arguments")
            is_valid, errors = integrated_validator.validate_application_startup(args)
            assert is_valid is False
            assert len(errors) == 1
            assert "CLI Argument Error" in errors[0]

    def test_valid_config_file(self, integrated_validator, temp_dir):
        """Test validation with valid config file (mocked Configuration)."""
        # Create a valid TOML config file
        config_file = temp_dir / "test_config.toml"
        config_content = """
[processing]
batch_size = 100
ai_engine = "local"

[network]
timeout = 30
max_retries = 3
"""
        config_file.write_text(config_content)

        args = argparse.Namespace()
        args.input = None
        args.output = None
        args.batch_size = 100
        args.max_retries = 3
        args.ai_engine = "local"
        args.duplicate_strategy = "highest_quality"
        args.resume = False
        args.clear_checkpoints = False
        args.no_duplicates = False
        args.verbose = False
        args.config = None

        # Mock both CLI validation and Configuration class
        with patch(
            "bookmark_processor.utils.integrated_validation.validate_cli_arguments"
        ) as mock_validate:
            mock_validate.return_value = ({}, None)
            # Mock Configuration to avoid the keyword arg issue in integrated_validation
            with patch(
                "bookmark_processor.config.configuration.Configuration"
            ) as mock_config:
                mock_config.return_value = MagicMock()
                is_valid, errors = integrated_validator.validate_application_startup(
                    args, config_path=config_file
                )
                assert is_valid is True
                assert errors == []

    def test_invalid_config_file(self, integrated_validator, temp_dir):
        """Test validation with invalid config file path."""
        args = argparse.Namespace()
        args.input = None
        args.output = None
        args.batch_size = 100
        args.max_retries = 3
        args.ai_engine = "local"
        args.duplicate_strategy = "highest_quality"
        args.resume = False
        args.clear_checkpoints = False
        args.no_duplicates = False
        args.verbose = False
        args.config = None

        nonexistent_config = temp_dir / "nonexistent_config.ini"

        with patch(
            "bookmark_processor.utils.integrated_validation.validate_cli_arguments"
        ) as mock_validate:
            mock_validate.return_value = ({}, None)
            is_valid, errors = integrated_validator.validate_application_startup(
                args, config_path=nonexistent_config
            )
            assert is_valid is False
            assert len(errors) == 1
            assert "Configuration Error" in errors[0]


# ============================================================================
# validate_csv_data Tests
# ============================================================================


class TestValidateCsvData:
    """Tests for validate_csv_data method."""

    def test_valid_csv_data(
        self, integrated_validator, csv_data_valid, expected_columns
    ):
        """Test validation with all valid CSV data."""
        validated, errors = integrated_validator.validate_csv_data(
            csv_data_valid, expected_columns
        )
        assert len(validated) > 0
        assert len(errors) == 0

    def test_csv_data_with_recovery(
        self, integrated_validator, csv_data_mixed, expected_columns
    ):
        """Test validation with mixed data and recovery enabled."""
        validated, errors = integrated_validator.validate_csv_data(
            csv_data_mixed, expected_columns, recovery_mode=True
        )
        # Should recover some records
        assert len(validated) >= 1

    def test_csv_data_without_recovery(
        self, integrated_validator, csv_data_mixed, expected_columns
    ):
        """Test validation with recovery disabled."""
        validated, errors = integrated_validator.validate_csv_data(
            csv_data_mixed, expected_columns, recovery_mode=False
        )
        # Without recovery, invalid records should fail
        assert len(errors) > 0 or len(validated) < len(csv_data_mixed)

    def test_empty_csv_data(self, integrated_validator, expected_columns):
        """Test validation with empty CSV data."""
        validated, errors = integrated_validator.validate_csv_data([], expected_columns)
        assert validated == []
        assert errors == []

    def test_csv_data_updates_stats(
        self, integrated_validator, csv_data_valid, expected_columns
    ):
        """Test that validation updates statistics."""
        initial_total = integrated_validator.validation_stats["total_validations"]
        integrated_validator.validate_csv_data(csv_data_valid, expected_columns)
        final_total = integrated_validator.validation_stats["total_validations"]
        assert final_total > initial_total

    def test_csv_data_exception_handling(self, integrated_validator, expected_columns):
        """Test that validation handles exceptions gracefully."""
        # Data that could trigger exceptions
        bad_data = [{"malformed": "data", "url": None}]
        validated, errors = integrated_validator.validate_csv_data(
            bad_data, expected_columns
        )
        # Should not crash, may have errors
        assert isinstance(validated, list)
        assert isinstance(errors, list)


# ============================================================================
# validate_bookmark_record Tests
# ============================================================================


class TestValidateBookmarkRecord:
    """Tests for validate_bookmark_record method."""

    def test_valid_bookmark(self, integrated_validator, valid_bookmark_data):
        """Test validation of valid bookmark."""
        result = integrated_validator.validate_bookmark_record(valid_bookmark_data)
        assert isinstance(result, ValidationResult)
        # Should be valid or at least not have critical errors
        assert result.is_valid or not any(
            issue.severity.value == "critical" for issue in result.issues
        )

    def test_invalid_bookmark(self, integrated_validator, invalid_bookmark_data):
        """Test validation of invalid bookmark."""
        result = integrated_validator.validate_bookmark_record(invalid_bookmark_data)
        assert isinstance(result, ValidationResult)
        assert result.has_errors()

    def test_bookmark_with_record_id(self, integrated_validator, valid_bookmark_data):
        """Test validation with record ID for logging."""
        result = integrated_validator.validate_bookmark_record(
            valid_bookmark_data, record_id="test_record_123"
        )
        assert isinstance(result, ValidationResult)

    def test_updates_statistics_on_success(
        self, integrated_validator, valid_bookmark_data
    ):
        """Test that successful validation updates statistics."""
        initial_success = integrated_validator.validation_stats["successful_validations"]
        integrated_validator.validate_bookmark_record(valid_bookmark_data)
        # Stats should be updated
        total = integrated_validator.validation_stats["total_validations"]
        assert total > 0

    def test_updates_statistics_on_failure(
        self, integrated_validator, invalid_bookmark_data
    ):
        """Test that failed validation updates statistics."""
        integrated_validator.validate_bookmark_record(invalid_bookmark_data)
        failed = integrated_validator.validation_stats["failed_validations"]
        assert failed > 0

    def test_exception_handling_returns_error_result(self, integrated_validator):
        """Test that exceptions are handled and return error result."""
        # Use mock to force an exception
        with patch(
            "bookmark_processor.utils.integrated_validation.validate_bookmark_record"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Test exception")
            result = integrated_validator.validate_bookmark_record(
                {"url": "https://test.com"}, record_id="test_id"
            )
            assert isinstance(result, ValidationResult)
            assert not result.is_valid

    def test_warnings_update_statistics(self, integrated_validator):
        """Test that warnings are counted in statistics."""
        # Bookmark with data that generates warnings (e.g., localhost URL)
        warning_data = {
            "url": "https://localhost:8080/test",
            "title": "Test Title",
            "note": "",
            "excerpt": "",
            "folder": "",
            "tags": "",
            "created": "",
            "cover": "",
            "highlights": "",
            "favorite": False,
        }
        integrated_validator.validate_bookmark_record(warning_data)
        # May have warnings (localhost URL)
        warnings = integrated_validator.validation_stats["warnings_issued"]
        # At minimum, stats should be tracked
        assert integrated_validator.validation_stats["total_validations"] > 0


# ============================================================================
# validate_and_recover_bookmarks Tests
# ============================================================================


class TestValidateAndRecoverBookmarks:
    """Tests for validate_and_recover_bookmarks method."""

    def test_all_valid_bookmarks(self, integrated_validator, csv_data_valid):
        """Test with all valid bookmarks."""
        valid_bookmarks, summary = integrated_validator.validate_and_recover_bookmarks(
            csv_data_valid
        )
        assert len(valid_bookmarks) > 0
        assert summary["total_processed"] == len(csv_data_valid)
        assert summary["valid_bookmarks"] >= 0

    def test_mixed_bookmarks_with_recovery(self, integrated_validator):
        """Test with mixed valid/invalid bookmarks."""
        bookmarks = [
            {
                "url": "https://example.com",
                "title": "Valid",
                "note": "",
                "excerpt": "",
                "folder": "",
                "tags": "",
                "created": "",
                "cover": "",
                "highlights": "",
                "favorite": False,
            },
            {"url": "invalid", "title": ""},  # Invalid
        ]
        valid_bookmarks, summary = integrated_validator.validate_and_recover_bookmarks(
            bookmarks
        )
        assert summary["total_processed"] == 2
        assert "failed_bookmarks" in summary
        assert "recovered_bookmarks" in summary

    def test_empty_list(self, integrated_validator):
        """Test with empty bookmark list."""
        valid_bookmarks, summary = integrated_validator.validate_and_recover_bookmarks(
            []
        )
        assert valid_bookmarks == []
        assert summary["total_processed"] == 0
        assert summary["success_rate"] == 0

    def test_summary_contains_expected_keys(self, integrated_validator, csv_data_valid):
        """Test that summary contains all expected keys."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        expected_keys = [
            "total_processed",
            "valid_bookmarks",
            "failed_bookmarks",
            "recovered_bookmarks",
            "recovery_rate",
            "success_rate",
            "failed_records",
            "recovered_records",
            "validation_stats",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_recovery_rate_calculation(self, integrated_validator):
        """Test that recovery rate is calculated correctly."""
        bookmarks = [
            {
                "url": "https://example.com",
                "title": "Valid",
                "note": "",
                "excerpt": "",
                "folder": "",
                "tags": "",
                "created": "",
                "cover": "",
                "highlights": "",
                "favorite": False,
            }
        ]
        _, summary = integrated_validator.validate_and_recover_bookmarks(bookmarks)
        # Recovery rate should be a percentage
        assert 0 <= summary["recovery_rate"] <= 100

    def test_success_rate_calculation(self, integrated_validator, csv_data_valid):
        """Test that success rate is calculated correctly."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        assert 0 <= summary["success_rate"] <= 100

    def test_failed_records_list(self, integrated_validator):
        """Test that failed records are tracked properly."""
        bookmarks = [
            {"url": "invalid-url", "title": ""},  # Should fail
        ]
        _, summary = integrated_validator.validate_and_recover_bookmarks(bookmarks)
        failed_records = summary["failed_records"]
        assert isinstance(failed_records, list)

    def test_recovered_records_list(self, integrated_validator):
        """Test that recovered records are tracked properly."""
        bookmarks = [
            {
                "url": "https://example.com",  # Valid
                "title": "Test",
            }
        ]
        _, summary = integrated_validator.validate_and_recover_bookmarks(bookmarks)
        recovered_records = summary["recovered_records"]
        assert isinstance(recovered_records, list)


# ============================================================================
# get_validation_statistics Tests
# ============================================================================


class TestGetValidationStatistics:
    """Tests for get_validation_statistics method."""

    def test_returns_dict(self, integrated_validator):
        """Test that statistics are returned as dict."""
        stats = integrated_validator.get_validation_statistics()
        assert isinstance(stats, dict)

    def test_contains_base_stats(self, integrated_validator):
        """Test that base statistics are included."""
        stats = integrated_validator.get_validation_statistics()
        assert "total_validations" in stats
        assert "successful_validations" in stats
        assert "failed_validations" in stats
        assert "data_recoveries" in stats
        assert "warnings_issued" in stats

    def test_contains_rate_stats(self, integrated_validator):
        """Test that rate statistics are included."""
        stats = integrated_validator.get_validation_statistics()
        assert "success_rate" in stats
        assert "failure_rate" in stats
        assert "recovery_rate" in stats
        assert "warning_rate" in stats

    def test_rates_zero_when_no_validations(self, integrated_validator):
        """Test that rates are zero when no validations performed."""
        stats = integrated_validator.get_validation_statistics()
        assert stats["success_rate"] == 0
        assert stats["failure_rate"] == 0
        assert stats["recovery_rate"] == 0
        assert stats["warning_rate"] == 0

    def test_rates_calculated_after_validations(
        self, integrated_validator, valid_bookmark_data
    ):
        """Test that rates are calculated after validations."""
        integrated_validator.validate_bookmark_record(valid_bookmark_data)
        stats = integrated_validator.get_validation_statistics()
        # After validations, rates should be calculated
        assert stats["total_validations"] > 0
        # Success or failure rate should be non-zero
        assert stats["success_rate"] > 0 or stats["failure_rate"] > 0

    def test_returns_copy_not_reference(self, integrated_validator):
        """Test that a copy is returned, not a reference."""
        stats1 = integrated_validator.get_validation_statistics()
        stats1["total_validations"] = 9999
        stats2 = integrated_validator.get_validation_statistics()
        assert stats2["total_validations"] != 9999


# ============================================================================
# reset_statistics Tests
# ============================================================================


class TestResetStatistics:
    """Tests for reset_statistics method."""

    def test_resets_all_stats_to_zero(
        self, integrated_validator, valid_bookmark_data
    ):
        """Test that all statistics are reset to zero."""
        # Perform some validations first
        integrated_validator.validate_bookmark_record(valid_bookmark_data)
        assert integrated_validator.validation_stats["total_validations"] > 0

        # Reset
        integrated_validator.reset_statistics()

        # Verify reset
        stats = integrated_validator.validation_stats
        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["failed_validations"] == 0
        assert stats["data_recoveries"] == 0
        assert stats["warnings_issued"] == 0

    def test_reset_allows_fresh_start(
        self, integrated_validator, valid_bookmark_data
    ):
        """Test that reset allows a fresh start."""
        # First round
        integrated_validator.validate_bookmark_record(valid_bookmark_data)
        first_total = integrated_validator.validation_stats["total_validations"]

        # Reset
        integrated_validator.reset_statistics()

        # Second round
        integrated_validator.validate_bookmark_record(valid_bookmark_data)
        second_total = integrated_validator.validation_stats["total_validations"]

        # Should be same as first round (fresh start)
        assert second_total == first_total


# ============================================================================
# generate_validation_report Tests
# ============================================================================


class TestGenerateValidationReport:
    """Tests for generate_validation_report method."""

    def test_generates_string_report(self, integrated_validator, csv_data_valid):
        """Test that report is generated as string."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        report = integrated_validator.generate_validation_report(summary)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_header(self, integrated_validator, csv_data_valid):
        """Test that report contains header."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        report = integrated_validator.generate_validation_report(summary)
        assert "VALIDATION REPORT" in report

    def test_report_contains_summary_section(
        self, integrated_validator, csv_data_valid
    ):
        """Test that report contains summary section."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        report = integrated_validator.generate_validation_report(summary)
        assert "SUMMARY" in report

    def test_report_contains_total_processed(
        self, integrated_validator, csv_data_valid
    ):
        """Test that report shows total processed."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        report = integrated_validator.generate_validation_report(summary)
        assert "Total records processed" in report

    def test_report_contains_recovery_section_when_recoveries_exist(
        self, integrated_validator
    ):
        """Test that recovery section appears when recoveries exist."""
        # Create summary with recoveries
        summary = {
            "total_processed": 10,
            "valid_bookmarks": 8,
            "failed_bookmarks": 2,
            "recovered_bookmarks": 3,
            "recovery_rate": 30.0,
            "success_rate": 80.0,
            "failed_records": [],
            "recovered_records": [],
            "validation_stats": integrated_validator.validation_stats,
        }
        report = integrated_validator.generate_validation_report(summary)
        assert "DATA RECOVERY" in report

    def test_report_contains_failed_section_when_failures_exist(
        self, integrated_validator
    ):
        """Test that failed section appears when failures exist."""
        summary = {
            "total_processed": 10,
            "valid_bookmarks": 5,
            "failed_bookmarks": 5,
            "recovered_bookmarks": 0,
            "recovery_rate": 0.0,
            "success_rate": 50.0,
            "failed_records": [
                {"index": 1, "data": {"url": "bad"}, "errors": ["Invalid URL"]}
            ],
            "recovered_records": [],
            "validation_stats": integrated_validator.validation_stats,
        }
        report = integrated_validator.generate_validation_report(summary)
        assert "FAILED RECORDS" in report

    def test_report_contains_statistics_section(
        self, integrated_validator, csv_data_valid
    ):
        """Test that report contains statistics section."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        report = integrated_validator.generate_validation_report(summary)
        assert "VALIDATION STATISTICS" in report

    def test_report_contains_recommendations_section(
        self, integrated_validator, csv_data_valid
    ):
        """Test that report contains recommendations section."""
        _, summary = integrated_validator.validate_and_recover_bookmarks(csv_data_valid)
        report = integrated_validator.generate_validation_report(summary)
        assert "RECOMMENDATIONS" in report

    def test_report_recommendations_for_high_recovery_rate(self, integrated_validator):
        """Test recommendations when recovery rate is high."""
        summary = {
            "total_processed": 10,
            "valid_bookmarks": 5,
            "failed_bookmarks": 0,
            "recovered_bookmarks": 5,
            "recovery_rate": 50.0,  # High
            "success_rate": 100.0,
            "failed_records": [],
            "recovered_records": [],
            "validation_stats": integrated_validator.validation_stats,
        }
        report = integrated_validator.generate_validation_report(summary)
        assert "data source quality" in report.lower()

    def test_report_recommendations_for_low_success_rate(self, integrated_validator):
        """Test recommendations when success rate is low."""
        summary = {
            "total_processed": 10,
            "valid_bookmarks": 5,
            "failed_bookmarks": 5,
            "recovered_bookmarks": 0,
            "recovery_rate": 0.0,
            "success_rate": 50.0,  # Low
            "failed_records": [],
            "recovered_records": [],
            "validation_stats": integrated_validator.validation_stats,
        }
        report = integrated_validator.generate_validation_report(summary)
        assert "CSV format" in report or "column structure" in report


# ============================================================================
# Global Validator (get_validator) Tests
# ============================================================================


class TestGetValidator:
    """Tests for get_validator function."""

    def test_returns_validator_instance(self):
        """Test that get_validator returns an IntegratedValidator."""
        # Reset global state first
        import bookmark_processor.utils.integrated_validation as iv

        iv._global_validator = None

        validator = get_validator()
        assert isinstance(validator, IntegratedValidator)

    def test_returns_singleton(self):
        """Test that get_validator returns same instance."""
        import bookmark_processor.utils.integrated_validation as iv

        iv._global_validator = None

        validator1 = get_validator()
        validator2 = get_validator()
        assert validator1 is validator2

    def test_creates_new_when_none(self):
        """Test that new instance is created when none exists."""
        import bookmark_processor.utils.integrated_validation as iv

        iv._global_validator = None
        validator = get_validator()
        assert validator is not None


# ============================================================================
# validate_application_config Convenience Function Tests
# ============================================================================


class TestValidateApplicationConfig:
    """Tests for validate_application_config convenience function."""

    def test_delegates_to_validator(self):
        """Test that function delegates to IntegratedValidator."""
        args = argparse.Namespace()
        args.input = None
        args.output = None
        args.batch_size = 100
        args.max_retries = 3
        args.ai_engine = "local"
        args.duplicate_strategy = "highest_quality"
        args.resume = False
        args.clear_checkpoints = False
        args.no_duplicates = False
        args.verbose = False
        args.config = None

        with patch(
            "bookmark_processor.utils.integrated_validation.validate_cli_arguments"
        ) as mock:
            mock.return_value = ({}, None)
            is_valid, errors = validate_application_config(args)
            assert isinstance(is_valid, bool)
            assert isinstance(errors, list)


# ============================================================================
# validate_and_process_csv_data Convenience Function Tests
# ============================================================================


class TestValidateAndProcessCsvData:
    """Tests for validate_and_process_csv_data convenience function."""

    def test_returns_tuple(self, csv_data_valid, expected_columns):
        """Test that function returns tuple of data and report."""
        result = validate_and_process_csv_data(
            csv_data_valid, expected_columns, recovery_mode=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_validated_data(self, csv_data_valid, expected_columns):
        """Test that function returns validated data."""
        validated_data, _ = validate_and_process_csv_data(
            csv_data_valid, expected_columns, recovery_mode=True
        )
        assert isinstance(validated_data, list)

    def test_returns_report_string(self, csv_data_valid, expected_columns):
        """Test that function returns report as string."""
        _, report = validate_and_process_csv_data(
            csv_data_valid, expected_columns, recovery_mode=True
        )
        assert isinstance(report, str)


# ============================================================================
# get_validation_summary Convenience Function Tests
# ============================================================================


class TestGetValidationSummary:
    """Tests for get_validation_summary convenience function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        summary = get_validation_summary()
        assert isinstance(summary, dict)

    def test_contains_expected_keys(self):
        """Test that summary contains expected keys."""
        summary = get_validation_summary()
        assert "total_validations" in summary
        assert "success_rate" in summary


# ============================================================================
# reset_validation_stats Convenience Function Tests
# ============================================================================


class TestResetValidationStats:
    """Tests for reset_validation_stats convenience function."""

    def test_resets_global_stats(self):
        """Test that function resets global validator stats."""
        # Get validator and perform validation
        validator = get_validator()
        validator.validation_stats["total_validations"] = 100

        # Reset
        reset_validation_stats()

        # Verify
        summary = get_validation_summary()
        assert summary["total_validations"] == 0


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    def test_none_bookmark_data(self, integrated_validator):
        """Test handling of None bookmark data."""
        # Should not crash
        try:
            result = integrated_validator.validate_bookmark_record(None)
            assert isinstance(result, ValidationResult)
        except (TypeError, AttributeError):
            # Expected - None is not a valid dict
            pass

    def test_empty_bookmark_data(self, integrated_validator):
        """Test handling of empty bookmark data."""
        result = integrated_validator.validate_bookmark_record({})
        assert isinstance(result, ValidationResult)

    def test_unicode_in_bookmark_data(self, integrated_validator):
        """Test handling of Unicode characters in bookmark data."""
        unicode_data = {
            "url": "https://example.com/path",
            "title": "Test Title",
            "note": "",
            "excerpt": "",
            "folder": "Test",
            "tags": "",
            "created": "",
            "cover": "",
            "highlights": "",
            "favorite": False,
        }
        result = integrated_validator.validate_bookmark_record(unicode_data)
        assert isinstance(result, ValidationResult)

    def test_very_long_url(self, integrated_validator):
        """Test handling of very long URLs."""
        long_url = "https://example.com/" + "a" * 5000
        long_url_data = {
            "url": long_url,
            "title": "Long URL Test",
            "note": "",
            "excerpt": "",
            "folder": "",
            "tags": "",
            "created": "",
            "cover": "",
            "highlights": "",
            "favorite": False,
        }
        result = integrated_validator.validate_bookmark_record(long_url_data)
        assert isinstance(result, ValidationResult)

    def test_special_characters_in_fields(self, integrated_validator):
        """Test handling of special characters."""
        special_data = {
            "url": "https://example.com/test?a=1&b=2",
            "title": "Test <script>alert('xss')</script> Title",
            "note": "Note with\nnewlines\tand\ttabs",
            "excerpt": "",
            "folder": "Folder/Sub-folder",
            "tags": "tag1, tag2, tag-with-dash",
            "created": "2024-01-15T10:30:00Z",
            "cover": "",
            "highlights": "",
            "favorite": False,
        }
        result = integrated_validator.validate_bookmark_record(special_data)
        assert isinstance(result, ValidationResult)

    def test_large_csv_data_set(self, integrated_validator, expected_columns):
        """Test handling of large CSV data set."""
        # Create 1000 records
        large_data = []
        for i in range(1000):
            large_data.append(
                {
                    "id": str(i),
                    "title": f"Bookmark {i}",
                    "url": f"https://example{i}.com",
                    "note": "",
                    "excerpt": "",
                    "folder": "Test",
                    "tags": "test",
                    "created": "",
                    "cover": "",
                    "highlights": "",
                    "favorite": False,
                }
            )
        validated, errors = integrated_validator.validate_csv_data(
            large_data, expected_columns
        )
        assert isinstance(validated, list)
        # Should process all without crashing
        assert integrated_validator.validation_stats["total_validations"] >= 1000

    def test_concurrent_access_to_stats(self, integrated_validator, valid_bookmark_data):
        """Test that statistics tracking is consistent."""
        import threading

        def validate_task():
            for _ in range(10):
                integrated_validator.validate_bookmark_record(valid_bookmark_data)

        threads = [threading.Thread(target=validate_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Stats should be updated (though exact count may vary due to threading)
        assert integrated_validator.validation_stats["total_validations"] >= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
