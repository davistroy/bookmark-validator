"""
Integrated Validation System

This module provides a unified interface for all validation components,
integrating input validation, CSV field validation, data recovery,
command-line validation, and configuration validation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .cli_validators import validate_cli_arguments

# NOTE: config_validators was removed during Pydantic migration - validation moved to Configuration class
from .csv_field_validators import validate_bookmark_record, validate_csv_row
from .data_recovery import (
    DataRecoveryManager,
    MalformedDataDetector,
    create_error_report,
)
from .input_validator import ValidationResult, ValidationSeverity


class IntegratedValidator:
    """Unified validation system for the entire application"""

    def __init__(self):
        """Initialize the integrated validator"""
        self.logger = logging.getLogger(__name__)
        self.data_recovery = DataRecoveryManager()
        self.malformed_detector = MalformedDataDetector()

        # Statistics tracking
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "data_recoveries": 0,
            "warnings_issued": 0,
        }

    def validate_application_startup(
        self, cli_args, config_path: Optional[Path] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate application startup including CLI args and configuration

        Args:
            cli_args: Parsed command-line arguments
            config_path: Optional path to configuration file

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Validate CLI arguments
        validated_args, cli_error = validate_cli_arguments(cli_args)
        if cli_error:
            errors.append(f"CLI Argument Error: {cli_error}")
            return False, errors

        # Configuration validation now handled by Configuration class during Pydantic migration
        # The Configuration class validates itself during initialization
        if config_path:
            try:
                # Import here to avoid circular imports
                from bookmark_processor.config.configuration import Configuration

                # Test loading the configuration - it will validate itself
                config = Configuration(config_file=config_path)
                self.logger.info(
                    f"Configuration loaded successfully from {config_path}"
                )
            except Exception as e:
                errors.append(f"Configuration Error: {e}")
                return False, errors

        return True, []

    def validate_csv_data(
        self,
        csv_data: List[Dict[str, Any]],
        expected_columns: List[str],
        recovery_mode: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate and potentially recover CSV data

        Args:
            csv_data: List of dictionaries from CSV rows
            expected_columns: List of expected column names
            recovery_mode: Whether to attempt data recovery

        Returns:
            Tuple of (validated_data, list_of_error_messages)
        """
        validated_data = []
        error_messages = []

        for row_index, row_data in enumerate(csv_data):
            row_number = row_index + 1
            self.validation_stats["total_validations"] += 1

            try:
                # Validate CSV row structure and basic content
                row_result = validate_csv_row(row_data, expected_columns)

                if row_result.is_valid:
                    # Row is valid, add to results
                    validated_data.append(row_result.sanitized_value)
                    self.validation_stats["successful_validations"] += 1

                    # Log any warnings
                    warnings = row_result.get_warnings()
                    if warnings:
                        self.validation_stats["warnings_issued"] += len(warnings)
                        for warning in warnings:
                            self.logger.warning(f"Row {row_number}: {warning.message}")

                elif recovery_mode:
                    # Attempt data recovery
                    self.logger.info(f"Attempting data recovery for row {row_number}")

                    # Check for corruption patterns
                    corruption_issues = self.malformed_detector.detect_corruption(
                        row_data
                    )
                    if corruption_issues:
                        self.logger.warning(
                            f"Row {row_number} corruption detected: {', '.join(corruption_issues)}"
                        )

                    # Attempt recovery
                    recovered_data, recovery_messages = (
                        self.data_recovery.recover_partial_record(row_data)
                    )

                    if recovered_data and recovered_data.get("url"):
                        # Recovery successful
                        validated_data.append(recovered_data)
                        self.validation_stats["data_recoveries"] += 1

                        # Create informative message
                        recovery_summary = (
                            f"Row {row_number}: Data recovered successfully"
                        )
                        if recovery_messages:
                            recovery_summary += (
                                f" ({len(recovery_messages)} issues resolved)"
                            )

                        self.logger.info(recovery_summary)
                    else:
                        # Recovery failed
                        self.validation_stats["failed_validations"] += 1
                        error_report = create_error_report(
                            [issue.message for issue in row_result.get_errors()]
                            + recovery_messages,
                            row_data,
                            row_number,
                        )
                        error_messages.append(error_report)

                else:
                    # No recovery, record as failed
                    self.validation_stats["failed_validations"] += 1
                    error_report = create_error_report(
                        [issue.message for issue in row_result.get_errors()],
                        row_data,
                        row_number,
                    )
                    error_messages.append(error_report)

            except Exception as e:
                # Unexpected error during validation
                self.validation_stats["failed_validations"] += 1
                error_message = f"Row {row_number}: Unexpected validation error: {e}"
                error_messages.append(error_message)
                self.logger.error(error_message, exc_info=True)

        return validated_data, error_messages

    def validate_bookmark_record(
        self, bookmark_data: Dict[str, Any], record_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a single bookmark record

        Args:
            bookmark_data: Dictionary containing bookmark data
            record_id: Optional identifier for the record

        Returns:
            ValidationResult with validation details
        """
        self.validation_stats["total_validations"] += 1

        try:
            result = validate_bookmark_record(bookmark_data)

            if result.is_valid:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1

            # Count warnings
            warnings = result.get_warnings()
            if warnings:
                self.validation_stats["warnings_issued"] += len(warnings)

            return result

        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            error_message = f"Unexpected error validating bookmark"
            if record_id:
                error_message += f" {record_id}"
            error_message += f": {e}"

            self.logger.error(error_message, exc_info=True)

            # Return error result
            result = ValidationResult(is_valid=False)
            result.add_critical(error_message)
            return result

    def validate_and_recover_bookmarks(
        self, bookmarks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate and recover a list of bookmark records

        Args:
            bookmarks: List of bookmark dictionaries

        Returns:
            Tuple of (valid_bookmarks, validation_summary)
        """
        valid_bookmarks = []
        failed_bookmarks = []
        recovered_bookmarks = []

        for i, bookmark in enumerate(bookmarks):
            result = self.validate_bookmark_record(bookmark, str(i + 1))

            if result.is_valid:
                valid_bookmarks.append(result.sanitized_value or bookmark)
            else:
                # Attempt recovery
                try:
                    recovered_data, recovery_messages = (
                        self.data_recovery.recover_partial_record(bookmark)
                    )

                    if recovered_data and recovered_data.get("url"):
                        # Validate recovered data
                        recovery_result = validate_bookmark_record(recovered_data)

                        if recovery_result.is_valid:
                            valid_bookmarks.append(recovery_result.sanitized_value)
                            recovered_bookmarks.append(
                                {
                                    "index": i + 1,
                                    "original": bookmark,
                                    "recovered": recovery_result.sanitized_value,
                                    "messages": recovery_messages,
                                }
                            )
                            self.validation_stats["data_recoveries"] += 1
                        else:
                            failed_bookmarks.append(
                                {
                                    "index": i + 1,
                                    "data": bookmark,
                                    "errors": [
                                        issue.message for issue in result.get_errors()
                                    ],
                                    "recovery_failed": True,
                                }
                            )
                    else:
                        failed_bookmarks.append(
                            {
                                "index": i + 1,
                                "data": bookmark,
                                "errors": [
                                    issue.message for issue in result.get_errors()
                                ],
                                "recovery_failed": True,
                            }
                        )

                except Exception as e:
                    failed_bookmarks.append(
                        {
                            "index": i + 1,
                            "data": bookmark,
                            "errors": [f"Validation and recovery failed: {e}"],
                            "recovery_failed": True,
                        }
                    )

        # Create summary
        summary = {
            "total_processed": len(bookmarks),
            "valid_bookmarks": len(valid_bookmarks),
            "failed_bookmarks": len(failed_bookmarks),
            "recovered_bookmarks": len(recovered_bookmarks),
            "recovery_rate": (
                len(recovered_bookmarks) / len(bookmarks) * 100 if bookmarks else 0
            ),
            "success_rate": (
                len(valid_bookmarks) / len(bookmarks) * 100 if bookmarks else 0
            ),
            "failed_records": failed_bookmarks,
            "recovered_records": recovered_bookmarks,
            "validation_stats": self.validation_stats.copy(),
        }

        return valid_bookmarks, summary

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive validation statistics

        Returns:
            Dictionary with validation statistics
        """
        stats = self.validation_stats.copy()

        # Calculate rates
        total = stats["total_validations"]
        if total > 0:
            stats["success_rate"] = (stats["successful_validations"] / total) * 100
            stats["failure_rate"] = (stats["failed_validations"] / total) * 100
            stats["recovery_rate"] = (stats["data_recoveries"] / total) * 100
            stats["warning_rate"] = (stats["warnings_issued"] / total) * 100
        else:
            stats["success_rate"] = 0
            stats["failure_rate"] = 0
            stats["recovery_rate"] = 0
            stats["warning_rate"] = 0

        return stats

    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "data_recoveries": 0,
            "warnings_issued": 0,
        }

    def generate_validation_report(self, validation_summary: Dict[str, Any]) -> str:
        """
        Generate a comprehensive validation report

        Args:
            validation_summary: Summary from validate_and_recover_bookmarks

        Returns:
            Formatted validation report
        """
        report_parts = [
            "📊 BOOKMARK VALIDATION REPORT",
            "=" * 50,
            "",
            f"📈 SUMMARY:",
            f"  Total records processed: {validation_summary['total_processed']:,}",
            f"  Valid bookmarks: {validation_summary['valid_bookmarks']:,} ({validation_summary['success_rate']:.1f}%)",
            f"  Failed bookmarks: {validation_summary['failed_bookmarks']:,}",
            f"  Recovered bookmarks: {validation_summary['recovered_bookmarks']:,} ({validation_summary['recovery_rate']:.1f}%)",
            "",
        ]

        # Add recovery details if any
        if validation_summary["recovered_bookmarks"] > 0:
            report_parts.extend(
                [
                    "🔧 DATA RECOVERY:",
                    f"  Successfully recovered {validation_summary['recovered_bookmarks']} records",
                    f"  Recovery rate: {validation_summary['recovery_rate']:.1f}%",
                    "",
                ]
            )

        # Add failure details if any
        if validation_summary["failed_bookmarks"] > 0:
            report_parts.extend(
                [
                    "❌ FAILED RECORDS:",
                    f"  {validation_summary['failed_bookmarks']} records could not be processed",
                    "",
                ]
            )

            # Show first few failed records
            failed_records = validation_summary.get("failed_records", [])
            if failed_records:
                report_parts.append("  Sample failed records:")
                for record in failed_records[:3]:  # Show first 3
                    url = record["data"].get("url", "N/A")
                    errors = record["errors"][:2]  # Show first 2 errors
                    report_parts.append(f"    Record {record['index']}: {url}")
                    for error in errors:
                        report_parts.append(f"      - {error}")

                if len(failed_records) > 3:
                    report_parts.append(
                        f"    ... and {len(failed_records) - 3} more failed records"
                    )
                report_parts.append("")

        # Add overall statistics
        stats = validation_summary.get("validation_stats", {})
        report_parts.extend(
            [
                "📊 VALIDATION STATISTICS:",
                f"  Total validations performed: {stats.get('total_validations', 0):,}",
                f"  Successful validations: {stats.get('successful_validations', 0):,}",
                f"  Data recoveries performed: {stats.get('data_recoveries', 0):,}",
                f"  Warnings issued: {stats.get('warnings_issued', 0):,}",
                "",
            ]
        )

        # Add recommendations
        report_parts.extend(
            [
                "💡 RECOMMENDATIONS:",
            ]
        )

        if validation_summary["recovery_rate"] > 10:
            report_parts.append(
                "  • Review data source quality - high recovery rate indicates input issues"
            )

        if validation_summary["success_rate"] < 90:
            report_parts.append("  • Check CSV format and column structure")
            report_parts.append("  • Verify data encoding (use UTF-8)")

        if (
            stats.get("warnings_issued", 0)
            > validation_summary["total_processed"] * 0.1
        ):
            report_parts.append(
                "  • Review warning messages for data quality improvements"
            )

        report_parts.extend(
            [
                "  • Enable verbose logging for detailed issue tracking",
                "  • Use checkpoint functionality for large datasets",
                "",
            ]
        )

        return "\n".join(report_parts)


# Global validator instance
_global_validator = None


def get_validator() -> IntegratedValidator:
    """Get the global validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = IntegratedValidator()
    return _global_validator


def validate_application_config(
    cli_args, config_path: Optional[Path] = None
) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate application configuration

    Args:
        cli_args: Command-line arguments
        config_path: Optional configuration file path

    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = get_validator()
    return validator.validate_application_startup(cli_args, config_path)


def validate_and_process_csv_data(
    csv_data: List[Dict[str, Any]],
    expected_columns: List[str],
    recovery_mode: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Convenience function to validate and process CSV data

    Args:
        csv_data: List of CSV row dictionaries
        expected_columns: Expected column names
        recovery_mode: Whether to attempt data recovery

    Returns:
        Tuple of (validated_data, validation_report)
    """
    validator = get_validator()
    validated_data, summary = validator.validate_and_recover_bookmarks(csv_data)
    report = validator.generate_validation_report(summary)

    return validated_data, report


def get_validation_summary() -> Dict[str, Any]:
    """
    Get current validation statistics

    Returns:
        Dictionary with validation statistics
    """
    validator = get_validator()
    return validator.get_validation_statistics()


def reset_validation_stats() -> None:
    """Reset global validation statistics"""
    validator = get_validator()
    validator.reset_statistics()
