"""
Test Validators for Integration Testing

Provides validation classes for different aspects of integration test results,
including result validation, checkpoint validation, and error validation.
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bookmark_processor.core.data_models import Bookmark, ProcessingResults


class ValidationResult:
    """Result of a validation check."""

    def __init__(
        self,
        validator_name: str,
        passed: bool,
        message: str = "",
        details: Dict[str, Any] = None,
    ):
        self.validator_name = validator_name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.validator_name}: {self.message}"


class BaseValidator(ABC):
    """Abstract base class for validators."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"validator.{name}")

    @abstractmethod
    def validate(self, *args, **kwargs) -> ValidationResult:
        """Perform validation and return result."""
        pass

    def _create_result(
        self, passed: bool, message: str, details: Dict[str, Any] = None
    ) -> ValidationResult:
        """Create a validation result."""
        return ValidationResult(self.name, passed, message, details)


class ResultValidator(BaseValidator):
    """Validates processing results and output files."""

    def __init__(self):
        super().__init__("result_validator")

    def validate(
        self,
        results: ProcessingResults,
        output_file: Path,
        expected_results: Dict[str, Any] = None,
    ) -> ValidationResult:
        """Validate processing results and output file."""

        if not results:
            return self._create_result(False, "No processing results provided")

        errors = []
        details = {}

        # Validate processing results structure
        result_errors = self._validate_processing_results(results)
        errors.extend(result_errors)

        # Validate output file
        if output_file:
            file_errors, file_details = self._validate_output_file(output_file, results)
            errors.extend(file_errors)
            details.update(file_details)

        # Validate against expected results
        if expected_results:
            expected_errors = self._validate_expected_results(results, expected_results)
            errors.extend(expected_errors)

        # Add metrics to details
        details.update(self._calculate_validation_metrics(results))

        passed = len(errors) == 0
        message = (
            "All validations passed"
            if passed
            else f"Found {len(errors)} validation errors: {'; '.join(errors[:3])}"
        )

        if len(errors) > 3:
            message += f" (and {len(errors) - 3} more)"

        return self._create_result(passed, message, details)

    def _validate_processing_results(self, results: ProcessingResults) -> List[str]:
        """Validate the structure and content of processing results."""
        errors = []

        # Basic structure validation
        if not hasattr(results, "total_bookmarks"):
            errors.append("Missing total_bookmarks field")
        elif results.total_bookmarks < 0:
            errors.append("total_bookmarks cannot be negative")

        if not hasattr(results, "processed_bookmarks"):
            errors.append("Missing processed_bookmarks field")
        elif results.processed_bookmarks < 0:
            errors.append("processed_bookmarks cannot be negative")
        elif results.processed_bookmarks > results.total_bookmarks:
            errors.append("processed_bookmarks cannot exceed total_bookmarks")

        if not hasattr(results, "valid_bookmarks"):
            errors.append("Missing valid_bookmarks field")
        elif results.valid_bookmarks < 0:
            errors.append("valid_bookmarks cannot be negative")
        elif results.valid_bookmarks > results.processed_bookmarks:
            errors.append("valid_bookmarks cannot exceed processed_bookmarks")

        if not hasattr(results, "invalid_bookmarks"):
            errors.append("Missing invalid_bookmarks field")
        elif results.invalid_bookmarks < 0:
            errors.append("invalid_bookmarks cannot be negative")

        # Logical validation
        if (
            hasattr(results, "processed_bookmarks")
            and hasattr(results, "valid_bookmarks")
            and hasattr(results, "invalid_bookmarks")
        ):
            if (
                results.valid_bookmarks + results.invalid_bookmarks
                != results.processed_bookmarks
            ):
                errors.append(
                    "valid_bookmarks + invalid_bookmarks should equal processed_bookmarks"
                )

        if not hasattr(results, "processing_time"):
            errors.append("Missing processing_time field")
        elif results.processing_time < 0:
            errors.append("processing_time cannot be negative")

        if not hasattr(results, "errors"):
            errors.append("Missing errors field")
        elif not isinstance(results.errors, list):
            errors.append("errors field must be a list")

        return errors

    def _validate_output_file(
        self, output_file: Path, results: ProcessingResults
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Validate the output CSV file."""
        errors = []
        details = {}

        if not output_file.exists():
            errors.append("Output file does not exist")
            return errors, details

        try:
            # Read and validate CSV structure
            df = pd.read_csv(output_file)

            # Check required columns
            expected_columns = ["url", "folder", "title", "note", "tags", "created"]
            actual_columns = list(df.columns)

            if actual_columns != expected_columns:
                errors.append(
                    f"Output columns incorrect. Expected: {expected_columns}, Got: {actual_columns}"
                )

            # Check row count consistency
            output_row_count = len(df)
            if output_row_count != results.valid_bookmarks:
                errors.append(
                    f"Output row count ({output_row_count}) doesn't match valid_bookmarks ({results.valid_bookmarks})"
                )

            # Validate data quality
            data_errors = self._validate_output_data_quality(df)
            errors.extend(data_errors)

            # Store file details
            details["output_file_size"] = output_file.stat().st_size
            details["output_row_count"] = output_row_count
            details["output_columns"] = actual_columns

        except Exception as e:
            errors.append(f"Failed to read/parse output file: {str(e)}")

        return errors, details

    def _validate_output_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Validate the quality of data in the output file."""
        errors = []

        # Check for empty URLs
        empty_urls = df["url"].isna().sum() + (df["url"] == "").sum()
        if empty_urls > 0:
            errors.append(f"Found {empty_urls} empty URLs in output")

        # Check for empty titles
        empty_titles = df["title"].isna().sum() + (df["title"] == "").sum()
        if empty_titles > 0:
            errors.append(f"Found {empty_titles} empty titles in output")

        # Check URL format (basic validation)
        if "url" in df.columns:
            invalid_urls = 0
            for url in df["url"].dropna():
                if not isinstance(url, str) or not (
                    url.startswith("http://") or url.startswith("https://")
                ):
                    invalid_urls += 1

            if invalid_urls > 0:
                errors.append(f"Found {invalid_urls} invalid URL formats in output")

        # Check date format
        if "created" in df.columns:
            invalid_dates = 0
            for date_str in df["created"].dropna():
                try:
                    datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    invalid_dates += 1

            if invalid_dates > 0:
                errors.append(f"Found {invalid_dates} invalid date formats in output")

        return errors

    def _validate_expected_results(
        self, results: ProcessingResults, expected: Dict[str, Any]
    ) -> List[str]:
        """Validate results against expected values."""
        errors = []

        # Check minimum success rate
        if "min_success_rate" in expected and results.total_bookmarks > 0:
            actual_rate = results.valid_bookmarks / results.total_bookmarks
            if actual_rate < expected["min_success_rate"]:
                errors.append(
                    f"Success rate {actual_rate:.2%} below minimum {expected['min_success_rate']:.2%}"
                )

        # Check maximum duration
        if (
            "max_duration" in expected
            and results.processing_time > expected["max_duration"]
        ):
            errors.append(
                f"Processing time {results.processing_time:.2f}s exceeds maximum {expected['max_duration']:.2f}s"
            )

        # Check should complete flag
        if (
            expected.get("should_complete", True)
            and results.processed_bookmarks != results.total_bookmarks
        ):
            errors.append(
                f"Processing incomplete: {results.processed_bookmarks}/{results.total_bookmarks} processed"
            )

        # Check minimum processed count
        if (
            "min_processed" in expected
            and results.processed_bookmarks < expected["min_processed"]
        ):
            errors.append(
                f"Processed count {results.processed_bookmarks} below minimum {expected['min_processed']}"
            )

        return errors

    def _calculate_validation_metrics(
        self, results: ProcessingResults
    ) -> Dict[str, Any]:
        """Calculate validation metrics."""
        metrics = {
            "total_bookmarks": results.total_bookmarks,
            "processed_bookmarks": results.processed_bookmarks,
            "valid_bookmarks": results.valid_bookmarks,
            "invalid_bookmarks": results.invalid_bookmarks,
            "processing_time": results.processing_time,
            "error_count": len(results.errors) if results.errors else 0,
        }

        if results.total_bookmarks > 0:
            metrics["success_rate"] = results.valid_bookmarks / results.total_bookmarks
            metrics["processing_rate"] = results.total_bookmarks / max(
                results.processing_time, 0.01
            )

        return metrics


class CheckpointValidator(BaseValidator):
    """Validates checkpoint functionality and files."""

    def __init__(self):
        super().__init__("checkpoint_validator")

    def validate(
        self,
        checkpoint_dir: Path,
        expected_checkpoints: int = None,
        validate_content: bool = True,
    ) -> ValidationResult:
        """Validate checkpoint files and functionality."""

        if not checkpoint_dir.exists():
            return self._create_result(False, "Checkpoint directory does not exist")

        errors = []
        details = {}

        # Find checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        details["checkpoint_count"] = len(checkpoint_files)
        details["checkpoint_files"] = [f.name for f in checkpoint_files]

        # Validate checkpoint count
        if expected_checkpoints is not None:
            if len(checkpoint_files) != expected_checkpoints:
                errors.append(
                    f"Expected {expected_checkpoints} checkpoint files, found {len(checkpoint_files)}"
                )
        elif len(checkpoint_files) == 0:
            errors.append("No checkpoint files found")

        # Validate checkpoint file content
        if validate_content:
            for checkpoint_file in checkpoint_files:
                file_errors = self._validate_checkpoint_file(checkpoint_file)
                errors.extend(file_errors)

        # Check for resume capability
        resume_errors = self._validate_resume_capability(checkpoint_files)
        errors.extend(resume_errors)

        passed = len(errors) == 0
        message = (
            "Checkpoint validation passed"
            if passed
            else f"Found {len(errors)} checkpoint issues"
        )

        return self._create_result(passed, message, details)

    def _validate_checkpoint_file(self, checkpoint_file: Path) -> List[str]:
        """Validate a single checkpoint file."""
        errors = []

        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            # Check required fields
            required_fields = ["checkpoint_id", "timestamp", "progress"]
            for field in required_fields:
                if field not in checkpoint_data:
                    errors.append(
                        f"Checkpoint file {checkpoint_file.name} missing required field: {field}"
                    )

            # Validate progress data
            if "progress" in checkpoint_data:
                progress = checkpoint_data["progress"]
                if not isinstance(progress, dict):
                    errors.append(
                        f"Checkpoint file {checkpoint_file.name} has invalid progress format"
                    )
                else:
                    # Check for common progress fields
                    if "processed_count" not in progress:
                        errors.append(
                            f"Checkpoint file {checkpoint_file.name} missing processed_count in progress"
                        )

            # Validate timestamp
            if "timestamp" in checkpoint_data:
                try:
                    datetime.fromisoformat(checkpoint_data["timestamp"])
                except ValueError:
                    errors.append(
                        f"Checkpoint file {checkpoint_file.name} has invalid timestamp format"
                    )

        except json.JSONDecodeError as e:
            errors.append(
                f"Checkpoint file {checkpoint_file.name} is not valid JSON: {str(e)}"
            )
        except Exception as e:
            errors.append(
                f"Failed to validate checkpoint file {checkpoint_file.name}: {str(e)}"
            )

        return errors

    def _validate_resume_capability(self, checkpoint_files: List[Path]) -> List[str]:
        """Validate that checkpoints can be used for resuming."""
        errors = []

        if not checkpoint_files:
            return errors

        # Find the most recent checkpoint
        try:
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)

            with open(latest_checkpoint, "r") as f:
                checkpoint_data = json.load(f)

            # Validate resume data completeness
            if "progress" in checkpoint_data:
                progress = checkpoint_data["progress"]

                # Check for essential resume information
                essential_fields = ["processed_count", "last_processed_id"]
                missing_fields = [
                    field for field in essential_fields if field not in progress
                ]

                if missing_fields:
                    errors.append(
                        f"Latest checkpoint missing essential resume fields: {missing_fields}"
                    )

        except Exception as e:
            errors.append(f"Failed to validate resume capability: {str(e)}")

        return errors


class ErrorValidator(BaseValidator):
    """Validates error handling and error reporting."""

    def __init__(self):
        super().__init__("error_validator")

    def validate(
        self,
        results: ProcessingResults,
        log_files: List[Path] = None,
        expected_error_types: List[str] = None,
    ) -> ValidationResult:
        """Validate error handling and reporting."""

        errors = []
        details = {}

        # Validate error structure in results
        if not results or not hasattr(results, "errors"):
            errors.append("Results missing error information")
        else:
            error_errors = self._validate_error_structure(results.errors)
            errors.extend(error_errors)
            details["result_error_count"] = len(results.errors)

        # Validate log files if provided
        if log_files:
            log_errors, log_details = self._validate_log_files(log_files)
            errors.extend(log_errors)
            details.update(log_details)

        # Validate expected error types
        if expected_error_types and results and results.errors:
            type_errors = self._validate_expected_error_types(
                results.errors, expected_error_types
            )
            errors.extend(type_errors)

        # Validate error handling behavior
        if results:
            behavior_errors = self._validate_error_handling_behavior(results)
            errors.extend(behavior_errors)

        passed = len(errors) == 0
        message = (
            "Error validation passed"
            if passed
            else f"Found {len(errors)} error handling issues"
        )

        return self._create_result(passed, message, details)

    def _validate_error_structure(self, error_list: List[str]) -> List[str]:
        """Validate the structure of error messages."""
        errors = []

        if not isinstance(error_list, list):
            errors.append("Errors should be provided as a list")
            return errors

        for i, error in enumerate(error_list):
            if not isinstance(error, str):
                errors.append(f"Error {i} is not a string")
            elif len(error.strip()) == 0:
                errors.append(f"Error {i} is empty")

        return errors

    def _validate_log_files(
        self, log_files: List[Path]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Validate log files for error information."""
        errors = []
        details = {}

        log_error_counts = {}
        total_log_errors = 0

        for log_file in log_files:
            if not log_file.exists():
                errors.append(f"Log file does not exist: {log_file}")
                continue

            try:
                with open(log_file, "r") as f:
                    log_content = f.read()

                # Count error messages in log
                error_indicators = [
                    "ERROR",
                    "CRITICAL",
                    "EXCEPTION",
                    "Failed",
                    "Error:",
                ]
                file_error_count = sum(
                    log_content.count(indicator) for indicator in error_indicators
                )

                log_error_counts[log_file.name] = file_error_count
                total_log_errors += file_error_count

            except Exception as e:
                errors.append(f"Failed to read log file {log_file}: {str(e)}")

        details["log_error_counts"] = log_error_counts
        details["total_log_errors"] = total_log_errors

        return errors, details

    def _validate_expected_error_types(
        self, error_list: List[str], expected_types: List[str]
    ) -> List[str]:
        """Validate that expected error types are present."""
        errors = []

        found_types = set()

        for error in error_list:
            error_lower = error.lower()
            for expected_type in expected_types:
                if expected_type.lower() in error_lower:
                    found_types.add(expected_type)

        missing_types = set(expected_types) - found_types
        if missing_types:
            errors.append(f"Expected error types not found: {list(missing_types)}")

        return errors

    def _validate_error_handling_behavior(
        self, results: ProcessingResults
    ) -> List[str]:
        """Validate that error handling behavior is appropriate."""
        errors = []

        # If there are errors, processing should still have attempted all items
        if results.errors and len(results.errors) > 0:
            # Should have processed some items despite errors
            if results.processed_bookmarks == 0:
                errors.append("No items processed despite having error messages")

            # Should have some invalid bookmarks if there are errors
            if results.invalid_bookmarks == 0 and len(results.errors) > 3:
                errors.append("No invalid bookmarks recorded despite multiple errors")

        # If there are invalid bookmarks, there should be corresponding errors
        if results.invalid_bookmarks > 0 and len(results.errors) == 0:
            errors.append("Invalid bookmarks recorded but no error messages found")

        return errors


class CompositeValidator:
    """Combines multiple validators for comprehensive validation."""

    def __init__(self):
        self.validators = {
            "result": ResultValidator(),
            "checkpoint": CheckpointValidator(),
            "error": ErrorValidator(),
        }
        self.logger = logging.getLogger("composite_validator")

    def validate_integration_test(
        self,
        results: ProcessingResults,
        output_file: Path,
        checkpoint_dir: Path = None,
        log_files: List[Path] = None,
        expected_results: Dict[str, Any] = None,
        expected_error_types: List[str] = None,
    ) -> Dict[str, ValidationResult]:
        """Perform comprehensive validation of an integration test."""

        validation_results = {}

        # Result validation
        validation_results["result"] = self.validators["result"].validate(
            results, output_file, expected_results
        )

        # Checkpoint validation (if applicable)
        if checkpoint_dir and checkpoint_dir.exists():
            validation_results["checkpoint"] = self.validators["checkpoint"].validate(
                checkpoint_dir, validate_content=True
            )

        # Error validation
        validation_results["error"] = self.validators["error"].validate(
            results, log_files, expected_error_types
        )

        return validation_results

    def get_overall_result(
        self, validation_results: Dict[str, ValidationResult]
    ) -> bool:
        """Get overall validation result."""
        return all(result.passed for result in validation_results.values())

    def generate_validation_report(
        self, validation_results: Dict[str, ValidationResult]
    ) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        report = {
            "overall_passed": self.get_overall_result(validation_results),
            "validation_count": len(validation_results),
            "passed_count": sum(1 for r in validation_results.values() if r.passed),
            "failed_count": sum(1 for r in validation_results.values() if not r.passed),
            "validations": {},
        }

        for name, result in validation_results.items():
            report["validations"][name] = {
                "passed": result.passed,
                "message": result.message,
                "details": result.details,
                "timestamp": result.timestamp.isoformat(),
            }

        return report
