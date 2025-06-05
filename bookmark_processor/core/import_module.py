"""
Main CSV import module interface for raindrop.io bookmark processing.

This module provides the high-level API for importing raindrop.io CSV files
and transforming them into the application's internal data structure.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .csv_handler import CSVError, RaindropCSVHandler
from .data_models import Bookmark


class ValidationMode(Enum):
    """Validation mode options for CSV import."""

    STRICT = "strict"  # Fail on any validation error
    BEST_EFFORT = "best_effort"  # Continue processing despite some errors
    PERMISSIVE = "permissive"  # Accept files with significant issues


@dataclass
class ImportOptions:
    """Configuration options for CSV import."""

    validation_mode: ValidationMode = ValidationMode.BEST_EFFORT
    max_errors: Optional[int] = None  # Maximum errors before failing
    encoding: Optional[str] = None  # Force specific encoding
    progress_callback: Optional[Callable[[int, int], None]] = None  # (processed, total)
    error_callback: Optional[Callable[[str], None]] = None  # Called for each error
    include_invalid: bool = False  # Include invalid bookmarks in results
    transform_urls: bool = True  # Apply URL normalization/cleaning
    parse_dates: bool = True  # Parse datetime fields
    normalize_tags: bool = True  # Clean and normalize tags


class BookmarkImporter:
    """
    High-level interface for importing raindrop.io CSV files.

    This class provides a simple, clean API for importing and validating
    raindrop.io export CSV files with comprehensive error handling and
    progress reporting.
    """

    def __init__(self, options: Optional[ImportOptions] = None):
        """
        Initialize the bookmark importer.

        Args:
            options: Import configuration options
        """
        self.options = options or ImportOptions()
        self.logger = logging.getLogger(__name__)
        self.csv_handler = RaindropCSVHandler()

        # Import statistics
        self.reset_statistics()

    def reset_statistics(self) -> None:
        """Reset import statistics."""
        self.stats = {
            "total_rows": 0,
            "valid_bookmarks": 0,
            "invalid_bookmarks": 0,
            "errors": [],
            "warnings": [],
            "processing_time": 0.0,
            "file_size_mb": 0.0,
            "encoding_detected": None,
        }

    def import_csv(
        self, file_path: Union[str, Path], options: Optional[ImportOptions] = None
    ) -> List[Bookmark]:
        """
        Import raindrop.io CSV file and return list of bookmarks.

        This is the main entry point for importing CSV files. It handles
        all aspects of loading, validation, and transformation.

        Args:
            file_path: Path to CSV file to import
            options: Override default import options

        Returns:
            List of Bookmark objects

        Raises:
            CSVError: If import fails based on validation mode
            FileNotFoundError: If file doesn't exist
            PermissionError: If file cannot be read
        """
        import time

        start_time = time.time()

        # Use provided options or instance default
        import_options = options or self.options

        try:
            self.reset_statistics()
            file_path = Path(file_path)

            self.logger.info(f"Starting CSV import: {file_path}")
            self.logger.info(f"Validation mode: {import_options.validation_mode.value}")

            # Get file statistics
            self.stats["file_size_mb"] = file_path.stat().st_size / (1024 * 1024)

            # Step 1: Load and validate CSV structure
            bookmarks = self._load_and_transform(file_path, import_options)

            # Step 2: Apply additional validation based on mode
            validated_bookmarks = self._apply_validation_mode(bookmarks, import_options)

            # Step 3: Record final statistics
            self.stats["processing_time"] = time.time() - start_time
            self.stats["valid_bookmarks"] = len(validated_bookmarks)

            self.logger.info(
                f"Import completed: {len(validated_bookmarks)} bookmarks processed"
            )
            self.logger.info(f"Processing time: {self.stats['processing_time']:.2f}s")

            return validated_bookmarks

        except Exception as e:
            self.stats["processing_time"] = time.time() - start_time
            self.logger.error(f"Import failed: {e}")
            raise

    def _load_and_transform(
        self, file_path: Path, options: ImportOptions
    ) -> List[Bookmark]:
        """
        Load CSV file and transform to bookmarks.

        Args:
            file_path: Path to CSV file
            options: Import options

        Returns:
            List of Bookmark objects
        """
        try:
            # Detect encoding first
            if options.encoding:
                self.stats["encoding_detected"] = options.encoding
                self.logger.info(f"Using forced encoding: {options.encoding}")
            else:
                self.stats["encoding_detected"] = self.csv_handler.detect_encoding(
                    file_path
                )
                self.logger.info(
                    f"Detected encoding: {self.stats['encoding_detected']}"
                )

            # Load CSV with appropriate error handling
            if options.validation_mode == ValidationMode.PERMISSIVE:
                # Use recovery mode for permissive validation
                df = self.csv_handler.attempt_recovery(
                    file_path,
                    fill_missing_columns=True,
                    ignore_data_quality=True,
                    skip_validation=False,
                )
            else:
                # Standard loading
                df = self.csv_handler.load_export_csv(file_path)

            self.stats["total_rows"] = len(df)

            # Transform to bookmarks with progress reporting
            bookmarks = []
            errors = []

            for index, row in df.iterrows():
                try:
                    # Report progress
                    if options.progress_callback:
                        options.progress_callback(index + 1, len(df))

                    bookmark = self.csv_handler.transform_row_to_bookmark(row)
                    bookmarks.append(bookmark)

                except Exception as e:
                    error_msg = f"Row {index + 1}: {e}"
                    errors.append(error_msg)

                    # Call error callback if provided
                    if options.error_callback:
                        options.error_callback(error_msg)

                    # Check if we should continue based on validation mode
                    if options.validation_mode == ValidationMode.STRICT:
                        raise CSVError(f"Strict validation failed: {error_msg}")

                    if options.max_errors and len(errors) >= options.max_errors:
                        raise CSVError(
                            f"Too many errors ({len(errors)}), stopping import"
                        )

            # Store errors and warnings
            self.stats["errors"] = errors
            self.stats["invalid_bookmarks"] = len(errors)

            if errors:
                self.logger.warning(f"Import completed with {len(errors)} errors")

            return bookmarks

        except CSVError:
            # Re-raise CSV errors
            raise
        except Exception as e:
            raise CSVError(f"Failed to load and transform CSV: {e}")

    def _apply_validation_mode(
        self, bookmarks: List[Bookmark], options: ImportOptions
    ) -> List[Bookmark]:
        """
        Apply validation rules based on validation mode.

        Args:
            bookmarks: List of bookmarks to validate
            options: Import options

        Returns:
            Filtered list of valid bookmarks
        """
        if options.validation_mode == ValidationMode.PERMISSIVE:
            # In permissive mode, return all bookmarks (even invalid ones)
            if options.include_invalid:
                return bookmarks

        # Filter out invalid bookmarks
        valid_bookmarks = []
        invalid_count = 0

        for bookmark in bookmarks:
            if bookmark.is_valid():
                valid_bookmarks.append(bookmark)
            else:
                invalid_count += 1
                if options.include_invalid:
                    valid_bookmarks.append(bookmark)

        if invalid_count > 0:
            warning_msg = f"Filtered out {invalid_count} invalid bookmarks"
            self.stats["warnings"].append(warning_msg)
            self.logger.warning(warning_msg)

            # In strict mode, fail if any bookmarks are invalid
            if options.validation_mode == ValidationMode.STRICT and invalid_count > 0:
                raise CSVError(
                    f"Strict validation failed: {invalid_count} invalid bookmarks found"
                )

        return valid_bookmarks

    def get_import_statistics(self) -> Dict[str, Any]:
        """
        Get detailed import statistics.

        Returns:
            Dictionary with import statistics
        """
        return self.stats.copy()

    def validate_csv_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate CSV file without importing (dry run).

        Args:
            file_path: Path to CSV file

        Returns:
            Validation report dictionary
        """
        try:
            file_path = Path(file_path)

            # Use the CSV handler's diagnosis function
            diagnosis = self.csv_handler.diagnose_csv_issues(file_path)

            # Add our own analysis
            validation_report = {
                "file_path": str(file_path),
                "file_exists": diagnosis["file_exists"],
                "file_size_mb": (
                    file_path.stat().st_size / (1024 * 1024)
                    if diagnosis["file_exists"]
                    else 0
                ),
                "encoding_detected": diagnosis["encoding_detected"],
                "encoding_confidence": diagnosis["encoding_confidence"],
                "structure_valid": len(diagnosis["structure_issues"]) == 0,
                "structure_issues": diagnosis["structure_issues"],
                "data_quality_issues": diagnosis["data_quality_issues"],
                "parsing_errors": diagnosis["parsing_errors"],
                "recommendations": diagnosis["suggestions"],
                "can_import": (
                    diagnosis["file_exists"]
                    and len(diagnosis["parsing_errors"]) == 0
                    and len(diagnosis["structure_issues"]) == 0
                ),
                "import_mode_recommended": self._recommend_import_mode(diagnosis),
            }

            return validation_report

        except Exception as e:
            return {
                "file_path": str(file_path),
                "validation_error": str(e),
                "can_import": False,
                "import_mode_recommended": ValidationMode.STRICT,
            }

    def _recommend_import_mode(self, diagnosis: Dict[str, Any]) -> ValidationMode:
        """
        Recommend import validation mode based on file diagnosis.

        Args:
            diagnosis: File diagnosis from CSV handler

        Returns:
            Recommended validation mode
        """
        # Count issues
        total_issues = (
            len(diagnosis["structure_issues"])
            + len(diagnosis["data_quality_issues"])
            + len(diagnosis["parsing_errors"])
        )

        if total_issues == 0:
            return ValidationMode.STRICT
        elif total_issues <= 2:
            return ValidationMode.BEST_EFFORT
        else:
            return ValidationMode.PERMISSIVE

    def export_bookmarks(
        self, bookmarks: List[Bookmark], output_path: Union[str, Path]
    ) -> None:
        """
        Export bookmarks to raindrop.io import CSV format.

        Args:
            bookmarks: List of bookmarks to export
            output_path: Path to save CSV file

        Raises:
            CSVError: If export fails
        """
        try:
            self.csv_handler.save_import_csv(bookmarks, output_path)
            self.logger.info(f"Exported {len(bookmarks)} bookmarks to {output_path}")

        except Exception as e:
            raise CSVError(f"Failed to export bookmarks: {e}")


# Convenience functions for simple use cases


def import_raindrop_csv(
    file_path: Union[str, Path],
    validation_mode: ValidationMode = ValidationMode.BEST_EFFORT,
) -> List[Bookmark]:
    """
    Simple function to import a raindrop.io CSV file.

    Args:
        file_path: Path to CSV file
        validation_mode: How strict to be with validation

    Returns:
        List of Bookmark objects

    Raises:
        CSVError: If import fails
    """
    options = ImportOptions(validation_mode=validation_mode)
    importer = BookmarkImporter(options)
    return importer.import_csv(file_path)


def validate_raindrop_csv(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a raindrop.io CSV file without importing.

    Args:
        file_path: Path to CSV file

    Returns:
        Validation report dictionary
    """
    importer = BookmarkImporter()
    return importer.validate_csv_file(file_path)


def convert_raindrop_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    validation_mode: ValidationMode = ValidationMode.BEST_EFFORT,
) -> int:
    """
    Convert raindrop.io export CSV to import CSV format.

    Args:
        input_path: Path to export CSV file
        output_path: Path to save import CSV file
        validation_mode: How strict to be with validation

    Returns:
        Number of bookmarks processed

    Raises:
        CSVError: If conversion fails
    """
    # Import bookmarks
    bookmarks = import_raindrop_csv(input_path, validation_mode)

    # Export in import format
    importer = BookmarkImporter()
    importer.export_bookmarks(bookmarks, output_path)

    return len(bookmarks)
