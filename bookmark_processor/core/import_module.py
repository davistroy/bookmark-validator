"""
Multi-format bookmark import module.

This module provides a unified interface for importing bookmarks from multiple
file formats including raindrop.io CSV exports and Chrome HTML exports.
It also maintains the original CSV-specific functionality for backward compatibility.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .chrome_html_parser import ChromeHTMLParser
from .csv_handler import RaindropCSVHandler
from .data_models import Bookmark
from ..utils.error_handler import (
    ChromeHTMLError,
    CSVError,
    UnsupportedFormatError,
    BookmarkImportError,
)


class MultiFormatImporter:
    """
    Unified importer for multiple bookmark file formats.

    Supports:
    - Raindrop.io CSV exports (11-column format)
    - Chrome HTML bookmark exports (Netscape format)
    """

    def __init__(self):
        """Initialize the multi-format importer."""
        self.logger = logging.getLogger(__name__)
        self.csv_handler = RaindropCSVHandler()
        self.chrome_parser = ChromeHTMLParser()

    def import_bookmarks(self, file_path: Union[str, Path]) -> List[Bookmark]:
        """
        Import bookmarks from a file, auto-detecting the format.

        Args:
            file_path: Path to the bookmark file

        Returns:
            List of Bookmark objects

        Raises:
            BookmarkImportError: If import fails
            UnsupportedFormatError: If file format is not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise BookmarkImportError(f"File not found: {file_path}")

        # Detect file format and use appropriate parser
        file_format = self.detect_format(file_path)

        try:
            if file_format == "csv":
                return self._import_csv(file_path)
            elif file_format == "html":
                return self._import_html(file_path)
            else:
                raise UnsupportedFormatError(f"Unsupported file format: {file_format}")

        except Exception as e:
            self.logger.error(f"Failed to import bookmarks from {file_path}: {str(e)}")
            raise BookmarkImportError(f"Failed to import bookmarks: {str(e)}") from e

    def detect_format(self, file_path: Path) -> str:
        """
        Detect the format of a bookmark file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Format string ("csv", "html", or "unknown")
        """
        try:
            # Check file extension first
            extension = file_path.suffix.lower()

            if extension == ".csv":
                # Verify it's a valid raindrop.io CSV
                if self._is_raindrop_csv(file_path):
                    return "csv"
            elif extension in [".html", ".htm"]:
                # Verify it's a valid Chrome bookmark HTML
                if self.chrome_parser.validate_file(file_path):
                    return "html"

            # Try content-based detection if extension check fails
            return self._detect_by_content(file_path)

        except Exception as e:
            self.logger.warning(f"Error detecting format for {file_path}: {str(e)}")
            return "unknown"

    def _detect_by_content(self, file_path: Path) -> str:
        """
        Detect format by analyzing file content.

        Args:
            file_path: Path to the file

        Returns:
            Format string
        """
        try:
            # Read first few lines to detect format
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                first_lines = []
                for _ in range(10):  # Read first 10 lines
                    line = f.readline()
                    if not line:
                        break
                    first_lines.append(line.strip())

            content_sample = "\n".join(first_lines)

            # Check for Chrome bookmark HTML markers
            if "DOCTYPE NETSCAPE-Bookmark-file-1" in content_sample:
                return "html"
            elif "<DL>" in content_sample.upper() and "<DT>" in content_sample.upper():
                return "html"

            # Check for CSV with raindrop.io header
            if (
                first_lines
                and "id,title,note,excerpt,url,folder,tags,created" in first_lines[0]
            ):
                return "csv"

            # Check if it looks like CSV (comma-separated values)
            if first_lines and "," in first_lines[0] and '"' in content_sample:
                # Could be CSV, try to validate
                if self._is_raindrop_csv(file_path):
                    return "csv"

            return "unknown"

        except Exception as e:
            self.logger.warning(f"Error analyzing content of {file_path}: {str(e)}")
            return "unknown"

    def _is_raindrop_csv(self, file_path: Path) -> bool:
        """
        Check if file is a valid raindrop.io CSV export.

        Args:
            file_path: Path to the CSV file

        Returns:
            True if valid raindrop.io CSV
        """
        try:
            # Use the existing CSV handler validation
            self.csv_handler.validate_export_format(str(file_path))
            return True
        except CSVError:
            return False
        except Exception:
            return False

    def _import_csv(self, file_path: Path) -> List[Bookmark]:
        """
        Import bookmarks from raindrop.io CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of Bookmark objects
        """
        self.logger.info(f"Importing bookmarks from CSV file: {file_path}")

        try:
            # Load CSV data
            df = self.csv_handler.load_export_csv(str(file_path))

            # Convert to bookmark objects
            bookmarks = []
            for _, row in df.iterrows():
                try:
                    bookmark = Bookmark.from_raindrop_export(row.to_dict())
                    bookmarks.append(bookmark)
                except Exception as e:
                    self.logger.warning(f"Failed to create bookmark from CSV row: {e}")

            self.logger.info(
                f"Successfully imported {len(bookmarks)} bookmarks from CSV"
            )
            return bookmarks

        except Exception as e:
            raise BookmarkImportError(f"Failed to import CSV file: {str(e)}") from e

    def _import_html(self, file_path: Path) -> List[Bookmark]:
        """
        Import bookmarks from Chrome HTML file.

        Args:
            file_path: Path to the HTML file

        Returns:
            List of Bookmark objects
        """
        self.logger.info(f"Importing bookmarks from Chrome HTML file: {file_path}")

        try:
            bookmarks = self.chrome_parser.parse_file(file_path)
            self.logger.info(
                f"Successfully imported {len(bookmarks)} bookmarks from Chrome HTML"
            )
            return bookmarks

        except ChromeHTMLError as e:
            raise BookmarkImportError(f"Failed to import Chrome HTML file: {str(e)}") from e

    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get information about a bookmark file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)

        base_info = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size_bytes": 0,
            "format": "unknown",
            "estimated_bookmarks": 0,
            "is_supported": False,
        }

        if not file_path.exists():
            return base_info

        try:
            base_info["size_bytes"] = file_path.stat().st_size
            base_info["format"] = self.detect_format(file_path)
            base_info["is_supported"] = base_info["format"] in ["csv", "html"]

            # Get format-specific info
            if base_info["format"] == "csv":
                try:
                    df = self.csv_handler.load_export_csv(str(file_path))
                    base_info["estimated_bookmarks"] = len(df)
                except Exception:
                    pass
            elif base_info["format"] == "html":
                html_info = self.chrome_parser.get_file_info(file_path)
                base_info["estimated_bookmarks"] = html_info.get(
                    "estimated_bookmark_count", 0
                )

        except Exception as e:
            self.logger.warning(f"Error getting file info for {file_path}: {str(e)}")

        return base_info

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.

        Returns:
            List of supported format names
        """
        return ["csv", "html"]

    def get_format_descriptions(self) -> dict:
        """
        Get descriptions of supported formats.

        Returns:
            Dictionary mapping format names to descriptions
        """
        return {
            "csv": "Raindrop.io CSV export (11-column format)",
            "html": "Chrome HTML bookmark export (Netscape format)",
        }


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
