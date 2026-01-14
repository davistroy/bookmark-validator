"""
CSV handling module for raindrop.io bookmark import/export.

This module handles reading 11-column raindrop.io export CSV files
and writing 6-column raindrop.io import CSV files.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chardet
import pandas as pd

from .data_models import Bookmark
from ..utils.error_handler import (
    CSVError,
    CSVStructureError,
    CSVParsingError,
    CSVValidationError,
    CSVEncodingError,
    CSVFormatError,
)


class RaindropCSVHandler:
    """Handles raindrop.io specific CSV formats."""

    # Raindrop.io export format (11 columns)
    EXPORT_COLUMNS = [
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

    # Raindrop.io import format (6 columns)
    IMPORT_COLUMNS = ["url", "folder", "title", "note", "tags", "created"]

    def __init__(self):
        """Initialize the CSV handler."""
        self.logger = logging.getLogger(__name__)
        # Provide instance properties for backward compatibility with tests
        self.export_columns = self.EXPORT_COLUMNS
        self.import_columns = self.IMPORT_COLUMNS
        self.required_export_columns = ["url"]
        self.required_import_columns = ["url"]

    def detect_encoding(self, file_path: Union[str, Path]) -> str:
        """
        Detect the encoding of a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Detected encoding string
        """
        path = Path(file_path)

        # Try to detect encoding
        try:
            with open(path, "rb") as f:
                # Read first 64KB for detection
                sample = f.read(65536)
                result = chardet.detect(sample)

            encoding = result.get("encoding", "utf-8")
            confidence = result.get("confidence", 0.0)

            self.logger.debug(
                f"Detected encoding: {encoding} (confidence: {confidence:.2f})"
            )

            # If confidence is low, fallback to utf-8
            if confidence < 0.7:
                self.logger.warning(
                    f"Low encoding confidence ({confidence:.2f}), using utf-8"
                )
                encoding = "utf-8"

            return encoding

        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return "utf-8"

    def read_csv_file(
        self, file_path: Union[str, Path], encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read CSV file with proper encoding support.

        Args:
            file_path: Path to the CSV file
            encoding: Specific encoding to use (auto-detected if None)

        Returns:
            Pandas DataFrame with CSV data

        Raises:
            CSVError: If file cannot be read or parsed
        """
        path = Path(file_path)

        if not path.exists():
            raise CSVError(f"CSV file does not exist: {file_path}")

        if not path.is_file():
            raise CSVError(f"Path is not a file: {file_path}")

        # Detect encoding if not provided
        if encoding is None:
            encoding = self.detect_encoding(path)

        self.logger.info(f"Reading CSV file: {path} (encoding: {encoding})")

        # Try multiple encoding strategies
        encodings_to_try = [encoding]
        if encoding != "utf-8":
            encodings_to_try.extend(["utf-8", "utf-8-sig", "latin1", "cp1252"])
        else:
            encodings_to_try.extend(["utf-8-sig", "latin1", "cp1252"])

        last_error = None

        for enc in encodings_to_try:
            try:
                self.logger.debug(f"Trying encoding: {enc}")

                df = pd.read_csv(
                    path,
                    encoding=enc,
                    dtype=str,  # Read all columns as strings initially
                    na_filter=False,  # Don't convert empty strings to NaN
                )

                self.logger.info(f"Successfully read CSV with encoding: {enc}")
                self.logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

                return df

            except UnicodeDecodeError as e:
                last_error = e
                self.logger.debug(f"Encoding {enc} failed: {e}")
                continue
            except pd.errors.EmptyDataError:
                raise CSVParsingError(f"CSV file is empty: {file_path}")
            except pd.errors.ParserError as parse_error:
                raise CSVParsingError(
                    f"CSV parsing error in file {file_path}: {parse_error}"
                )
            except pd.errors.DtypeWarning as dtype_warning:
                self.logger.warning(
                    f"Data type warning for encoding {enc}: {dtype_warning}"
                )
                # Continue processing despite dtype warning
            except Exception as e:
                last_error = e
                self.logger.debug(f"Encoding {enc} failed with unexpected error: {e}")
                continue

        # If we get here, all encodings failed
        raise CSVEncodingError(
            f"Could not read CSV file '{file_path}' with any encoding. "
            f"Tried encodings: {', '.join(encodings_to_try)}. "
            f"Last error: {last_error}"
        )

    def _try_parse_csv(
        self, file_path: Union[str, Path], encodings: List[str]
    ) -> pd.DataFrame:
        """
        Try to parse CSV with different encodings.

        Args:
            file_path: Path to CSV file
            encodings: List of encodings to try

        Returns:
            Successfully parsed DataFrame

        Raises:
            CSVParsingError: If all encodings fail
        """
        path = Path(file_path)

        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except Exception as e:
                self.logger.debug(f"Failed to parse with encoding {encoding}: {e}")
                continue

        raise CSVParsingError(
            f"Could not parse CSV with any of the provided encodings: {encodings}"
        )

    def _clean_dataframe_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize DataFrame values.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()

        # Clean string columns
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == "object":
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: str(x).strip() if pd.notnull(x) else ""
                )

        return cleaned_df

    def read_csv_with_fallback(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read CSV file with multiple fallback strategies.

        Args:
            file_path: Path to the CSV file

        Returns:
            Pandas DataFrame with CSV data

        Raises:
            CSVError: If file cannot be read with any strategy
        """
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise CSVError(f"CSV file not found: {path}")

        # Check if file is empty
        if path.stat().st_size == 0:
            raise CSVValidationError(f"CSV file is empty: {path}")

        # Strategy 1: Auto-detect encoding
        try:
            return self.read_csv_file(path)
        except (CSVParsingError, CSVEncodingError) as auto_detect_error:
            self.logger.warning(f"Auto-detect strategy failed: {auto_detect_error}")
        except CSVError:
            # Re-raise validation errors immediately
            raise

        # Strategy 2: Try with different separators
        separators = [",", ";", "\t"]
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

        for sep in separators:
            for enc in encodings:
                try:
                    self.logger.debug(f"Trying separator '{sep}' with encoding '{enc}'")

                    df = pd.read_csv(
                        path, encoding=enc, sep=sep, dtype=str, na_filter=False
                    )

                    # Check if we got reasonable results
                    if df.shape[1] >= 5:  # At least 5 columns expected
                        self.logger.info(
                            f"Success with separator '{sep}' and encoding '{enc}'"
                        )
                        return df

                except Exception as e:
                    self.logger.debug(
                        f"Separator '{sep}' with encoding '{enc}' failed: {e}"
                    )
                    continue

        raise CSVParsingError(
            f"Could not read CSV file with any parsing strategy: {path}"
        )

    def validate_export_dataframe(self, df: pd.DataFrame) -> None:
        """Alias for validate_export_structure for backward compatibility."""
        return self.validate_export_structure(df)

    def validate_export_structure(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame matches raindrop.io export structure.

        Args:
            df: DataFrame to validate

        Raises:
            CSVError: If structure doesn't match expected format
        """
        if df.empty:
            raise CSVValidationError("CSV file contains no data rows")

        # Check number of columns
        if len(df.columns) != len(self.EXPORT_COLUMNS):
            raise CSVFormatError(
                f"Missing required columns: expected {len(self.EXPORT_COLUMNS)} "
                f"columns, got {len(df.columns)}. "
                f"Expected columns: {', '.join(self.EXPORT_COLUMNS)}. "
                f"Found columns: {', '.join(df.columns)}"
            )

        # Check column names (case-insensitive)
        df_columns_lower = [col.lower().strip() for col in df.columns]
        expected_columns_lower = [col.lower() for col in self.EXPORT_COLUMNS]

        missing_columns = []
        extra_columns = []

        for expected_col in expected_columns_lower:
            if expected_col not in df_columns_lower:
                # Find the original case version
                original = self.EXPORT_COLUMNS[
                    expected_columns_lower.index(expected_col)
                ]
                missing_columns.append(original)

        for i, df_col in enumerate(df_columns_lower):
            if df_col not in expected_columns_lower:
                extra_columns.append(df.columns[i])

        error_messages = []
        if missing_columns:
            error_messages.append(f"Missing columns: {', '.join(missing_columns)}")
        if extra_columns:
            error_messages.append(f"Unexpected columns: {', '.join(extra_columns)}")

        if error_messages:
            raise CSVFormatError(
                f"Missing required columns. {' '.join(error_messages)}. "
                f"Expected columns: {', '.join(self.EXPORT_COLUMNS)}"
            )

        # Check for required column - at minimum 'url' should have some data
        if "url" in df.columns:
            # Check for any empty URLs
            empty_urls = df["url"].isna() | (df["url"].str.strip() == "")
            if empty_urls.any():
                raise CSVValidationError(
                    "Empty URL found in row(s). All bookmarks must have valid URLs"
                )

            # Check for duplicate URLs
            duplicates = df["url"].duplicated()
            if duplicates.any():
                raise CSVValidationError(
                    "Duplicate URLs found. All bookmark URLs must be unique"
                )

        self.logger.info("CSV structure validation passed")

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to match expected format.

        Args:
            df: DataFrame with potentially misnamed columns

        Returns:
            DataFrame with normalized column names
        """
        # Create mapping from current columns to expected columns
        expected_columns_lower = [col.lower() for col in self.EXPORT_COLUMNS]

        column_mapping = {}
        for i, df_col in enumerate(df.columns):
            df_col_lower = df_col.lower().strip()
            if df_col_lower in expected_columns_lower:
                expected_idx = expected_columns_lower.index(df_col_lower)
                expected_col = self.EXPORT_COLUMNS[expected_idx]
                if df_col != expected_col:
                    column_mapping[df_col] = expected_col

        if column_mapping:
            self.logger.info(f"Normalizing column names: {column_mapping}")
            df = df.rename(columns=column_mapping)

        return df

    def load_export_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and validate raindrop.io export CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Validated and normalized DataFrame

        Raises:
            CSVEncodingError: If file cannot be read due to encoding issues
            CSVParsingError: If file cannot be parsed due to malformation
            CSVStructureError: If file structure doesn't match expected format
            CSVValidationError: If file data fails validation rules
        """
        try:
            # Read the CSV file
            df = self.read_csv_with_fallback(file_path)

            # Validate we got some data
            if df is None:
                raise CSVParsingError(f"Failed to read CSV file: {file_path}")

            # Check for completely empty DataFrame
            if df.empty:
                raise CSVValidationError(f"CSV file contains no data: {file_path}")

            # Normalize column names
            df = self.normalize_column_names(df)

            # Validate structure
            self.validate_export_structure(df)

        except CSVError:
            # Re-raise our custom exceptions
            raise
        except pd.errors.EmptyDataError:
            raise CSVValidationError(
                f"CSV file is empty or contains only headers: {file_path}"
            )
        except pd.errors.ParserError as e:
            raise CSVParsingError(f"CSV parsing failed for {file_path}: {str(e)}")
        except Exception as e:
            raise CSVError(f"Unexpected error reading CSV file {file_path}: {str(e)}")

        # Additional data quality checks
        self._perform_data_quality_checks(df, file_path)

        # Log statistics
        self.logger.info(f"Loaded {len(df)} bookmarks from {file_path}")

        # Count non-empty URLs
        if "url" in df.columns:
            valid_urls = df["url"].notna() & (df["url"].str.strip() != "")
            self.logger.info(f"Found {valid_urls.sum()} bookmarks with URLs")

        return df

    def _perform_data_quality_checks(
        self, df: pd.DataFrame, file_path: Union[str, Path]
    ) -> None:
        """
        Perform additional data quality checks on the loaded DataFrame.

        Args:
            df: DataFrame to check
            file_path: Path to the original file (for error messages)

        Raises:
            CSVValidationError: If data quality issues are found
        """
        quality_issues = []

        # Check for rows with all empty values
        empty_rows = df.isnull().all(axis=1) | (df == "").all(axis=1)
        if empty_rows.any():
            empty_count = empty_rows.sum()
            quality_issues.append(f"{empty_count} completely empty rows found")
            self.logger.warning(f"Found {empty_count} empty rows in {file_path}")

        # Check for excessive missing data in critical columns
        critical_columns = ["id", "title", "url"]
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum() + (df[col] == "").sum()
                missing_pct = (missing_count / len(df)) * 100

                if missing_pct > 90:  # More than 90% missing
                    quality_issues.append(f"Column '{col}' is {missing_pct:.1f}% empty")
                elif missing_pct > 50:  # More than 50% missing - warning only
                    self.logger.warning(
                        f"Column '{col}' is {missing_pct:.1f}% empty in {file_path}"
                    )

        # Check for extremely long values that might indicate data corruption
        text_columns = ["title", "note", "excerpt"]
        for col in text_columns:
            if col in df.columns:
                max_length = df[col].astype(str).str.len().max()
                if max_length > 10000:  # Extremely long text
                    self.logger.warning(
                        f"Column '{col}' contains very long values "
                        f"(max: {max_length} chars) in {file_path}"
                    )

        # Check for duplicate IDs if ID column exists
        if "id" in df.columns:
            non_empty_ids = df["id"].dropna()
            if len(non_empty_ids) != len(non_empty_ids.unique()):
                duplicate_count = len(non_empty_ids) - len(non_empty_ids.unique())
                quality_issues.append(f"{duplicate_count} duplicate ID values found")

        # Check for malformed URLs in URL column
        if "url" in df.columns:
            urls = df["url"].dropna()
            urls = urls[urls.str.strip() != ""]  # Remove empty strings

            if len(urls) > 0:
                # Basic URL format check - more permissive
                # Allow URLs that start with common protocols or domains
                def is_valid_url_format(url):
                    if not url or not isinstance(url, str):
                        return False
                    url = url.strip()
                    return (
                        url.startswith(("http://", "https://"))  # Standard URLs
                        or url.startswith(("www.", "ftp://"))  # Common patterns
                        or (
                            "." in url and not url.startswith(("javascript:", "data:"))
                        )  # Domain-like
                    )

                valid_urls = urls.apply(is_valid_url_format)
                invalid_count = (~valid_urls).sum()

                if invalid_count > 0:
                    invalid_pct = (invalid_count / len(urls)) * 100
                    if invalid_pct > 75:  # More than 75% clearly invalid URLs
                        quality_issues.append(
                            f"{invalid_count} URLs appear to be invalid or malformed"
                        )
                    elif (
                        invalid_pct > 25
                    ):  # More than 25% suspicious URLs - warning only
                        self.logger.warning(
                            f"Found {invalid_count} potentially unusual URLs "
                            f"in {file_path}"
                        )

        # Check for inconsistent row lengths
        # (this should be caught by pandas, but double-check)
        row_lengths = df.count(axis=1)
        if row_lengths.nunique() > 1:
            self.logger.warning(f"Inconsistent row lengths detected in {file_path}")

        # If we have critical quality issues, raise an error
        if quality_issues:
            error_msg = (
                f"Data quality issues in {file_path}: {'; '.join(quality_issues)}"
            )
            self.logger.error(error_msg)
            raise CSVValidationError(error_msg)

        self.logger.info(f"Data quality checks passed for {file_path}")

    def create_sample_export_csv(self, file_path: Union[str, Path]) -> None:
        """
        Create a sample raindrop.io export CSV for testing.

        Args:
            file_path: Path where to save the sample CSV
        """
        sample_data = [
            {
                "id": "1",
                "title": "Example Website",
                "note": "This is a test bookmark",
                "excerpt": "Example excerpt from the page",
                "url": "https://example.com",
                "folder": "Test Folder",
                "tags": "test, example, bookmark",
                "created": "2024-01-01T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            {
                "id": "2",
                "title": "Another Example",
                "note": "Second test bookmark",
                "excerpt": "",
                "url": "https://test.example.org",
                "folder": "Test Folder/Subfolder",
                "tags": "test, demo",
                "created": "2024-01-02T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "true",
            },
        ]

        df = pd.DataFrame(sample_data)
        df.to_csv(file_path, index=False, encoding="utf-8")
        self.logger.info(f"Created sample CSV: {file_path}")

    def diagnose_csv_issues(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Diagnose CSV file issues without raising exceptions.

        Args:
            file_path: Path to the CSV file to diagnose

        Returns:
            Dictionary with diagnosis information
        """
        path = Path(file_path)
        diagnosis = {
            "file_exists": path.exists(),
            "is_file": path.is_file() if path.exists() else False,
            "file_size": path.stat().st_size if path.exists() and path.is_file() else 0,
            "encoding_detected": None,
            "encoding_confidence": 0.0,
            "parsing_errors": [],
            "structure_issues": [],
            "data_quality_issues": [],
            "suggestions": [],
        }

        if not diagnosis["file_exists"]:
            diagnosis["suggestions"].append("Check if the file path is correct")
            return diagnosis

        if not diagnosis["is_file"]:
            diagnosis["suggestions"].append("Path exists but is not a file")
            return diagnosis

        if diagnosis["file_size"] == 0:
            diagnosis["suggestions"].append("File is empty")
            return diagnosis

        # Try encoding detection
        try:
            encoding = self.detect_encoding(path)
            diagnosis["encoding_detected"] = encoding

            # Get confidence from chardet
            with open(path, "rb") as f:
                sample = f.read(65536)
                result = chardet.detect(sample)
                diagnosis["encoding_confidence"] = result.get("confidence", 0.0)

        except Exception as e:
            diagnosis["parsing_errors"].append(f"Encoding detection failed: {e}")

        # Try reading with different strategies
        parsing_successful = False

        try:
            df = self.read_csv_with_fallback(path)
            parsing_successful = True

            # Check structure
            try:
                self.validate_export_structure(df)
            except CSVStructureError as e:
                diagnosis["structure_issues"].append(str(e))
            except CSVValidationError as e:
                diagnosis["data_quality_issues"].append(str(e))

        except Exception as e:
            diagnosis["parsing_errors"].append(str(e))

        # Generate suggestions based on findings
        if not parsing_successful:
            diagnosis["suggestions"].extend(
                [
                    "Try opening the file in a text editor to check for obvious issues",
                    "Check if the file uses a different delimiter (semicolon, tab)",
                    "Verify the file is a proper CSV format",
                ]
            )

        if diagnosis["encoding_confidence"] < 0.7:
            diagnosis["suggestions"].append(
                "File encoding detection has low confidence - try saving as UTF-8"
            )

        if diagnosis["structure_issues"]:
            diagnosis["suggestions"].append(
                "File structure doesn't match raindrop.io export format"
            )

        if diagnosis["data_quality_issues"]:
            diagnosis["suggestions"].append(
                "File has data quality issues that need attention"
            )

        return diagnosis

    def attempt_recovery(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Attempt to recover a problematic CSV file with relaxed validation.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional options for recovery
                - skip_validation: Skip structure validation
                - fill_missing_columns: Add missing columns with empty values
                - ignore_data_quality: Skip data quality checks

        Returns:
            DataFrame with recovered data

        Raises:
            CSVError: If recovery is not possible
        """
        path = Path(file_path)
        self.logger.info(f"Attempting recovery of CSV file: {path}")

        # Try to read with fallback methods
        try:
            df = self.read_csv_with_fallback(path)
        except Exception as e:
            raise CSVParsingError(
                f"Could not read file even with recovery methods: {e}"
            )

        if df.empty:
            raise CSVValidationError("File contains no data to recover")

        # Normalize column names if possible
        df = self.normalize_column_names(df)

        # Handle missing columns if requested
        if kwargs.get("fill_missing_columns", False):
            missing_cols = set(self.EXPORT_COLUMNS) - set(df.columns)
            for col in missing_cols:
                df[col] = ""
                self.logger.info(f"Added missing column '{col}' with empty values")

            # Reorder columns to match expected format
            df = df[self.EXPORT_COLUMNS]

        # Skip validation if requested
        if not kwargs.get("skip_validation", False):
            try:
                self.validate_export_structure(df)
            except CSVStructureError as e:
                if not kwargs.get("fill_missing_columns", False):
                    raise CSVError(
                        f"Structure validation failed and "
                        f"fill_missing_columns=False: {e}"
                    )

        # Skip data quality checks if requested
        if not kwargs.get("ignore_data_quality", False):
            try:
                self._perform_data_quality_checks(df, path)
            except CSVValidationError as e:
                self.logger.warning(f"Data quality issues found but ignored: {e}")

        self.logger.info(f"Successfully recovered {len(df)} rows from {path}")
        return df

    def transform_row_to_bookmark(self, row: pd.Series) -> Bookmark:
        """
        Transform a pandas Series row to a Bookmark object.

        Args:
            row: Pandas Series containing bookmark data

        Returns:
            Bookmark object with transformed data

        Raises:
            CSVValidationError: If required fields are missing or invalid
        """
        try:
            # Extract and clean basic fields
            bookmark = Bookmark(
                id=self._clean_string_field(row.get("id", "")),
                title=self._clean_string_field(row.get("title", "")),
                note=self._clean_string_field(row.get("note", "")),
                excerpt=self._clean_string_field(row.get("excerpt", "")),
                url=self._clean_url_field(row.get("url", "")),
                folder=self._clean_string_field(row.get("folder", "")),
                tags=self._parse_tags_field(row.get("tags", "")),
                created=self._parse_datetime_field(row.get("created", "")),
                cover=self._clean_string_field(row.get("cover", "")),
                highlights=self._clean_string_field(row.get("highlights", "")),
                favorite=self._parse_boolean_field(row.get("favorite", False)),
            )

            # Validate minimum requirements
            if not bookmark.url:
                raise CSVValidationError(
                    f"Bookmark missing required URL: {row.to_dict()}"
                )

            return bookmark

        except Exception as e:
            if isinstance(e, CSVValidationError):
                raise
            raise CSVValidationError(f"Error transforming row to bookmark: {e}")

    def _clean_string_field(self, value: Any) -> str:
        """
        Clean and normalize string field.

        Args:
            value: Raw field value

        Returns:
            Cleaned string value
        """
        if pd.isna(value) or value is None:
            return ""

        # Convert to string and strip whitespace
        cleaned = str(value).strip()

        # Remove null bytes and other problematic characters
        cleaned = cleaned.replace("\x00", "").replace("\r", "").strip()

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned

    def _clean_url_field(self, value: Any) -> str:
        """
        Clean and validate URL field.

        Args:
            value: Raw URL value

        Returns:
            Cleaned URL string
        """
        url = self._clean_string_field(value)

        if not url:
            return ""

        # Basic URL cleaning
        url = url.strip()

        # Remove common URL artifacts
        if url.startswith(("http://", "https://")):
            # URL seems valid, just clean it up
            url = re.sub(r"\s", "", url)  # Remove any whitespace

        elif url.startswith(
            ("javascript:", "data:", "mailto:", "ftp:", "file:")
        ):
            # Special protocols - don't modify
            pass

        elif url.startswith("www."):
            # Add https prefix
            url = f"https://{url}"

        elif "." in url:
            # Looks like a domain, add https prefix
            url = f"https://{url}"

        return url

    def _parse_tags_field(self, value: Any) -> List[str]:
        """
        Parse tags field into list of individual tags.

        Args:
            value: Raw tags value (string or list)

        Returns:
            List of cleaned tag strings
        """
        if pd.isna(value) or value is None:
            return []

        # Handle different tag formats
        if isinstance(value, list):
            tags = value
        else:
            tags_str = self._clean_string_field(value)
            if not tags_str:
                return []

            # Split on various delimiters
            tags = re.split(r"[,;|]", tags_str)

        # Clean and filter tags
        cleaned_tags = []
        for tag in tags:
            if isinstance(tag, str):
                cleaned_tag = tag.strip().lower()
                # Remove quotes and extra whitespace
                cleaned_tag = cleaned_tag.strip("\"'")

                if cleaned_tag and len(cleaned_tag) > 1:
                    cleaned_tags.append(cleaned_tag)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in cleaned_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags

    def _parse_datetime_field(self, value: Any) -> Optional[datetime]:
        """
        Parse datetime field.

        Args:
            value: Raw datetime value

        Returns:
            Parsed datetime object or None
        """
        if pd.isna(value) or value is None:
            return None

        datetime_str = self._clean_string_field(value)
        if not datetime_str:
            return None

        # Try various datetime formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format with Z
            "%Y-%m-%dT%H:%M:%S+00:00",  # ISO format with timezone
            "%Y-%m-%dT%H:%M:%S",  # ISO format
            "%Y-%m-%d %H:%M:%S",  # Standard format
            "%Y/%m/%d %H:%M:%S",  # Slash format with time
            "%Y-%m-%d",  # Date only
            "%m/%d/%Y",  # US format
            "%d/%m/%Y",  # European format
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                # If timezone aware, convert to UTC naive
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue

        # Try to handle timezone formats manually
        try:
            # Handle formats like "2024-01-01T00:00:00+00:00"
            if "+" in datetime_str or datetime_str.endswith("Z"):
                # Remove timezone info for basic parsing
                clean_dt = datetime_str.replace("+00:00", "").replace("Z", "")
                return datetime.fromisoformat(clean_dt)
        except (ValueError, AttributeError):
            pass

        # If all formats fail, log warning and return None
        self.logger.warning(f"Could not parse datetime: {datetime_str}")
        return None

    def _parse_boolean_field(self, value: Any) -> bool:
        """
        Parse boolean field.

        Args:
            value: Raw boolean value

        Returns:
            Boolean value
        """
        if pd.isna(value) or value is None:
            return False

        if isinstance(value, bool):
            return value

        # Convert string representations
        str_value = str(value).lower().strip()
        return str_value in ("true", "1", "yes", "on", "enabled")

    def transform_export_to_bookmarks(self, df: pd.DataFrame) -> List[Bookmark]:
        """Alias for transform_dataframe_to_bookmarks for backward compatibility."""
        return self.transform_dataframe_to_bookmarks(df)

    def transform_dataframe_to_bookmarks(self, df: pd.DataFrame) -> List[Bookmark]:
        """
        Transform entire DataFrame to list of Bookmark objects.

        Args:
            df: DataFrame with raindrop.io export data

        Returns:
            List of Bookmark objects

        Raises:
            CSVValidationError: If transformation fails
        """
        bookmarks = []
        errors = []

        self.logger.info(f"Transforming {len(df)} rows to Bookmark objects")

        for index, row in df.iterrows():
            try:
                bookmark = self.transform_row_to_bookmark(row)
                bookmarks.append(bookmark)

            except CSVValidationError as e:
                error_msg = f"Row {index + 1}: {e}"
                errors.append(error_msg)
                self.logger.warning(error_msg)
                continue

            except Exception as e:
                error_msg = f"Row {index + 1}: Unexpected error: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                continue

        self.logger.info(
            f"Successfully transformed {len(bookmarks)} bookmarks, {len(errors)} errors"
        )

        if errors and len(bookmarks) == 0:
            raise CSVValidationError(
                f"All rows failed transformation. Errors: {errors[:5]}"
            )

        return bookmarks

    def bookmarks_to_dataframe(self, bookmarks: List[Bookmark]) -> pd.DataFrame:
        """
        Convert list of Bookmark objects to DataFrame for export.

        Args:
            bookmarks: List of Bookmark objects

        Returns:
            DataFrame in raindrop.io import format
        """
        if not bookmarks:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=self.IMPORT_COLUMNS)

        # Convert bookmarks to export dictionaries
        export_data = []
        for bookmark in bookmarks:
            if bookmark.is_valid():
                export_data.append(bookmark.to_export_dict())
            else:
                self.logger.warning(f"Skipping invalid bookmark: {bookmark.url}")

        if not export_data:
            self.logger.warning("No valid bookmarks to export")
            return pd.DataFrame(columns=self.IMPORT_COLUMNS)

        # Create DataFrame
        df = pd.DataFrame(export_data)

        # Ensure column order matches import format
        df = df[self.IMPORT_COLUMNS]

        self.logger.info(f"Created export DataFrame with {len(df)} bookmarks")
        return df

    def save_import_csv(
        self, bookmarks: List[Bookmark], file_path: Union[str, Path]
    ) -> None:
        """
        Save bookmarks as raindrop.io import CSV file.

        Args:
            bookmarks: List of processed bookmarks
            file_path: Path to save the CSV file

        Raises:
            CSVError: If saving fails
        """
        try:
            df = self.bookmarks_to_dataframe(bookmarks)

            if df.empty:
                raise CSVError("No valid bookmarks to export")

            # Save with proper encoding and formatting
            df.to_csv(
                file_path,
                index=False,
                encoding="utf-8-sig",  # UTF-8 with BOM for better compatibility
                quoting=1,  # Quote all non-numeric fields
                na_rep="",  # Empty string for NaN values
                escapechar=None,  # Don't escape quotes
            )

            self.logger.info(f"Saved {len(df)} bookmarks to {file_path}")

        except Exception as e:
            raise CSVError(f"Failed to save CSV file {file_path}: {e}")

    def load_and_transform_csv(self, file_path: Union[str, Path]) -> List[Bookmark]:
        """
        Load CSV file and transform to list of Bookmark objects.

        This is the main entry point for loading and transforming
        raindrop.io export CSV files.

        Args:
            file_path: Path to CSV file

        Returns:
            List of Bookmark objects

        Raises:
            CSVError: If loading or transformation fails
        """
        # Load the CSV file
        df = self.load_export_csv(file_path)

        # Transform to bookmarks
        bookmarks = self.transform_dataframe_to_bookmarks(df)

        return bookmarks


# For backward compatibility and easier imports
def load_export_csv(file_path):
    """Load export CSV using default handler."""
    return RaindropCSVHandler().load_export_csv(file_path)
