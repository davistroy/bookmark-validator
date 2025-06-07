"""
Data Recovery and Malformed Data Handling

This module provides strategies for handling missing, null, or malformed data
in CSV imports. It includes sanitization functions, recovery mechanisms,
and clear error messages to help users understand and fix data issues.
"""

import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from .csv_field_validators import get_field_validator
from .input_validator import ValidationResult, ValidationSeverity


class DataRecoveryStrategy:
    """Defines how to handle missing or invalid data for a specific field"""

    def __init__(
        self,
        strategy: str,
        default_value: Any = None,
        recovery_function: Optional[Callable] = None,
        required: bool = False,
    ):
        """
        Initialize data recovery strategy

        Args:
            strategy: Recovery strategy ('skip', 'default', 'derive', 'error')
            default_value: Default value to use
            recovery_function: Function to derive/recover value
            required: Whether field is absolutely required
        """
        self.strategy = strategy
        self.default_value = default_value
        self.recovery_function = recovery_function
        self.required = required


class DataRecoveryManager:
    """Manages data recovery strategies for different fields and situations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recovery_strategies = self._init_default_strategies()
        self.sanitization_functions = self._init_sanitization_functions()

    def _init_default_strategies(self) -> Dict[str, DataRecoveryStrategy]:
        """Initialize default recovery strategies for each field"""
        return {
            "id": DataRecoveryStrategy("derive", recovery_function=self._generate_id),
            "title": DataRecoveryStrategy(
                "derive", recovery_function=self._derive_title
            ),
            "note": DataRecoveryStrategy("default", default_value=""),
            "excerpt": DataRecoveryStrategy("default", default_value=""),
            "url": DataRecoveryStrategy("error", required=True),  # URLs are critical
            "folder": DataRecoveryStrategy("default", default_value=""),
            "tags": DataRecoveryStrategy("default", default_value=[]),
            "created": DataRecoveryStrategy(
                "derive", recovery_function=self._derive_created_date
            ),
            "cover": DataRecoveryStrategy("default", default_value=""),
            "highlights": DataRecoveryStrategy("default", default_value=""),
            "favorite": DataRecoveryStrategy("default", default_value=False),
        }

    def _init_sanitization_functions(self) -> Dict[str, Callable]:
        """Initialize sanitization functions for each field type"""
        return {
            "string": self._sanitize_string,
            "url": self._sanitize_url,
            "tags": self._sanitize_tags,
            "boolean": self._sanitize_boolean,
            "datetime": self._sanitize_datetime,
            "folder": self._sanitize_folder_path,
        }

    def handle_missing_data(
        self, field_name: str, record: Dict[str, Any]
    ) -> Tuple[Any, List[str]]:
        """
        Handle missing data for a specific field

        Args:
            field_name: Name of the missing field
            record: Complete record data for context

        Returns:
            Tuple of (recovered_value, list_of_messages)
        """
        messages = []
        strategy = self.recovery_strategies.get(field_name)

        if not strategy:
            messages.append(
                f"No recovery strategy for field '{field_name}', using empty string"
            )
            return "", messages

        if strategy.required:
            messages.append(f"Required field '{field_name}' is missing")
            return None, messages

        if strategy.strategy == "default":
            messages.append(f"Using default value for missing field '{field_name}'")
            return strategy.default_value, messages

        elif strategy.strategy == "derive":
            if strategy.recovery_function:
                try:
                    derived_value = strategy.recovery_function(record)
                    messages.append(f"Derived value for missing field '{field_name}'")
                    return derived_value, messages
                except Exception as e:
                    messages.append(f"Failed to derive value for '{field_name}': {e}")
                    return strategy.default_value, messages
            else:
                messages.append(
                    f"No recovery function for field '{field_name}', using default"
                )
                return strategy.default_value, messages

        elif strategy.strategy == "skip":
            messages.append(f"Skipping missing field '{field_name}'")
            return None, messages

        elif strategy.strategy == "error":
            messages.append(f"Missing required field '{field_name}'")
            return None, messages

        else:
            messages.append(
                f"Unknown recovery strategy '{strategy.strategy}' for field '{field_name}'"
            )
            return strategy.default_value, messages

    def handle_malformed_data(
        self, field_name: str, value: Any, record: Dict[str, Any]
    ) -> Tuple[Any, List[str]]:
        """
        Handle malformed data for a specific field

        Args:
            field_name: Name of the field
            value: Malformed value
            record: Complete record data for context

        Returns:
            Tuple of (sanitized_value, list_of_messages)
        """
        messages = []

        # Try field-specific sanitization first
        field_type = self._get_field_type(field_name)
        sanitization_func = self.sanitization_functions.get(
            field_type, self._sanitize_string
        )

        try:
            sanitized_value = sanitization_func(value)
            if sanitized_value != value:
                messages.append(f"Sanitized malformed data in field '{field_name}'")

            # Validate the sanitized value
            validator = get_field_validator(field_name)
            validation_result = validator.validate(sanitized_value)

            if validation_result.is_valid:
                return validation_result.sanitized_value, messages
            else:
                # If still invalid after sanitization, try recovery
                error_messages = [
                    issue.message for issue in validation_result.get_errors()
                ]
                messages.extend(error_messages)

                # Attempt data recovery
                recovered_value, recovery_messages = self.handle_missing_data(
                    field_name, record
                )
                messages.extend(recovery_messages)
                messages.append(
                    f"Used recovery value for malformed field '{field_name}'"
                )

                return recovered_value, messages

        except Exception as e:
            messages.append(f"Sanitization failed for field '{field_name}': {e}")

            # Fall back to recovery strategy
            recovered_value, recovery_messages = self.handle_missing_data(
                field_name, record
            )
            messages.extend(recovery_messages)

            return recovered_value, messages

    def recover_partial_record(
        self, partial_record: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Recover a partially valid record by filling in missing/invalid fields

        Args:
            partial_record: Record with some missing or invalid fields

        Returns:
            Tuple of (recovered_record, list_of_messages)
        """
        messages = []
        recovered_record = {}

        # Define expected fields based on raindrop.io format
        expected_fields = [
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

        for field_name in expected_fields:
            if field_name in partial_record and partial_record[field_name] is not None:
                # Field exists, validate it
                value = partial_record[field_name]
                validator = get_field_validator(field_name)
                validation_result = validator.validate(value)

                if validation_result.is_valid:
                    recovered_record[field_name] = validation_result.sanitized_value
                else:
                    # Handle malformed data
                    recovered_value, field_messages = self.handle_malformed_data(
                        field_name, value, partial_record
                    )
                    recovered_record[field_name] = recovered_value
                    messages.extend(field_messages)
            else:
                # Field missing, handle accordingly
                recovered_value, field_messages = self.handle_missing_data(
                    field_name, partial_record
                )
                recovered_record[field_name] = recovered_value
                messages.extend(field_messages)

        return recovered_record, messages

    def _get_field_type(self, field_name: str) -> str:
        """Get the data type for a specific field"""
        field_types = {
            "id": "string",
            "title": "string",
            "note": "string",
            "excerpt": "string",
            "url": "url",
            "folder": "folder",
            "tags": "tags",
            "created": "datetime",
            "cover": "string",
            "highlights": "string",
            "favorite": "boolean",
        }
        return field_types.get(field_name, "string")

    # Sanitization functions
    def _sanitize_string(self, value: Any) -> str:
        """Sanitize general string values"""
        if value is None:
            return ""

        # Convert to string
        text = str(value)

        # Remove null bytes and control characters
        text = text.replace("\x00", "").replace("\r", "")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove HTML tags if present
        html_pattern = re.compile(r"<[^>]+>")
        if html_pattern.search(text):
            text = html_pattern.sub("", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def _sanitize_url(self, value: Any) -> str:
        """Sanitize URL values"""
        if value is None:
            return ""

        url = self._sanitize_string(value)
        if not url:
            return ""

        # Remove whitespace
        url = re.sub(r"\s", "", url)

        # Add protocol if missing
        if url and not url.startswith(("http://", "https://", "ftp://", "file://")):
            if url.startswith("www."):
                url = f"https://{url}"
            elif "." in url and not url.startswith(("mailto:", "javascript:", "data:")):
                url = f"https://{url}"

        return url

    def _sanitize_tags(self, value: Any) -> List[str]:
        """Sanitize tag values"""
        if value is None:
            return []

        if isinstance(value, list):
            tags = [self._sanitize_string(tag) for tag in value]
        else:
            # Parse string tags
            tags_str = self._sanitize_string(value)
            if not tags_str:
                return []

            # Handle quoted strings
            if tags_str.startswith('"') and tags_str.endswith('"'):
                tags_str = tags_str[1:-1]

            # Split on various delimiters
            if "," in tags_str:
                tags = [tag.strip() for tag in tags_str.split(",")]
            elif ";" in tags_str:
                tags = [tag.strip() for tag in tags_str.split(";")]
            else:
                tags = [tags_str]

        # Clean and filter tags
        cleaned_tags = []
        for tag in tags:
            if tag and len(tag.strip()) > 1:
                clean_tag = tag.strip().lower()
                clean_tag = re.sub(r"[^a-zA-Z0-9\s\-_.]", "", clean_tag)
                if clean_tag and clean_tag not in cleaned_tags:
                    cleaned_tags.append(clean_tag)

        return cleaned_tags

    def _sanitize_boolean(self, value: Any) -> bool:
        """Sanitize boolean values"""
        if value is None:
            return False

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            value_lower = value.strip().lower()
            return value_lower in ("true", "1", "yes", "on", "enabled", "y")

        return False

    def _sanitize_datetime(self, value: Any) -> Optional[datetime]:
        """Sanitize datetime values"""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        datetime_str = self._sanitize_string(value)
        if not datetime_str:
            return None

        # Try various datetime formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue

        return None

    def _sanitize_folder_path(self, value: Any) -> str:
        """Sanitize folder path values"""
        if value is None:
            return ""

        folder = self._sanitize_string(value)
        if not folder:
            return ""

        # Replace backslashes with forward slashes
        folder = folder.replace("\\", "/")

        # Remove leading/trailing slashes
        folder = folder.strip("/")

        # Clean up multiple consecutive slashes
        folder = re.sub(r"/+", "/", folder)

        # Remove invalid filesystem characters
        folder = re.sub(r'[<>:"|?*\x00-\x1f]', "", folder)

        return folder

    # Recovery functions for deriving missing values
    def _generate_id(self, record: Dict[str, Any]) -> str:
        """Generate ID from URL or other identifying information"""
        import hashlib

        # Try to generate ID from URL
        url = record.get("url", "")
        if url:
            # Use URL hash as ID
            url_hash = hashlib.md5(
                url.encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:8]
            return f"auto_{url_hash}"

        # Try to generate from title
        title = record.get("title", "")
        if title:
            # Sanitize title for ID
            clean_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title.lower())[:20]
            return f"auto_{clean_title}"

        # Generate random ID
        import time

        return f"auto_{int(time.time())}"

    def _derive_title(self, record: Dict[str, Any]) -> str:
        """Derive title from URL or other available data"""
        # Try to extract from URL
        url = record.get("url", "")
        if url:
            try:
                parsed = urlparse(url)
                # Use domain as title
                domain = parsed.netloc
                if domain:
                    # Clean up domain name
                    if domain.startswith("www."):
                        domain = domain[4:]
                    return domain.title()
            except Exception:
                pass

        # Try to use note or excerpt
        note = record.get("note", "")
        if note and len(note) > 5:
            # Use first line of note as title
            first_line = note.split("\n")[0].strip()
            if len(first_line) > 3:
                return first_line[:100]  # Limit length

        excerpt = record.get("excerpt", "")
        if excerpt and len(excerpt) > 5:
            # Use first part of excerpt as title
            return excerpt[:100].strip()

        return "Untitled Bookmark"

    def _derive_created_date(self, record: Dict[str, Any]) -> datetime:
        """Derive creation date (default to current time)"""
        return datetime.now()


class MalformedDataDetector:
    """Detects various types of malformed data patterns"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_encoding_issues(self, text: str) -> List[str]:
        """Detect encoding-related issues in text"""
        issues = []

        if not isinstance(text, str):
            return issues

        # Check for null bytes
        if "\x00" in text:
            issues.append("Contains null bytes")

        # Check for replacement characters (encoding errors)
        if "\ufffd" in text:
            issues.append("Contains replacement characters (encoding errors)")

        # Check for control characters
        control_chars = [
            chr(i) for i in range(32) if i not in (9, 10, 13)
        ]  # Exclude tab, LF, CR
        if any(char in text for char in control_chars):
            issues.append("Contains control characters")

        # Check for mixed encoding indicators
        if re.search(r"[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ].*[а-я]", text, re.IGNORECASE):
            issues.append("Contains mixed character sets (possible encoding issue)")

        return issues

    def detect_truncation(self, text: str, expected_patterns: List[str] = None) -> bool:
        """Detect if text appears to be truncated"""
        if not text:
            return False

        # Check for common truncation indicators
        truncation_indicators = ["...", "…", "[truncated]", "[cut off]", "(more)", ">>"]
        if any(indicator in text.lower() for indicator in truncation_indicators):
            return True

        # Check for abrupt ending mid-sentence
        if text.endswith((",", ";", "and", "or", "but", "the", "a", "an")):
            return True

        # Check against expected patterns
        if expected_patterns:
            for pattern in expected_patterns:
                if re.search(pattern, text) and not text.strip().endswith(
                    (".", "!", "?")
                ):
                    return True

        return False

    def detect_corruption(self, record: Dict[str, Any]) -> List[str]:
        """Detect various forms of data corruption in a record"""
        issues = []

        # Check for field mixing (content appears in wrong field)
        url = record.get("url", "")
        title = record.get("title", "")

        # URL in title field
        if title and (title.startswith("http://") or title.startswith("https://")):
            issues.append("URL found in title field")

        # Title in URL field
        if (
            url
            and not (url.startswith("http://") or url.startswith("https://"))
            and len(url) > 50
        ):
            issues.append("Long text found in URL field (possible title)")

        # Check for repeated content across fields
        fields_content = {
            k: str(v).strip().lower()
            for k, v in record.items()
            if v and isinstance(v, str)
        }
        for field1, content1 in fields_content.items():
            for field2, content2 in fields_content.items():
                if field1 != field2 and content1 == content2 and len(content1) > 10:
                    issues.append(f"Identical content in {field1} and {field2} fields")

        # Check for encoding issues in all text fields
        text_fields = ["title", "note", "excerpt", "folder"]
        for field in text_fields:
            value = record.get(field, "")
            if value:
                encoding_issues = self.detect_encoding_issues(value)
                if encoding_issues:
                    issues.extend([f"{field}: {issue}" for issue in encoding_issues])

        return issues


def create_error_report(
    issues: List[str], record: Dict[str, Any], row_number: Optional[int] = None
) -> str:
    """
    Create a detailed error report for a problematic record

    Args:
        issues: List of issue descriptions
        record: The problematic record
        row_number: Optional row number in CSV

    Returns:
        Formatted error report string
    """
    report_parts = []

    if row_number:
        report_parts.append(f"Row {row_number}:")

    # Add URL for identification
    url = record.get("url", "N/A")
    report_parts.append(f"  URL: {url}")

    # Add title if available
    title = record.get("title", "")
    if title:
        report_parts.append(
            f"  Title: {title[:100]}{'...' if len(title) > 100 else ''}"
        )

    # Add issues
    report_parts.append("  Issues:")
    for issue in issues:
        report_parts.append(f"    - {issue}")

    # Add suggestions
    suggestions = generate_fix_suggestions(issues, record)
    if suggestions:
        report_parts.append("  Suggestions:")
        for suggestion in suggestions:
            report_parts.append(f"    - {suggestion}")

    return "\n".join(report_parts)


def generate_fix_suggestions(issues: List[str], record: Dict[str, Any]) -> List[str]:
    """
    Generate suggestions for fixing data issues

    Args:
        issues: List of detected issues
        record: The problematic record

    Returns:
        List of fix suggestions
    """
    suggestions = []

    for issue in issues:
        issue_lower = issue.lower()

        if "missing" in issue_lower and "url" in issue_lower:
            suggestions.append("Add a valid URL for this bookmark")

        elif "url" in issue_lower and "title" in issue_lower:
            suggestions.append("Check if URL and title fields are swapped")

        elif "encoding" in issue_lower:
            suggestions.append("Re-save the CSV file with UTF-8 encoding")

        elif "truncated" in issue_lower:
            suggestions.append("Check source data for complete content")

        elif "html" in issue_lower:
            suggestions.append("Remove HTML tags from text fields")

        elif "tags" in issue_lower:
            suggestions.append("Format tags as comma-separated values")

        elif "date" in issue_lower:
            suggestions.append("Use ISO format (YYYY-MM-DDTHH:MM:SSZ) for dates")

    # Generic suggestions
    if not suggestions:
        suggestions.append("Verify data integrity and format consistency")

    return suggestions
