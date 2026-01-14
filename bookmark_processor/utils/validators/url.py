"""
URL Validators

This module provides consolidated URL validation functionality,
combining basic format validation with comprehensive security checks.
"""

import re
from typing import List, Optional
from urllib.parse import urlparse

from .base import ValidationResult, Validator
from .security import SecurityValidator


class URLValidator(Validator):
    """Validator for URL values with security checks"""

    def __init__(
        self,
        field_name: Optional[str] = None,
        required: bool = False,
        allow_none: bool = True,
        allowed_schemes: Optional[List[str]] = None,
        security_check: bool = True,
        normalize_url: bool = True,
    ):
        """
        Initialize URL validator

        Args:
            field_name: Name of the field being validated
            required: Whether the field is required
            allow_none: Whether None values are allowed
            allowed_schemes: List of allowed URL schemes
            security_check: Whether to perform security validation
            normalize_url: Whether to normalize the URL
        """
        super().__init__(field_name, required, allow_none)
        self.allowed_schemes = (
            set(allowed_schemes) if allowed_schemes else {"http", "https"}
        )
        self.security_check = security_check
        self.normalize_url = normalize_url
        self.security_validator = SecurityValidator() if security_check else None

    def validate(self, value) -> ValidationResult:
        """Validate URL value"""
        # Check basic requirements
        basic_check = self._check_required_and_none(value)
        if basic_check:
            return basic_check

        result = ValidationResult(is_valid=True)

        if value is None:
            result.sanitized_value = None
            return result

        # Convert to string and clean
        if not isinstance(value, str):
            value = str(value)

        value = value.strip()
        result.sanitized_value = value

        if not value:
            if self.required:
                result.add_error("URL cannot be empty", self.field_name)
            return result

        # Basic URL format validation
        try:
            parsed = urlparse(value)
        except Exception as e:
            result.add_error(f"Invalid URL format: {e}", self.field_name)
            return result

        # Check scheme
        if not parsed.scheme:
            result.add_warning("URL missing scheme, assuming https://", self.field_name)
            value = f"https://{value}"
            result.sanitized_value = value
            try:
                parsed = urlparse(value)
            except Exception as e:
                result.add_error(
                    f"Invalid URL after adding scheme: {e}", self.field_name
                )
                return result

        if parsed.scheme.lower() not in self.allowed_schemes:
            result.add_error(
                f"URL scheme '{parsed.scheme}' not allowed. "
                f"Allowed: {list(self.allowed_schemes)}",
                self.field_name,
            )

        # Check netloc (domain)
        if not parsed.netloc:
            result.add_error("URL missing domain/hostname", self.field_name)

        # Security validation
        if self.security_check and self.security_validator:
            security_result = self.security_validator.validate_url_security(value)
            if not security_result.is_safe:
                result.add_critical(
                    f"Security validation failed: {security_result.blocked_reason}",
                    self.field_name,
                )
                result.metadata["security_issues"] = security_result.issues
            elif security_result.issues:
                result.add_warning(
                    f"Security concerns: {', '.join(security_result.issues)}",
                    self.field_name,
                )
                result.metadata["security_issues"] = security_result.issues

        # URL normalization
        if self.normalize_url and result.is_valid:
            try:
                # Basic normalization: lowercase domain, remove trailing slash
                normalized = f"{parsed.scheme}://{parsed.netloc.lower()}"
                if parsed.path and parsed.path != "/":
                    normalized += parsed.path.rstrip("/")
                elif not parsed.path:
                    normalized += "/"

                if parsed.query:
                    normalized += f"?{parsed.query}"
                if parsed.fragment:
                    normalized += f"#{parsed.fragment}"

                result.sanitized_value = normalized

            except Exception as e:
                result.add_warning(f"URL normalization failed: {e}", self.field_name)

        return result


class BookmarkURLValidator(URLValidator):
    """Enhanced URL validator specifically for bookmarks"""

    def __init__(self):
        super().__init__(
            field_name="url",
            required=True,  # URLs are mandatory for bookmarks
            allow_none=False,
            allowed_schemes=["http", "https"],
            security_check=True,
            normalize_url=True,
        )

    def validate(self, value) -> ValidationResult:
        result = super().validate(value)

        if result.sanitized_value and isinstance(result.sanitized_value, str):
            url = result.sanitized_value

            # Additional bookmark-specific URL validation
            try:
                parsed = urlparse(url)

                # Check for localhost/development URLs
                if parsed.netloc.lower() in ["localhost", "127.0.0.1", "0.0.0.0"]:
                    result.add_warning(
                        "URL points to localhost/development server", self.field_name
                    )

                # Check for common development ports
                dev_ports = {3000, 3001, 8000, 8080, 8888, 9000}
                if parsed.port in dev_ports:
                    result.add_warning(
                        f"URL uses common development port {parsed.port}",
                        self.field_name,
                    )

                # Check for very long URLs
                if len(url) > 500:
                    result.add_warning(
                        f"URL is very long ({len(url)} chars)", self.field_name
                    )

                # Check for suspicious query parameters
                if parsed.query:
                    suspicious_params = [
                        "utm_source",
                        "utm_medium",
                        "utm_campaign",
                        "fbclid",
                        "gclid",
                    ]
                    query_lower = parsed.query.lower()
                    tracking_params = [
                        param for param in suspicious_params if param in query_lower
                    ]
                    if tracking_params:
                        result.add_info(
                            f"URL contains tracking parameters: "
                            f"{', '.join(tracking_params)}",
                            self.field_name,
                        )

                # Check for fragments that might indicate specific sections
                if parsed.fragment and len(parsed.fragment) > 50:
                    result.add_info("URL has long fragment identifier", self.field_name)

            except Exception:
                pass  # URL parsing already handled by parent validator

        return result


def validate_url_format(url: str) -> bool:
    """
    Validate URL format (backward compatible function).

    Args:
        url: URL string to validate

    Returns:
        True if valid, False if invalid

    Note:
        Only accepts HTTP(S) and FTP URLs for bookmark processing.
        Rejects javascript:, mailto:, and malformed URLs.
    """
    # Handle None, empty, or whitespace-only strings
    if not url or not isinstance(url, str) or not url.strip():
        return False

    url = url.strip()

    # Reject dangerous schemes
    dangerous_schemes = ["javascript:", "data:", "vbscript:"]
    for scheme in dangerous_schemes:
        if url.lower().startswith(scheme):
            return False

    # Accept HTTP(S) and FTP URLs
    valid_url_pattern = re.compile(
        r"^(https?|ftp)://"  # http://, https://, or ftp://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    # Reject mailto: URLs (not suitable for bookmarks)
    if url.lower().startswith("mailto:"):
        return False

    # Reject malformed URLs
    if url.count("://") != 1:
        return False

    if "://" in url and not url.split("://")[1]:
        return False

    return bool(valid_url_pattern.match(url))


def create_url_validator(field_name: str, required: bool = True) -> URLValidator:
    """Create a URL validator with security checks"""
    return URLValidator(
        field_name=field_name,
        required=required,
        allow_none=not required,
        security_check=True,
    )
