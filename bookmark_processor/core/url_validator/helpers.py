"""
Helper functions for URL validation.

Contains utility functions for URL format checking, error classification,
and other validation-related helpers.
"""

from typing import Optional
from urllib.parse import urlparse


# URLs to skip validation (known problematic patterns)
SKIP_PATTERNS = [
    "javascript:",
    "mailto:",
    "tel:",
    "ftp:",
    "file:",
    "data:",
    "#",  # Fragment-only URLs
    "about:blank",
]

# Valid HTTP status codes for success
SUCCESS_CODES = {
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    300,
    301,
    302,
    303,
    304,
    307,
    308,
}


def is_valid_url_format(url: str) -> bool:
    """
    Check if URL has valid format.

    Args:
        url: URL to validate

    Returns:
        True if URL format is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def should_skip_url(url: str) -> bool:
    """
    Check if URL should be skipped.

    Args:
        url: URL to check

    Returns:
        True if URL should be skipped, False otherwise
    """
    url_lower = url.lower().strip()

    for pattern in SKIP_PATTERNS:
        if url_lower.startswith(pattern):
            return True

    return False


def classify_http_error(status_code: int) -> str:
    """
    Classify HTTP error by status code.

    Args:
        status_code: HTTP status code

    Returns:
        Error type string
    """
    if 400 <= status_code < 500:
        error_types = {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
            408: "request_timeout",
            429: "rate_limited",
        }
        return error_types.get(status_code, "client_error")
    elif 500 <= status_code < 600:
        error_types = {
            500: "internal_server_error",
            501: "not_implemented",
            502: "bad_gateway",
            503: "service_unavailable",
            504: "gateway_timeout",
        }
        return error_types.get(status_code, "server_error")
    else:
        return "http_error"


def parse_content_length(content_length_header: Optional[str]) -> Optional[int]:
    """
    Parse content-length header.

    Args:
        content_length_header: Content-Length header value

    Returns:
        Content length as integer, or None if invalid
    """
    if not content_length_header:
        return None

    try:
        return int(content_length_header)
    except (ValueError, TypeError):
        return None


__all__ = [
    "SKIP_PATTERNS",
    "SUCCESS_CODES",
    "is_valid_url_format",
    "should_skip_url",
    "classify_http_error",
    "parse_content_length",
]
