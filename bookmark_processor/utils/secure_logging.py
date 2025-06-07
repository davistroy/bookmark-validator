"""
Secure Logging Module

Provides secure logging functionality that redacts sensitive information
and prevents information leakage in error messages and logs.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Pattern


@dataclass
class SecurityEvent:
    """Security event record"""

    event_id: str
    event_type: str
    timestamp: datetime
    severity: str
    details: Dict[str, Any]
    sanitized_details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "details": self.sanitized_details,  # Only log sanitized details
        }


class SecureLogger:
    """Secure logging with automatic data sanitization"""

    # Patterns for sensitive data that should be redacted
    SENSITIVE_PATTERNS = [
        # API Keys and tokens
        (
            re.compile(
                r'(?i)(api[_-]?key|token|secret|password|auth)["\s]*[:=]["\s]*([a-zA-Z0-9+/=_-]{10,})',
                re.IGNORECASE,
            ),
            r"\1=***REDACTED***",
        ),
        (
            re.compile(r"(sk-[a-zA-Z0-9_-]{10,})", re.IGNORECASE),
            r"***API_KEY_REDACTED***",
        ),
        (
            re.compile(r"(?i)(bearer\s+)([a-zA-Z0-9+/=]{10,})", re.IGNORECASE),
            r"\1***REDACTED***",
        ),
        # AWS credentials
        (
            re.compile(r"(?i)(AKIA[0-9A-Z]{16})", re.IGNORECASE),
            r"***AWS_ACCESS_KEY_REDACTED***",
        ),
        (
            re.compile(r"(?i)([A-Za-z0-9+/]{40})", re.IGNORECASE),
            r"***AWS_SECRET_REDACTED***",
        ),
        # Common passwords in URLs
        (
            re.compile(r"(?i)(://[^:]+:)([^@]+)(@)", re.IGNORECASE),
            r"\1***PASSWORD_REDACTED***\3",
        ),
        # Private IP addresses (for privacy)
        (
            re.compile(
                r"\b(?:192\.168|10\.|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+\b"
            ),
            r"***PRIVATE_IP***",
        ),
        # Email addresses (for privacy)
        (
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            r"***EMAIL_REDACTED***",
        ),
        # Phone numbers
        (
            re.compile(
                r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
            ),
            r"***PHONE_REDACTED***",
        ),
        # Credit card-like numbers
        (
            re.compile(r"\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b"),
            r"***CARD_REDACTED***",
        ),
        # Social Security-like numbers
        (
            re.compile(r"\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b"),
            r"***SSN_REDACTED***",
        ),
    ]

    # Additional patterns for URLs and file paths
    URL_PATTERNS = [
        # Query parameters that might contain sensitive data
        (
            re.compile(
                r"(?i)([?&](?:key|token|password|secret|auth)[=])([^&]+)", re.IGNORECASE
            ),
            r"\1***REDACTED***",
        ),
        # File paths that might contain usernames
        (
            re.compile(
                r"(?i)(\/users?\/|\/home\/|C:\\Users\\)([^\/\\]+)", re.IGNORECASE
            ),
            r"\1***USER***",
        ),
    ]

    def __init__(self, logger_name: str = __name__):
        """Initialize secure logger"""
        self.logger = logging.getLogger(logger_name)
        self.security_events: List[SecurityEvent] = []

        # Compile patterns for better performance
        self.compiled_patterns = [
            (pattern, replacement) for pattern, replacement in self.SENSITIVE_PATTERNS
        ]
        self.compiled_url_patterns = [
            (pattern, replacement) for pattern, replacement in self.URL_PATTERNS
        ]

    def sanitize_message(self, message: str) -> str:
        """
        Sanitize a message by redacting sensitive information.

        Args:
            message: Message to sanitize

        Returns:
            Sanitized message with sensitive data redacted
        """
        if not isinstance(message, str):
            message = str(message)

        sanitized = message

        # Apply all sanitization patterns
        for pattern, replacement in self.compiled_patterns:
            sanitized = pattern.sub(replacement, sanitized)

        for pattern, replacement in self.compiled_url_patterns:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def sanitize_data(self, data: Any) -> Any:
        """
        Recursively sanitize data structures.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            return self.sanitize_message(data)
        elif isinstance(data, dict):
            return {key: self.sanitize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.sanitize_data(item) for item in data)
        else:
            return data

    def log_security_event(
        self, event_type: str, details: Dict[str, Any], severity: str = "warning"
    ) -> str:
        """
        Log a security event with automatic sanitization.

        Args:
            event_type: Type of security event
            details: Event details (will be sanitized)
            severity: Event severity level

        Returns:
            Event ID for tracking
        """
        event_id = str(uuid.uuid4())

        # Create security event with both original and sanitized details
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            severity=severity,
            details=details.copy(),  # Store original for internal use if needed
            sanitized_details=self.sanitize_data(details),
        )

        self.security_events.append(event)

        # Log the sanitized event
        log_level = getattr(logging, severity.upper(), logging.WARNING)
        self.logger.log(
            log_level,
            f"Security Event [{event_id}] {event_type}: {json.dumps(event.sanitized_details)}",
        )

        return event_id

    def log_url_validation_failure(
        self, url: str, reason: str, security_issues: Optional[List[str]] = None
    ) -> str:
        """
        Log URL validation failure with security context.

        Args:
            url: URL that failed validation (will be sanitized)
            reason: Reason for failure
            security_issues: List of specific security issues detected

        Returns:
            Event ID
        """
        details = {
            "url": url,
            "failure_reason": reason,
            "security_issues": security_issues or [],
        }

        return self.log_security_event("url_validation_failure", details, "warning")

    def log_ssrf_attempt(
        self, url: str, blocked_reason: str, resolved_ips: Optional[List[str]] = None
    ) -> str:
        """
        Log potential SSRF attempt.

        Args:
            url: URL that was blocked
            blocked_reason: Reason for blocking
            resolved_ips: IPs the URL resolved to (if any)

        Returns:
            Event ID
        """
        details = {
            "attempted_url": url,
            "blocked_reason": blocked_reason,
            "resolved_ips": resolved_ips or [],
            "action_taken": "blocked",
        }

        return self.log_security_event("ssrf_attempt", details, "error")

    def log_malicious_input(
        self, input_data: Any, input_type: str, detected_patterns: List[str]
    ) -> str:
        """
        Log malicious input detection.

        Args:
            input_data: The malicious input (will be sanitized)
            input_type: Type of input (url, query_param, etc.)
            detected_patterns: List of malicious patterns detected

        Returns:
            Event ID
        """
        details = {
            "input_data": input_data,
            "input_type": input_type,
            "detected_patterns": detected_patterns,
            "action_taken": "rejected",
        }

        return self.log_security_event("malicious_input", details, "error")

    def create_safe_error_response(
        self,
        internal_error: Exception,
        user_message: str = "An error occurred",
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a safe error response that doesn't leak sensitive information.

        Args:
            internal_error: The actual error (logged securely)
            user_message: Safe message to show to user
            correlation_id: Optional correlation ID for tracking

        Returns:
            Safe error response dictionary
        """
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Log the actual error securely
        details = {
            "error_type": type(internal_error).__name__,
            "error_message": str(internal_error),
            "correlation_id": correlation_id,
        }

        self.log_security_event("internal_error", details, "error")

        # Return safe response
        return {
            "error": user_message,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
        }

    def get_security_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent security events (sanitized).

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            limit: Maximum number of events to return

        Returns:
            List of sanitized security events
        """
        events = self.security_events

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if severity:
            events = [e for e in events if e.severity == severity]

        # Sort by timestamp (most recent first) and limit
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]

        return [event.to_dict() for event in events]

    def clear_old_events(self, max_age_hours: int = 24) -> int:
        """
        Clear security events older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of events cleared
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        old_events = [e for e in self.security_events if e.timestamp < cutoff_time]
        self.security_events = [
            e for e in self.security_events if e.timestamp >= cutoff_time
        ]

        return len(old_events)


# Global secure logger instance
secure_logger = SecureLogger("bookmark_processor.security")


def log_security_event(
    event_type: str, details: Dict[str, Any], severity: str = "warning"
) -> str:
    """Convenience function for logging security events"""
    return secure_logger.log_security_event(event_type, details, severity)


def create_safe_error_response(
    error: Exception, user_message: str = "An error occurred"
) -> Dict[str, Any]:
    """Convenience function for creating safe error responses"""
    return secure_logger.create_safe_error_response(error, user_message)


def sanitize_for_logging(data: Any) -> Any:
    """Convenience function for sanitizing data before logging"""
    return secure_logger.sanitize_data(data)
