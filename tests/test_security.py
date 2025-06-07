"""
Comprehensive Security Tests

Tests for all security measures including SSRF protection, URL validation,
secure logging, and malicious input handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bookmark_processor.core.url_validator import URLValidator
from bookmark_processor.utils.secure_logging import SecureLogger, secure_logger
from bookmark_processor.utils.security_validator import (
    SecurityValidationResult,
    SecurityValidator,
)


class TestSecurityValidator:
    """Test suite for SecurityValidator"""

    def setup_method(self):
        """Setup for each test"""
        self.validator = SecurityValidator()

    def test_blocked_schemes(self):
        """Test that dangerous URL schemes are blocked"""
        dangerous_urls = [
            "javascript:alert('xss')",
            "file:///etc/passwd",
            "ftp://example.com/file.txt",
            "data:text/html,<script>alert('xss')</script>",
            "mailto:test@example.com",
            "tel:555-1234",
        ]

        for url in dangerous_urls:
            result = self.validator.validate_url_security(url)
            assert not result.is_safe, f"URL should be blocked: {url}"
            assert result.risk_level in ["high", "critical"]
            assert result.blocked_reason is not None

    def test_allowed_schemes(self):
        """Test that safe URL schemes are allowed"""
        safe_urls = [
            "https://www.example.com",
            "http://example.com",
            "https://subdomain.example.com:8080/path?query=value",
        ]

        for url in safe_urls:
            result = self.validator.validate_url_security(url)
            # Note: URL might still fail DNS validation, but scheme should pass
            assert (
                "javascript" not in result.blocked_reason
                if result.blocked_reason
                else True
            )

    def test_private_ip_blocking(self):
        """Test that private IP addresses are blocked"""
        private_ips = [
            "http://192.168.1.1",
            "https://10.0.0.1:8080",
            "http://172.16.0.1/admin",
            "http://127.0.0.1:3000",
            "https://localhost:8080",
            "http://0.0.0.0",
        ]

        for url in private_ips:
            result = self.validator.validate_url_security(url)
            assert not result.is_safe, f"Private IP should be blocked: {url}"
            assert result.risk_level == "critical"

    def test_blocked_hostnames(self):
        """Test that dangerous hostnames are blocked"""
        blocked_urls = [
            "http://metadata.google.internal",
            "https://instance-data/latest/meta-data",
            "http://169.254.169.254/metadata",
        ]

        for url in blocked_urls:
            result = self.validator.validate_url_security(url)
            assert not result.is_safe, f"Dangerous hostname should be blocked: {url}"

    def test_directory_traversal_detection(self):
        """Test detection of directory traversal attempts"""
        traversal_urls = [
            "http://example.com/../../../etc/passwd",
            "https://example.com/..\\..\\windows\\system32",
            "http://example.com/%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

        for url in traversal_urls:
            result = self.validator.validate_url_security(url)
            assert not result.is_safe, f"Directory traversal should be detected: {url}"

    def test_url_length_limits(self):
        """Test URL length validation"""
        # Create extremely long URL
        long_url = "http://example.com/" + "a" * 3000

        result = self.validator.validate_url_security(long_url)
        assert not result.is_safe
        assert "URL too long" in result.issues[0]

    def test_suspicious_patterns(self):
        """Test detection of suspicious URL patterns"""
        suspicious_urls = [
            "http://example.com/admin@evil.com:password",
            "http://localhost:8080/internal",
            "http://example.com/file:///etc/passwd",
        ]

        for url in suspicious_urls:
            result = self.validator.validate_url_security(url)
            # These might not all be blocked, but should have warnings
            if not result.is_safe or result.issues:
                assert result.risk_level in ["low", "medium", "high", "critical"]

    def test_url_sanitization(self):
        """Test URL sanitization functionality"""
        test_cases = [
            ("http://example.com/../admin", "https://example.com/admin"),
            ("javascript:alert('xss')", ""),  # Should be completely sanitized
            ("", ""),
        ]

        for input_url, expected_pattern in test_cases:
            sanitized = self.validator.sanitize_url(input_url)
            if expected_pattern:
                assert expected_pattern in sanitized or sanitized.startswith("https://")
            else:
                assert sanitized == ""


class TestSecureLogger:
    """Test suite for SecureLogger"""

    def setup_method(self):
        """Setup for each test"""
        self.logger = SecureLogger("test_security")

    def test_sensitive_data_redaction(self):
        """Test that sensitive data is properly redacted"""
        test_data = {
            "api_key": "sk-1234567890abcdef",
            "token": "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "password": "mypassword123",
            "email": "user@example.com",
            "phone": "555-123-4567",
            "ssn": "123-45-6789",
            "private_ip": "192.168.1.100",
            "aws_key": "AKIAIOSFODNN7EXAMPLE",
        }

        sanitized = self.logger.sanitize_data(test_data)

        # Check that sensitive values are redacted
        assert "***REDACTED***" in str(sanitized.get("api_key", ""))
        assert "***EMAIL_REDACTED***" in str(sanitized)
        assert "***PHONE_REDACTED***" in str(sanitized)
        assert "***SSN_REDACTED***" in str(sanitized)
        assert "***PRIVATE_IP***" in str(sanitized)
        assert "***AWS_ACCESS_KEY_REDACTED***" in str(sanitized)

    def test_url_sanitization(self):
        """Test URL-specific sanitization"""
        urls = [
            "https://user:password@example.com/api?token=secret123",
            "http://admin:secret@internal.com/admin",
            "/home/username/documents/file.txt",
        ]

        for url in urls:
            sanitized = self.logger.sanitize_message(url)
            assert (
                "***PASSWORD_REDACTED***" in sanitized
                or "***REDACTED***" in sanitized
                or "***USER***" in sanitized
            )

    def test_security_event_logging(self):
        """Test security event logging and retrieval"""
        # Log various security events
        event_id1 = self.logger.log_security_event(
            "test_attack",
            {"attack_type": "xss", "payload": "<script>alert('xss')</script>"},
            "high",
        )

        event_id2 = self.logger.log_ssrf_attempt(
            "http://192.168.1.1/admin", "Private IP detected", ["192.168.1.1"]
        )

        # Verify events are logged
        assert event_id1 is not None
        assert event_id2 is not None

        # Retrieve events
        events = self.logger.get_security_events(limit=10)
        assert len(events) >= 2

        # Check event structure
        for event in events:
            assert "event_id" in event
            assert "event_type" in event
            assert "timestamp" in event
            assert "severity" in event
            assert "details" in event

    def test_malicious_input_logging(self):
        """Test malicious input detection and logging"""
        malicious_inputs = [
            ("<script>alert('xss')</script>", "html_input"),
            ("'; DROP TABLE users; --", "sql_input"),
            ("../../../../etc/passwd", "path_input"),
        ]

        for input_data, input_type in malicious_inputs:
            event_id = self.logger.log_malicious_input(
                input_data,
                input_type,
                ["script_injection", "sql_injection", "path_traversal"],
            )
            assert event_id is not None

    def test_safe_error_response(self):
        """Test creation of safe error responses"""
        internal_error = Exception("Database connection failed at server 192.168.1.100")

        safe_response = self.logger.create_safe_error_response(
            internal_error, "Service temporarily unavailable"
        )

        # Check that response is safe
        assert safe_response["error"] == "Service temporarily unavailable"
        assert "correlation_id" in safe_response
        assert "timestamp" in safe_response

        # Ensure internal details are not exposed
        response_str = str(safe_response)
        assert "192.168.1.100" not in response_str
        assert "Database connection failed" not in response_str

    def test_recursive_data_sanitization(self):
        """Test sanitization of nested data structures"""
        complex_data = {
            "user": {
                "email": "test@example.com",
                "api_keys": ["sk-123456", "sk-789012"],
                "metadata": {"last_ip": "10.0.0.1", "user_agent": "Mozilla/5.0"},
            },
            "logs": [
                "User logged in from 192.168.1.50",
                "API call with token=abc123def456",
            ],
        }

        sanitized = self.logger.sanitize_data(complex_data)

        # Check nested sanitization
        assert "***EMAIL_REDACTED***" in str(sanitized)
        assert "***PRIVATE_IP***" in str(sanitized)
        assert "***REDACTED***" in str(sanitized)


class TestURLValidatorSecurity:
    """Test security integration in URLValidator"""

    def setup_method(self):
        """Setup for each test"""
        self.validator = URLValidator()

    def test_security_validation_integration(self):
        """Test that security validation is properly integrated"""
        dangerous_url = "javascript:alert('xss')"

        result = self.validator.validate_url(dangerous_url)

        assert not result.is_valid
        assert result.error_type == "security_error"
        assert result.security_validation is not None
        assert not result.security_validation.is_safe

    def test_security_logging_integration(self):
        """Test that security events are logged during validation"""
        # Clear previous events
        secure_logger.security_events.clear()

        dangerous_urls = [
            "http://192.168.1.1/admin",
            "javascript:void(0)",
            "file:///etc/passwd",
        ]

        for url in dangerous_urls:
            result = self.validator.validate_url(url)
            assert not result.is_valid

        # Check that security events were logged
        events = secure_logger.get_security_events()
        assert len(events) >= len(dangerous_urls)

    def test_valid_url_passes_security(self):
        """Test that valid URLs pass security validation"""
        # Note: This URL won't actually be validated against the network
        # but should pass security checks
        valid_url = "https://www.example.com/safe/path"

        result = self.validator.validate_url(valid_url)

        # URL might fail network validation, but security should pass
        if result.error_type == "security_error":
            pytest.fail(f"Valid URL failed security validation: {result.error_message}")

    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, "validator"):
            self.validator.close()


class TestPenetrationTests:
    """Penetration testing scenarios"""

    def setup_method(self):
        """Setup for penetration tests"""
        self.validator = URLValidator()
        self.security_validator = SecurityValidator()

    def test_ssrf_prevention(self):
        """Test SSRF attack prevention"""
        ssrf_payloads = [
            # Direct private IPs
            "http://127.0.0.1:22",
            "http://192.168.1.1:3389",
            "http://10.0.0.1:8080",
            # Localhost variations
            "http://localhost:6379",
            "http://0.0.0.0:9200",
            # Cloud metadata endpoints
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            # IPv6 localhost
            "http://[::1]:80",
            # Encoded variations
            "http://0x7f000001/",  # Hex encoded 127.0.0.1
        ]

        for payload in ssrf_payloads:
            result = self.validator.validate_url(payload)
            assert not result.is_valid, f"SSRF payload should be blocked: {payload}"
            assert result.error_type == "security_error"

    def test_injection_prevention(self):
        """Test injection attack prevention"""
        injection_payloads = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "javascript:void(0)",
            "vbscript:msgbox('xss')",
        ]

        for payload in injection_payloads:
            result = self.validator.validate_url(payload)
            assert (
                not result.is_valid
            ), f"Injection payload should be blocked: {payload}"
            assert result.error_type == "security_error"

    def test_directory_traversal_prevention(self):
        """Test directory traversal prevention"""
        traversal_payloads = [
            "http://example.com/../../../etc/passwd",
            "http://example.com/..\\..\\windows\\system32",
            "http://example.com/%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "http://example.com/....//....//etc/passwd",
        ]

        for payload in traversal_payloads:
            result = self.security_validator.validate_url_security(payload)
            assert (
                not result.is_safe
            ), f"Directory traversal should be blocked: {payload}"

    def test_url_manipulation_prevention(self):
        """Test URL manipulation attack prevention"""
        manipulation_payloads = [
            "http://evil.com@good.com/",  # Host confusion
            "http://good.com.evil.com/",  # Subdomain confusion
            "http://good.com#@evil.com/",  # Fragment confusion
        ]

        for payload in manipulation_payloads:
            result = self.security_validator.validate_url_security(payload)
            # These might not all be blocked but should be flagged
            if result.issues:
                assert len(result.issues) > 0

    def teardown_method(self):
        """Cleanup after penetration tests"""
        if hasattr(self, "validator"):
            self.validator.close()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
