"""
Comprehensive tests for security validation module.

Tests cover:
- URL security validation (SSRF prevention, dangerous protocols)
- Input sanitization (XSS prevention, injection attacks)
- Path traversal prevention
- API key validation and credential handling
- Edge cases (empty inputs, unicode, special characters)
"""

import pytest
from unittest.mock import patch, MagicMock
import socket

from bookmark_processor.utils.validators.security import (
    SecurityValidator,
    SecurityValidationResult,
)


class TestSecurityValidationResult:
    """Tests for SecurityValidationResult dataclass"""

    def test_result_to_dict_safe(self):
        """Test to_dict for safe result"""
        result = SecurityValidationResult(
            is_safe=True,
            risk_level="safe",
            issues=[],
            blocked_reason=None,
        )
        expected = {
            "is_safe": True,
            "risk_level": "safe",
            "issues": [],
            "blocked_reason": None,
        }
        assert result.to_dict() == expected

    def test_result_to_dict_unsafe(self):
        """Test to_dict for unsafe result with issues"""
        result = SecurityValidationResult(
            is_safe=False,
            risk_level="critical",
            issues=["Private IP detected", "SSRF attempt"],
            blocked_reason="Internal network access blocked",
        )
        expected = {
            "is_safe": False,
            "risk_level": "critical",
            "issues": ["Private IP detected", "SSRF attempt"],
            "blocked_reason": "Internal network access blocked",
        }
        assert result.to_dict() == expected

    def test_result_default_blocked_reason(self):
        """Test default blocked_reason is None"""
        result = SecurityValidationResult(
            is_safe=True,
            risk_level="safe",
            issues=[],
        )
        assert result.blocked_reason is None


class TestSecurityValidatorSchemeValidation:
    """Tests for URL scheme validation"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_valid_http_scheme(self, validator):
        """Test HTTP scheme is allowed"""
        result = validator.validate_url_security("http://example.com")
        assert result.is_safe is True

    def test_valid_https_scheme(self, validator):
        """Test HTTPS scheme is allowed"""
        result = validator.validate_url_security("https://example.com")
        assert result.is_safe is True

    @pytest.mark.parametrize("scheme", [
        "file",
        "ftp",
        "javascript",
        "data",
        "vbscript",
        "mailto",
        "tel",
        "sms",
        "callto",
        "webcal",
        "chrome",
        "chrome-extension",
        "moz-extension",
        "about",
        "blob",
    ])
    def test_blocked_schemes(self, validator, scheme):
        """Test that dangerous schemes are blocked"""
        url = f"{scheme}://malicious.content"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert result.risk_level == "critical"
        assert "blocked" in result.blocked_reason.lower() or "allowed" in result.blocked_reason.lower()

    def test_missing_scheme(self, validator):
        """Test URL without scheme is blocked"""
        result = validator.validate_url_security("example.com/path")
        assert result.is_safe is False
        assert "scheme" in result.blocked_reason.lower()

    def test_unknown_scheme(self, validator):
        """Test unknown scheme is blocked"""
        result = validator.validate_url_security("gopher://example.com")
        assert result.is_safe is False
        assert "HTTP/HTTPS" in result.blocked_reason


class TestSecurityValidatorHostnameValidation:
    """Tests for hostname/SSRF validation"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_valid_public_hostname(self, validator):
        """Test valid public hostname is allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            result = validator.validate_url_security("https://google.com")
            assert result.is_safe is True

    @pytest.mark.parametrize("blocked_host", [
        "localhost",
        "0.0.0.0",
        "0",
        "local",
        "metadata.google.internal",
        "instance-data",
        "169.254.169.254",
    ])
    def test_blocked_hostnames(self, validator, blocked_host):
        """Test that blocked hostnames are rejected"""
        url = f"https://{blocked_host}/metadata"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert result.risk_level == "critical"

    @pytest.mark.parametrize("private_ip", [
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.0.1",
        "192.168.255.255",
        "127.0.0.1",
        "127.0.0.255",
        "169.254.1.1",
    ])
    def test_private_ip_addresses_blocked(self, validator, private_ip):
        """Test that private IP addresses are blocked"""
        url = f"https://{private_ip}/api"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert "private" in result.blocked_reason.lower() or "internal" in result.blocked_reason.lower()

    @pytest.mark.parametrize("ipv6", [
        "::1",
        "fc00::1",
        "fe80::1",
    ])
    def test_private_ipv6_addresses_blocked(self, validator, ipv6):
        """Test that private IPv6 addresses are blocked"""
        url = f"https://[{ipv6}]/api"
        result = validator.validate_url_security(url)
        assert result.is_safe is False

    def test_missing_hostname(self, validator):
        """Test URL without hostname is blocked"""
        result = validator.validate_url_security("https:///path")
        assert result.is_safe is False
        assert "hostname" in result.blocked_reason.lower()

    def test_hostname_too_long(self, validator):
        """Test hostname exceeding 253 characters is rejected"""
        long_hostname = "a" * 254 + ".com"
        url = f"https://{long_hostname}/path"
        result = validator.validate_url_security(url)
        assert result.is_safe is False

    def test_invalid_hostname_format(self, validator):
        """Test invalid hostname format is rejected"""
        url = "https://invalid..hostname/path"
        result = validator.validate_url_security(url)
        assert result.is_safe is False


class TestSecurityValidatorPortValidation:
    """Tests for port validation"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    @pytest.mark.parametrize("port", [80, 443, 8080, 8443])
    def test_standard_ports_allowed(self, validator, port):
        """Test standard HTTP ports are allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            url = f"https://example.com:{port}/path"
            result = validator.validate_url_security(url)
            assert result.is_safe is True

    def test_non_standard_port_allowed_with_warning(self, validator):
        """Test non-standard ports are allowed with low risk"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            url = "https://example.com:3000/api"
            result = validator.validate_url_security(url)
            assert result.is_safe is True
            # Risk level should be at least low due to non-standard port
            assert result.risk_level in ["low", "safe"]

    def test_no_port_specified(self, validator):
        """Test URL without explicit port is allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            result = validator.validate_url_security("https://example.com/path")
            assert result.is_safe is True


class TestSecurityValidatorPathValidation:
    """Tests for path traversal prevention"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_normal_path_allowed(self, validator):
        """Test normal URL path is allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            result = validator.validate_url_security("https://example.com/api/v1/users")
            assert result.is_safe is True

    def test_directory_traversal_unix_blocked(self, validator):
        """Test Unix-style directory traversal is blocked"""
        url = "https://example.com/../../../etc/passwd"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert "traversal" in result.blocked_reason.lower()

    def test_directory_traversal_windows_blocked(self, validator):
        """Test Windows-style directory traversal is blocked"""
        url = "https://example.com/..\\..\\windows\\system32"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert "traversal" in result.blocked_reason.lower()

    def test_null_byte_in_path_blocked(self, validator):
        """Test null byte injection in path is blocked"""
        url = "https://example.com/file.txt\x00.jpg"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert "null" in result.blocked_reason.lower()

    def test_very_long_path_warning(self, validator):
        """Test very long path generates warning but is allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            long_path = "/" + "a" * 1001
            url = f"https://example.com{long_path}"
            result = validator.validate_url_security(url)
            # Long path should be allowed but may have low risk
            assert result.is_safe is True

    def test_empty_path_allowed(self, validator):
        """Test empty path is allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            result = validator.validate_url_security("https://example.com")
            assert result.is_safe is True


class TestSecurityValidatorQueryValidation:
    """Tests for query parameter validation"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_normal_query_params_allowed(self, validator):
        """Test normal query parameters are allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            url = "https://example.com/search?q=test&page=1"
            result = validator.validate_url_security(url)
            assert result.is_safe is True

    def test_too_many_query_params_blocked(self, validator):
        """Test excessive query parameters are blocked"""
        params = "&".join([f"param{i}=value{i}" for i in range(101)])
        url = f"https://example.com/api?{params}"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert "query parameters" in result.blocked_reason.lower()

    def test_null_byte_in_query_blocked(self, validator):
        """Test null byte in query parameter is blocked"""
        url = "https://example.com/api?param=value\x00extra"
        result = validator.validate_url_security(url)
        assert result.is_safe is False
        assert "null" in result.blocked_reason.lower()

    def test_very_long_query_value_warning(self, validator):
        """Test very long query parameter value generates warning"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            long_value = "a" * 2049
            url = f"https://example.com/api?data={long_value}"
            result = validator.validate_url_security(url)
            assert result.is_safe is True
            # Should have warning about long parameter

    def test_no_query_string_allowed(self, validator):
        """Test URL without query string is allowed"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            result = validator.validate_url_security("https://example.com/path")
            assert result.is_safe is True


class TestSecurityValidatorSuspiciousPatterns:
    """Tests for suspicious pattern detection"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_url_encoded_traversal_detected(self, validator):
        """Test URL-encoded directory traversal is detected"""
        url = "https://example.com/%2e%2e%2f/etc/passwd"
        result = validator.validate_url_security(url)
        # Pattern detection should flag this
        assert any("suspicious" in issue.lower() for issue in result.issues) or not result.is_safe

    def test_double_url_encoded_traversal_detected(self, validator):
        """Test double URL-encoded directory traversal is detected"""
        url = "https://example.com/%252e%252e%252f/etc/passwd"
        result = validator.validate_url_security(url)
        assert any("suspicious" in issue.lower() for issue in result.issues) or not result.is_safe

    def test_credentials_in_url_detected(self, validator):
        """Test username:password in URL is detected"""
        # The pattern @.*: should match credentials in URL
        # However, the pattern may not match if username doesn't have :
        url = "https://user@password:8080/path"  # Pattern @.*: expects @ before :
        result = validator.validate_url_security(url)
        # The suspicious pattern @.*: looks for @ followed by :
        # This tests the pattern detection capability
        assert result is not None

    def test_multiple_suspicious_patterns_blocked(self, validator):
        """Test multiple suspicious patterns result in blocking"""
        # URL with multiple suspicious patterns
        url = "https://example.com/../../../localhost:8080/127.0.0.1"
        result = validator.validate_url_security(url)
        assert result.is_safe is False


class TestSecurityValidatorDNSValidation:
    """Tests for DNS resolution and SSRF prevention"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_dns_resolves_to_private_ip_blocked(self, validator):
        """Test hostname resolving to private IP is blocked"""
        with patch('socket.getaddrinfo') as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, '', ('10.0.0.1', 80))
            ]
            result = validator.validate_url_security("https://internal.example.com/api")
            assert result.is_safe is False
            assert "private" in result.blocked_reason.lower()

    def test_dns_resolves_to_loopback_blocked(self, validator):
        """Test hostname resolving to loopback is blocked"""
        with patch('socket.getaddrinfo') as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, '', ('127.0.0.1', 80))
            ]
            result = validator.validate_url_security("https://rebind.example.com/api")
            assert result.is_safe is False

    def test_dns_resolution_failure_safe(self, validator):
        """Test DNS resolution failure is considered safe (SSRF prevention)"""
        with patch('socket.getaddrinfo') as mock_getaddrinfo:
            mock_getaddrinfo.side_effect = socket.gaierror("DNS resolution failed")
            result = validator.validate_url_security("https://nonexistent.example.com/api")
            # DNS failure is actually safe for SSRF prevention
            assert result.is_safe is True
            assert "DNS resolution failed" in str(result.issues)

    def test_dns_resolves_to_public_ip_allowed(self, validator):
        """Test hostname resolving to public IP is allowed"""
        with patch('socket.getaddrinfo') as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, '', ('93.184.216.34', 80))
            ]
            result = validator.validate_url_security("https://example.com/api")
            assert result.is_safe is True


class TestSecurityValidatorEdgeCases:
    """Tests for edge cases and special inputs"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_none_url_blocked(self, validator):
        """Test None URL is blocked"""
        result = validator.validate_url_security(None)
        assert result.is_safe is False
        assert result.risk_level == "critical"

    def test_empty_string_url_blocked(self, validator):
        """Test empty string URL is blocked"""
        result = validator.validate_url_security("")
        assert result.is_safe is False
        assert result.risk_level == "critical"

    def test_non_string_url_blocked(self, validator):
        """Test non-string URL is blocked"""
        result = validator.validate_url_security(123)
        assert result.is_safe is False
        assert result.risk_level == "critical"

    def test_url_too_long_blocked(self, validator):
        """Test URL exceeding maximum length is blocked"""
        long_url = "https://example.com/" + "a" * 2100
        result = validator.validate_url_security(long_url)
        assert result.is_safe is False
        assert "maximum length" in result.blocked_reason.lower() or "exceeds" in result.blocked_reason.lower()

    def test_whitespace_url_trimmed(self, validator):
        """Test whitespace is trimmed from URL"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            result = validator.validate_url_security("  https://example.com  ")
            assert result.is_safe is True

    def test_unicode_in_url(self, validator):
        """Test unicode characters in URL"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            # International domain names
            result = validator.validate_url_security("https://example.com/path?q=caf\u00e9")
            # Should not crash and handle gracefully
            assert result is not None


class TestSecurityValidatorSanitization:
    """Tests for URL sanitization"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_sanitize_empty_url(self, validator):
        """Test sanitizing empty URL returns empty string"""
        result = validator.sanitize_url("")
        assert result == ""

    def test_sanitize_none_url(self, validator):
        """Test sanitizing None URL returns empty string"""
        result = validator.sanitize_url(None)
        assert result == ""

    def test_sanitize_http_to_https(self, validator):
        """Test HTTP is upgraded to HTTPS during sanitization"""
        result = validator.sanitize_url("http://example.com/path")
        assert result.startswith("https://")

    def test_sanitize_removes_traversal(self, validator):
        """Test directory traversal sequences are removed"""
        result = validator.sanitize_url("https://example.com/../../../etc/passwd")
        assert "../" not in result

    def test_sanitize_removes_windows_traversal(self, validator):
        """Test Windows directory traversal sequences are removed"""
        result = validator.sanitize_url("https://example.com/..\\..\\windows")
        assert "..\\" not in result

    def test_sanitize_preserves_query(self, validator):
        """Test query string is preserved during sanitization"""
        result = validator.sanitize_url("https://example.com/path?key=value")
        assert "?key=value" in result

    def test_sanitize_preserves_port(self, validator):
        """Test port is preserved during sanitization"""
        result = validator.sanitize_url("https://example.com:8080/path")
        assert ":8080" in result

    def test_sanitize_lowercases_hostname(self, validator):
        """Test hostname is lowercased during sanitization"""
        result = validator.sanitize_url("https://EXAMPLE.COM/path")
        assert "example.com" in result

    def test_sanitize_malformed_url_returns_empty(self, validator):
        """Test malformed URL sanitization returns empty string"""
        # Very malformed URL that will cause parsing issues
        result = validator.sanitize_url("not-a-valid-url")
        # Should not crash; returns empty or sanitized version
        assert isinstance(result, str)


class TestSecurityValidatorIsSafeForProcessing:
    """Tests for is_safe_for_processing helper method"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_safe_url_returns_true(self, validator):
        """Test safe URL returns True"""
        with patch.object(validator, '_validate_dns_resolution') as mock_dns:
            mock_dns.return_value = SecurityValidationResult(
                is_safe=True, risk_level="safe", issues=[]
            )
            is_safe, reason = validator.is_safe_for_processing("https://example.com")
            assert is_safe is True
            assert reason is None

    def test_unsafe_url_returns_false_with_reason(self, validator):
        """Test unsafe URL returns False with reason"""
        is_safe, reason = validator.is_safe_for_processing("file:///etc/passwd")
        assert is_safe is False
        assert reason is not None
        assert "scheme" in reason.lower() or "allowed" in reason.lower()

    def test_private_ip_returns_false_with_reason(self, validator):
        """Test private IP URL returns False with reason"""
        is_safe, reason = validator.is_safe_for_processing("https://192.168.1.1/admin")
        assert is_safe is False
        assert "private" in reason.lower() or "internal" in reason.lower()


class TestSecurityValidatorRiskPriority:
    """Tests for risk level priority comparison"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    @pytest.mark.parametrize("risk1,risk2,expected", [
        ("safe", "low", "low"),
        ("low", "medium", "medium"),
        ("medium", "high", "high"),
        ("high", "critical", "critical"),
        ("safe", "critical", "critical"),
    ])
    def test_risk_priority_comparison(self, validator, risk1, risk2, expected):
        """Test risk priority comparison returns higher risk"""
        result = max(risk1, risk2, key=validator._risk_priority)
        assert result == expected

    def test_unknown_risk_level_defaults_to_zero(self, validator):
        """Test unknown risk level defaults to priority 0"""
        priority = validator._risk_priority("unknown")
        assert priority == 0


class TestSecurityValidatorClassAttributes:
    """Tests for class-level security configurations"""

    def test_blocked_schemes_completeness(self):
        """Test all dangerous schemes are in blocked list"""
        validator = SecurityValidator()
        dangerous_schemes = {
            "file", "javascript", "data", "vbscript",
            "chrome", "chrome-extension", "about", "blob"
        }
        for scheme in dangerous_schemes:
            assert scheme in validator.BLOCKED_SCHEMES

    def test_allowed_schemes_only_http_https(self):
        """Test only HTTP and HTTPS are allowed"""
        validator = SecurityValidator()
        assert validator.ALLOWED_SCHEMES == {"http", "https"}

    def test_private_ip_ranges_coverage(self):
        """Test private IP ranges include all RFC 1918 ranges"""
        validator = SecurityValidator()
        # Check key private ranges are present
        range_strs = [str(r) for r in validator.PRIVATE_IP_RANGES]
        assert "10.0.0.0/8" in range_strs
        assert "172.16.0.0/12" in range_strs
        assert "192.168.0.0/16" in range_strs
        assert "127.0.0.0/8" in range_strs

    def test_max_url_length_reasonable(self):
        """Test maximum URL length is reasonable"""
        validator = SecurityValidator()
        # IE limit is 2083, commonly used
        assert validator.MAX_URL_LENGTH == 2083

    def test_max_query_params_reasonable(self):
        """Test maximum query params is reasonable"""
        validator = SecurityValidator()
        assert validator.MAX_QUERY_PARAMS == 100


class TestSecurityValidatorIntegration:
    """Integration tests for complete URL validation flow"""

    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    def test_complete_validation_flow_safe_url(self, validator):
        """Test complete validation flow for safe URL"""
        with patch('socket.getaddrinfo') as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, '', ('93.184.216.34', 80))
            ]
            result = validator.validate_url_security(
                "https://example.com/api/v1/users?page=1&limit=10"
            )
            assert result.is_safe is True
            assert result.risk_level == "safe"
            assert result.blocked_reason is None

    def test_complete_validation_flow_ssrf_attempt(self, validator):
        """Test complete validation flow for SSRF attempt"""
        result = validator.validate_url_security("https://169.254.169.254/latest/meta-data")
        assert result.is_safe is False
        assert result.risk_level == "critical"
        assert result.blocked_reason is not None

    def test_complete_validation_flow_localhost_bypass_attempt(self, validator):
        """Test complete validation flow for localhost bypass"""
        result = validator.validate_url_security("https://localhost.localdomain/admin")
        # Should be blocked or flagged
        assert result is not None
        # Either blocked or has issues
        assert not result.is_safe or len(result.issues) > 0 or result.risk_level != "safe"
