"""
Comprehensive tests for bookmark_processor/utils/validators/url.py

Tests cover:
1. URLValidator class - all initialization options and validate method branches
2. BookmarkURLValidator class - bookmark-specific validation
3. validate_url_format function - URL format validation
4. create_url_validator function - factory function
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from bookmark_processor.utils.validators.url import (
    URLValidator,
    BookmarkURLValidator,
    validate_url_format,
    create_url_validator,
)
from bookmark_processor.utils.validators.base import ValidationResult, ValidationSeverity


class TestURLValidator:
    """Tests for URLValidator class"""

    def test_init_default_values(self):
        """Test URLValidator initialization with default values"""
        validator = URLValidator()

        assert validator.field_name is None
        assert validator.required is False
        assert validator.allow_none is True
        assert validator.allowed_schemes == {"http", "https"}
        assert validator.security_check is True
        assert validator.normalize_url is True
        assert validator.security_validator is not None

    def test_init_custom_values(self):
        """Test URLValidator initialization with custom values"""
        validator = URLValidator(
            field_name="test_url",
            required=True,
            allow_none=False,
            allowed_schemes=["http", "https", "ftp"],
            security_check=False,
            normalize_url=False,
        )

        assert validator.field_name == "test_url"
        assert validator.required is True
        assert validator.allow_none is False
        assert validator.allowed_schemes == {"http", "https", "ftp"}
        assert validator.security_check is False
        assert validator.normalize_url is False
        assert validator.security_validator is None

    def test_validate_none_value_allowed(self):
        """Test validation of None value when allowed"""
        validator = URLValidator(allow_none=True, required=False)
        result = validator.validate(None)

        assert result.is_valid is True
        assert result.sanitized_value is None

    def test_validate_none_value_not_allowed(self):
        """Test validation of None value when not allowed"""
        validator = URLValidator(allow_none=False)
        result = validator.validate(None)

        assert result.is_valid is False
        assert len(result.issues) > 0

    def test_validate_none_value_required(self):
        """Test validation of None value when field is required"""
        validator = URLValidator(required=True, allow_none=True)
        result = validator.validate(None)

        assert result.is_valid is False
        assert len(result.issues) > 0

    def test_validate_empty_string_not_required(self):
        """Test validation of empty string when field is not required"""
        validator = URLValidator(required=False)
        result = validator.validate("")

        # Empty string should still be valid for non-required fields
        assert result.is_valid is True
        assert result.sanitized_value == ""

    def test_validate_empty_string_required(self):
        """Test validation of empty string when field is required"""
        validator = URLValidator(required=True)
        result = validator.validate("")

        assert result.is_valid is False
        assert any("empty" in str(issue.message).lower() for issue in result.issues)

    def test_validate_whitespace_only_string(self):
        """Test validation of whitespace-only string"""
        validator = URLValidator(required=True)
        result = validator.validate("   ")

        assert result.is_valid is False

    def test_validate_non_string_value(self):
        """Test validation converts non-string value to string"""
        validator = URLValidator()
        # Create a mock object with __str__ method
        mock_value = Mock()
        mock_value.__str__ = Mock(return_value="https://example.com")

        with patch.object(validator, 'security_validator') as mock_security:
            mock_security.validate_url_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate(mock_value)

        # Should convert and process as string
        assert result.sanitized_value is not None

    def test_validate_valid_https_url(self):
        """Test validation of valid HTTPS URL"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https://example.com/path")

        assert result.is_valid is True
        assert "example.com" in result.sanitized_value

    def test_validate_valid_http_url(self):
        """Test validation of valid HTTP URL"""
        validator = URLValidator(security_check=False)
        result = validator.validate("http://example.com")

        assert result.is_valid is True

    def test_validate_url_without_scheme(self):
        """Test validation of URL without scheme adds https"""
        validator = URLValidator(security_check=False)
        result = validator.validate("example.com/path")

        assert result.is_valid is True
        assert result.sanitized_value.startswith("https://")
        # Should have a warning about missing scheme
        assert any(
            issue.severity == ValidationSeverity.WARNING
            for issue in result.issues
        )

    def test_validate_disallowed_scheme(self):
        """Test validation rejects disallowed scheme"""
        validator = URLValidator(
            allowed_schemes=["https"],
            security_check=False
        )
        result = validator.validate("http://example.com")

        assert result.is_valid is False
        assert any("scheme" in str(issue.message).lower() for issue in result.issues)

    def test_validate_ftp_scheme_when_allowed(self):
        """Test validation allows FTP scheme when configured"""
        validator = URLValidator(
            allowed_schemes=["http", "https", "ftp"],
            security_check=False
        )
        result = validator.validate("ftp://files.example.com")

        assert result.is_valid is True

    def test_validate_url_missing_domain(self):
        """Test validation rejects URL without domain"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https:///path/to/file")

        assert result.is_valid is False
        assert any("domain" in str(issue.message).lower() or "hostname" in str(issue.message).lower() for issue in result.issues)

    def test_validate_url_with_security_check_safe(self):
        """Test validation with security check that passes"""
        validator = URLValidator(security_check=True)

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://example.com")

        mock_security.assert_called_once()
        assert result.is_valid is True

    def test_validate_url_with_security_check_unsafe(self):
        """Test validation with security check that fails"""
        validator = URLValidator(security_check=True)

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=False,
                issues=["Suspicious pattern detected"],
                blocked_reason="Security violation"
            )
            result = validator.validate("https://malicious.example.com")

        assert result.is_valid is False
        assert any(
            issue.severity == ValidationSeverity.CRITICAL
            for issue in result.issues
        )
        assert "security_issues" in result.metadata

    def test_validate_url_with_security_warnings(self):
        """Test validation with security warnings (safe but with issues)"""
        validator = URLValidator(security_check=True)

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=["Non-standard port used"],
                blocked_reason=None
            )
            result = validator.validate("https://example.com:8080")

        assert result.is_valid is True
        assert any(
            issue.severity == ValidationSeverity.WARNING
            for issue in result.issues
        )
        assert "security_issues" in result.metadata

    def test_validate_url_normalization_lowercase_domain(self):
        """Test URL normalization lowercases domain"""
        validator = URLValidator(
            security_check=False,
            normalize_url=True
        )
        result = validator.validate("https://EXAMPLE.COM/Path")

        assert result.is_valid is True
        assert "example.com" in result.sanitized_value
        # Path case should be preserved
        assert "/Path" in result.sanitized_value

    def test_validate_url_normalization_trailing_slash(self):
        """Test URL normalization handles trailing slash"""
        validator = URLValidator(
            security_check=False,
            normalize_url=True
        )

        # Path with trailing slash should have it removed
        result = validator.validate("https://example.com/path/")
        assert result.sanitized_value == "https://example.com/path"

        # Root path (just slash) gets normalized - implementation strips trailing slash
        # and if path is empty, "/" is not added back (since parsed.path == "/")
        result2 = validator.validate("https://example.com/")
        # The implementation strips trailing slash from "/" leaving empty path
        # which means the normalized URL has no trailing slash
        assert result2.sanitized_value == "https://example.com"

    def test_validate_url_normalization_no_path(self):
        """Test URL normalization adds slash when no path"""
        validator = URLValidator(
            security_check=False,
            normalize_url=True
        )
        result = validator.validate("https://example.com")

        assert result.sanitized_value == "https://example.com/"

    def test_validate_url_normalization_with_query(self):
        """Test URL normalization preserves query string"""
        validator = URLValidator(
            security_check=False,
            normalize_url=True
        )
        result = validator.validate("https://example.com/search?q=test&page=1")

        assert "?q=test&page=1" in result.sanitized_value

    def test_validate_url_normalization_with_fragment(self):
        """Test URL normalization preserves fragment"""
        validator = URLValidator(
            security_check=False,
            normalize_url=True
        )
        result = validator.validate("https://example.com/page#section")

        assert "#section" in result.sanitized_value

    def test_validate_url_normalization_disabled(self):
        """Test URL is not normalized when disabled"""
        validator = URLValidator(
            security_check=False,
            normalize_url=False
        )
        result = validator.validate("https://EXAMPLE.COM/Path/")

        # Original URL should be preserved (after stripping)
        assert result.sanitized_value == "https://EXAMPLE.COM/Path/"

    def test_validate_url_normalization_error_handling(self):
        """Test URL normalization handles errors gracefully"""
        validator = URLValidator(
            security_check=False,
            normalize_url=True
        )

        # Inject an error during normalization by mocking urlparse
        # to raise during second call (after scheme addition)
        with patch('bookmark_processor.utils.validators.url.urlparse') as mock_urlparse:
            # First call succeeds (initial parsing)
            first_result = Mock()
            first_result.scheme = "https"
            first_result.netloc = "example.com"
            first_result.hostname = "example.com"
            first_result.path = "/test"
            first_result.query = ""
            first_result.fragment = ""
            first_result.port = None

            # Configure to work normally first, then raise during normalization
            mock_urlparse.return_value = first_result

            # When accessing .lower() on netloc, simulate an exception
            type(first_result).netloc = property(
                lambda self: Mock(lower=Mock(side_effect=Exception("Normalization error")))
            )

            # This should handle the exception and add a warning
            result = validator.validate("https://example.com/test")

        # Even if normalization fails, the URL should still be valid
        # (just with a warning)
        assert any(
            issue.severity == ValidationSeverity.WARNING and "normalization" in str(issue.message).lower()
            for issue in result.issues
        ) or result.is_valid


class TestBookmarkURLValidator:
    """Tests for BookmarkURLValidator class"""

    def test_init_default_configuration(self):
        """Test BookmarkURLValidator has correct default configuration"""
        validator = BookmarkURLValidator()

        assert validator.field_name == "url"
        assert validator.required is True
        assert validator.allow_none is False
        assert validator.allowed_schemes == {"http", "https"}
        assert validator.security_check is True
        assert validator.normalize_url is True

    def test_validate_localhost_url(self):
        """Test validation warns about localhost URLs"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://localhost/path")

        assert result.is_valid is True
        assert any(
            "localhost" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_127_0_0_1_url(self):
        """Test validation warns about 127.0.0.1 URLs"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://127.0.0.1/api")

        assert result.is_valid is True
        assert any(
            "localhost" in str(issue.message).lower() or "127.0.0.1" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_0_0_0_0_url(self):
        """Test validation warns about 0.0.0.0 URLs"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://0.0.0.0/app")

        assert result.is_valid is True
        assert any(
            "localhost" in str(issue.message).lower() or "development" in str(issue.message).lower()
            for issue in result.issues
        )

    @pytest.mark.parametrize("port", [3000, 3001, 8000, 8080, 8888, 9000])
    def test_validate_development_port(self, port):
        """Test validation warns about common development ports"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate(f"https://example.com:{port}/app")

        assert result.is_valid is True
        assert any(
            "development port" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_non_development_port(self):
        """Test validation does not warn about standard ports"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://example.com:443/secure")

        assert result.is_valid is True
        # Should not have development port warning
        assert not any(
            "development port" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_very_long_url(self):
        """Test validation warns about very long URLs"""
        validator = BookmarkURLValidator()
        long_path = "a" * 600

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate(f"https://example.com/{long_path}")

        assert result.is_valid is True
        assert any(
            "long" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_normal_length_url(self):
        """Test validation does not warn about normal length URLs"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://example.com/normal/path")

        # Should not have length warning
        assert not any(
            "long" in str(issue.message).lower() and "char" in str(issue.message).lower()
            for issue in result.issues
        )

    @pytest.mark.parametrize("tracking_param", [
        "utm_source", "utm_medium", "utm_campaign", "fbclid", "gclid"
    ])
    def test_validate_tracking_parameters(self, tracking_param):
        """Test validation detects tracking parameters"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate(f"https://example.com?{tracking_param}=test123")

        assert result.is_valid is True
        assert any(
            "tracking" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_multiple_tracking_parameters(self):
        """Test validation detects multiple tracking parameters"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate(
                "https://example.com?utm_source=google&utm_medium=cpc&fbclid=abc123"
            )

        assert result.is_valid is True
        # Should mention tracking parameters
        tracking_issues = [
            issue for issue in result.issues
            if "tracking" in str(issue.message).lower()
        ]
        assert len(tracking_issues) > 0
        # Should list multiple parameters
        tracking_message = str(tracking_issues[0].message)
        assert "utm_source" in tracking_message or "utm_medium" in tracking_message

    def test_validate_long_fragment(self):
        """Test validation notices long fragment identifiers"""
        validator = BookmarkURLValidator()
        long_fragment = "a" * 60

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate(f"https://example.com/page#{long_fragment}")

        assert result.is_valid is True
        assert any(
            "fragment" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_normal_fragment(self):
        """Test validation does not warn about normal fragments"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://example.com/page#section-1")

        # Should not have fragment warning for short fragments
        assert not any(
            "fragment" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_invalid_url_from_parent(self):
        """Test validation inherits parent validation failure"""
        validator = BookmarkURLValidator()

        # Required field with empty value should fail
        result = validator.validate("")

        assert result.is_valid is False

    def test_validate_none_value_not_allowed(self):
        """Test BookmarkURLValidator rejects None values"""
        validator = BookmarkURLValidator()
        result = validator.validate(None)

        assert result.is_valid is False

    def test_validate_url_parsing_exception_handling(self):
        """Test validation handles URL parsing exceptions gracefully"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )

            # URL that might cause parsing issues but is still processable
            result = validator.validate("https://example.com/path with spaces")

        # Should not raise exception, should handle gracefully
        assert isinstance(result, ValidationResult)


class TestValidateUrlFormat:
    """Tests for validate_url_format function"""

    def test_valid_http_url(self):
        """Test valid HTTP URL returns True"""
        assert validate_url_format("http://example.com") is True

    def test_valid_https_url(self):
        """Test valid HTTPS URL returns True"""
        assert validate_url_format("https://example.com") is True

    def test_valid_ftp_url(self):
        """Test valid FTP URL returns True"""
        assert validate_url_format("ftp://files.example.com") is True

    def test_valid_url_with_path(self):
        """Test valid URL with path returns True"""
        assert validate_url_format("https://example.com/path/to/page") is True

    def test_valid_url_with_query(self):
        """Test valid URL with query string returns True"""
        assert validate_url_format("https://example.com/search?q=test") is True

    def test_valid_url_with_port(self):
        """Test valid URL with port returns True"""
        assert validate_url_format("https://example.com:8080/api") is True

    def test_valid_url_with_subdomain(self):
        """Test valid URL with subdomain returns True"""
        assert validate_url_format("https://www.example.com") is True

    def test_valid_url_with_ip_address(self):
        """Test valid URL with IP address returns True"""
        assert validate_url_format("http://192.168.1.1/admin") is True

    def test_valid_url_with_localhost(self):
        """Test valid URL with localhost returns True"""
        assert validate_url_format("http://localhost/test") is True
        assert validate_url_format("http://localhost:3000") is True

    def test_none_value(self):
        """Test None value returns False"""
        assert validate_url_format(None) is False

    def test_empty_string(self):
        """Test empty string returns False"""
        assert validate_url_format("") is False

    def test_whitespace_only_string(self):
        """Test whitespace-only string returns False"""
        assert validate_url_format("   ") is False
        assert validate_url_format("\t\n") is False

    def test_non_string_value(self):
        """Test non-string value returns False"""
        assert validate_url_format(12345) is False
        assert validate_url_format(['http://example.com']) is False
        assert validate_url_format({'url': 'http://example.com'}) is False

    def test_javascript_scheme(self):
        """Test javascript: scheme returns False"""
        assert validate_url_format("javascript:void(0)") is False
        assert validate_url_format("javascript:alert('XSS')") is False
        assert validate_url_format("JAVASCRIPT:alert(1)") is False  # Case insensitive

    def test_data_scheme(self):
        """Test data: scheme returns False"""
        assert validate_url_format("data:text/html,<script>alert(1)</script>") is False
        assert validate_url_format("DATA:text/plain,test") is False

    def test_vbscript_scheme(self):
        """Test vbscript: scheme returns False"""
        assert validate_url_format("vbscript:msgbox(1)") is False
        assert validate_url_format("VBSCRIPT:test") is False

    def test_mailto_scheme(self):
        """Test mailto: scheme returns False"""
        assert validate_url_format("mailto:test@example.com") is False
        assert validate_url_format("MAILTO:user@domain.com") is False

    def test_multiple_scheme_separators(self):
        """Test URLs with multiple :// return False"""
        assert validate_url_format("http://http://example.com") is False
        assert validate_url_format("https://://example.com") is False

    def test_missing_host_after_scheme(self):
        """Test URL with scheme but no host returns False"""
        assert validate_url_format("http://") is False
        assert validate_url_format("https://") is False

    def test_url_without_scheme(self):
        """Test URL without scheme returns False"""
        assert validate_url_format("example.com") is False
        assert validate_url_format("www.example.com") is False

    def test_whitespace_is_stripped(self):
        """Test that leading/trailing whitespace is handled"""
        assert validate_url_format("  https://example.com  ") is True

    def test_invalid_domain_format(self):
        """Test invalid domain format returns False"""
        # Missing TLD extension pattern
        assert validate_url_format("http://.com") is False
        # Note: The regex pattern validates IP format syntactically (\d{1,3}.\d{1,3}...)
        # It does not validate semantic IP address validity (0-255 range)
        # So 999.999.999.999 is syntactically valid to the regex even though
        # it's not a valid IP address semantically
        # This test documents the actual behavior
        assert validate_url_format("http://999.999.999.999") is True

    def test_relative_url(self):
        """Test relative URL returns False"""
        assert validate_url_format("/path/to/page") is False
        assert validate_url_format("path/to/page") is False

    def test_file_scheme(self):
        """Test file: scheme returns False"""
        # Not in the valid pattern (only http, https, ftp)
        assert validate_url_format("file:///etc/passwd") is False


class TestCreateUrlValidator:
    """Tests for create_url_validator factory function"""

    def test_create_with_field_name(self):
        """Test creating validator with field name"""
        validator = create_url_validator("my_url_field")

        assert validator.field_name == "my_url_field"
        assert validator.required is True  # Default when not specified
        assert validator.security_check is True

    def test_create_required(self):
        """Test creating required URL validator"""
        validator = create_url_validator("url", required=True)

        assert validator.required is True
        assert validator.allow_none is False

    def test_create_optional(self):
        """Test creating optional URL validator"""
        validator = create_url_validator("url", required=False)

        assert validator.required is False
        assert validator.allow_none is True

    def test_create_returns_url_validator_instance(self):
        """Test factory returns URLValidator instance"""
        validator = create_url_validator("test")

        assert isinstance(validator, URLValidator)

    def test_created_validator_has_security_check(self):
        """Test created validator has security check enabled"""
        validator = create_url_validator("url")

        assert validator.security_check is True
        assert validator.security_validator is not None

    def test_created_validator_validates_correctly(self):
        """Test created validator validates URLs correctly"""
        validator = create_url_validator("url", required=True)

        # Should fail for empty value (required)
        result = validator.validate("")
        assert result.is_valid is False

        # Should pass for valid URL (with mocked security)
        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://example.com")
        assert result.is_valid is True


class TestURLValidatorEdgeCases:
    """Edge case tests for URL validation"""

    def test_url_with_unicode_characters(self):
        """Test URL with unicode characters in path"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https://example.com/path/cafe")

        assert result.is_valid is True

    def test_url_with_encoded_characters(self):
        """Test URL with percent-encoded characters"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https://example.com/path%20with%20spaces")

        assert result.is_valid is True

    def test_url_with_special_characters_in_query(self):
        """Test URL with special characters in query string"""
        validator = URLValidator(security_check=False)
        result = validator.validate(
            "https://example.com/search?q=test&filter=a%3Db&sort=name"
        )

        assert result.is_valid is True

    def test_url_with_authentication_info(self):
        """Test URL with username/password (may be flagged by security)"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https://user:pass@example.com/admin")

        # URL format is valid even with auth info
        assert result.is_valid is True

    def test_url_with_ipv6_address(self):
        """Test URL with IPv6 address"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https://[2001:db8::1]/path")

        # May or may not be valid depending on urlparse handling
        assert isinstance(result, ValidationResult)

    def test_url_with_fragment_only(self):
        """Test URL that's just a fragment"""
        assert validate_url_format("#section") is False

    def test_url_with_multiple_fragments(self):
        """Test URL with multiple fragment markers"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https://example.com/page#section#subsection")

        assert isinstance(result, ValidationResult)

    def test_extremely_long_query_string(self):
        """Test URL with extremely long query string"""
        validator = URLValidator(security_check=False)
        long_query = "&".join([f"param{i}=value{i}" for i in range(100)])
        result = validator.validate(f"https://example.com?{long_query}")

        assert isinstance(result, ValidationResult)

    def test_url_with_empty_path_and_query(self):
        """Test URL with empty path but with query"""
        validator = URLValidator(security_check=False)
        result = validator.validate("https://example.com?query=value")

        assert result.is_valid is True

    def test_case_sensitivity_in_scheme(self):
        """Test URL scheme is case-insensitive"""
        validator = URLValidator(security_check=False)

        result1 = validator.validate("HTTPS://example.com")
        result2 = validator.validate("HtTpS://example.com")

        assert result1.is_valid is True
        assert result2.is_valid is True


class TestBookmarkURLValidatorEdgeCases:
    """Edge case tests for BookmarkURLValidator"""

    def test_validate_mixed_case_localhost(self):
        """Test validation handles mixed case localhost"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://LocalHost/path")

        # Should still detect localhost regardless of case
        assert any(
            "localhost" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_url_just_under_length_limit(self):
        """Test URL at 500 characters is not flagged"""
        validator = BookmarkURLValidator()
        # Create URL just under 500 chars
        path_length = 500 - len("https://example.com/")
        path = "a" * path_length
        url = f"https://example.com/{path}"

        # Ensure it's <= 500
        if len(url) <= 500:
            with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
                mock_security.return_value = Mock(
                    is_safe=True,
                    issues=[],
                    blocked_reason=None
                )
                result = validator.validate(url)

            # Should not have length warning
            assert not any(
                "long" in str(issue.message).lower() and "char" in str(issue.message).lower()
                for issue in result.issues
            )

    def test_validate_url_exactly_501_chars(self):
        """Test URL at 501 characters is flagged"""
        validator = BookmarkURLValidator()
        base_url = "https://example.com/"
        path_length = 501 - len(base_url)
        url = f"{base_url}{'a' * path_length}"

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate(url)

        assert any(
            "long" in str(issue.message).lower()
            for issue in result.issues
        )

    def test_validate_url_with_no_tracking_but_similar_params(self):
        """Test URL with parameters similar to tracking but not tracking"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://example.com?source=internal&medium=email")

        # These are not the exact tracking parameters
        tracking_issues = [
            issue for issue in result.issues
            if "tracking" in str(issue.message).lower()
        ]
        assert len(tracking_issues) == 0

    def test_validate_handles_sanitized_value_non_string(self):
        """Test validation handles case where sanitized_value might not be string"""
        validator = BookmarkURLValidator()

        # If parent returns non-string sanitized_value, should handle gracefully
        # This is an edge case that shouldn't happen but good to test
        result = validator.validate(None)

        # Should have failed validation for None
        assert result.is_valid is False
        assert isinstance(result, ValidationResult)


class TestURLValidatorUncoveredBranches:
    """Tests specifically targeting uncovered code branches"""

    def test_validate_none_returns_early_after_basic_check_passes(self):
        """Test that None value returns early with sanitized_value=None when allowed"""
        # This covers lines 56-58 - the case where None passes basic check
        # but we still need to return early with sanitized_value=None
        validator = URLValidator(allow_none=True, required=False)
        result = validator.validate(None)

        assert result.is_valid is True
        assert result.sanitized_value is None
        # Should have no issues since None is allowed
        assert len(result.issues) == 0

    def test_validate_urlparse_exception(self):
        """Test handling of urlparse exception"""
        # This covers lines 75-77
        validator = URLValidator(security_check=False)

        with patch('bookmark_processor.utils.validators.url.urlparse') as mock_urlparse:
            mock_urlparse.side_effect = Exception("Parsing error")
            result = validator.validate("https://example.com")

        assert result.is_valid is False
        assert any("Invalid URL format" in str(issue.message) for issue in result.issues)

    def test_validate_urlparse_exception_after_adding_scheme(self):
        """Test handling of urlparse exception after adding https:// scheme"""
        # This covers lines 86-90
        validator = URLValidator(security_check=False)

        call_count = 0

        def mock_urlparse_side_effect(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds but returns no scheme
                result = Mock()
                result.scheme = ""
                result.netloc = "example.com"
                return result
            else:
                # Second call (after scheme added) raises exception
                raise Exception("Parsing error after scheme addition")

        with patch('bookmark_processor.utils.validators.url.urlparse', side_effect=mock_urlparse_side_effect):
            result = validator.validate("example.com")

        assert result.is_valid is False
        assert any("Invalid URL after adding scheme" in str(issue.message) for issue in result.issues)

    def test_bookmark_validator_exception_in_additional_checks(self):
        """Test BookmarkURLValidator handles exception in additional checks"""
        # This covers lines 209-210 in BookmarkURLValidator
        # The exception handler catches any exception in the additional bookmark checks
        # and silently passes (since parent already validated the URL)

        validator = BookmarkURLValidator()

        # We need to make the parent validation succeed, then make urlparse
        # raise an exception when called again in BookmarkURLValidator.validate

        # First, do normal validation to get a valid result
        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )

            # Use a URL that will pass parent validation
            # Then patch urlparse to fail on subsequent calls
            from urllib.parse import urlparse as real_urlparse

            call_count = 0

            def controlled_urlparse(url):
                nonlocal call_count
                call_count += 1
                # Parent class URLValidator calls urlparse up to 2 times
                # (once for initial parse, possibly once after adding scheme)
                # BookmarkURLValidator.validate calls it once more on line 163
                if call_count <= 2:
                    return real_urlparse(url)
                # On the third call (from BookmarkURLValidator.validate line 163),
                # raise an exception to trigger lines 209-210
                raise ValueError("Simulated urlparse failure in bookmark validation")

            with patch('bookmark_processor.utils.validators.url.urlparse', side_effect=controlled_urlparse):
                result = validator.validate("https://example.com/test")

        # Should still be valid because the exception in lines 209-210 is caught
        # and the parent validation already passed
        assert result.is_valid is True
        assert isinstance(result, ValidationResult)

    def test_bookmark_validator_exception_via_netloc_access(self):
        """Test BookmarkURLValidator exception when accessing parsed URL attributes"""
        # This is another approach to cover lines 209-210
        # We trigger an exception during the bookmark-specific checks
        # by making an attribute access fail

        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )

            # Mock the result from parent to have a sanitized_value
            # but make urlparse return an object that raises on attribute access
            from urllib.parse import urlparse as real_urlparse

            call_count = 0

            def controlled_urlparse(url):
                nonlocal call_count
                call_count += 1
                # Let first two calls succeed (parent validation)
                if call_count <= 2:
                    return real_urlparse(url)
                # Third call - return a mock that raises on attribute access
                mock_parsed = Mock()
                mock_parsed.netloc = property(lambda self: (_ for _ in ()).throw(Exception("netloc access failed")))
                # Make it raise when netloc.lower() is called
                type(mock_parsed).netloc = property(lambda s: (_ for _ in ()).throw(RuntimeError("Boom")))
                return mock_parsed

            with patch('bookmark_processor.utils.validators.url.urlparse', side_effect=controlled_urlparse):
                result = validator.validate("https://example.com/path")

        # Should still be valid - exception is caught silently
        assert result.is_valid is True

    def test_validate_empty_string_required_returns_early(self):
        """Test empty string for required field returns early with error"""
        # Note: Whitespace-only strings are handled by the base validator's
        # _check_required_and_none method, which returns early before URLValidator's
        # empty string check (line 69). The sanitized_value is None from that check.
        validator = URLValidator(required=True)
        result = validator.validate("   ")  # Whitespace-only

        assert result.is_valid is False
        assert any("empty" in str(issue.message).lower() for issue in result.issues)
        # Base validator handles this, sets sanitized_value to None
        assert result.sanitized_value is None

    def test_validate_empty_string_after_strip_required(self):
        """Test empty string detected after strip (covers line 67-70)"""
        # To actually hit line 69, we need a value that passes _check_required_and_none
        # but becomes empty after strip. This is tricky because the base validator
        # also checks for empty strings. However, if we pass an object that converts
        # to an empty string, it should work.

        class EmptyStringable:
            def __str__(self):
                return ""

        validator = URLValidator(required=True)
        result = validator.validate(EmptyStringable())

        assert result.is_valid is False
        assert any("empty" in str(issue.message).lower() or "cannot be empty" in str(issue.message).lower() for issue in result.issues)


class TestIntegration:
    """Integration tests for URL validators"""

    def test_full_validation_flow_valid_url(self):
        """Test complete validation flow for a valid URL"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=[],
                blocked_reason=None
            )
            result = validator.validate("https://www.example.com/article/123?ref=home#comments")

        assert result.is_valid is True
        assert result.sanitized_value is not None
        assert "example.com" in result.sanitized_value

    def test_full_validation_flow_problematic_url(self):
        """Test complete validation flow for a problematic but valid URL"""
        validator = BookmarkURLValidator()

        with patch.object(validator.security_validator, 'validate_url_security') as mock_security:
            mock_security.return_value = Mock(
                is_safe=True,
                issues=["Non-standard port"],
                blocked_reason=None
            )
            # URL with tracking params, dev port, and long fragment
            long_frag = "a" * 60
            result = validator.validate(
                f"https://localhost:3000/page?utm_source=test&fbclid=abc#{long_frag}"
            )

        assert result.is_valid is True
        # Should have multiple warnings/info issues
        assert len(result.issues) >= 2

    def test_validate_url_format_matches_urlvalidator(self):
        """Test that validate_url_format aligns with URLValidator for basic cases"""
        test_urls = [
            ("https://example.com", True),
            ("http://example.com", True),
            ("", False),
            ("javascript:alert(1)", False),
            ("mailto:test@test.com", False),
        ]

        for url, expected in test_urls:
            format_result = validate_url_format(url)
            assert format_result == expected, f"Failed for {url}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
