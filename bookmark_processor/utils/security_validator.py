"""
Security Validation Module

Provides comprehensive security validation for URLs and user input to protect against
SSRF attacks, malicious URLs, and other security vulnerabilities.
"""

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class SecurityValidationResult:
    """Result of security validation"""

    is_safe: bool
    risk_level: str  # 'safe', 'low', 'medium', 'high', 'critical'
    issues: List[str]
    blocked_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "risk_level": self.risk_level,
            "issues": self.issues,
            "blocked_reason": self.blocked_reason,
        }


class SecurityValidator:
    """Comprehensive security validator for URLs and user input"""

    # Dangerous URL schemes that should never be processed
    BLOCKED_SCHEMES = {
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
        "wyciwyg",
        "chrome",
        "chrome-extension",
        "moz-extension",
        "ms-browser-extension",
        "about",
        "blob",
    }

    # Only allow these schemes
    ALLOWED_SCHEMES = {"http", "https"}

    # Private/internal IP ranges (RFC 1918, RFC 3927, etc.)
    PRIVATE_IP_RANGES = [
        ipaddress.IPv4Network("10.0.0.0/8"),  # RFC 1918
        ipaddress.IPv4Network("172.16.0.0/12"),  # RFC 1918
        ipaddress.IPv4Network("192.168.0.0/16"),  # RFC 1918
        ipaddress.IPv4Network("127.0.0.0/8"),  # Loopback
        ipaddress.IPv4Network("169.254.0.0/16"),  # Link-local
        ipaddress.IPv4Network("224.0.0.0/4"),  # Multicast
        ipaddress.IPv4Network("240.0.0.0/4"),  # Reserved
        ipaddress.IPv6Network("::1/128"),  # IPv6 loopback
        ipaddress.IPv6Network("fc00::/7"),  # IPv6 private
        ipaddress.IPv6Network("fe80::/10"),  # IPv6 link-local
    ]

    # Dangerous hostnames/domains
    BLOCKED_HOSTNAMES = {
        "localhost",
        "0.0.0.0",
        "0",
        "local",
        "metadata.google.internal",  # GCP metadata
        "instance-data",  # AWS metadata
        "169.254.169.254",  # AWS/GCP metadata IP
    }

    # Suspicious URL patterns
    SUSPICIOUS_PATTERNS = [
        r"\.\./",  # Directory traversal
        r"%2e%2e%2f",  # URL-encoded directory traversal
        r"%252e%252e%252f",  # Double URL-encoded directory traversal
        r"localhost",  # Localhost references
        r"127\.0\.0\.1",  # Loopback IP
        r"0\.0\.0\.0",  # Null route
        r"@.*:",  # Username:password in URL
        r"javascript:",  # JavaScript protocol
        r"data:",  # Data URLs
        r"file:",  # File protocol
    ]

    # Maximum URL length to prevent buffer overflow attacks
    MAX_URL_LENGTH = 2083  # IE limit, good general limit

    # Maximum number of query parameters
    MAX_QUERY_PARAMS = 100

    def __init__(self):
        """Initialize security validator"""
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SUSPICIOUS_PATTERNS
        ]
        logger.info("Security validator initialized with comprehensive protection")

    def validate_url_security(self, url: str) -> SecurityValidationResult:
        """
        Perform comprehensive security validation on a URL.

        Args:
            url: URL to validate

        Returns:
            SecurityValidationResult with validation details
        """
        issues = []
        risk_level = "safe"

        # Basic format validation
        if not url or not isinstance(url, str):
            return SecurityValidationResult(
                is_safe=False,
                risk_level="critical",
                issues=["Invalid URL: None or non-string value"],
                blocked_reason="Invalid input format",
            )

        # Length check
        if len(url) > self.MAX_URL_LENGTH:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="high",
                issues=[f"URL too long: {len(url)} > {self.MAX_URL_LENGTH}"],
                blocked_reason="URL exceeds maximum length",
            )

        # Parse URL
        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="critical",
                issues=[f"URL parsing failed: {str(e)}"],
                blocked_reason="Malformed URL",
            )

        # Scheme validation
        scheme_result = self._validate_scheme(parsed.scheme)
        if not scheme_result.is_safe:
            return scheme_result
        issues.extend(scheme_result.issues)
        risk_level = max(risk_level, scheme_result.risk_level, key=self._risk_priority)

        # Hostname validation
        hostname_result = self._validate_hostname(parsed.hostname)
        if not hostname_result.is_safe:
            return hostname_result
        issues.extend(hostname_result.issues)
        risk_level = max(
            risk_level, hostname_result.risk_level, key=self._risk_priority
        )

        # Port validation
        port_result = self._validate_port(parsed.port)
        if not port_result.is_safe:
            return port_result
        issues.extend(port_result.issues)
        risk_level = max(risk_level, port_result.risk_level, key=self._risk_priority)

        # Path validation
        path_result = self._validate_path(parsed.path)
        if not path_result.is_safe:
            return path_result
        issues.extend(path_result.issues)
        risk_level = max(risk_level, path_result.risk_level, key=self._risk_priority)

        # Query validation
        query_result = self._validate_query(parsed.query)
        if not query_result.is_safe:
            return query_result
        issues.extend(query_result.issues)
        risk_level = max(risk_level, query_result.risk_level, key=self._risk_priority)

        # Pattern matching for suspicious content
        pattern_result = self._check_suspicious_patterns(url)
        if not pattern_result.is_safe:
            return pattern_result
        issues.extend(pattern_result.issues)
        risk_level = max(risk_level, pattern_result.risk_level, key=self._risk_priority)

        # DNS resolution check (if hostname is provided)
        if parsed.hostname:
            dns_result = self._validate_dns_resolution(parsed.hostname)
            if not dns_result.is_safe:
                return dns_result
            issues.extend(dns_result.issues)
            risk_level = max(risk_level, dns_result.risk_level, key=self._risk_priority)

        return SecurityValidationResult(
            is_safe=True, risk_level=risk_level, issues=issues
        )

    def _validate_scheme(self, scheme: str) -> SecurityValidationResult:
        """Validate URL scheme"""
        if not scheme:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="high",
                issues=["Missing URL scheme"],
                blocked_reason="No scheme specified",
            )

        scheme = scheme.lower()

        if scheme in self.BLOCKED_SCHEMES:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="critical",
                issues=[f"Blocked scheme: {scheme}"],
                blocked_reason=f'Scheme "{scheme}" is not allowed',
            )

        if scheme not in self.ALLOWED_SCHEMES:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="high",
                issues=[f"Unsupported scheme: {scheme}"],
                blocked_reason="Only HTTP/HTTPS schemes are allowed",
            )

        return SecurityValidationResult(is_safe=True, risk_level="safe", issues=[])

    def _validate_hostname(self, hostname: Optional[str]) -> SecurityValidationResult:
        """Validate hostname for security issues"""
        if not hostname:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="high",
                issues=["Missing hostname"],
                blocked_reason="No hostname specified",
            )

        hostname = hostname.lower().strip()

        # Check blocked hostnames
        if hostname in self.BLOCKED_HOSTNAMES:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="critical",
                issues=[f"Blocked hostname: {hostname}"],
                blocked_reason=f'Hostname "{hostname}" is not allowed',
            )

        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            # Check if IP is in private ranges
            for private_range in self.PRIVATE_IP_RANGES:
                if ip in private_range:
                    return SecurityValidationResult(
                        is_safe=False,
                        risk_level="critical",
                        issues=[f"Private IP address: {hostname}"],
                        blocked_reason="Private/internal IP addresses are not allowed",
                    )
        except ValueError:
            # Not an IP address, continue with hostname validation
            pass

        # Check hostname format
        if not self._is_valid_hostname_format(hostname):
            return SecurityValidationResult(
                is_safe=False,
                risk_level="medium",
                issues=[f"Invalid hostname format: {hostname}"],
                blocked_reason="Invalid hostname format",
            )

        return SecurityValidationResult(is_safe=True, risk_level="safe", issues=[])

    def _validate_port(self, port: Optional[int]) -> SecurityValidationResult:
        """Validate port number"""
        if port is None:
            return SecurityValidationResult(is_safe=True, risk_level="safe", issues=[])

        # Check for valid port range
        if not (1 <= port <= 65535):
            return SecurityValidationResult(
                is_safe=False,
                risk_level="medium",
                issues=[f"Invalid port: {port}"],
                blocked_reason="Port must be between 1 and 65535",
            )

        # Warn about non-standard HTTP ports
        if port not in {80, 443, 8080, 8443}:
            return SecurityValidationResult(
                is_safe=True, risk_level="low", issues=[f"Non-standard port: {port}"]
            )

        return SecurityValidationResult(is_safe=True, risk_level="safe", issues=[])

    def _validate_path(self, path: str) -> SecurityValidationResult:
        """Validate URL path for traversal attacks"""
        if not path:
            return SecurityValidationResult(is_safe=True, risk_level="safe", issues=[])

        issues = []
        risk_level = "safe"

        # Check for directory traversal
        if "../" in path or "..\\" in path:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="high",
                issues=["Directory traversal detected in path"],
                blocked_reason="Path contains directory traversal sequences",
            )

        # Check for null bytes
        if "\x00" in path:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="high",
                issues=["Null byte detected in path"],
                blocked_reason="Path contains null bytes",
            )

        # Check path length
        if len(path) > 1000:  # Reasonable path length limit
            issues.append("Very long path detected")
            risk_level = "low"

        return SecurityValidationResult(
            is_safe=True, risk_level=risk_level, issues=issues
        )

    def _validate_query(self, query: str) -> SecurityValidationResult:
        """Validate query parameters"""
        if not query:
            return SecurityValidationResult(is_safe=True, risk_level="safe", issues=[])

        issues = []
        risk_level = "safe"

        try:
            params = parse_qs(query, keep_blank_values=True)

            # Check number of parameters
            if len(params) > self.MAX_QUERY_PARAMS:
                return SecurityValidationResult(
                    is_safe=False,
                    risk_level="medium",
                    issues=[f"Too many query parameters: {len(params)}"],
                    blocked_reason=(
                        f"Maximum {self.MAX_QUERY_PARAMS} query parameters allowed"
                    ),
                )

            # Check for suspicious parameter values
            for key, values in params.items():
                for value in values:
                    if "\x00" in value:  # Null bytes
                        return SecurityValidationResult(
                            is_safe=False,
                            risk_level="high",
                            issues=["Null byte in query parameter"],
                            blocked_reason="Query parameters contain null bytes",
                        )

                    if len(value) > 2048:  # Very long parameter value
                        issues.append(f"Very long query parameter value: {key}")
                        risk_level = "low"

        except Exception as e:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="medium",
                issues=[f"Query parsing error: {str(e)}"],
                blocked_reason="Malformed query parameters",
            )

        return SecurityValidationResult(
            is_safe=True, risk_level=risk_level, issues=issues
        )

    def _check_suspicious_patterns(self, url: str) -> SecurityValidationResult:
        """Check URL against suspicious patterns"""
        issues = []
        risk_level = "safe"

        for pattern in self.compiled_patterns:
            if pattern.search(url):
                issues.append(f"Suspicious pattern detected: {pattern.pattern}")
                risk_level = "medium"

        # If multiple suspicious patterns detected, increase risk
        if len(issues) > 2:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="high",
                issues=issues,
                blocked_reason="Multiple suspicious patterns detected",
            )

        return SecurityValidationResult(
            is_safe=True, risk_level=risk_level, issues=issues
        )

    def _validate_dns_resolution(self, hostname: str) -> SecurityValidationResult:
        """Validate DNS resolution to prevent SSRF via DNS rebinding"""
        try:
            # Resolve hostname to IP addresses
            ip_addresses = socket.getaddrinfo(hostname, None)

            for family, type_, proto, canonname, sockaddr in ip_addresses:
                ip_str = sockaddr[0]

                try:
                    ip = ipaddress.ip_address(ip_str)

                    # Check if resolved IP is in private ranges
                    for private_range in self.PRIVATE_IP_RANGES:
                        if ip in private_range:
                            return SecurityValidationResult(
                                is_safe=False,
                                risk_level="critical",
                                issues=[f"Hostname resolves to private IP: {ip_str}"],
                                blocked_reason=(
                                    f'Hostname "{hostname}" resolves to '
                                    f"private/internal IP"
                                ),
                            )

                except ValueError:
                    # Invalid IP format, skip
                    continue

        except socket.gaierror:
            # DNS resolution failed - this is actually safer for SSRF prevention
            return SecurityValidationResult(
                is_safe=True,
                risk_level="low",
                issues=["DNS resolution failed (safer for SSRF prevention)"],
            )
        except Exception as e:
            return SecurityValidationResult(
                is_safe=False,
                risk_level="medium",
                issues=[f"DNS validation error: {str(e)}"],
                blocked_reason="DNS validation failed",
            )

        return SecurityValidationResult(is_safe=True, risk_level="safe", issues=[])

    def _is_valid_hostname_format(self, hostname: str) -> bool:
        """Check if hostname has valid format"""
        if not hostname or len(hostname) > 253:
            return False

        # Allow common domain patterns including example.com
        # More permissive pattern that allows standard domain names
        hostname_pattern = re.compile(
            r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
            r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
        )

        return bool(hostname_pattern.match(hostname))

    def _risk_priority(self, risk_level: str) -> int:
        """Get numeric priority for risk level comparison"""
        priorities = {"safe": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        return priorities.get(risk_level, 0)

    def sanitize_url(self, url: str) -> str:
        """
        Sanitize URL by removing potentially dangerous components.

        Args:
            url: URL to sanitize

        Returns:
            Sanitized URL string
        """
        if not url:
            return ""

        try:
            parsed = urlparse(url.strip())

            # Force HTTPS if HTTP
            scheme = "https" if parsed.scheme in ["http", "https"] else "https"

            # Clean hostname
            hostname = parsed.hostname.lower() if parsed.hostname else ""

            # Clean path (remove potential traversal)
            path = parsed.path.replace("../", "").replace("..\\", "")

            # Reconstruct URL with safe components
            port_part = f":{parsed.port}" if parsed.port else ""

            sanitized = f"{scheme}://{hostname}{port_part}{path}"

            # Add query if it exists and is safe
            if parsed.query:
                sanitized += f"?{parsed.query}"

            return sanitized

        except Exception as e:
            logger.warning(f"Failed to sanitize URL: {e}")
            return ""

    def is_safe_for_processing(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Quick check if URL is safe for processing.

        Args:
            url: URL to check

        Returns:
            Tuple of (is_safe, blocking_reason)
        """
        result = self.validate_url_security(url)
        return result.is_safe, result.blocked_reason
