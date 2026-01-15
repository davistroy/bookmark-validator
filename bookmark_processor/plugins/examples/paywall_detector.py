"""
Paywall Detector Plugin

Detects paywalled content and adds metadata about paywall status
to bookmarks during validation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from ..base import PluginHook, ValidationResult, ValidatorPlugin


class PaywallDetectorPlugin(ValidatorPlugin):
    """
    Plugin that detects paywalled content.

    Identifies common paywall patterns in URLs and content,
    marking bookmarks that may require subscription access.
    """

    # Known paywall domains
    PAYWALL_DOMAINS: Set[str] = {
        "nytimes.com",
        "wsj.com",
        "washingtonpost.com",
        "ft.com",
        "economist.com",
        "bloomberg.com",
        "theathletic.com",
        "newyorker.com",
        "wired.com",
        "medium.com",
        "hbr.org",
        "thetimes.co.uk",
        "telegraph.co.uk",
        "theinformation.com",
        "businessinsider.com",
        "seekingalpha.com",
    }

    # Patterns that indicate paywall in content
    PAYWALL_CONTENT_PATTERNS: List[str] = [
        r"subscribe\s+to\s+(continue|read|access)",
        r"(sign|log)\s*(in|up)\s+to\s+(continue|read|access)",
        r"(this|full)\s+(article|story|content)\s+is\s+(for\s+)?subscribers?\s+only",
        r"become\s+a\s+(member|subscriber)",
        r"unlimited\s+(access|reading)",
        r"free\s+(trial|articles?)\s+(remaining|left)",
        r"you('ve|\s+have)\s+reached\s+(your|the)\s+(free\s+)?limit",
        r"paywall",
        r"premium\s+(content|article|access)",
        r"members(-|\s+)only",
        r"subscription\s+required",
    ]

    # URL patterns that may indicate non-paywalled content
    BYPASS_PATTERNS: List[str] = [
        r"/gift/",
        r"/free/",
        r"/open/",
        r"/public/",
        r"[?&]gift=",
        r"[?&]unlocked=",
    ]

    def __init__(self):
        super().__init__()
        self._compiled_patterns: List[re.Pattern] = []
        self._compiled_bypass: List[re.Pattern] = []
        self._custom_domains: Set[str] = set()
        self._detected_count: int = 0
        self._checked_count: int = 0

    @property
    def name(self) -> str:
        return "paywall-detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Detects paywalled content and marks bookmarks with paywall metadata"

    @property
    def author(self) -> str:
        return "Bookmark Processor Team"

    @property
    def provides(self) -> List[str]:
        return ["validation", "paywall_detection"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [
            PluginHook.PRE_VALIDATION,
            PluginHook.POST_VALIDATION,
            PluginHook.VALIDATION_FILTER,
        ]

    def on_load(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        super().on_load(config)

        # Compile regex patterns
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PAYWALL_CONTENT_PATTERNS
        ]
        self._compiled_bypass = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.BYPASS_PATTERNS
        ]

        # Add custom domains from config
        custom_domains = config.get("additional_domains", [])
        self._custom_domains = set(custom_domains)

        # Configuration options
        self._mark_as_invalid = config.get("mark_as_invalid", False)
        self._confidence_threshold = config.get("confidence_threshold", 0.7)

        self._logger.info(
            f"Paywall detector loaded with {len(self.PAYWALL_DOMAINS) + len(self._custom_domains)} "
            f"tracked domains"
        )

    def validate(
        self, url: str, content: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a URL for paywall indicators.

        Args:
            url: URL to check
            content: Optional page content for deeper analysis

        Returns:
            ValidationResult with paywall metadata
        """
        self._checked_count += 1

        # Check for bypass patterns first
        if self._has_bypass_pattern(url):
            return ValidationResult(
                is_valid=True,
                url=url,
                metadata={
                    "paywall_detected": False,
                    "has_bypass": True,
                },
                plugin_name=self.name,
                confidence=1.0,
            )

        # Check domain
        domain_match = self._check_domain(url)

        # Check content if available
        content_match = False
        content_confidence = 0.0
        matched_patterns: List[str] = []

        if content:
            content_match, content_confidence, matched_patterns = self._check_content(
                content
            )

        # Calculate overall confidence
        if domain_match and content_match:
            confidence = 0.95
        elif content_match:
            confidence = content_confidence
        elif domain_match:
            confidence = 0.7
        else:
            confidence = 0.0

        is_paywalled = confidence >= self._confidence_threshold

        if is_paywalled:
            self._detected_count += 1

        # Determine validity based on config
        is_valid = not (is_paywalled and self._mark_as_invalid)

        return ValidationResult(
            is_valid=is_valid,
            url=url,
            error_message="Paywall detected" if is_paywalled and not is_valid else None,
            error_type="paywall" if is_paywalled and not is_valid else None,
            metadata={
                "paywall_detected": is_paywalled,
                "paywall_confidence": confidence,
                "is_known_paywall_domain": domain_match,
                "content_indicators": matched_patterns[:3],  # Limit to top 3
            },
            plugin_name=self.name,
            confidence=confidence,
        )

    def should_validate(self, url: str) -> bool:
        """Check if this validator should process the URL."""
        # Validate all HTTP(S) URLs
        return url.startswith("http://") or url.startswith("https://")

    def get_priority(self) -> int:
        """Lower priority (runs after basic validation)."""
        return 200

    def _check_domain(self, url: str) -> bool:
        """Check if URL domain is a known paywall site."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check against known domains
            all_domains = self.PAYWALL_DOMAINS | self._custom_domains

            for known_domain in all_domains:
                if domain == known_domain or domain.endswith("." + known_domain):
                    return True

            return False

        except Exception:
            return False

    def _check_content(
        self, content: str
    ) -> tuple[bool, float, List[str]]:
        """
        Check content for paywall indicators.

        Returns:
            Tuple of (has_paywall, confidence, matched_patterns)
        """
        matched: List[str] = []

        for pattern in self._compiled_patterns:
            if pattern.search(content):
                matched.append(pattern.pattern)

        if not matched:
            return False, 0.0, []

        # Calculate confidence based on number of matches
        confidence = min(0.5 + (len(matched) * 0.15), 0.95)

        return True, confidence, matched

    def _has_bypass_pattern(self, url: str) -> bool:
        """Check if URL has a bypass pattern."""
        for pattern in self._compiled_bypass:
            if pattern.search(url):
                return True
        return False

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration."""
        errors = []

        if "additional_domains" in config:
            if not isinstance(config["additional_domains"], list):
                errors.append("additional_domains must be a list")

        if "confidence_threshold" in config:
            threshold = config["confidence_threshold"]
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                errors.append("confidence_threshold must be a number between 0 and 1")

        return errors

    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "checked_count": self._checked_count,
            "detected_count": self._detected_count,
            "detection_rate": (
                self._detected_count / self._checked_count
                if self._checked_count > 0
                else 0
            ),
            "tracked_domains": len(self.PAYWALL_DOMAINS) + len(self._custom_domains),
        }

    # Hook methods
    def on_pre_validation(self, url: str) -> Optional[str]:
        """Called before URL validation."""
        return url

    def on_post_validation(
        self, url: str, result: ValidationResult
    ) -> ValidationResult:
        """Called after URL validation."""
        return result

    def filter_validation(
        self, results: List[ValidationResult]
    ) -> List[ValidationResult]:
        """Filter validation results."""
        return results


__all__ = ["PaywallDetectorPlugin"]
