"""
Intelligent Rate Limiter Module

Provides domain-aware rate limiting with special handling for major websites.
Implements smart delays and concurrent request management.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Set
from urllib.parse import urlparse


@dataclass
class DomainStats:
    """Statistics for domain access patterns"""

    last_request_time: float = 0.0
    request_count: int = 0
    total_wait_time: float = 0.0
    error_count: int = 0


class IntelligentRateLimiter:
    """Domain-aware rate limiting with special handling for major sites"""

    # Special delays for major sites (in seconds)
    MAJOR_SITE_DELAYS = {
        "google.com": 2.0,
        "www.google.com": 2.0,
        "github.com": 1.5,
        "stackoverflow.com": 1.0,
        "youtube.com": 2.0,
        "www.youtube.com": 2.0,
        "facebook.com": 3.0,
        "www.facebook.com": 3.0,
        "linkedin.com": 2.0,
        "www.linkedin.com": 2.0,
        "twitter.com": 1.5,
        "x.com": 1.5,
        "reddit.com": 1.0,
        "www.reddit.com": 1.0,
        "medium.com": 1.0,
        "amazon.com": 1.5,
        "www.amazon.com": 1.5,
        "wikipedia.org": 0.5,
        "news.ycombinator.com": 1.0,
        "ycombinator.com": 1.0,
    }

    def __init__(self, default_delay: float = 0.5, max_concurrent: int = 10):
        """
        Initialize the rate limiter.

        Args:
            default_delay: Default delay between requests to the same domain
            max_concurrent: Maximum number of concurrent active domains
        """
        self.default_delay = default_delay
        self.max_concurrent = max_concurrent
        self.domain_stats: Dict[str, DomainStats] = defaultdict(DomainStats)
        self.active_domains: Set[str] = set()
        self.lock = threading.RLock()

        # Load additional delays from configuration if needed
        self.domain_delays = self.MAJOR_SITE_DELAYS.copy()

        logging.info(
            f"Initialized rate limiter with "
            f"{len(self.MAJOR_SITE_DELAYS)} special domain rules"
        )

    def wait_if_needed(self, url: str) -> float:
        """
        Apply intelligent rate limiting based on domain.

        Args:
            url: The URL to check

        Returns:
            float: Actual wait time applied
        """
        domain = self._extract_domain(url)

        with self.lock:
            stats = self.domain_stats[domain]
            delay = self._get_domain_delay(domain)

            # Calculate required wait time
            current_time = time.time()
            time_since_last = current_time - stats.last_request_time
            wait_time = max(0, delay - time_since_last)

            # Apply wait if needed
            if wait_time > 0:
                logging.debug(f"Rate limiting {domain}: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                stats.total_wait_time += wait_time

            # Update statistics
            stats.last_request_time = time.time()
            stats.request_count += 1
            self.active_domains.add(domain)

            return wait_time

    def record_error(self, url: str) -> None:
        """Record an error for domain statistics"""
        domain = self._extract_domain(url)
        with self.lock:
            self.domain_stats[domain].error_count += 1

    def record_success(self, url: str) -> None:
        """Record a successful request completion"""
        domain = self._extract_domain(url)
        with self.lock:
            # Remove from active domains
            self.active_domains.discard(domain)

    def get_domain_stats(self, domain: str) -> DomainStats:
        """Get statistics for a specific domain"""
        with self.lock:
            return self.domain_stats[domain]

    def get_all_stats(self) -> Dict[str, DomainStats]:
        """Get statistics for all domains"""
        with self.lock:
            return dict(self.domain_stats)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove port numbers
            if ":" in domain:
                domain = domain.split(":")[0]

            return domain
        except Exception:
            # Fallback for malformed URLs
            return "unknown"

    def _get_domain_delay(self, domain: str) -> float:
        """Get delay for specific domain"""
        # Check for exact match first
        if domain in self.domain_delays:
            return self.domain_delays[domain]

        # Check for parent domain matches
        for special_domain, delay in self.domain_delays.items():
            if domain.endswith("." + special_domain):
                return delay

        # Adaptive delay based on error rate
        stats = self.domain_stats[domain]
        if stats.request_count > 5:  # Only adjust after some requests
            error_rate = stats.error_count / stats.request_count
            if error_rate > 0.2:  # High error rate
                return self.default_delay * 2.0
            elif error_rate > 0.1:  # Moderate error rate
                return self.default_delay * 1.5

        return self.default_delay

    def is_at_capacity(self) -> bool:
        """Check if we're at maximum concurrent capacity"""
        with self.lock:
            return len(self.active_domains) >= self.max_concurrent

    def wait_for_capacity(self, timeout: float = 30.0) -> bool:
        """
        Wait for capacity to become available.

        Args:
            timeout: Maximum time to wait

        Returns:
            bool: True if capacity available, False if timeout
        """
        start_time = time.time()

        while self.is_at_capacity():
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)

        return True

    def reset_domain_stats(self, domain: Optional[str] = None) -> None:
        """Reset statistics for a domain or all domains"""
        with self.lock:
            if domain:
                if domain in self.domain_stats:
                    del self.domain_stats[domain]
                self.active_domains.discard(domain)
            else:
                self.domain_stats.clear()
                self.active_domains.clear()
                logging.info("Reset all domain statistics")
