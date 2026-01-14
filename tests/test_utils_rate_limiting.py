"""
Unit tests for rate limiting utilities.

Tests for intelligent rate limiter and basic rate limiter.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from bookmark_processor.utils.intelligent_rate_limiter import (
    DomainStats,
    IntelligentRateLimiter,
)
from bookmark_processor.utils.rate_limiter import RateLimiter, ServiceRateLimiters


class TestRateLimiter:
    """Test basic rate limiter."""

    def test_init_default(self):
        """Test RateLimiter initialization with defaults."""
        limiter = RateLimiter(requests_per_minute=60)

        assert limiter.requests_per_minute == 60
        assert limiter.min_interval == 1.0
        assert limiter.last_request_time == 0

    def test_init_custom(self):
        """Test RateLimiter initialization with custom values."""
        limiter = RateLimiter(requests_per_minute=120, burst_size=20, name="TestLimiter")

        assert limiter.requests_per_minute == 120
        assert limiter.burst_size == 20
        assert limiter.name == "TestLimiter"
        assert limiter.min_interval == 0.5  # 60/120

    @pytest.mark.asyncio
    async def test_acquire_no_delay_needed(self):
        """Test acquire when no delay is needed."""
        limiter = RateLimiter(requests_per_minute=600)  # Very permissive

        # First request should not need delay
        result = await limiter.acquire()
        assert result is True
        assert limiter.total_requests == 1

    @pytest.mark.asyncio
    async def test_acquire_with_delay(self):
        """Test acquire when delay is needed."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 request per second

        # Make first request
        await limiter.acquire()
        first_time = limiter.last_request_time

        # Second request should need delay
        await limiter.acquire()
        second_time = limiter.last_request_time

        # Time difference should be approximately the min_interval
        time_diff = second_time - first_time
        assert time_diff >= limiter.min_interval * 0.9  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquire with timeout."""
        limiter = RateLimiter(requests_per_minute=1)  # Very restrictive

        # Make first request
        await limiter.acquire()

        # Second request with very short timeout should fail
        result = await limiter.acquire(timeout=0.1)
        assert result is False
        assert limiter.requests_denied == 1

    def test_get_status(self):
        """Test get_status returns correct information."""
        limiter = RateLimiter(requests_per_minute=60, name="TestStatus")

        status = limiter.get_status()

        assert status["name"] == "TestStatus"
        assert status["requests_per_minute"] == 60
        assert status["total_requests"] == 0
        assert "requests_in_window" in status
        assert "utilization_percent" in status

    def test_reset(self):
        """Test resetting the rate limiter."""
        limiter = RateLimiter(requests_per_minute=60)

        # Simulate some activity
        limiter.request_times.append(time.time())
        limiter.last_request_time = time.time()
        limiter.total_requests = 5

        # Reset
        limiter.reset()
        assert len(limiter.request_times) == 0
        assert limiter.last_request_time == 0.0
        assert limiter.total_requests == 0


class TestIntelligentRateLimiter:
    """Test intelligent rate limiter."""

    def test_init_default(self):
        """Test IntelligentRateLimiter initialization with defaults."""
        limiter = IntelligentRateLimiter()

        assert limiter.default_delay == 0.5
        assert limiter.max_concurrent == 10
        assert len(limiter.domain_delays) > 0  # Should have predefined delays
        assert isinstance(limiter.domain_stats, dict)

    def test_init_custom(self):
        """Test IntelligentRateLimiter initialization with custom values."""
        limiter = IntelligentRateLimiter(default_delay=1.0, max_concurrent=5)

        assert limiter.default_delay == 1.0
        assert limiter.max_concurrent == 5

    def test_extract_domain(self):
        """Test domain extraction from URLs."""
        limiter = IntelligentRateLimiter()

        test_cases = [
            ("https://example.com/path", "example.com"),
            ("http://subdomain.example.com", "subdomain.example.com"),
            ("https://github.com/user/repo", "github.com"),
            ("https://www.google.com:8080/search", "www.google.com"),
            ("invalid-url", ""),  # No netloc, returns empty string
            ("", ""),  # Empty URL, returns empty string
        ]

        for url, expected_domain in test_cases:
            domain = limiter._extract_domain(url)
            assert domain == expected_domain, f"Failed for {url}"

    def test_get_domain_delay_default(self):
        """Test get_domain_delay with default delay."""
        limiter = IntelligentRateLimiter(default_delay=1.0)

        # Unknown domain should use default delay
        delay = limiter._get_domain_delay("unknown-domain.com")
        assert delay == 1.0

    def test_get_domain_delay_special_site(self):
        """Test get_domain_delay with special site."""
        limiter = IntelligentRateLimiter(default_delay=0.5)

        # GitHub should have special delay
        delay = limiter._get_domain_delay("github.com")
        assert delay == 1.5

        # Google should have special delay
        delay = limiter._get_domain_delay("www.google.com")
        assert delay == 2.0

    def test_get_domain_delay_adaptive(self):
        """Test adaptive delay based on error rate."""
        limiter = IntelligentRateLimiter(default_delay=1.0)
        domain = "example.com"

        # Record some requests and errors to trigger adaptive behavior
        stats = limiter.domain_stats[domain]
        stats.request_count = 10
        stats.error_count = 3  # 30% error rate > 20%

        delay = limiter._get_domain_delay(domain)
        assert delay == 2.0  # Should be doubled

        # Moderate error rate (>10% but <=20%)
        stats.error_count = 2  # 20% error rate (not >20%, but >10%)
        delay = limiter._get_domain_delay(domain)
        assert delay == 1.5  # Should be 1.5x

        # Low error rate (<=10%)
        stats.error_count = 1  # 10% error rate (not >10%)
        delay = limiter._get_domain_delay(domain)
        assert delay == 1.0  # Should be default

    def test_wait_if_needed_first_request(self):
        """Test wait_if_needed for first request to domain."""
        limiter = IntelligentRateLimiter(default_delay=1.0)
        url = "https://example.com/page"

        # First request should have minimal wait
        wait_time = limiter.wait_if_needed(url)
        assert wait_time == 0.0

        # Check stats were updated
        stats = limiter.domain_stats["example.com"]
        assert stats.request_count == 1
        assert stats.last_request_time > 0

    def test_wait_if_needed_with_delay(self):
        """Test wait_if_needed applies delay correctly."""
        limiter = IntelligentRateLimiter(default_delay=0.1)  # Short delay for testing
        url = "https://example.com/page"

        # First request
        limiter.wait_if_needed(url)

        # Immediate second request should require wait
        start_time = time.time()
        wait_time = limiter.wait_if_needed(url)
        actual_time = time.time() - start_time

        # Should have waited approximately the delay time
        assert wait_time > 0
        assert actual_time >= 0.05  # At least half the delay

    def test_record_error(self):
        """Test recording errors."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"

        # Record error
        limiter.record_error(url)

        # Check error was recorded
        stats = limiter.domain_stats["example.com"]
        assert stats.error_count == 1

    def test_record_success(self):
        """Test recording successful requests."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"

        # Add to active domains first
        limiter.active_domains.add("example.com")
        assert "example.com" in limiter.active_domains

        # Record success
        limiter.record_success(url)

        # Should be removed from active domains
        assert "example.com" not in limiter.active_domains

    def test_get_domain_stats(self):
        """Test getting domain statistics."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"

        # Make some requests
        limiter.wait_if_needed(url)
        limiter.record_error(url)

        # Get stats
        stats = limiter.get_domain_stats("example.com")
        assert isinstance(stats, DomainStats)
        assert stats.request_count == 1
        assert stats.error_count == 1

    def test_get_all_stats(self):
        """Test getting all domain statistics."""
        limiter = IntelligentRateLimiter()

        # Make requests to different domains
        urls = [
            "https://example.com",
            "https://github.com/user/repo",
            "https://stackoverflow.com/questions",
        ]

        for url in urls:
            limiter.wait_if_needed(url)

        # Get all stats
        all_stats = limiter.get_all_stats()
        assert isinstance(all_stats, dict)
        assert len(all_stats) == 3
        assert "example.com" in all_stats
        assert "github.com" in all_stats
        assert "stackoverflow.com" in all_stats

    def test_is_at_capacity(self):
        """Test capacity checking."""
        limiter = IntelligentRateLimiter(max_concurrent=2)

        # No active domains initially
        assert limiter.is_at_capacity() is False

        # Add domains to reach capacity
        limiter.active_domains.add("domain1.com")
        limiter.active_domains.add("domain2.com")

        assert limiter.is_at_capacity() is True

    def test_wait_for_capacity(self):
        """Test waiting for capacity."""
        limiter = IntelligentRateLimiter(max_concurrent=1)

        # Fill capacity
        limiter.active_domains.add("domain1.com")
        assert limiter.is_at_capacity() is True

        # Try to wait with very short timeout
        result = limiter.wait_for_capacity(timeout=0.1)
        assert result is False  # Should timeout

        # Clear capacity
        limiter.active_domains.clear()
        result = limiter.wait_for_capacity(timeout=0.1)
        assert result is True  # Should succeed immediately

    def test_reset_domain_stats(self):
        """Test resetting statistics for a domain."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"

        # Make some requests
        limiter.wait_if_needed(url)
        limiter.record_error(url)
        limiter.active_domains.add("example.com")

        # Verify data exists
        assert "example.com" in limiter.domain_stats
        assert "example.com" in limiter.active_domains

        # Reset specific domain
        limiter.reset_domain_stats("example.com")

        # Should be cleaned up
        assert "example.com" not in limiter.domain_stats
        assert "example.com" not in limiter.active_domains

    def test_reset_all_domain_stats(self):
        """Test resetting all statistics."""
        limiter = IntelligentRateLimiter()

        # Make requests to multiple domains
        urls = ["https://example.com", "https://github.com"]
        for url in urls:
            limiter.wait_if_needed(url)
            limiter.active_domains.add(limiter._extract_domain(url))

        # Verify data exists
        assert len(limiter.domain_stats) > 0
        assert len(limiter.active_domains) > 0

        # Reset all
        limiter.reset_domain_stats()

        # Everything should be cleared
        assert len(limiter.domain_stats) == 0
        assert len(limiter.active_domains) == 0

    @pytest.mark.slow
    def test_wait_timing_accuracy(self):
        """Test actual wait timing accuracy (slow test)."""
        limiter = IntelligentRateLimiter(default_delay=0.1)
        url = "https://example.com"

        # First request - should be fast
        start_time = time.time()
        limiter.wait_if_needed(url)
        first_duration = time.time() - start_time
        assert first_duration < 0.05

        # Second request - should wait
        start_time = time.time()
        limiter.wait_if_needed(url)
        second_duration = time.time() - start_time

        # Should have waited approximately the delay time
        assert 0.05 <= second_duration <= 0.2


class TestServiceRateLimiters:
    """Test service rate limiters container."""

    def test_init_creates_default_limiters(self):
        """Test initialization creates default limiters."""
        limiters = ServiceRateLimiters()

        assert "claude" in limiters.limiters
        assert "openai" in limiters.limiters
        assert "local" in limiters.limiters

    def test_get_limiter(self):
        """Test getting a limiter by service name."""
        limiters = ServiceRateLimiters()

        claude_limiter = limiters.get_limiter("claude")
        assert isinstance(claude_limiter, RateLimiter)
        assert claude_limiter.name == "Claude"

    def test_get_limiter_invalid_service(self):
        """Test getting limiter for invalid service raises error."""
        limiters = ServiceRateLimiters()

        with pytest.raises(ValueError, match="Unsupported service"):
            limiters.get_limiter("invalid_service")

    def test_add_custom_limiter(self):
        """Test adding a custom limiter."""
        limiters = ServiceRateLimiters()

        limiters.add_custom_limiter("custom", requests_per_minute=100, burst_size=10)

        custom_limiter = limiters.get_limiter("custom")
        assert custom_limiter.requests_per_minute == 100
        assert custom_limiter.burst_size == 10

    def test_get_all_status(self):
        """Test getting status for all limiters."""
        limiters = ServiceRateLimiters()

        all_status = limiters.get_all_status()

        assert isinstance(all_status, dict)
        assert "claude" in all_status
        assert "openai" in all_status
        assert "local" in all_status
        assert all(isinstance(status, dict) for status in all_status.values())

    def test_reset_all(self):
        """Test resetting all limiters."""
        limiters = ServiceRateLimiters()

        # Modify some state
        for limiter in limiters.limiters.values():
            limiter.total_requests = 10

        # Reset all
        limiters.reset_all()

        # Verify all are reset
        for limiter in limiters.limiters.values():
            assert limiter.total_requests == 0


if __name__ == "__main__":
    pytest.main([__file__])
