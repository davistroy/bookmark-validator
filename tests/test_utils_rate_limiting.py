"""
Unit tests for rate limiting utilities.

Tests for intelligent rate limiter and basic rate limiter.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from bookmark_processor.utils.rate_limiter import RateLimiter
from bookmark_processor.utils.intelligent_rate_limiter import IntelligentRateLimiter


class TestRateLimiter:
    """Test basic rate limiter."""
    
    def test_init_default(self):
        """Test RateLimiter initialization with defaults."""
        limiter = RateLimiter()
        
        assert limiter.requests_per_second == 2.0
        assert limiter.last_request_time == 0
    
    def test_init_custom(self):
        """Test RateLimiter initialization with custom values."""
        limiter = RateLimiter(requests_per_second=5.0)
        
        assert limiter.requests_per_second == 5.0
        assert limiter.last_request_time == 0
    
    def test_wait_no_delay_needed(self):
        """Test wait when no delay is needed."""
        limiter = RateLimiter(requests_per_second=10.0)  # Very permissive
        
        # First request should not need delay
        with patch('time.sleep') as mock_sleep:
            limiter.wait()
            mock_sleep.assert_not_called()
    
    def test_wait_delay_needed(self):
        """Test wait when delay is needed."""
        limiter = RateLimiter(requests_per_second=1.0)  # 1 request per second
        
        # Make first request
        limiter.wait()
        
        # Second request immediately should need delay
        with patch('time.sleep') as mock_sleep:
            with patch('time.time', return_value=limiter.last_request_time + 0.5):  # 0.5 seconds later
                limiter.wait()
                mock_sleep.assert_called_once()
                # Should sleep for approximately 0.5 seconds (1.0 - 0.5)
                sleep_duration = mock_sleep.call_args[0][0]
                assert 0.4 <= sleep_duration <= 0.6
    
    def test_get_delay(self):
        """Test get_delay calculation."""
        limiter = RateLimiter(requests_per_second=2.0)  # 0.5 seconds between requests
        
        # No delay needed for first request
        delay = limiter.get_delay()
        assert delay == 0
        
        # Record a request
        limiter.wait()
        
        # Check delay needed immediately after
        with patch('time.time', return_value=limiter.last_request_time + 0.25):  # 0.25 seconds later
            delay = limiter.get_delay()
            assert 0.2 <= delay <= 0.3  # Should need about 0.25 seconds delay
    
    def test_update_rate(self):
        """Test updating the rate limit."""
        limiter = RateLimiter(requests_per_second=1.0)
        
        assert limiter.requests_per_second == 1.0
        
        limiter.update_rate(5.0)
        assert limiter.requests_per_second == 5.0
    
    def test_reset(self):
        """Test resetting the rate limiter."""
        limiter = RateLimiter()
        
        # Make a request
        limiter.wait()
        assert limiter.last_request_time > 0
        
        # Reset
        limiter.reset()
        assert limiter.last_request_time == 0


class TestIntelligentRateLimiter:
    """Test intelligent rate limiter."""
    
    def test_init_default(self):
        """Test IntelligentRateLimiter initialization with defaults."""
        limiter = IntelligentRateLimiter()
        
        assert limiter.default_delay == 0.5
        assert limiter.max_delay == 60.0
        assert limiter.adaptive_factor == 1.5
        assert len(limiter.domain_delays) > 0  # Should have some predefined delays
    
    def test_init_custom(self):
        """Test IntelligentRateLimiter initialization with custom values."""
        custom_delays = {"example.com": 2.0}
        
        limiter = IntelligentRateLimiter(
            default_delay=1.0,
            max_delay=30.0,
            adaptive_factor=2.0,
            domain_delays=custom_delays
        )
        
        assert limiter.default_delay == 1.0
        assert limiter.max_delay == 30.0
        assert limiter.adaptive_factor == 2.0
        assert limiter.domain_delays["example.com"] == 2.0
    
    def test_get_domain_from_url(self):
        """Test domain extraction from URLs."""
        limiter = IntelligentRateLimiter()
        
        test_cases = [
            ("https://example.com/path", "example.com"),
            ("http://subdomain.example.com", "subdomain.example.com"),
            ("https://github.com/user/repo", "github.com"),
            ("ftp://files.example.com", "files.example.com"),
            ("invalid-url", ""),
            ("", ""),
        ]
        
        for url, expected_domain in test_cases:
            domain = limiter._get_domain_from_url(url)
            assert domain == expected_domain
    
    def test_get_delay_default(self):
        """Test get_delay with default delay."""
        limiter = IntelligentRateLimiter(default_delay=1.0)
        
        # Unknown domain should use default delay
        delay = limiter.get_delay("https://unknown-domain.com")
        assert delay == 1.0
    
    def test_get_delay_predefined_domain(self):
        """Test get_delay with predefined domain delays."""
        custom_delays = {"github.com": 2.0}
        limiter = IntelligentRateLimiter(default_delay=0.5, domain_delays=custom_delays)
        
        # Known domain should use specific delay
        delay = limiter.get_delay("https://github.com/user/repo")
        assert delay == 2.0
        
        # Unknown domain should use default
        delay = limiter.get_delay("https://example.com")
        assert delay == 0.5
    
    def test_get_delay_adaptive_increase(self):
        """Test adaptive delay increase after errors."""
        limiter = IntelligentRateLimiter(default_delay=1.0, adaptive_factor=2.0)
        url = "https://example.com"
        
        # Initial delay
        initial_delay = limiter.get_delay(url)
        assert initial_delay == 1.0
        
        # Record error and check increased delay
        limiter.record_error(url, 429)  # Rate limited
        increased_delay = limiter.get_delay(url)
        assert increased_delay == 2.0  # Should be doubled
        
        # Record another error
        limiter.record_error(url, 503)  # Service unavailable
        further_increased = limiter.get_delay(url)
        assert further_increased == 4.0  # Should be doubled again
    
    def test_get_delay_max_limit(self):
        """Test that delay doesn't exceed maximum."""
        limiter = IntelligentRateLimiter(default_delay=1.0, max_delay=5.0, adaptive_factor=10.0)
        url = "https://example.com"
        
        # Record multiple errors to try to exceed max
        for _ in range(10):
            limiter.record_error(url, 429)
        
        delay = limiter.get_delay(url)
        assert delay <= 5.0  # Should not exceed max_delay
    
    def test_record_request_success(self):
        """Test recording successful requests."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"
        
        # Record successful request
        limiter.record_request(url, success=True)
        
        # Check that request was recorded
        domain = limiter._get_domain_from_url(url)
        assert domain in limiter.request_history
        assert len(limiter.request_history[domain]) == 1
        
        # Check success count
        assert limiter.success_counts.get(domain, 0) == 1
        assert limiter.error_counts.get(domain, 0) == 0
    
    def test_record_request_failure(self):
        """Test recording failed requests."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"
        
        # Record failed request
        limiter.record_request(url, success=False)
        
        # Check that request was recorded
        domain = limiter._get_domain_from_url(url)
        assert domain in limiter.request_history
        assert len(limiter.request_history[domain]) == 1
        
        # Check error count
        assert limiter.success_counts.get(domain, 0) == 0
        assert limiter.error_counts.get(domain, 0) == 1
    
    def test_record_error_rate_limit(self):
        """Test recording rate limit errors."""
        limiter = IntelligentRateLimiter(default_delay=1.0, adaptive_factor=2.0)
        url = "https://example.com"
        
        initial_delay = limiter.get_delay(url)
        
        # Record rate limit error
        limiter.record_error(url, 429)
        
        # Delay should increase significantly for rate limits
        new_delay = limiter.get_delay(url)
        assert new_delay > initial_delay
    
    def test_record_error_server_error(self):
        """Test recording server errors."""
        limiter = IntelligentRateLimiter(default_delay=1.0, adaptive_factor=2.0)
        url = "https://example.com"
        
        initial_delay = limiter.get_delay(url)
        
        # Record server error
        limiter.record_error(url, 503)
        
        # Delay should increase for server errors
        new_delay = limiter.get_delay(url)
        assert new_delay > initial_delay
    
    def test_record_error_client_error(self):
        """Test recording client errors (should not increase delay much)."""
        limiter = IntelligentRateLimiter(default_delay=1.0, adaptive_factor=2.0)
        url = "https://example.com"
        
        initial_delay = limiter.get_delay(url)
        
        # Record client error (not our fault)
        limiter.record_error(url, 404)
        
        # Delay might increase slightly but not as much as server errors
        new_delay = limiter.get_delay(url)
        # For client errors, delay increase should be minimal
        assert new_delay <= initial_delay * 1.5
    
    def test_should_retry_rate_limit(self):
        """Test retry decision for rate limit errors."""
        limiter = IntelligentRateLimiter()
        
        # Rate limit errors should generally allow retry
        assert limiter.should_retry("https://example.com", 429, attempt=1) is True
        assert limiter.should_retry("https://example.com", 429, attempt=3) is True
        
        # But not after too many attempts
        assert limiter.should_retry("https://example.com", 429, attempt=10) is False
    
    def test_should_retry_server_error(self):
        """Test retry decision for server errors."""
        limiter = IntelligentRateLimiter()
        
        # Server errors should allow some retries
        assert limiter.should_retry("https://example.com", 503, attempt=1) is True
        assert limiter.should_retry("https://example.com", 502, attempt=2) is True
        
        # But not after too many attempts
        assert limiter.should_retry("https://example.com", 503, attempt=8) is False
    
    def test_should_retry_client_error(self):
        """Test retry decision for client errors."""
        limiter = IntelligentRateLimiter()
        
        # Client errors should generally not retry
        assert limiter.should_retry("https://example.com", 404, attempt=1) is False
        assert limiter.should_retry("https://example.com", 403, attempt=1) is False
        assert limiter.should_retry("https://example.com", 400, attempt=1) is False
    
    def test_should_retry_success(self):
        """Test retry decision for successful responses."""
        limiter = IntelligentRateLimiter()
        
        # Successful responses should not retry
        assert limiter.should_retry("https://example.com", 200, attempt=1) is False
        assert limiter.should_retry("https://example.com", 201, attempt=1) is False
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"
        
        # No requests yet
        assert limiter.get_success_rate(url) == 0.0
        
        # Record some successful and failed requests
        for _ in range(7):
            limiter.record_request(url, success=True)
        for _ in range(3):
            limiter.record_request(url, success=False)
        
        # Should be 70% success rate
        success_rate = limiter.get_success_rate(url)
        assert success_rate == 0.7
    
    def test_get_request_count(self):
        """Test request count tracking."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"
        
        # No requests yet
        assert limiter.get_request_count(url) == 0
        
        # Record some requests
        for _ in range(5):
            limiter.record_request(url, success=True)
        
        assert limiter.get_request_count(url) == 5
    
    def test_reset_domain_stats(self):
        """Test resetting statistics for a domain."""
        limiter = IntelligentRateLimiter()
        url = "https://example.com"
        domain = limiter._get_domain_from_url(url)
        
        # Record some requests and errors
        limiter.record_request(url, success=True)
        limiter.record_error(url, 429)
        
        # Verify data exists
        assert limiter.get_request_count(url) > 0
        assert domain in limiter.success_counts
        assert domain in limiter.error_counts
        
        # Reset and verify clean state
        limiter.reset_domain_stats(url)
        
        assert limiter.get_request_count(url) == 0
        assert limiter.success_counts.get(domain, 0) == 0
        assert limiter.error_counts.get(domain, 0) == 0
    
    def test_get_statistics(self):
        """Test getting comprehensive statistics."""
        limiter = IntelligentRateLimiter()
        
        # Record various requests
        urls = [
            "https://example.com",
            "https://github.com/user/repo",
            "https://stackoverflow.com/questions"
        ]
        
        for url in urls:
            limiter.record_request(url, success=True)
            limiter.record_request(url, success=False)
        
        stats = limiter.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "total_domains" in stats
        assert "average_success_rate" in stats
        assert "domain_stats" in stats
        
        assert stats["total_requests"] == 6  # 2 requests per URL
        assert stats["total_domains"] == 3  # 3 different domains
    
    @pytest.mark.slow
    def test_wait_timing(self):
        """Test actual wait timing (slow test)."""
        limiter = IntelligentRateLimiter(default_delay=0.1)  # Short delay for testing
        url = "https://example.com"
        
        start_time = time.time()
        limiter.wait(url)
        first_duration = time.time() - start_time
        
        # First wait should be minimal
        assert first_duration < 0.05
        
        # Second wait should include delay
        start_time = time.time()
        limiter.wait(url)
        second_duration = time.time() - start_time
        
        # Should have waited approximately the delay time
        assert 0.05 <= second_duration <= 0.2


if __name__ == "__main__":
    pytest.main([__file__])