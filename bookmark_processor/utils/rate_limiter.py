"""
Rate Limiter Module for API Request Management

This module provides rate limiting functionality to ensure API requests
stay within allowed limits and implement proper backoff strategies.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Dict, Optional


class RateLimiter:
    """
    Rate limiter to control API request frequency.

    Uses a sliding window approach to track requests and enforce limits.
    """

    def __init__(
        self,
        requests_per_minute: int,
        burst_size: Optional[int] = None,
        name: str = "RateLimiter",
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
            burst_size: Maximum burst requests (defaults to requests_per_minute)
            name: Name for logging purposes
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.name = name

        # Time interval between requests in seconds
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0

        # Track request timestamps (sliding window)
        self.request_times: deque = deque(maxlen=self.burst_size)
        self.last_request_time = 0.0

        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.requests_denied = 0

        # Logging
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Thread safety
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait for permission (None = no timeout)

        Returns:
            True if permission granted, False if timeout exceeded
        """
        async with self._lock:
            start_time = time.time()

            while True:
                current_time = time.time()

                # Remove old requests from the sliding window
                self._cleanup_old_requests(current_time)

                # Check if we can make a request
                can_proceed, wait_time = self._can_make_request(current_time)

                if can_proceed:
                    # Record the request
                    self.request_times.append(current_time)
                    self.last_request_time = current_time
                    self.total_requests += 1
                    return True

                # Check timeout
                if timeout is not None:
                    elapsed = current_time - start_time
                    if elapsed + wait_time > timeout:
                        self.requests_denied += 1
                        self.logger.warning(
                            f"Rate limit timeout exceeded for {self.name} "
                            f"(waited {elapsed:.2f}s, need {wait_time:.2f}s more)"
                        )
                        return False

                # Wait before checking again
                self.logger.debug(
                    f"Rate limit hit for {self.name}, waiting {wait_time:.2f}s"
                )
                self.total_wait_time += wait_time
                await asyncio.sleep(wait_time)

    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove requests older than 1 minute from the sliding window."""
        cutoff_time = current_time - 60.0
        while self.request_times and self.request_times[0] <= cutoff_time:
            self.request_times.popleft()

    def _can_make_request(self, current_time: float) -> tuple[bool, float]:
        """
        Check if a request can be made now.

        Args:
            current_time: Current timestamp

        Returns:
            Tuple of (can_proceed, wait_time_if_not)
        """
        # Check minimum interval since last request
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            return False, wait_time

        # Check if we're within the sliding window limit
        if len(self.request_times) < self.requests_per_minute:
            return True, 0.0

        # We're at the limit, need to wait for the oldest request to expire
        oldest_request = self.request_times[0]
        wait_time = max(0.0, oldest_request + 60.0 - current_time)

        if wait_time == 0.0:
            return True, 0.0
        else:
            return False, wait_time

    def get_status(self) -> Dict[str, any]:
        """
        Get current rate limiter status.

        Returns:
            Dictionary with status information
        """
        current_time = time.time()
        self._cleanup_old_requests(current_time)

        requests_in_window = len(self.request_times)
        utilization = (
            (requests_in_window / self.requests_per_minute)
            if self.requests_per_minute > 0
            else 0
        )

        # Calculate time until next request can be made
        time_since_last = current_time - self.last_request_time
        time_until_next = max(0.0, self.min_interval - time_since_last)

        # Calculate time until window resets
        if self.request_times:
            oldest_request = self.request_times[0]
            time_until_reset = max(0.0, oldest_request + 60.0 - current_time)
        else:
            time_until_reset = 0.0

        return {
            "name": self.name,
            "requests_in_window": requests_in_window,
            "requests_per_minute": self.requests_per_minute,
            "utilization_percent": utilization * 100,
            "time_until_next_request": time_until_next,
            "time_until_window_reset": time_until_reset,
            "total_requests": self.total_requests,
            "total_wait_time": self.total_wait_time,
            "requests_denied": self.requests_denied,
            "can_make_request_now": self._can_make_request(current_time)[0],
        }

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self.request_times.clear()
        self.last_request_time = 0.0
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.requests_denied = 0
        self.logger.info(f"Rate limiter {self.name} reset")


class ServiceRateLimiters:
    """
    Container for managing multiple service-specific rate limiters.
    """

    def __init__(self):
        """Initialize with default rate limiters for supported services."""
        self.limiters: Dict[str, RateLimiter] = {}
        self.logger = logging.getLogger(__name__)

        # Initialize default rate limiters
        self._setup_default_limiters()

    def _setup_default_limiters(self) -> None:
        """Set up default rate limiters for supported AI services."""
        # Claude API - 50 requests per minute
        self.limiters["claude"] = RateLimiter(
            requests_per_minute=50, burst_size=10, name="Claude"  # Allow small bursts
        )

        # OpenAI API - 60 requests per minute for GPT-3.5-turbo
        self.limiters["openai"] = RateLimiter(
            requests_per_minute=60,
            burst_size=20,  # Allow larger bursts for OpenAI
            name="OpenAI",
        )

        # Local AI - no rate limiting
        self.limiters["local"] = RateLimiter(
            requests_per_minute=10000, name="Local"  # Effectively unlimited
        )

    def get_limiter(self, service: str) -> RateLimiter:
        """
        Get rate limiter for a service.

        Args:
            service: Service name (claude, openai, local)

        Returns:
            RateLimiter instance

        Raises:
            ValueError: If service is not supported
        """
        if service not in self.limiters:
            raise ValueError(
                f"Unsupported service: {service}. Available: {list(self.limiters.keys())}"
            )

        return self.limiters[service]

    def add_custom_limiter(
        self, service: str, requests_per_minute: int, burst_size: Optional[int] = None
    ) -> None:
        """
        Add a custom rate limiter for a service.

        Args:
            service: Service name
            requests_per_minute: Rate limit
            burst_size: Optional burst size
        """
        self.limiters[service] = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            name=service.title(),
        )
        self.logger.info(
            f"Added custom rate limiter for {service}: {requests_per_minute} req/min"
        )

    def get_all_status(self) -> Dict[str, Dict[str, any]]:
        """
        Get status for all rate limiters.

        Returns:
            Dictionary mapping service names to their status
        """
        return {
            service: limiter.get_status() for service, limiter in self.limiters.items()
        }

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self.limiters.values():
            limiter.reset()
        self.logger.info("All rate limiters reset")


# Global instance for easy access
_global_rate_limiters = ServiceRateLimiters()


def get_rate_limiter(service: str) -> RateLimiter:
    """
    Get rate limiter for a service (convenience function).

    Args:
        service: Service name

    Returns:
        RateLimiter instance
    """
    return _global_rate_limiters.get_limiter(service)


def get_all_rate_limiter_status() -> Dict[str, Dict[str, any]]:
    """
    Get status for all rate limiters (convenience function).

    Returns:
        Dictionary mapping service names to their status
    """
    return _global_rate_limiters.get_all_status()


async def with_rate_limit(service: str, timeout: Optional[float] = None):
    """
    Context manager for rate-limited operations.

    Usage:
        async with with_rate_limit("claude"):
            # Make API request
            response = await api_call()

    Args:
        service: Service name
        timeout: Maximum time to wait for rate limit
    """

    class RateLimitContext:
        def __init__(self, limiter: RateLimiter, timeout: Optional[float]):
            self.limiter = limiter
            self.timeout = timeout
            self.acquired = False

        async def __aenter__(self):
            self.acquired = await self.limiter.acquire(self.timeout)
            if not self.acquired:
                raise asyncio.TimeoutError(f"Rate limit timeout for {service}")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Nothing to clean up for rate limiter
            pass

    limiter = get_rate_limiter(service)
    return RateLimitContext(limiter, timeout)
