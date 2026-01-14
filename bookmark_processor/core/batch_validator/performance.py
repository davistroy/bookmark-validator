"""
Performance Optimization for Batch Processing

Provides auto-tuning, concurrency adaptation, and performance tracking
for batch operations.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Tuple
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..batch_types import BatchResult


class PerformanceOptimizationMixin:
    """
    Mixin providing performance optimization capabilities for batch processing.

    This mixin adds auto-tuning, concurrency adaptation, and performance
    tracking methods to the EnhancedBatchProcessor.
    """

    def _update_performance_metrics(self, batch_result: "BatchResult") -> None:
        """
        Update performance metrics based on batch result.

        Args:
            batch_result: Result from a completed batch
        """
        with self.lock:
            if batch_result.average_item_time > 0:
                self.performance_history.append(
                    (batch_result.items_processed, batch_result.average_item_time)
                )

                # Keep only recent history (last 20 batches)
                if len(self.performance_history) > 20:
                    self.performance_history = self.performance_history[-20:]

    def _auto_tune_batch_size(self) -> None:
        """Automatically tune batch size based on performance history."""
        if len(self.performance_history) < 3:
            return

        # Calculate average time per item for different batch sizes
        size_performance = {}
        for batch_size, avg_time in self.performance_history[-10:]:
            if batch_size not in size_performance:
                size_performance[batch_size] = []
            size_performance[batch_size].append(avg_time)

        # Find optimal batch size (lowest average time per item)
        best_size = self.current_batch_size
        best_time = float("inf")

        for size, times in size_performance.items():
            if len(times) >= 2:  # Need at least 2 samples
                avg_time = sum(times) / len(times)
                if avg_time < best_time:
                    best_time = avg_time
                    best_size = size

        # Adjust batch size within constraints
        new_size = max(
            self.config.min_batch_size, min(self.config.max_batch_size, best_size)
        )

        if new_size != self.current_batch_size:
            logging.info(
                f"Auto-tuning batch size from {self.current_batch_size} to {new_size}"
            )
            self.current_batch_size = new_size

    def _adapt_concurrency_limits(self) -> None:
        """Adapt concurrency limits based on performance metrics."""
        if len(self.performance_history) < 3:
            return

        # Calculate recent performance
        recent_performance = self.performance_history[-5:]
        avg_time = sum(time for _, time in recent_performance) / len(recent_performance)

        # Adjust concurrency based on performance
        if avg_time > 5.0:  # Slow performance
            new_limit = max(10, self.current_concurrency_limit - 10)
        elif avg_time < 1.0:  # Fast performance
            new_limit = min(100, self.current_concurrency_limit + 10)
        else:
            return  # No change needed

        if new_limit != self.current_concurrency_limit:
            logging.info(
                f"Adapting concurrency limit from "
                f"{self.current_concurrency_limit} to {new_limit}"
            )
            self.current_concurrency_limit = new_limit

            # Update semaphore if it exists
            if self.async_semaphore:
                # Note: Can't directly change semaphore limit,
                # will be updated on next initialization
                pass

    async def _apply_domain_rate_limiting(self, url: str) -> None:
        """
        Apply domain-specific rate limiting for async requests.

        Args:
            url: URL to apply rate limiting for
        """
        import asyncio

        if not self.config.rate_limit_respect:
            return

        try:
            domain = urlparse(url).netloc.lower()

            # Get or create domain semaphore
            if domain not in self.domain_semaphores:
                # Limit concurrent requests per domain
                from asyncio import Semaphore

                self.domain_semaphores[domain] = Semaphore(
                    2
                )  # Max 2 concurrent per domain

            # Apply rate limiting
            async with self.domain_semaphores[domain]:
                # Check if we need to wait based on last request time
                current_time = time.time()
                last_request = self.rate_limit_tracker.get(domain, 0)

                # Apply domain-specific delays (mirroring IntelligentRateLimiter logic)
                domain_delays = {
                    "google.com": 2.0,
                    "github.com": 1.5,
                    "youtube.com": 2.0,
                    "facebook.com": 3.0,
                    "linkedin.com": 2.0,
                    "twitter.com": 1.5,
                    "x.com": 1.5,
                    "reddit.com": 1.0,
                    "medium.com": 1.0,
                }

                required_delay = domain_delays.get(domain, 0.5)
                time_since_last = current_time - last_request

                if time_since_last < required_delay:
                    wait_time = required_delay - time_since_last
                    await asyncio.sleep(wait_time)

                # Update last request time
                self.rate_limit_tracker[domain] = time.time()

        except Exception as e:
            logging.debug(f"Rate limiting failed for {url}: {e}")
            # Don't fail the whole request for rate limiting issues
            await asyncio.sleep(0.1)  # Minimal fallback delay


__all__ = ["PerformanceOptimizationMixin"]
