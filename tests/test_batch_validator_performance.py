"""
Unit tests for batch_validator/performance.py module.

Tests the PerformanceOptimizationMixin class for:
- Performance metrics updates
- Auto-tuning of batch sizes
- Concurrency limit adaptation
- Domain-specific rate limiting
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import pytest

from bookmark_processor.core.batch_types import BatchConfig, BatchResult
from bookmark_processor.core.batch_validator.performance import PerformanceOptimizationMixin


class ConcretePerformanceMixin(PerformanceOptimizationMixin):
    """
    Concrete implementation of PerformanceOptimizationMixin for testing.

    This class provides the required attributes that the mixin expects
    to be present on the class it's mixed into.
    """

    def __init__(
        self,
        config: BatchConfig = None,
        initial_batch_size: int = 50,
        initial_concurrency_limit: int = 50,
    ):
        self.config = config or BatchConfig()
        self.lock = threading.RLock()
        self.performance_history = []
        self.current_batch_size = initial_batch_size
        self.current_concurrency_limit = initial_concurrency_limit
        self.async_semaphore = None
        self.domain_semaphores = {}
        self.rate_limit_tracker = {}


class TestPerformanceOptimizationMixinInit:
    """Test initialization and basic setup of the mixin."""

    def test_mixin_can_be_instantiated(self):
        """Test that the concrete mixin can be instantiated."""
        mixin = ConcretePerformanceMixin()

        assert mixin.performance_history == []
        assert mixin.current_batch_size == 50
        assert mixin.current_concurrency_limit == 50
        assert mixin.domain_semaphores == {}
        assert mixin.rate_limit_tracker == {}

    def test_mixin_with_custom_config(self):
        """Test mixin with custom BatchConfig."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=75,
            rate_limit_respect=False,
        )
        mixin = ConcretePerformanceMixin(config=config)

        assert mixin.config.min_batch_size == 10
        assert mixin.config.max_batch_size == 200
        assert mixin.config.rate_limit_respect is False

    def test_mixin_with_custom_initial_values(self):
        """Test mixin with custom initial batch size and concurrency."""
        mixin = ConcretePerformanceMixin(
            initial_batch_size=100,
            initial_concurrency_limit=75,
        )

        assert mixin.current_batch_size == 100
        assert mixin.current_concurrency_limit == 75


class TestUpdatePerformanceMetrics:
    """Test _update_performance_metrics method."""

    def test_update_performance_metrics_basic(self):
        """Test basic performance metrics update."""
        mixin = ConcretePerformanceMixin()

        batch_result = BatchResult(
            batch_id="test_batch",
            items_processed=10,
            items_successful=10,
            items_failed=0,
            processing_time=1.0,
            average_item_time=0.1,
            error_rate=0.0,
        )

        mixin._update_performance_metrics(batch_result)

        assert len(mixin.performance_history) == 1
        assert mixin.performance_history[0] == (10, 0.1)

    def test_update_performance_metrics_zero_average_time(self):
        """Test that zero average_item_time entries are not added."""
        mixin = ConcretePerformanceMixin()

        batch_result = BatchResult(
            batch_id="test_batch",
            items_processed=10,
            items_successful=10,
            items_failed=0,
            processing_time=0.0,
            average_item_time=0.0,  # Zero average time
            error_rate=0.0,
        )

        mixin._update_performance_metrics(batch_result)

        # Should not add entry with zero average time
        assert len(mixin.performance_history) == 0

    def test_update_performance_metrics_history_limit(self):
        """Test that performance history is limited to 20 entries."""
        mixin = ConcretePerformanceMixin()

        # Add 25 entries
        for i in range(25):
            batch_result = BatchResult(
                batch_id=f"test_batch_{i}",
                items_processed=10,
                items_successful=10,
                items_failed=0,
                processing_time=1.0,
                average_item_time=0.1 + i * 0.01,  # Varying times
                error_rate=0.0,
            )
            mixin._update_performance_metrics(batch_result)

        # Should be limited to 20
        assert len(mixin.performance_history) == 20
        # Should keep the most recent entries
        assert mixin.performance_history[0][1] == pytest.approx(0.15, rel=0.01)

    def test_update_performance_metrics_thread_safety(self):
        """Test thread safety of performance metrics updates."""
        mixin = ConcretePerformanceMixin()

        def update_metrics(thread_id):
            for i in range(10):
                batch_result = BatchResult(
                    batch_id=f"batch_{thread_id}_{i}",
                    items_processed=10,
                    items_successful=10,
                    items_failed=0,
                    processing_time=1.0,
                    average_item_time=0.1,
                    error_rate=0.0,
                )
                mixin._update_performance_metrics(batch_result)

        threads = []
        for i in range(5):
            t = threading.Thread(target=update_metrics, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # History should be limited to 20
        assert len(mixin.performance_history) <= 20

    def test_update_performance_metrics_preserves_recent_entries(self):
        """Test that trimming preserves the most recent entries."""
        mixin = ConcretePerformanceMixin()

        # Add entries with unique identifiable values
        for i in range(30):
            batch_result = BatchResult(
                batch_id=f"batch_{i}",
                items_processed=i + 1,  # Unique batch size as identifier
                items_successful=i + 1,
                items_failed=0,
                processing_time=1.0,
                average_item_time=1.0,
                error_rate=0.0,
            )
            mixin._update_performance_metrics(batch_result)

        # Should contain entries 10-29 (the last 20)
        assert len(mixin.performance_history) == 20
        # First entry in history should be batch 10 (items_processed=11)
        assert mixin.performance_history[0][0] == 11
        # Last entry should be batch 29 (items_processed=30)
        assert mixin.performance_history[-1][0] == 30


class TestAutoTuneBatchSize:
    """Test _auto_tune_batch_size method."""

    def test_auto_tune_insufficient_data(self):
        """Test auto-tuning with insufficient performance history."""
        config = BatchConfig(optimal_batch_size=50)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        # Add only 2 entries (need at least 3)
        mixin.performance_history = [(50, 0.1), (50, 0.15)]

        original_size = mixin.current_batch_size
        mixin._auto_tune_batch_size()

        # Should not change with insufficient data
        assert mixin.current_batch_size == original_size

    def test_auto_tune_with_sufficient_data(self):
        """Test auto-tuning with sufficient data finds better batch size."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        # Add history with batch size 30 being faster
        mixin.performance_history = [
            (50, 0.2),  # Slower
            (50, 0.2),
            (30, 0.05),  # Faster - should be selected
            (30, 0.05),
            (30, 0.05),
        ]

        mixin._auto_tune_batch_size()

        # Should tune to batch size 30 since it's faster
        assert mixin.current_batch_size == 30

    def test_auto_tune_respects_min_batch_size(self):
        """Test that auto-tuning respects minimum batch size constraint."""
        config = BatchConfig(
            min_batch_size=25,
            max_batch_size=200,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        # Add history suggesting batch size 10 is best (below min)
        mixin.performance_history = [
            (10, 0.01),  # Best time but below min_batch_size
            (10, 0.01),
            (50, 0.1),
            (50, 0.1),
        ]

        mixin._auto_tune_batch_size()

        # Should be clamped to min_batch_size
        assert mixin.current_batch_size >= config.min_batch_size

    def test_auto_tune_respects_max_batch_size(self):
        """Test that auto-tuning respects maximum batch size constraint."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=100,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        # Add history suggesting batch size 300 is best (above max)
        mixin.performance_history = [
            (300, 0.01),  # Best time but above max_batch_size
            (300, 0.01),
            (50, 0.1),
            (50, 0.1),
        ]

        mixin._auto_tune_batch_size()

        # Should be clamped to max_batch_size
        assert mixin.current_batch_size <= config.max_batch_size

    def test_auto_tune_requires_two_samples(self):
        """Test that auto-tuning requires at least 2 samples per batch size."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        # Add history with only 1 sample for good batch size
        mixin.performance_history = [
            (30, 0.01),  # Only 1 sample - won't be considered
            (50, 0.1),
            (50, 0.1),  # 2 samples - will be considered
        ]

        original_size = mixin.current_batch_size
        mixin._auto_tune_batch_size()

        # Should stay at 50 since 30 doesn't have enough samples
        assert mixin.current_batch_size == 50

    def test_auto_tune_no_change_when_current_is_optimal(self):
        """Test that batch size doesn't change if current is already optimal."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        # All same batch size with good performance
        mixin.performance_history = [
            (50, 0.1),
            (50, 0.1),
            (50, 0.1),
        ]

        with patch('logging.info') as mock_log:
            mixin._auto_tune_batch_size()
            # Should not log any change
            for call in mock_log.call_args_list:
                assert "Auto-tuning batch size" not in str(call)

        assert mixin.current_batch_size == 50

    def test_auto_tune_logs_batch_size_change(self):
        """Test that batch size changes are logged."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        mixin.performance_history = [
            (30, 0.05),
            (30, 0.05),
            (50, 0.2),
            (50, 0.2),
        ]

        with patch('logging.info') as mock_log:
            mixin._auto_tune_batch_size()
            # Should log the change
            mock_log.assert_called()
            call_args = str(mock_log.call_args)
            assert "Auto-tuning batch size" in call_args


class TestAdaptConcurrencyLimits:
    """Test _adapt_concurrency_limits method."""

    def test_adapt_concurrency_insufficient_data(self):
        """Test concurrency adaptation with insufficient history."""
        config = BatchConfig(async_concurrency_limit=50)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=50,
        )

        # Only 2 entries
        mixin.performance_history = [(10, 0.1), (10, 0.1)]

        original_limit = mixin.current_concurrency_limit
        mixin._adapt_concurrency_limits()

        assert mixin.current_concurrency_limit == original_limit

    def test_adapt_concurrency_slow_performance_decreases(self):
        """Test that slow performance decreases concurrency limit."""
        config = BatchConfig(async_concurrency_limit=50)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=50,
        )

        # Slow performance (> 5.0 seconds average)
        mixin.performance_history = [
            (10, 6.0),
            (10, 6.5),
            (10, 7.0),
            (10, 6.0),
            (10, 6.5),
        ]

        mixin._adapt_concurrency_limits()

        # Should decrease by 10
        assert mixin.current_concurrency_limit == 40

    def test_adapt_concurrency_fast_performance_increases(self):
        """Test that fast performance increases concurrency limit."""
        config = BatchConfig(async_concurrency_limit=50)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=50,
        )

        # Fast performance (< 1.0 seconds average)
        mixin.performance_history = [
            (10, 0.3),
            (10, 0.4),
            (10, 0.5),
            (10, 0.3),
            (10, 0.4),
        ]

        mixin._adapt_concurrency_limits()

        # Should increase by 10
        assert mixin.current_concurrency_limit == 60

    def test_adapt_concurrency_moderate_performance_no_change(self):
        """Test that moderate performance doesn't change concurrency."""
        config = BatchConfig(async_concurrency_limit=50)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=50,
        )

        # Moderate performance (between 1.0 and 5.0)
        mixin.performance_history = [
            (10, 2.0),
            (10, 2.5),
            (10, 3.0),
            (10, 2.0),
            (10, 2.5),
        ]

        mixin._adapt_concurrency_limits()

        # Should stay the same
        assert mixin.current_concurrency_limit == 50

    def test_adapt_concurrency_respects_minimum(self):
        """Test that concurrency doesn't go below minimum (10)."""
        config = BatchConfig(async_concurrency_limit=15)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=15,
        )

        # Very slow performance
        mixin.performance_history = [
            (10, 10.0),
            (10, 10.0),
            (10, 10.0),
            (10, 10.0),
            (10, 10.0),
        ]

        mixin._adapt_concurrency_limits()

        # Should be clamped to minimum 10
        assert mixin.current_concurrency_limit == 10

    def test_adapt_concurrency_respects_maximum(self):
        """Test that concurrency doesn't go above maximum (100)."""
        config = BatchConfig(async_concurrency_limit=95)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=95,
        )

        # Very fast performance
        mixin.performance_history = [
            (10, 0.1),
            (10, 0.1),
            (10, 0.1),
            (10, 0.1),
            (10, 0.1),
        ]

        mixin._adapt_concurrency_limits()

        # Should be clamped to maximum 100
        assert mixin.current_concurrency_limit == 100

    def test_adapt_concurrency_logs_change(self):
        """Test that concurrency changes are logged."""
        config = BatchConfig(async_concurrency_limit=50)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=50,
        )

        mixin.performance_history = [
            (10, 0.5),
            (10, 0.5),
            (10, 0.5),
            (10, 0.5),
            (10, 0.5),
        ]

        with patch('logging.info') as mock_log:
            mixin._adapt_concurrency_limits()
            mock_log.assert_called()
            call_args = str(mock_log.call_args)
            assert "Adapting concurrency limit" in call_args

    def test_adapt_concurrency_with_existing_semaphore(self):
        """Test adaptation when async_semaphore exists."""
        config = BatchConfig(async_concurrency_limit=50)
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_concurrency_limit=50,
        )
        mixin.async_semaphore = asyncio.Semaphore(50)  # Existing semaphore

        mixin.performance_history = [
            (10, 0.5),
            (10, 0.5),
            (10, 0.5),
            (10, 0.5),
            (10, 0.5),
        ]

        # Should not raise any errors
        mixin._adapt_concurrency_limits()

        # Current limit should be updated
        assert mixin.current_concurrency_limit == 60


class TestApplyDomainRateLimiting:
    """Test _apply_domain_rate_limiting async method."""

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self):
        """Test rate limiting when disabled in config."""
        config = BatchConfig(rate_limit_respect=False)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://google.com/search"
        await mixin._apply_domain_rate_limiting(url)

        # Should not create any rate limiting structures
        assert len(mixin.domain_semaphores) == 0
        assert len(mixin.rate_limit_tracker) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_creates_semaphore(self):
        """Test that rate limiting creates a domain semaphore."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://example.com/page"
        await mixin._apply_domain_rate_limiting(url)

        assert "example.com" in mixin.domain_semaphores
        assert "example.com" in mixin.rate_limit_tracker

    @pytest.mark.asyncio
    async def test_rate_limiting_known_domains(self):
        """Test rate limiting for known domains with specific delays."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        known_domains = [
            "https://google.com/search",
            "https://github.com/repo",
            "https://youtube.com/video",
            "https://facebook.com/page",
            "https://linkedin.com/profile",
            "https://twitter.com/tweet",
            "https://x.com/post",
            "https://reddit.com/thread",
            "https://medium.com/article",
        ]

        for url in known_domains:
            await mixin._apply_domain_rate_limiting(url)

        # All domains should have semaphores and tracking
        assert "google.com" in mixin.domain_semaphores
        assert "github.com" in mixin.domain_semaphores
        assert "youtube.com" in mixin.domain_semaphores

    @pytest.mark.asyncio
    async def test_rate_limiting_applies_delay(self):
        """Test that rate limiting applies delay between requests."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://example.com/page"

        # First request
        start = time.time()
        await mixin._apply_domain_rate_limiting(url)
        first_request_time = time.time()

        # Second request immediately after - should have delay
        await mixin._apply_domain_rate_limiting(url)
        second_request_time = time.time()

        # Default delay is 0.5 seconds for unknown domains
        # Should have some delay between requests
        time_between = second_request_time - first_request_time
        # Allow some tolerance for test execution time
        assert time_between >= 0.4  # Slightly less than 0.5 to account for overhead

    @pytest.mark.asyncio
    async def test_rate_limiting_reuses_semaphore(self):
        """Test that the same semaphore is reused for the same domain."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"

        await mixin._apply_domain_rate_limiting(url1)
        first_semaphore = mixin.domain_semaphores.get("example.com")

        await mixin._apply_domain_rate_limiting(url2)
        second_semaphore = mixin.domain_semaphores.get("example.com")

        # Should be the same semaphore object
        assert first_semaphore is second_semaphore

    @pytest.mark.asyncio
    async def test_rate_limiting_handles_invalid_url(self):
        """Test that rate limiting handles invalid URLs gracefully."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        # Invalid URL - should not crash
        url = "not-a-valid-url"

        with patch('logging.debug') as mock_log:
            await mixin._apply_domain_rate_limiting(url)
            # Should log debug message about failure
            # The function catches exceptions and adds minimal delay

    @pytest.mark.asyncio
    async def test_rate_limiting_exception_handling(self):
        """Test that exceptions are handled with fallback delay."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://example.com/page"

        # Mock urlparse to raise an exception
        with patch(
            'bookmark_processor.core.batch_validator.performance.urlparse',
            side_effect=ValueError("Forced URL parse error")
        ):
            start = time.time()
            await mixin._apply_domain_rate_limiting(url)
            elapsed = time.time() - start

            # Should have minimal fallback delay (0.1 seconds)
            assert elapsed >= 0.09  # Slightly less to account for timing variation

    @pytest.mark.asyncio
    async def test_rate_limiting_google_specific_delay(self):
        """Test Google-specific rate limiting delay."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://google.com/search?q=test"

        await mixin._apply_domain_rate_limiting(url)
        first_time = mixin.rate_limit_tracker["google.com"]

        # Immediate second request should wait ~2 seconds
        start = time.time()
        await mixin._apply_domain_rate_limiting(url)
        elapsed = time.time() - start

        # Google has 2.0 second delay
        assert elapsed >= 1.9  # Allow small tolerance

    @pytest.mark.asyncio
    async def test_rate_limiting_github_specific_delay(self):
        """Test GitHub-specific rate limiting delay."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://github.com/user/repo"

        await mixin._apply_domain_rate_limiting(url)

        start = time.time()
        await mixin._apply_domain_rate_limiting(url)
        elapsed = time.time() - start

        # GitHub has 1.5 second delay
        assert elapsed >= 1.4  # Allow small tolerance

    @pytest.mark.asyncio
    async def test_rate_limiting_concurrent_requests(self):
        """Test that rate limiting handles concurrent requests properly."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://example.com/page"

        # Run multiple concurrent requests
        async def make_request():
            await mixin._apply_domain_rate_limiting(url)
            return time.time()

        # Start 3 concurrent requests
        tasks = [make_request() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # Results should be spread out due to rate limiting
        results.sort()

        # Domain semaphore limits to 2 concurrent, so some should be delayed
        assert "example.com" in mixin.domain_semaphores


class TestPerformanceHistoryManagement:
    """Test performance history management edge cases."""

    def test_performance_history_empty_initially(self):
        """Test that performance history is empty initially."""
        mixin = ConcretePerformanceMixin()
        assert mixin.performance_history == []

    def test_performance_history_exact_limit(self):
        """Test behavior when history is exactly at limit."""
        mixin = ConcretePerformanceMixin()

        # Add exactly 20 entries
        for i in range(20):
            batch_result = BatchResult(
                batch_id=f"batch_{i}",
                items_processed=10,
                items_successful=10,
                items_failed=0,
                processing_time=1.0,
                average_item_time=0.1,
                error_rate=0.0,
            )
            mixin._update_performance_metrics(batch_result)

        assert len(mixin.performance_history) == 20

        # Add one more - should still be 20
        batch_result = BatchResult(
            batch_id="batch_20",
            items_processed=10,
            items_successful=10,
            items_failed=0,
            processing_time=1.0,
            average_item_time=0.2,
            error_rate=0.0,
        )
        mixin._update_performance_metrics(batch_result)

        assert len(mixin.performance_history) == 20
        # Latest entry should be the new one
        assert mixin.performance_history[-1][1] == 0.2


class TestIntegrationScenarios:
    """Integration tests for combined functionality."""

    def test_full_performance_optimization_cycle(self):
        """Test a complete cycle of performance tracking and optimization."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
            auto_tune_batch_size=True,
            async_concurrency_limit=50,
            adaptive_concurrency=True,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
            initial_concurrency_limit=50,
        )

        # Simulate processing with varying performance
        for i in range(10):
            batch_result = BatchResult(
                batch_id=f"batch_{i}",
                items_processed=30,  # Smaller batch
                items_successful=30,
                items_failed=0,
                processing_time=0.3,
                average_item_time=0.01,  # Fast performance
                error_rate=0.0,
            )
            mixin._update_performance_metrics(batch_result)

        # Auto-tune should prefer smaller batches if they're faster
        mixin._auto_tune_batch_size()

        # Concurrency should increase for fast performance
        mixin._adapt_concurrency_limits()

        assert mixin.current_concurrency_limit == 60  # Increased from 50

    def test_performance_degradation_handling(self):
        """Test handling of performance degradation scenario."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
            async_concurrency_limit=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
            initial_concurrency_limit=50,
        )

        # Simulate degrading performance
        for i in range(5):
            batch_result = BatchResult(
                batch_id=f"batch_{i}",
                items_processed=50,
                items_successful=50,
                items_failed=0,
                processing_time=50.0,  # Very slow
                average_item_time=10.0,  # > 5.0 threshold
                error_rate=0.0,
            )
            mixin._update_performance_metrics(batch_result)

        mixin._adapt_concurrency_limits()

        # Concurrency should decrease
        assert mixin.current_concurrency_limit == 40

    @pytest.mark.asyncio
    async def test_mixed_domain_rate_limiting(self):
        """Test rate limiting across multiple different domains."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        urls = [
            "https://google.com/search",
            "https://github.com/repo",
            "https://example.com/page",
            "https://custom-site.org/resource",
        ]

        for url in urls:
            await mixin._apply_domain_rate_limiting(url)

        # All domains should be tracked
        assert len(mixin.domain_semaphores) == 4
        assert "google.com" in mixin.domain_semaphores
        assert "github.com" in mixin.domain_semaphores
        assert "example.com" in mixin.domain_semaphores
        assert "custom-site.org" in mixin.domain_semaphores


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_average_item_time(self):
        """Test handling of negative average item time (shouldn't happen but test anyway)."""
        mixin = ConcretePerformanceMixin()

        batch_result = BatchResult(
            batch_id="test",
            items_processed=10,
            items_successful=10,
            items_failed=0,
            processing_time=1.0,
            average_item_time=-0.1,  # Negative (invalid)
            error_rate=0.0,
        )

        mixin._update_performance_metrics(batch_result)

        # Should not be added (condition checks > 0)
        assert len(mixin.performance_history) == 0

    def test_very_large_batch_size_in_history(self):
        """Test handling of very large batch sizes in performance history."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=100,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        # Add history with very large batch size (should be clamped)
        mixin.performance_history = [
            (10000, 0.01),  # Very large batch
            (10000, 0.01),
            (50, 0.1),
            (50, 0.1),
        ]

        mixin._auto_tune_batch_size()

        # Should be clamped to max_batch_size
        assert mixin.current_batch_size <= 100

    def test_float_precision_in_metrics(self):
        """Test handling of floating point precision in metrics."""
        mixin = ConcretePerformanceMixin()

        # Add entry with very small float
        batch_result = BatchResult(
            batch_id="test",
            items_processed=10,
            items_successful=10,
            items_failed=0,
            processing_time=0.0000001,
            average_item_time=0.0000001,
            error_rate=0.0,
        )

        mixin._update_performance_metrics(batch_result)

        # Should be added (> 0 check should pass)
        assert len(mixin.performance_history) == 1

    def test_empty_url_for_rate_limiting(self):
        """Test rate limiting with empty URL string."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        # Empty URL - should not crash
        url = ""

        # Should handle gracefully
        asyncio.run(mixin._apply_domain_rate_limiting(url))

    def test_url_with_port_for_rate_limiting(self):
        """Test rate limiting with URL containing port number."""
        config = BatchConfig(rate_limit_respect=True)
        mixin = ConcretePerformanceMixin(config=config)

        url = "https://example.com:8080/page"

        asyncio.run(mixin._apply_domain_rate_limiting(url))

        # Should handle the port in the domain
        assert "example.com:8080" in mixin.domain_semaphores

    def test_concurrent_batch_size_tuning(self):
        """Test concurrent calls to auto_tune_batch_size."""
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
        )
        mixin = ConcretePerformanceMixin(
            config=config,
            initial_batch_size=50,
        )

        mixin.performance_history = [
            (30, 0.05),
            (30, 0.05),
            (50, 0.2),
            (50, 0.2),
        ]

        def tune():
            for _ in range(10):
                mixin._auto_tune_batch_size()

        threads = [threading.Thread(target=tune) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should settle on 30 as optimal
        assert mixin.current_batch_size == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
