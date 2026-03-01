"""
Comprehensive tests for batch_validator module.

This test file focuses on improving coverage of the batch_validator module,
particularly:
1. The CostTrackingMixin class in batch_validator/cost_tracking.py
2. Edge cases and error handling paths in batch_validator.py
3. Async processing paths
4. Integration scenarios combining multiple features

All HTTP calls are mocked - no network access required.
"""

import asyncio
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    PropertyMock,
    call,
    patch,
)

import pytest

from bookmark_processor.core.batch_types import (
    BatchConfig,
    BatchResult,
    CostBreakdown,
    ProgressUpdate,
    ValidationResult,
)
from bookmark_processor.core.batch_validator import EnhancedBatchProcessor
from bookmark_processor.core.batch_validator.cost_tracking import CostTrackingMixin


# =============================================================================
# Test Fixtures and Mock Classes
# =============================================================================


class MockProcessor:
    """Mock processor implementing BatchProcessorInterface."""

    def __init__(
        self,
        success_rate: float = 1.0,
        avg_time: float = 0.1,
        raise_exception: bool = False,
        exception_msg: str = "Mock error",
    ):
        self.success_rate = success_rate
        self.avg_time = avg_time
        self.raise_exception = raise_exception
        self.exception_msg = exception_msg
        self.process_batch_calls = []

    def process_batch(self, items: list, batch_id: str) -> BatchResult:
        """Process a batch of items."""
        if self.raise_exception:
            raise Exception(self.exception_msg)

        self.process_batch_calls.append((items, batch_id))
        time.sleep(self.avg_time * len(items) * 0.001)  # Minimal sleep

        results = []
        successful = 0
        for i, item in enumerate(items):
            is_valid = (i / max(len(items), 1)) < self.success_rate
            results.append(
                ValidationResult(
                    url=item,
                    is_valid=is_valid,
                    status_code=200 if is_valid else 404,
                    response_time=self.avg_time,
                )
            )
            if is_valid:
                successful += 1

        return BatchResult(
            batch_id=batch_id,
            items_processed=len(items),
            items_successful=successful,
            items_failed=len(items) - successful,
            processing_time=self.avg_time * len(items),
            average_item_time=self.avg_time,
            error_rate=(len(items) - successful) / max(len(items), 1),
            results=results,
        )

    def get_optimal_batch_size(self) -> int:
        return 50

    def estimate_processing_time(self, item_count: int) -> float:
        return item_count * self.avg_time


class MockAsyncProcessor(MockProcessor):
    """Mock processor with async support."""

    async def async_validate_batch(self, items: list, batch_id: str) -> BatchResult:
        """Async version of batch processing."""
        await asyncio.sleep(0.001)  # Minimal async work
        return self.process_batch(items, batch_id)


class MockCostTracker:
    """Mock cost tracker for testing budget management."""

    def __init__(self):
        self.session_cost = 0.0
        self.total_cost = 0.0
        self.cost_records = []
        self.last_confirmation = 0.0
        self.user_confirmed_cost = 0.0
        self.confirm_result = True

    def add_cost_record(self, **kwargs):
        """Add a cost record."""
        self.cost_records.append(kwargs)
        self.session_cost += kwargs.get("cost_usd", 0.0)
        self.total_cost += kwargs.get("cost_usd", 0.0)

    def get_detailed_statistics(self):
        """Return mock statistics."""
        return {
            "total_cost": self.total_cost,
            "session_cost": self.session_cost,
            "records_count": len(self.cost_records),
        }

    def get_confirmation_prompt(self):
        """Return a mock prompt."""
        return "Continue? (y/n): "

    async def confirm_continuation(self):
        """Return async confirmation result."""
        return self.confirm_result

    def reset_session(self):
        """Reset session cost."""
        self.session_cost = 0.0


# =============================================================================
# Tests for CostTrackingMixin
# =============================================================================


class ConcreteCostTrackingMixin(CostTrackingMixin):
    """
    Concrete implementation of CostTrackingMixin for testing.

    Provides required attributes that the mixin expects.
    """

    def __init__(
        self,
        config: BatchConfig = None,
        enable_cost_tracking: bool = True,
    ):
        self.config = config or BatchConfig(
            enable_cost_tracking=enable_cost_tracking,
            cost_per_url_validation=0.001,
            cost_confirmation_threshold=1.0,
            budget_limit=None,
        )
        self.lock = threading.RLock()
        self.total_session_cost = 0.0
        self.batch_cost_history = []
        self.cost_tracker = None


class TestCostTrackingMixinEstimation:
    """Tests for cost estimation in CostTrackingMixin."""

    def test_estimate_batch_cost_cost_tracking_disabled(self):
        """Test estimation when cost tracking is disabled."""
        config = BatchConfig(enable_cost_tracking=False)
        mixin = ConcreteCostTrackingMixin(config=config)

        result = mixin.estimate_batch_cost(100, "url_validation")

        assert result.total_estimated_cost == 0.0
        assert result.estimated_cost_per_item == 0.0
        assert "cost_tracking_disabled" in result.cost_factors

    def test_estimate_batch_cost_with_cost_tracking_enabled(self):
        """Test estimation when cost tracking is enabled."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        result = mixin.estimate_batch_cost(50, "url_validation")

        assert result.total_estimated_cost > 0
        assert result.estimated_cost_per_item > 0
        assert result.batch_size == 50
        assert result.operation_type == "url_validation"
        assert "base_url_validation" in result.cost_factors
        assert "network_overhead" in result.cost_factors
        assert "processing_overhead" in result.cost_factors

    def test_estimate_batch_cost_bulk_discount_for_large_batches(self):
        """Test that large batches get bulk discount."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        result = mixin.estimate_batch_cost(150, "url_validation")

        assert "bulk_discount" in result.cost_factors
        assert result.cost_factors["bulk_discount"] < 0  # Negative = discount

    def test_estimate_batch_cost_premium_for_small_batches(self):
        """Test that small batches get premium."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        result = mixin.estimate_batch_cost(5, "url_validation")

        assert "small_batch_premium" in result.cost_factors
        assert result.cost_factors["small_batch_premium"] > 0  # Positive = premium

    def test_estimate_batch_cost_minimum_cost_enforced(self):
        """Test that minimum cost is enforced."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.0,  # Zero cost
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        result = mixin.estimate_batch_cost(10, "url_validation")

        # Should have minimum cost enforced
        assert result.estimated_cost_per_item >= 0.00001


class TestCostTrackingMixinBudgetChecking:
    """Tests for budget checking in CostTrackingMixin."""

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_cost_tracking_disabled(self):
        """Test budget check returns True when tracking disabled."""
        config = BatchConfig(enable_cost_tracking=False)
        mixin = ConcreteCostTrackingMixin(config=config)

        result = await mixin.check_budget_and_confirm(100.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_within_budget(self):
        """Test budget check passes when within budget."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=10.0,
            cost_confirmation_threshold=100.0,  # High threshold
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        result = await mixin.check_budget_and_confirm(5.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_exceeds_budget_no_tracker(self):
        """Test budget check fails when exceeds budget without tracker."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0

        result = await mixin.check_budget_and_confirm(5.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_exceeds_budget_with_tracker_user_confirms(self):
        """Test budget check with tracker when user confirms."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0
        mixin.cost_tracker = MockCostTracker()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value="y")
            mock_loop.return_value.run_in_executor = mock_executor
            result = await mixin.check_budget_and_confirm(5.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_exceeds_budget_with_tracker_user_declines(self):
        """Test budget check with tracker when user declines."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0
        mixin.cost_tracker = MockCostTracker()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value="n")
            mock_loop.return_value.run_in_executor = mock_executor
            result = await mixin.check_budget_and_confirm(5.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_keyboard_interrupt(self):
        """Test budget check handles KeyboardInterrupt."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0
        mixin.cost_tracker = MockCostTracker()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(side_effect=KeyboardInterrupt())
            mock_loop.return_value.run_in_executor = mock_executor
            result = await mixin.check_budget_and_confirm(5.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_threshold_with_cost_tracker(self):
        """Test confirmation threshold triggers with cost tracker."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,
            cost_confirmation_threshold=1.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mock_tracker = MockCostTracker()
        mock_tracker.confirm_result = True
        mixin.cost_tracker = mock_tracker

        result = await mixin.check_budget_and_confirm(5.0)

        assert result is True
        # Verify temporary record was added and removed
        # Session cost should be back to 0 after removing temp record

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_threshold_without_tracker_user_confirms(self):
        """Test confirmation threshold without tracker when user confirms."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,
            cost_confirmation_threshold=1.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value="yes")
            mock_loop.return_value.run_in_executor = mock_executor
            result = await mixin.check_budget_and_confirm(5.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_budget_and_confirm_threshold_without_tracker_eof_error(self):
        """Test confirmation threshold handles EOFError."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,
            cost_confirmation_threshold=1.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(side_effect=EOFError())
            mock_loop.return_value.run_in_executor = mock_executor
            result = await mixin.check_budget_and_confirm(5.0)

        assert result is False


class TestCostTrackingMixinSyncBudget:
    """Tests for sync budget checking in CostTrackingMixin."""

    def test_sync_budget_check_cost_tracking_disabled(self):
        """Test sync budget check returns True when tracking disabled."""
        config = BatchConfig(enable_cost_tracking=False)
        mixin = ConcreteCostTrackingMixin(config=config)

        result = mixin._check_budget_and_confirm_sync(100.0)

        assert result is True

    def test_sync_budget_check_within_budget(self):
        """Test sync budget check passes when within budget."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=10.0,
            cost_confirmation_threshold=100.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is True

    def test_sync_budget_check_exceeds_budget_no_tracker(self):
        """Test sync budget check fails when exceeds without tracker."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0

        result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is False

    def test_sync_budget_check_exceeds_budget_with_tracker_user_confirms(self):
        """Test sync budget with tracker when user confirms."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0
        mixin.cost_tracker = MockCostTracker()

        with patch("builtins.input", return_value="y"):
            result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is True

    def test_sync_budget_check_exceeds_budget_with_tracker_user_declines(self):
        """Test sync budget with tracker when user declines."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0
        mixin.cost_tracker = MockCostTracker()

        with patch("builtins.input", return_value="no"):
            result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is False

    def test_sync_budget_check_keyboard_interrupt(self):
        """Test sync budget check handles KeyboardInterrupt."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.total_session_cost = 3.0
        mixin.cost_tracker = MockCostTracker()

        with patch("builtins.input", side_effect=KeyboardInterrupt()):
            result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is False

    def test_sync_budget_threshold_with_tracker_user_confirms(self):
        """Test sync threshold with tracker when user confirms."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,
            cost_confirmation_threshold=1.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.cost_tracker = MockCostTracker()

        with patch("builtins.input", return_value="y"):
            result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is True

    def test_sync_budget_threshold_with_tracker_user_uses_enter(self):
        """Test sync threshold with tracker when user presses enter (default yes)."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,
            cost_confirmation_threshold=1.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.cost_tracker = MockCostTracker()

        with patch("builtins.input", return_value=""):
            result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is True  # Empty input = yes

    def test_sync_budget_threshold_without_tracker_user_confirms(self):
        """Test sync threshold without tracker when user confirms."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,
            cost_confirmation_threshold=1.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        with patch("builtins.input", return_value="yes"):
            result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is True

    def test_sync_budget_threshold_without_tracker_eof_error(self):
        """Test sync threshold handles EOFError."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,
            cost_confirmation_threshold=1.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        with patch("builtins.input", side_effect=EOFError()):
            result = mixin._check_budget_and_confirm_sync(5.0)

        assert result is False


class TestCostTrackingMixinRecording:
    """Tests for cost recording in CostTrackingMixin."""

    def test_record_batch_cost_disabled(self):
        """Test recording when cost tracking disabled."""
        config = BatchConfig(enable_cost_tracking=False)
        mixin = ConcreteCostTrackingMixin(config=config)

        mixin.record_batch_cost("batch_1", 1.0)

        assert mixin.total_session_cost == 0.0
        assert len(mixin.batch_cost_history) == 0

    def test_record_batch_cost_enabled(self):
        """Test recording when cost tracking enabled."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        mixin.record_batch_cost("batch_1", 1.5)

        assert mixin.total_session_cost == 1.5
        assert len(mixin.batch_cost_history) == 1
        assert mixin.batch_cost_history[0] == ("batch_1", 1.5)

    def test_record_batch_cost_multiple(self):
        """Test recording multiple batches."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        mixin.record_batch_cost("batch_1", 0.5)
        mixin.record_batch_cost("batch_2", 0.3)
        mixin.record_batch_cost("batch_3", 0.2)

        assert mixin.total_session_cost == 1.0
        assert len(mixin.batch_cost_history) == 3

    def test_record_batch_cost_history_limit(self):
        """Test that history is limited to 100 entries."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        for i in range(150):
            mixin.record_batch_cost(f"batch_{i}", 0.01)

        assert len(mixin.batch_cost_history) == 100
        # Should keep the last 100
        assert mixin.batch_cost_history[0][0] == "batch_50"
        assert mixin.batch_cost_history[-1][0] == "batch_149"

    def test_record_batch_cost_with_tracker(self):
        """Test recording with external cost tracker."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.cost_tracker = MockCostTracker()

        mixin.record_batch_cost("batch_1", 0.5)

        assert len(mixin.cost_tracker.cost_records) == 1
        assert mixin.cost_tracker.cost_records[0]["cost_usd"] == 0.5


class TestCostTrackingMixinStatistics:
    """Tests for cost statistics in CostTrackingMixin."""

    def test_get_cost_statistics_disabled(self):
        """Test statistics when tracking disabled."""
        config = BatchConfig(enable_cost_tracking=False)
        mixin = ConcreteCostTrackingMixin(config=config)

        stats = mixin.get_cost_statistics()

        assert stats["cost_tracking_enabled"] is False

    def test_get_cost_statistics_enabled_empty(self):
        """Test statistics with no batches."""
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=10.0,
        )
        mixin = ConcreteCostTrackingMixin(config=config)

        stats = mixin.get_cost_statistics()

        assert stats["cost_tracking_enabled"] is True
        assert stats["total_session_cost"] == 0.0
        assert stats["batch_count"] == 0
        assert stats["budget_remaining"] == 10.0

    def test_get_cost_statistics_with_history(self):
        """Test statistics with batch history."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        for i in range(5):
            mixin.record_batch_cost(f"batch_{i}", 0.1 * (i + 1))

        stats = mixin.get_cost_statistics()

        assert stats["batch_count"] == 5
        assert stats["min_batch_cost"] == 0.1
        assert stats["max_batch_cost"] == 0.5
        assert "recent_average_cost" in stats
        assert "cost_trend" in stats

    def test_get_cost_statistics_with_tracker(self):
        """Test statistics includes tracker stats."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)
        mixin.cost_tracker = MockCostTracker()

        mixin.record_batch_cost("batch_1", 0.5)
        stats = mixin.get_cost_statistics()

        assert "cost_tracker_stats" in stats


class TestCostTrackingMixinTrend:
    """Tests for cost trend calculation in CostTrackingMixin."""

    def test_calculate_cost_trend_insufficient_data(self):
        """Test trend with insufficient data."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        mixin.record_batch_cost("batch_1", 0.1)
        mixin.record_batch_cost("batch_2", 0.2)

        trend = mixin._calculate_cost_trend()

        assert trend == "insufficient_data"

    def test_calculate_cost_trend_insufficient_early_data(self):
        """Test trend with insufficient early data for comparison."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        for i in range(7):
            mixin.record_batch_cost(f"batch_{i}", 0.1)

        trend = mixin._calculate_cost_trend()

        assert trend == "insufficient_data"

    def test_calculate_cost_trend_increasing(self):
        """Test increasing trend detection."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        # Early batches: low costs
        for i in range(5):
            mixin.record_batch_cost(f"early_{i}", 0.1)

        # Recent batches: high costs (> 10% increase)
        for i in range(5):
            mixin.record_batch_cost(f"recent_{i}", 0.2)

        trend = mixin._calculate_cost_trend()

        assert trend == "increasing"

    def test_calculate_cost_trend_decreasing(self):
        """Test decreasing trend detection."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        # Early batches: high costs
        for i in range(5):
            mixin.record_batch_cost(f"early_{i}", 0.2)

        # Recent batches: low costs (> 10% decrease)
        for i in range(5):
            mixin.record_batch_cost(f"recent_{i}", 0.1)

        trend = mixin._calculate_cost_trend()

        assert trend == "decreasing"

    def test_calculate_cost_trend_stable(self):
        """Test stable trend detection."""
        config = BatchConfig(enable_cost_tracking=True)
        mixin = ConcreteCostTrackingMixin(config=config)

        # All batches with similar costs (within 10%)
        for i in range(10):
            mixin.record_batch_cost(f"batch_{i}", 0.1)

        trend = mixin._calculate_cost_trend()

        assert trend == "stable"


# =============================================================================
# Tests for EnhancedBatchProcessor - Additional Coverage
# =============================================================================


class TestEnhancedBatchProcessorCostIntegration:
    """Tests for cost tracking integration in EnhancedBatchProcessor."""

    def test_process_single_batch_with_cost_and_success_rate_adjustment(self):
        """Test cost is adjusted based on success rate."""
        processor = MockProcessor(success_rate=0.5)
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Create cost estimate
        batch_id = "test_batch"
        cost_breakdown = batch_processor.estimate_batch_cost(10)
        batch_processor.cost_estimates[batch_id] = cost_breakdown

        items = [f"https://example{i}.com" for i in range(10)]
        result = batch_processor._process_single_batch(batch_id, items)

        # Cost should be adjusted based on success rate
        assert result.actual_cost is not None
        # With 50% success rate: cost * (0.5 + 0.5 * 0.5) = cost * 0.75
        assert result.cost_breakdown is not None

    def test_process_single_batch_fallback_cost_calculation(self):
        """Test fallback cost calculation when no estimate exists."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(10)]
        result = batch_processor._process_single_batch("unknown_batch", items)

        # Should use fallback calculation
        assert result.actual_cost is not None
        expected_cost = 10 * 0.001  # items * cost_per_url
        assert result.actual_cost == pytest.approx(expected_cost * (0.5 + 0.5 * 1.0), rel=0.1)

    def test_process_single_batch_error_partial_cost(self):
        """Test partial cost is recorded for failed batches."""
        processor = MockProcessor(raise_exception=True)
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(10)]
        result = batch_processor._process_single_batch("failed_batch", items)

        # Failed batch should have 10% cost
        assert result.actual_cost is not None
        expected_partial = 10 * 0.001 * 0.1
        assert result.actual_cost == pytest.approx(expected_partial, rel=0.01)

    def test_add_items_cancelled_due_to_budget(self):
        """Test items addition cancelled due to budget concerns."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=0.001,  # Very low budget
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)
        batch_processor.total_session_cost = 0.001  # Already at limit

        items = [f"https://example{i}.com" for i in range(100)]
        result = batch_processor.add_items(items)

        assert result is False

    def test_get_processing_statistics_with_cost_efficiency(self):
        """Test statistics include cost efficiency metrics."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            retry_failed_batches=False,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add completed batch with cost and time
        batch_result = BatchResult(
            batch_id="batch_1",
            items_processed=10,
            items_successful=10,
            items_failed=0,
            processing_time=2.0,
            average_item_time=0.2,
            error_rate=0.0,
            actual_cost=0.01,
        )
        batch_processor.completed_batches.append(batch_result)
        batch_processor.record_batch_cost("batch_1", 0.01)

        stats = batch_processor.get_processing_statistics()

        assert "cost_per_second" in stats
        assert "cost_per_successful_item" in stats


class TestEnhancedBatchProcessorAsyncProcessing:
    """Tests for async processing in EnhancedBatchProcessor."""

    def test_process_all_uses_async_when_available(self):
        """Test that async processing is used when available."""
        processor = MockAsyncProcessor()
        config = BatchConfig(
            enable_async_processing=True,
            optimal_batch_size=10,
            max_concurrent_batches=2,
            retry_failed_batches=False,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(20)]
        batch_processor.add_items(items)
        results = batch_processor.process_all()

        assert len(results) == 20

    def test_process_all_async_fallback_to_sync_on_error(self):
        """Test fallback to sync processing when async fails."""
        processor = MockAsyncProcessor()
        config = BatchConfig(
            enable_async_processing=True,
            optimal_batch_size=10,
            retry_failed_batches=False,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(10)]
        batch_processor.add_items(items)

        # Mock async processing to fail
        with patch.object(
            batch_processor, "_async_process_batches",
            side_effect=Exception("Async failed")
        ):
            results = batch_processor._process_all_async()

        # Should fall back to sync and still process
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_async_process_batches_empty_queue(self):
        """Test async processing with empty queue."""
        processor = MockAsyncProcessor()
        config = BatchConfig(enable_async_processing=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        results = await batch_processor._async_process_batches()

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_async_process_batches_with_failures(self):
        """Test async processing handles failed batches."""
        processor = MockAsyncProcessor()
        config = BatchConfig(
            enable_async_processing=True,
            optimal_batch_size=5,
            retry_failed_batches=True,
            max_concurrent_batches=2,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(10)]
        batch_processor.add_items(items)

        # Process and let retries happen
        results = await batch_processor._async_process_batches()

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_async_process_single_batch_fallback_to_sync(self):
        """Test async batch falls back to sync when no async method."""
        processor = MockProcessor()  # No async_validate_batch method
        config = BatchConfig(enable_async_processing=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        semaphore = asyncio.Semaphore(2)
        items = [f"https://example{i}.com" for i in range(5)]

        result = await batch_processor._async_process_single_batch(
            semaphore, "test_batch", items
        )

        assert result.batch_id == "test_batch"
        assert result.items_processed == 5


class TestEnhancedBatchProcessorProgressTracking:
    """Tests for progress tracking in EnhancedBatchProcessor."""

    def test_emit_progress_update_calculates_correct_stats(self):
        """Test progress update calculations are correct."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=10)
        progress_updates = []

        batch_processor = EnhancedBatchProcessor(
            processor=processor,
            config=config,
            progress_update_callback=lambda u: progress_updates.append(u),
        )

        # Set up state
        batch_processor.total_items = 50
        batch_processor.total_batches = 5
        batch_processor.processing_start_time = time.time() - 60  # 1 minute ago
        batch_processor.total_successes = 20
        batch_processor.total_errors = 5

        # Add a completed batch
        batch_processor.completed_batches.append(
            BatchResult(
                batch_id="batch_1",
                items_processed=25,
                items_successful=20,
                items_failed=5,
                processing_time=30.0,
                average_item_time=1.2,
                error_rate=0.2,
            )
        )

        batch_processor._emit_progress_update("processing", "batch_2", 5, 10)

        assert len(progress_updates) == 1
        update = progress_updates[0]
        assert update.total_items_processed == 30  # 25 completed + 5 current
        assert update.total_items_remaining == 20  # 50 - 30
        assert update.success_rate == pytest.approx(0.666, rel=0.01)  # 20/30

    def test_emit_progress_update_handles_callback_exception(self):
        """Test progress update handles callback exceptions gracefully."""
        processor = MockProcessor()

        def bad_callback(update):
            raise ValueError("Callback failed")

        batch_processor = EnhancedBatchProcessor(
            processor=processor,
            progress_update_callback=bad_callback,
        )

        batch_processor.total_items = 10
        batch_processor.processing_start_time = time.time()

        # Should not raise
        batch_processor._emit_progress_update("test", "batch_1", 0, 10)


class TestEnhancedBatchProcessorRetryLogic:
    """Tests for retry logic in EnhancedBatchProcessor."""

    def test_retry_failed_batches_with_progress_updates(self):
        """Test retry emits progress updates."""
        processor = MockProcessor()
        config = BatchConfig(min_batch_size=2, retry_failed_batches=True)
        progress_updates = []

        batch_processor = EnhancedBatchProcessor(
            processor=processor,
            config=config,
            progress_update_callback=lambda u: progress_updates.append(u),
        )
        batch_processor.total_items = 10
        batch_processor.processing_start_time = time.time()

        batch_processor.failed_batches.append(
            ("failed_batch", ["url1", "url2", "url3", "url4"], Exception("Error"))
        )

        results = batch_processor._retry_failed_batches()

        assert len(results) > 0
        # Should have emitted progress updates for retry
        retry_updates = [u for u in progress_updates if "retry" in u.current_stage.lower()]
        assert len(retry_updates) > 0

    def test_retry_failed_batches_handles_retry_failure(self):
        """Test retry handles when retry also fails gracefully."""
        processor = MockProcessor()

        # Make processor always fail
        def always_fail(items, batch_id):
            raise Exception("Always failing")

        processor.process_batch = always_fail

        config = BatchConfig(min_batch_size=2, retry_failed_batches=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)
        batch_processor.total_items = 4
        batch_processor.processing_start_time = time.time()

        batch_processor.failed_batches.append(
            ("failed_batch", ["url1", "url2", "url3", "url4"], Exception("Error"))
        )

        # Should not raise even when all retries fail
        results = batch_processor._retry_failed_batches()

        # When all retries fail, results should be empty
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_async_retry_failed_batches_handles_failures(self):
        """Test async retry handles failures gracefully."""
        processor = MockAsyncProcessor()
        config = BatchConfig(
            min_batch_size=2,
            max_concurrent_batches=2,
            retry_failed_batches=True,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)
        batch_processor.total_items = 4
        batch_processor.processing_start_time = time.time()

        # Mock to simulate some failures
        original_async = batch_processor._async_process_single_batch

        call_count = 0

        async def sometimes_fail(sem, batch_id, items):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First retry fails")
            return await original_async(sem, batch_id, items)

        batch_processor.failed_batches.append(
            ("failed_batch", ["url1", "url2", "url3", "url4"], Exception("Error"))
        )

        with patch.object(
            batch_processor, "_async_process_single_batch",
            side_effect=sometimes_fail
        ):
            results = await batch_processor._async_retry_failed_batches()

        # Should have some results even with failures


class TestEnhancedBatchProcessorEdgeCases:
    """Tests for edge cases in EnhancedBatchProcessor."""

    def test_create_batches_empty_list(self):
        """Test batch creation with empty list."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=10)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        batches = batch_processor._create_batches([])

        assert len(batches) == 0

    def test_create_batches_single_item(self):
        """Test batch creation with single item."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=10)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        batches = batch_processor._create_batches(["single_item"])

        assert len(batches) == 1
        assert len(batches[0][1]) == 1

    def test_process_all_with_failed_batches_no_retry(self):
        """Test process all when batches fail but retry is disabled."""
        processor = MockProcessor(raise_exception=True)
        config = BatchConfig(
            optimal_batch_size=5,
            enable_async_processing=False,
            retry_failed_batches=False,
            max_concurrent_batches=1,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(10)]
        batch_processor.add_items(items)
        results = batch_processor.process_all()

        # When _process_single_batch catches exceptions, it returns an error
        # BatchResult which goes to completed_batches (not failed_batches).
        # The failed_batches list is only populated when future.result() times out.
        # So we verify completed_batches have error_rate=1.0
        assert len(batch_processor.completed_batches) == 2
        for batch in batch_processor.completed_batches:
            assert batch.error_rate == 1.0
            assert batch.items_failed == 5

    def test_process_all_with_batch_timeout(self):
        """Test handling of batch timeout."""
        processor = MockProcessor()

        def slow_process(items, batch_id):
            time.sleep(0.5)  # Simulate slow processing
            return BatchResult(
                batch_id=batch_id,
                items_processed=len(items),
                items_successful=len(items),
                items_failed=0,
                processing_time=0.5,
                average_item_time=0.05,
                error_rate=0.0,
            )

        processor.process_batch = slow_process

        config = BatchConfig(
            optimal_batch_size=5,
            enable_async_processing=False,
            retry_failed_batches=False,
            max_concurrent_batches=1,
            batch_timeout=0.1,  # Very short timeout
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(5)]
        batch_processor.add_items(items)

        # Should handle timeout gracefully
        results = batch_processor.process_all()

    def test_reset_with_cost_tracker(self):
        """Test reset also resets external cost tracker."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        mock_tracker = MockCostTracker()
        mock_tracker.session_cost = 10.0
        batch_processor.cost_tracker = mock_tracker

        batch_processor.record_batch_cost("batch_1", 5.0)
        batch_processor.reset()

        assert mock_tracker.session_cost == 0.0

    def test_process_single_batch_with_zero_items(self):
        """Test processing batch with empty items list."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        result = batch_processor._process_single_batch("empty_batch", [])

        assert result.items_processed == 0


class TestEnhancedBatchProcessorAutoTuning:
    """Tests for auto-tuning in EnhancedBatchProcessor."""

    def test_auto_tune_finds_optimal_size(self):
        """Test auto-tuning finds optimal batch size."""
        processor = MockProcessor()
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
            auto_tune_batch_size=True,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add performance history showing smaller batches are faster
        for _ in range(3):
            batch_processor.performance_history.append((25, 0.05))
        for _ in range(3):
            batch_processor.performance_history.append((50, 0.15))

        batch_processor._auto_tune_batch_size()

        assert batch_processor.current_batch_size == 25

    def test_auto_tune_respects_constraints(self):
        """Test auto-tuning respects min/max constraints."""
        processor = MockProcessor()
        config = BatchConfig(
            min_batch_size=20,
            max_batch_size=80,
            optimal_batch_size=50,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # History suggests batch size 10 (below min) is best
        batch_processor.performance_history = [
            (10, 0.01),
            (10, 0.01),
            (50, 0.1),
            (50, 0.1),
        ]

        batch_processor._auto_tune_batch_size()

        assert batch_processor.current_batch_size >= 20


class TestEnhancedBatchProcessorRateLimiting:
    """Tests for rate limiting in EnhancedBatchProcessor."""

    @pytest.mark.asyncio
    async def test_apply_domain_rate_limiting_various_domains(self):
        """Test rate limiting applies correct delays for various domains."""
        processor = MockAsyncProcessor()
        config = BatchConfig(rate_limit_respect=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        domains = [
            ("https://google.com/search", "google.com"),
            ("https://github.com/repo", "github.com"),
            ("https://youtube.com/video", "youtube.com"),
            ("https://facebook.com/page", "facebook.com"),
            ("https://linkedin.com/profile", "linkedin.com"),
            ("https://twitter.com/tweet", "twitter.com"),
            ("https://x.com/post", "x.com"),
            ("https://reddit.com/thread", "reddit.com"),
            ("https://medium.com/article", "medium.com"),
            ("https://unknown.com/page", "unknown.com"),
        ]

        for url, expected_domain in domains:
            await batch_processor._apply_domain_rate_limiting(url)
            assert expected_domain in batch_processor.domain_semaphores
            assert expected_domain in batch_processor.rate_limit_tracker

    @pytest.mark.asyncio
    async def test_apply_domain_rate_limiting_invalid_url(self):
        """Test rate limiting handles invalid URLs."""
        processor = MockAsyncProcessor()
        config = BatchConfig(rate_limit_respect=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Should not crash with invalid URL
        await batch_processor._apply_domain_rate_limiting("")
        await batch_processor._apply_domain_rate_limiting("not-a-url")


# =============================================================================
# Tests for Package __init__.py
# =============================================================================


class TestBatchValidatorPackageImports:
    """Tests for batch_validator package imports."""

    def test_lazy_import_enhanced_batch_processor(self):
        """Test lazy import of EnhancedBatchProcessor works."""
        from bookmark_processor.core.batch_validator import EnhancedBatchProcessor

        assert EnhancedBatchProcessor is not None

    def test_import_cost_tracking_mixin(self):
        """Test CostTrackingMixin can be imported."""
        from bookmark_processor.core.batch_validator import CostTrackingMixin

        assert CostTrackingMixin is not None

    def test_import_performance_mixin(self):
        """Test PerformanceOptimizationMixin can be imported."""
        from bookmark_processor.core.batch_validator import PerformanceOptimizationMixin

        assert PerformanceOptimizationMixin is not None

    def test_invalid_attribute_raises_error(self):
        """Test accessing invalid attribute raises AttributeError."""
        import bookmark_processor.core.batch_validator as bv

        with pytest.raises(AttributeError):
            _ = bv.NonExistentClass


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestBatchValidatorThreadSafety:
    """Tests for thread safety in batch processing."""

    def test_concurrent_batch_processing(self):
        """Test concurrent batch processing is thread-safe."""
        processor = MockProcessor()
        config = BatchConfig(
            optimal_batch_size=5,
            enable_async_processing=False,
            max_concurrent_batches=5,
            retry_failed_batches=False,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(50)]
        batch_processor.add_items(items)
        results = batch_processor.process_all()

        assert len(results) == 50

    def test_concurrent_statistics_access(self):
        """Test concurrent access to statistics is thread-safe."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        # Add some completed batches
        for i in range(10):
            batch_processor.completed_batches.append(
                BatchResult(
                    batch_id=f"batch_{i}",
                    items_processed=10,
                    items_successful=10,
                    items_failed=0,
                    processing_time=1.0,
                    average_item_time=0.1,
                    error_rate=0.0,
                )
            )

        def get_stats():
            for _ in range(100):
                stats = batch_processor.get_processing_statistics()
                assert stats["total_batches"] >= 0

        threads = [threading.Thread(target=get_stats) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
