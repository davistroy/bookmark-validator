"""
Unit tests for batch_validator module.

Tests the EnhancedBatchProcessor class for batch processing capabilities including:
- Cost estimation and tracking
- Budget management
- Sync and async processing
- Progress tracking and callbacks
- Performance metrics and auto-tuning
- Error handling and retry logic
"""

import asyncio
import time
import threading
from datetime import datetime
from queue import Queue
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import pytest

from bookmark_processor.core.batch_types import (
    BatchConfig,
    BatchResult,
    CostBreakdown,
    ProgressUpdate,
    ValidationResult,
)
from bookmark_processor.core.batch_validator import EnhancedBatchProcessor


class MockProcessor:
    """Mock processor implementing BatchProcessorInterface."""

    def __init__(self, success_rate: float = 1.0, avg_time: float = 0.1):
        self.success_rate = success_rate
        self.avg_time = avg_time
        self.process_batch_calls = []

    def process_batch(self, items: list, batch_id: str) -> BatchResult:
        """Process a batch of items."""
        self.process_batch_calls.append((items, batch_id))
        time.sleep(self.avg_time * len(items) * 0.01)  # Simulate processing time

        # Create validation results
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
        await asyncio.sleep(0.01)  # Simulate async work
        return self.process_batch(items, batch_id)


class TestEnhancedBatchProcessorInit:
    """Test EnhancedBatchProcessor initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with default config."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        assert batch_processor.processor == processor
        assert batch_processor.config is not None
        assert batch_processor.progress_callback is None
        assert batch_processor.progress_update_callback is None
        assert batch_processor.total_items == 0
        assert batch_processor.total_batches == 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        processor = MockProcessor()
        config = BatchConfig(
            min_batch_size=5,
            max_batch_size=200,
            optimal_batch_size=50,
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        assert batch_processor.config.min_batch_size == 5
        assert batch_processor.config.max_batch_size == 200
        assert batch_processor.config.optimal_batch_size == 50
        assert batch_processor.config.enable_cost_tracking is True
        assert batch_processor.current_batch_size == 50

    def test_initialization_with_callbacks(self):
        """Test initialization with progress callbacks."""
        processor = MockProcessor()
        progress_callback = Mock()
        progress_update_callback = Mock()

        batch_processor = EnhancedBatchProcessor(
            processor=processor,
            progress_callback=progress_callback,
            progress_update_callback=progress_update_callback,
        )

        assert batch_processor.progress_callback == progress_callback
        assert batch_processor.progress_update_callback == progress_update_callback


class TestCostEstimation:
    """Test cost estimation methods."""

    def test_estimate_batch_cost_disabled(self):
        """Test cost estimation when cost tracking is disabled."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        cost_breakdown = batch_processor.estimate_batch_cost(100)

        assert cost_breakdown.total_estimated_cost == 0.0
        assert cost_breakdown.estimated_cost_per_item == 0.0
        assert "cost_tracking_disabled" in cost_breakdown.cost_factors

    def test_estimate_batch_cost_enabled(self):
        """Test cost estimation when cost tracking is enabled."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        cost_breakdown = batch_processor.estimate_batch_cost(100)

        assert cost_breakdown.total_estimated_cost > 0
        assert cost_breakdown.estimated_cost_per_item > 0
        assert cost_breakdown.batch_size == 100
        assert "base_url_validation" in cost_breakdown.cost_factors

    def test_estimate_batch_cost_bulk_discount(self):
        """Test cost estimation with bulk discount for large batches."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Large batch should get bulk discount
        large_batch_cost = batch_processor.estimate_batch_cost(150)
        assert "bulk_discount" in large_batch_cost.cost_factors
        assert large_batch_cost.cost_factors["bulk_discount"] < 0

    def test_estimate_batch_cost_small_batch_premium(self):
        """Test cost estimation with premium for small batches."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Small batch should have premium
        small_batch_cost = batch_processor.estimate_batch_cost(5)
        assert "small_batch_premium" in small_batch_cost.cost_factors
        assert small_batch_cost.cost_factors["small_batch_premium"] > 0


class TestBudgetManagement:
    """Test budget checking and confirmation logic."""

    def test_check_budget_cost_tracking_disabled(self):
        """Test budget check returns True when cost tracking disabled."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        result = batch_processor._check_budget_and_confirm_sync(100.0)
        assert result is True

    def test_check_budget_within_limit(self):
        """Test budget check passes when within limit."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=10.0,
            cost_confirmation_threshold=100.0,  # High threshold to avoid confirmation
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        result = batch_processor._check_budget_and_confirm_sync(5.0)
        assert result is True

    def test_check_budget_exceeds_limit_no_tracker(self):
        """Test budget check fails when exceeding limit without tracker."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=5.0,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)
        batch_processor.total_session_cost = 3.0

        # Without cost_tracker, should return False when budget exceeded
        result = batch_processor._check_budget_and_confirm_sync(5.0)
        assert result is False

    def test_check_budget_below_confirmation_threshold(self):
        """Test budget check passes when below confirmation threshold."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=100.0,  # High budget
            cost_confirmation_threshold=10.0,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Cost below threshold should pass without confirmation
        result = batch_processor._check_budget_and_confirm_sync(5.0)
        assert result is True


class TestRecordBatchCost:
    """Test batch cost recording."""

    def test_record_batch_cost_disabled(self):
        """Test cost recording when disabled."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        batch_processor.record_batch_cost("batch_1", 1.0)

        assert batch_processor.total_session_cost == 0.0
        assert len(batch_processor.batch_cost_history) == 0

    def test_record_batch_cost_enabled(self):
        """Test cost recording when enabled."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        batch_processor.record_batch_cost("batch_1", 1.0)
        batch_processor.record_batch_cost("batch_2", 2.0)

        assert batch_processor.total_session_cost == 3.0
        assert len(batch_processor.batch_cost_history) == 2

    def test_record_batch_cost_history_limit(self):
        """Test cost history is limited to 100 entries."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Record more than 100 batches
        for i in range(150):
            batch_processor.record_batch_cost(f"batch_{i}", 0.01)

        assert len(batch_processor.batch_cost_history) == 100


class TestCostStatistics:
    """Test cost statistics methods."""

    def test_get_cost_statistics_disabled(self):
        """Test cost stats when tracking disabled."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        stats = batch_processor.get_cost_statistics()

        assert stats["cost_tracking_enabled"] is False

    def test_get_cost_statistics_enabled_empty(self):
        """Test cost stats when enabled but no batches processed."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            budget_limit=10.0,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        stats = batch_processor.get_cost_statistics()

        assert stats["cost_tracking_enabled"] is True
        assert stats["total_session_cost"] == 0.0
        assert stats["batch_count"] == 0
        assert stats["budget_remaining"] == 10.0

    def test_get_cost_statistics_with_history(self):
        """Test cost stats with batch history."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Record some batches
        for i in range(5):
            batch_processor.record_batch_cost(f"batch_{i}", 0.1 * (i + 1))

        stats = batch_processor.get_cost_statistics()

        assert stats["batch_count"] == 5
        assert stats["min_batch_cost"] == 0.1
        assert stats["max_batch_cost"] == 0.5
        assert "recent_average_cost" in stats


class TestCostTrend:
    """Test cost trend calculation."""

    def test_calculate_cost_trend_insufficient_data(self):
        """Test trend calculation with insufficient data."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add only 2 batches (need at least 3)
        batch_processor.record_batch_cost("batch_1", 0.1)
        batch_processor.record_batch_cost("batch_2", 0.2)

        trend = batch_processor._calculate_cost_trend()
        assert trend == "insufficient_data"

    def test_calculate_cost_trend_increasing(self):
        """Test trend calculation when costs are increasing."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add batches with increasing costs
        for i in range(15):
            batch_processor.record_batch_cost(f"batch_{i}", 0.1 * (i + 1))

        trend = batch_processor._calculate_cost_trend()
        assert trend == "increasing"

    def test_calculate_cost_trend_decreasing(self):
        """Test trend calculation when costs are decreasing."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add batches with decreasing costs
        for i in range(15):
            batch_processor.record_batch_cost(f"batch_{i}", 0.1 * (15 - i))

        trend = batch_processor._calculate_cost_trend()
        assert trend == "decreasing"

    def test_calculate_cost_trend_stable(self):
        """Test trend calculation when costs are stable."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add batches with stable costs
        for i in range(15):
            batch_processor.record_batch_cost(f"batch_{i}", 0.1)

        trend = batch_processor._calculate_cost_trend()
        assert trend == "stable"


class TestAddItems:
    """Test adding items to processing queue."""

    def test_add_items_basic(self):
        """Test basic item addition."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=10)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(25)]
        result = batch_processor.add_items(items)

        assert result is True
        assert batch_processor.total_items == 25
        assert batch_processor.total_batches == 3  # 25 items / 10 batch size

    def test_add_items_with_cost_tracking(self):
        """Test item addition with cost tracking."""
        processor = MockProcessor()
        config = BatchConfig(
            optimal_batch_size=10,
            enable_cost_tracking=True,
            cost_confirmation_threshold=100.0,  # High threshold to avoid confirmation
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(25)]
        result = batch_processor.add_items(items)

        assert result is True
        assert len(batch_processor.cost_estimates) > 0


class TestCreateBatches:
    """Test batch creation logic."""

    def test_create_batches_even_split(self):
        """Test batch creation with even item count."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=10)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"item_{i}" for i in range(30)]
        batches = batch_processor._create_batches(items)

        assert len(batches) == 3
        for batch_id, batch_items in batches:
            assert len(batch_items) == 10

    def test_create_batches_uneven_split(self):
        """Test batch creation with uneven item count."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=10)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"item_{i}" for i in range(25)]
        batches = batch_processor._create_batches(items)

        assert len(batches) == 3
        assert len(batches[0][1]) == 10
        assert len(batches[1][1]) == 10
        assert len(batches[2][1]) == 5

    def test_create_batches_single_batch(self):
        """Test batch creation with fewer items than batch size."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=50)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"item_{i}" for i in range(10)]
        batches = batch_processor._create_batches(items)

        assert len(batches) == 1
        assert len(batches[0][1]) == 10


class TestProcessAllSync:
    """Test synchronous batch processing."""

    def test_process_all_sync_basic(self):
        """Test basic synchronous processing."""
        processor = MockProcessor()
        config = BatchConfig(
            optimal_batch_size=10,
            max_concurrent_batches=2,
            enable_async_processing=False,
            auto_tune_batch_size=False,
            retry_failed_batches=False,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(25)]
        batch_processor.add_items(items)
        results = batch_processor.process_all()

        assert len(results) == 25
        assert len(batch_processor.completed_batches) == 3

    def test_process_all_sync_with_progress_callback(self):
        """Test sync processing with progress callback."""
        processor = MockProcessor()
        config = BatchConfig(
            optimal_batch_size=10,
            enable_async_processing=False,
            retry_failed_batches=False,
        )
        progress_calls = []
        batch_processor = EnhancedBatchProcessor(
            processor=processor,
            config=config,
            progress_callback=lambda msg: progress_calls.append(msg),
        )

        items = [f"https://example{i}.com" for i in range(20)]
        batch_processor.add_items(items)
        batch_processor.process_all()

        assert len(progress_calls) > 0

    def test_process_all_sync_with_cost_tracking(self):
        """Test sync processing with cost tracking."""
        processor = MockProcessor()
        config = BatchConfig(
            optimal_batch_size=10,
            enable_async_processing=False,
            enable_cost_tracking=True,
            cost_confirmation_threshold=100.0,  # High threshold
            retry_failed_batches=False,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(20)]
        batch_processor.add_items(items)
        batch_processor.process_all()

        assert batch_processor.total_session_cost > 0
        assert len(batch_processor.batch_cost_history) > 0


class TestProcessSingleBatch:
    """Test single batch processing."""

    def test_process_single_batch_success(self):
        """Test successful single batch processing."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(5)]
        result = batch_processor._process_single_batch("test_batch", items)

        assert result.batch_id == "test_batch"
        assert result.items_processed == 5
        assert result.processing_time > 0

    def test_process_single_batch_with_cost_tracking(self):
        """Test single batch processing with cost tracking."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Create cost estimate first
        batch_id = "test_batch"
        cost_breakdown = batch_processor.estimate_batch_cost(5)
        batch_processor.cost_estimates[batch_id] = cost_breakdown

        items = [f"https://example{i}.com" for i in range(5)]
        result = batch_processor._process_single_batch(batch_id, items)

        assert result.actual_cost is not None
        assert result.cost_breakdown is not None

    def test_process_single_batch_error(self):
        """Test single batch processing with error."""
        processor = MockProcessor()
        processor.process_batch = Mock(side_effect=Exception("Test error"))
        config = BatchConfig(enable_cost_tracking=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        items = [f"https://example{i}.com" for i in range(5)]
        result = batch_processor._process_single_batch("test_batch", items)

        assert result.items_failed == 5
        assert result.error_rate == 1.0
        assert len(result.errors) > 0


class TestProgressTracking:
    """Test progress tracking functionality."""

    def test_emit_progress_update(self):
        """Test progress update emission."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=10)
        progress_updates = []
        batch_processor = EnhancedBatchProcessor(
            processor=processor,
            config=config,
            progress_update_callback=lambda update: progress_updates.append(update),
        )

        batch_processor.total_items = 20
        batch_processor.total_batches = 2
        batch_processor.processing_start_time = time.time() - 10

        batch_processor._emit_progress_update("processing", "batch_1", 5, 10)

        assert len(progress_updates) == 1
        assert progress_updates[0].batch_id == "batch_1"
        assert progress_updates[0].current_stage == "processing"

    def test_emit_progress_update_no_callback(self):
        """Test progress update with no callback set."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        # Should not raise error
        batch_processor._emit_progress_update("processing", "batch_1", 5, 10)

    def test_update_progress_counters(self):
        """Test progress counter updates from batch result."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        batch_result = BatchResult(
            batch_id="test",
            items_processed=10,
            items_successful=8,
            items_failed=2,
            processing_time=1.0,
            average_item_time=0.1,
            error_rate=0.2,
        )

        batch_processor._update_progress_counters(batch_result)

        assert batch_processor.total_successes == 8
        assert batch_processor.total_errors == 2


class TestPerformanceMetrics:
    """Test performance metrics and auto-tuning."""

    def test_update_performance_metrics(self):
        """Test performance metrics update."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        batch_result = BatchResult(
            batch_id="test",
            items_processed=10,
            items_successful=10,
            items_failed=0,
            processing_time=1.0,
            average_item_time=0.1,
            error_rate=0.0,
        )

        batch_processor._update_performance_metrics(batch_result)

        assert len(batch_processor.performance_history) == 1
        assert batch_processor.performance_history[0] == (10, 0.1)

    def test_performance_history_limit(self):
        """Test performance history is limited to 20 entries."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        # Add 25 entries
        for i in range(25):
            batch_result = BatchResult(
                batch_id=f"test_{i}",
                items_processed=10,
                items_successful=10,
                items_failed=0,
                processing_time=1.0,
                average_item_time=0.1,
                error_rate=0.0,
            )
            batch_processor._update_performance_metrics(batch_result)

        assert len(batch_processor.performance_history) == 20

    def test_auto_tune_batch_size_insufficient_data(self):
        """Test auto-tuning with insufficient data."""
        processor = MockProcessor()
        config = BatchConfig(optimal_batch_size=50)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add only 2 entries (need at least 3)
        for i in range(2):
            batch_result = BatchResult(
                batch_id=f"test_{i}",
                items_processed=50,
                items_successful=50,
                items_failed=0,
                processing_time=5.0,
                average_item_time=0.1,
                error_rate=0.0,
            )
            batch_processor._update_performance_metrics(batch_result)

        original_size = batch_processor.current_batch_size
        batch_processor._auto_tune_batch_size()

        # Should not change with insufficient data
        assert batch_processor.current_batch_size == original_size

    def test_auto_tune_batch_size_with_history(self):
        """Test auto-tuning with sufficient data."""
        processor = MockProcessor()
        config = BatchConfig(
            min_batch_size=10,
            max_batch_size=200,
            optimal_batch_size=50,
            auto_tune_batch_size=True,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add entries with different batch sizes
        for i in range(10):
            batch_size = 30  # Consistent size for samples
            batch_result = BatchResult(
                batch_id=f"test_{i}",
                items_processed=batch_size,
                items_successful=batch_size,
                items_failed=0,
                processing_time=batch_size * 0.05,
                average_item_time=0.05,
                error_rate=0.0,
            )
            batch_processor._update_performance_metrics(batch_result)

        batch_processor._auto_tune_batch_size()
        # Should find optimal batch size based on performance

    def test_adapt_concurrency_limits_insufficient_data(self):
        """Test concurrency adaptation with insufficient data."""
        processor = MockProcessor()
        config = BatchConfig(async_concurrency_limit=50)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Only 2 entries
        for i in range(2):
            batch_processor.performance_history.append((10, 0.1))

        original_limit = batch_processor.current_concurrency_limit
        batch_processor._adapt_concurrency_limits()

        assert batch_processor.current_concurrency_limit == original_limit

    def test_adapt_concurrency_limits_slow_performance(self):
        """Test concurrency adaptation for slow performance."""
        processor = MockProcessor()
        config = BatchConfig(async_concurrency_limit=50)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add entries with slow performance
        for i in range(5):
            batch_processor.performance_history.append((10, 6.0))  # > 5.0

        batch_processor._adapt_concurrency_limits()

        # Should decrease concurrency
        assert batch_processor.current_concurrency_limit < 50

    def test_adapt_concurrency_limits_fast_performance(self):
        """Test concurrency adaptation for fast performance."""
        processor = MockProcessor()
        config = BatchConfig(async_concurrency_limit=50)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add entries with fast performance
        for i in range(5):
            batch_processor.performance_history.append((10, 0.5))  # < 1.0

        batch_processor._adapt_concurrency_limits()

        # Should increase concurrency
        assert batch_processor.current_concurrency_limit > 50


class TestRetryLogic:
    """Test retry logic for failed batches."""

    def test_retry_failed_batches(self):
        """Test retrying failed batches."""
        processor = MockProcessor()
        config = BatchConfig(
            min_batch_size=5,
            retry_failed_batches=True,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add a failed batch
        batch_processor.failed_batches.append(
            ("failed_batch_1", ["url1", "url2", "url3"], Exception("Test error"))
        )

        results = batch_processor._retry_failed_batches()

        assert len(results) > 0


class TestProcessingStatistics:
    """Test processing statistics collection."""

    def test_get_processing_statistics_empty(self):
        """Test statistics with no processed batches."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        stats = batch_processor.get_processing_statistics()

        assert stats["total_batches"] == 0
        assert stats["total_items"] == 0
        assert stats["success_rate"] == 0

    def test_get_processing_statistics_with_batches(self):
        """Test statistics with processed batches."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        # Add completed batches
        batch_processor.completed_batches.append(
            BatchResult(
                batch_id="batch_1",
                items_processed=10,
                items_successful=8,
                items_failed=2,
                processing_time=1.0,
                average_item_time=0.1,
                error_rate=0.2,
            )
        )
        batch_processor.completed_batches.append(
            BatchResult(
                batch_id="batch_2",
                items_processed=10,
                items_successful=10,
                items_failed=0,
                processing_time=1.0,
                average_item_time=0.1,
                error_rate=0.0,
            )
        )

        stats = batch_processor.get_processing_statistics()

        assert stats["total_batches"] == 2
        assert stats["total_items"] == 20
        assert stats["successful_items"] == 18
        assert stats["success_rate"] == 0.9

    def test_get_processing_statistics_with_cost_tracking(self):
        """Test statistics with cost tracking enabled."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add completed batch with cost
        batch_processor.completed_batches.append(
            BatchResult(
                batch_id="batch_1",
                items_processed=10,
                items_successful=10,
                items_failed=0,
                processing_time=1.0,
                average_item_time=0.1,
                error_rate=0.0,
                actual_cost=0.01,
            )
        )
        batch_processor.record_batch_cost("batch_1", 0.01)

        stats = batch_processor.get_processing_statistics()

        assert "cost_tracking" in stats
        assert stats["total_batch_cost"] == 0.01


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        processor = MockProcessor()
        config = BatchConfig(
            enable_cost_tracking=True,
            optimal_batch_size=100,
            async_concurrency_limit=50,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add some state
        batch_processor.processing_queue.put(("batch_1", ["item1"]))
        batch_processor.results_queue.put("result1")
        batch_processor.completed_batches.append(
            BatchResult(
                batch_id="test",
                items_processed=1,
                items_successful=1,
                items_failed=0,
                processing_time=0.1,
                average_item_time=0.1,
                error_rate=0.0,
            )
        )
        batch_processor.failed_batches.append(("failed", [], Exception("err")))
        batch_processor.performance_history.append((10, 0.1))
        batch_processor.total_session_cost = 1.0
        batch_processor.batch_cost_history.append(("batch_1", 0.5))
        batch_processor.cost_estimates["test"] = CostBreakdown(
            operation_type="test",
            batch_size=10,
            estimated_cost_per_item=0.001,
            total_estimated_cost=0.01,
        )
        batch_processor.rate_limit_tracker["domain.com"] = time.time()
        batch_processor.current_batch_size = 200
        batch_processor.current_concurrency_limit = 100

        batch_processor.reset()

        assert batch_processor.processing_queue.empty()
        assert batch_processor.results_queue.empty()
        assert len(batch_processor.completed_batches) == 0
        assert len(batch_processor.failed_batches) == 0
        assert len(batch_processor.performance_history) == 0
        assert batch_processor.total_session_cost == 0.0
        assert len(batch_processor.batch_cost_history) == 0
        assert len(batch_processor.cost_estimates) == 0
        assert len(batch_processor.rate_limit_tracker) == 0
        assert batch_processor.current_batch_size == 100  # Reset to optimal
        assert batch_processor.current_concurrency_limit == 50  # Reset to config


class TestAsyncProcessing:
    """Test async processing functionality."""

    @pytest.mark.asyncio
    async def test_initialize_async_components(self):
        """Test async component initialization."""
        processor = MockAsyncProcessor()
        config = BatchConfig(async_concurrency_limit=20)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        await batch_processor._initialize_async_components()

        assert batch_processor.async_semaphore is not None
        assert batch_processor.async_session is not None

        # Cleanup
        await batch_processor._cleanup_async_components()

    @pytest.mark.asyncio
    async def test_cleanup_async_components(self):
        """Test async component cleanup."""
        processor = MockAsyncProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        await batch_processor._initialize_async_components()
        await batch_processor._cleanup_async_components()

        assert batch_processor.async_session is None
        assert len(batch_processor.domain_semaphores) == 0

    @pytest.mark.asyncio
    async def test_async_process_single_batch(self):
        """Test async single batch processing."""
        processor = MockAsyncProcessor()
        config = BatchConfig(enable_async_processing=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        semaphore = asyncio.Semaphore(2)
        items = [f"https://example{i}.com" for i in range(5)]

        result = await batch_processor._async_process_single_batch(
            semaphore, "test_batch", items
        )

        assert result.batch_id == "test_batch"
        assert result.items_processed == 5

    @pytest.mark.asyncio
    async def test_async_process_single_batch_error(self):
        """Test async single batch processing with error."""
        processor = MockAsyncProcessor()
        processor.async_validate_batch = AsyncMock(side_effect=Exception("Test error"))
        config = BatchConfig(enable_async_processing=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        semaphore = asyncio.Semaphore(2)
        items = [f"https://example{i}.com" for i in range(5)]

        result = await batch_processor._async_process_single_batch(
            semaphore, "test_batch", items
        )

        assert result.items_failed == 5
        assert result.error_rate == 1.0

    @pytest.mark.asyncio
    async def test_apply_domain_rate_limiting(self):
        """Test domain-specific rate limiting."""
        processor = MockAsyncProcessor()
        config = BatchConfig(rate_limit_respect=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        url = "https://google.com/search"
        await batch_processor._apply_domain_rate_limiting(url)

        assert "google.com" in batch_processor.domain_semaphores
        assert "google.com" in batch_processor.rate_limit_tracker

    @pytest.mark.asyncio
    async def test_apply_domain_rate_limiting_disabled(self):
        """Test rate limiting when disabled."""
        processor = MockAsyncProcessor()
        config = BatchConfig(rate_limit_respect=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        url = "https://google.com/search"
        await batch_processor._apply_domain_rate_limiting(url)

        # Should not create any rate limiting structures
        assert len(batch_processor.domain_semaphores) == 0

    @pytest.mark.asyncio
    async def test_async_retry_failed_batches(self):
        """Test async retry of failed batches."""
        processor = MockAsyncProcessor()
        config = BatchConfig(
            min_batch_size=2,
            max_concurrent_batches=2,
        )
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        # Add failed batches
        batch_processor.failed_batches.append(
            ("failed_batch", ["url1", "url2", "url3"], Exception("Test error"))
        )

        results = await batch_processor._async_retry_failed_batches()

        assert len(results) > 0


class TestThreadSafety:
    """Test thread safety of batch processor."""

    def test_concurrent_cost_recording(self):
        """Test concurrent access to cost recording."""
        processor = MockProcessor()
        config = BatchConfig(enable_cost_tracking=True)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        def record_cost(thread_id):
            for i in range(10):
                batch_processor.record_batch_cost(f"batch_{thread_id}_{i}", 0.01)

        threads = []
        for i in range(5):
            t = threading.Thread(target=record_cost, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have all 50 records
        assert len(batch_processor.batch_cost_history) == 50

    def test_concurrent_performance_updates(self):
        """Test concurrent performance metric updates."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

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
                batch_processor._update_performance_metrics(batch_result)

        threads = []
        for i in range(5):
            t = threading.Thread(target=update_metrics, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # History should be limited to 20
        assert len(batch_processor.performance_history) <= 20


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_items_list(self):
        """Test processing with empty items list."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        items = []
        result = batch_processor.add_items(items)

        assert result is True
        assert batch_processor.total_items == 0

    def test_process_all_with_empty_queue(self):
        """Test process_all with empty queue."""
        processor = MockProcessor()
        config = BatchConfig(enable_async_processing=False)
        batch_processor = EnhancedBatchProcessor(processor=processor, config=config)

        results = batch_processor.process_all()

        assert len(results) == 0

    def test_batch_result_with_zero_items(self):
        """Test handling batch result with zero items."""
        processor = MockProcessor()
        batch_processor = EnhancedBatchProcessor(processor=processor)

        batch_result = BatchResult(
            batch_id="empty_batch",
            items_processed=0,
            items_successful=0,
            items_failed=0,
            processing_time=0.0,
            average_item_time=0.0,
            error_rate=0.0,
        )

        # Should not crash
        batch_processor._update_progress_counters(batch_result)
        batch_processor._update_performance_metrics(batch_result)

    def test_progress_callback_exception(self):
        """Test handling exception in progress callback."""
        processor = MockProcessor()

        def bad_callback(msg):
            raise Exception("Callback error")

        batch_processor = EnhancedBatchProcessor(
            processor=processor,
            progress_callback=bad_callback,
        )

        items = [f"https://example{i}.com" for i in range(5)]
        batch_processor.add_items(items)

        # Should not crash even with bad callback
        # The _emit_progress_update catches exceptions
        batch_processor._emit_progress_update("test", "batch_1", 1, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
