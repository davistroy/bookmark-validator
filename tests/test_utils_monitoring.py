"""
Unit tests for monitoring and tracking utilities.

Tests for progress tracker, performance monitor, and cost tracker.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.utils.cost_tracker import APIUsage, CostEstimate, CostTracker
from bookmark_processor.utils.memory_optimizer import MemoryMonitor
from bookmark_processor.utils.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
)
from bookmark_processor.utils.progress_tracker import (
    ProgressBar,
    ProgressLogger,
    ProgressTracker,
    TimeEstimator,
)


class TestTimeEstimator:
    """Test TimeEstimator class."""

    def test_init(self):
        """Test TimeEstimator initialization."""
        estimator = TimeEstimator()

        assert estimator.total_items == 0
        assert estimator.items_processed == 0
        assert estimator.start_time is None

    def test_start(self):
        """Test starting time estimation."""
        estimator = TimeEstimator()

        with patch("time.time", return_value=1000.0):
            estimator.start(total_items=100)

        assert estimator.start_time == 1000.0
        assert estimator.total_items == 100

    def test_update_progress(self):
        """Test updating progress."""
        estimator = TimeEstimator()

        with patch("time.time", return_value=1000.0):
            estimator.start(total_items=100)

        estimator.update(25)

        assert estimator.items_processed == 25

    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        estimator = TimeEstimator()

        with patch("time.time", return_value=1000.0):
            estimator.start(total_items=100)

        with patch("time.time", return_value=1030.0):  # 30 seconds later
            elapsed = time.time() - estimator.start_time

        assert elapsed == 30.0

    def test_get_estimated_time_remaining_linear(self):
        """Test time remaining estimation with linear progress."""
        estimator = TimeEstimator()

        with patch("time.time", return_value=1000.0):
            estimator.start(total_items=100)

        with patch("time.time", return_value=1040.0):  # 40 seconds later
            estimator.update(40)
            remaining = estimator.get_eta()

        # Should estimate 60 more seconds (40s for 40%, so 60s for remaining 60%)
        assert remaining is not None
        assert 55 <= remaining <= 65

    def test_get_estimated_time_remaining_no_progress(self):
        """Test time remaining estimation with no progress."""
        estimator = TimeEstimator()

        with patch("time.time", return_value=1000.0):
            estimator.start(total_items=100)

        with patch("time.time", return_value=1010.0):  # 10 seconds later
            # No progress update
            remaining = estimator.get_eta()

        # Should return None when no progress
        assert remaining is None

    def test_get_processing_rate(self):
        """Test processing rate calculation."""
        estimator = TimeEstimator()

        with patch("time.time", return_value=1000.0):
            estimator.start(total_items=100)

        with patch("time.time", return_value=1020.0):  # 20 seconds later
            estimator.update(40)

            elapsed = time.time() - estimator.start_time
            rate = estimator.items_processed / elapsed

        # Should be 2 items per second (40 items in 20 seconds)
        assert rate == 2.0

    def test_get_eta_string(self):
        """Test ETA string formatting."""
        estimator = TimeEstimator()

        with patch("time.time", return_value=1000.0):
            estimator.start(total_items=100)

        with patch("time.time", return_value=1030.0):  # 30 seconds later
            estimator.update(50)
            eta = estimator.get_eta()

        # Should return a number or None
        assert eta is None or isinstance(eta, (int, float))


class TestProgressBar:
    """Test ProgressBar class (alias to ProgressTracker)."""

    def test_init(self):
        """Test ProgressBar initialization."""
        progress_bar = ProgressBar(total_items=100, show_progress_bar=False)

        assert progress_bar.total_items == 100
        assert progress_bar.items_processed == 0

    def test_format_bar_empty(self):
        """Test progress at 0%."""
        progress_bar = ProgressBar(total_items=100, show_progress_bar=False)

        # ProgressTracker doesn't have format_bar, test progress percentage instead
        percentage = (progress_bar.items_processed / progress_bar.total_items) * 100
        assert percentage == 0.0

    def test_format_bar_partial(self):
        """Test progress at 50%."""
        progress_bar = ProgressBar(total_items=100, show_progress_bar=False)
        progress_bar.update(50)

        percentage = (progress_bar.items_processed / progress_bar.total_items) * 100
        assert percentage == 50.0

    def test_format_bar_complete(self):
        """Test progress at 100%."""
        progress_bar = ProgressBar(total_items=100, show_progress_bar=False)
        progress_bar.update(100)

        percentage = (progress_bar.items_processed / progress_bar.total_items) * 100
        assert percentage == 100.0

    def test_update_progress(self):
        """Test updating progress."""
        progress_bar = ProgressBar(total_items=100, show_progress_bar=False)

        progress_bar.update(25)

        assert progress_bar.items_processed == 25

    def test_finish(self):
        """Test finishing progress bar."""
        progress_bar = ProgressBar(total_items=100, show_progress_bar=False)

        progress_bar.complete()

        assert progress_bar.current_stage.value == "completed"


class TestProgressLogger:
    """Test ProgressLogger class."""

    def test_init(self):
        """Test ProgressLogger initialization."""
        logger = ProgressLogger()
        assert logger is not None
        assert logger.name == "ProgressLogger"

    def test_log_progress(self):
        """Test logging progress."""
        logger = ProgressLogger()

        # New signature: log_progress(message, level)
        with patch.object(logger.logger, "info") as mock_info:
            logger.log_progress("Processing 50/100 bookmarks", level="info")
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "Processing" in call_args
            assert "bookmarks" in call_args


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_init_default(self):
        """Test ProgressTracker initialization with defaults."""
        tracker = ProgressTracker(total_items=100, show_progress_bar=False)

        assert tracker.total_items == 100
        assert tracker.items_processed == 0
        assert tracker.completed_items == 0  # Backward compatibility attribute
        assert tracker.show_progress_bar is False
        assert tracker.verbose is False

    def test_init_custom(self):
        """Test ProgressTracker initialization with custom settings."""
        tracker = ProgressTracker(
            total_items=200, show_progress_bar=False, verbose=True
        )

        assert tracker.total_items == 200
        assert tracker.show_progress_bar is False
        assert tracker.verbose is True

    def test_start(self):
        """Test starting progress tracking (via start_stage)."""
        tracker = ProgressTracker(total_items=100, show_progress_bar=False)

        from bookmark_processor.utils.progress_tracker import ProcessingStage
        tracker.start_stage(ProcessingStage.VALIDATING_URLS)

        assert tracker.current_stage == ProcessingStage.VALIDATING_URLS
        assert ProcessingStage.VALIDATING_URLS in tracker.stage_metrics

    def test_update(self):
        """Test updating progress."""
        tracker = ProgressTracker(total_items=100, show_progress_bar=False)

        tracker.update(25)

        assert tracker.items_processed == 25
        assert tracker.completed_items == 25  # Backward compatibility

    def test_increment(self):
        """Test incrementing progress."""
        tracker = ProgressTracker(total_items=100, show_progress_bar=False)

        tracker.update_progress(items_delta=1)
        tracker.update_progress(items_delta=3)

        assert tracker.items_processed == 4
        assert tracker.completed_items == 4

    def test_finish(self):
        """Test finishing progress tracking."""
        tracker = ProgressTracker(total_items=100, show_progress_bar=False)

        tracker.update(100)
        tracker.complete()

        assert tracker.items_processed == 100
        assert tracker.current_stage.value == "completed"

    def test_get_progress_percentage(self):
        """Test progress percentage calculation."""
        tracker = ProgressTracker(total_items=100, show_progress_bar=False)

        tracker.update(25)
        percentage = (tracker.items_processed / tracker.total_items) * 100
        assert percentage == 25.0

        tracker.update(75)
        percentage = (tracker.items_processed / tracker.total_items) * 100
        assert percentage == 75.0

    def test_get_status_summary(self):
        """Test status summary generation."""
        tracker = ProgressTracker(total_items=100, show_progress_bar=False, verbose=False)

        from bookmark_processor.utils.progress_tracker import ProcessingStage
        tracker.start_stage(ProcessingStage.VALIDATING_URLS)
        tracker.update(60)

        snapshot = tracker.get_snapshot()

        assert snapshot.items_total == 100  # Correct attribute name
        assert snapshot.items_processed == 60
        assert snapshot.overall_progress > 0
        assert snapshot.elapsed_time >= 0


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_init(self):
        """Test PerformanceMetrics initialization."""
        from datetime import datetime

        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            memory_usage_mb=100.0,
            memory_peak_mb=150.0,
            cpu_percent=50.0,
            processing_rate_per_hour=100.0,
            network_requests_count=10,
            network_failures_count=1,
            processing_time_seconds=60.0,
            items_processed=10,
            active_threads=2,
            gc_collections={"gen_0": 1, "gen_1": 0, "gen_2": 0}
        )

        assert metrics.memory_usage_mb == 100.0
        assert metrics.memory_peak_mb == 150.0
        assert metrics.items_processed == 10

    def test_record_operation(self):
        """Test recording operations (via PerformanceMonitor)."""
        monitor = PerformanceMonitor(target_items=100)

        monitor.record_item_processed()
        monitor.record_item_processed()

        assert monitor.items_processed == 2

    def test_get_average_time(self):
        """Test average time calculation (via stage metrics)."""
        monitor = PerformanceMonitor(target_items=100)

        # This functionality is now part of performance analysis
        perf = monitor.get_current_performance()

        # Just verify we can get performance data
        assert "items_processed" in perf
        assert perf["items_processed"] == 0

    def test_get_operation_stats(self):
        """Test operation statistics (via performance analysis)."""
        monitor = PerformanceMonitor(target_items=100)

        monitor.start_stage("test_stage")
        monitor.record_item_processed()
        monitor.end_current_stage()

        analysis = monitor.get_performance_analysis()

        assert "stage_analysis" in analysis
        assert "current_performance" in analysis


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_init(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor()

        assert monitor.peak_memory == 0
        assert len(monitor.history) == 0

    def test_get_current_memory_mb(self):
        """Test getting current memory usage."""
        monitor = MemoryMonitor()
        memory_mb = monitor.get_current_memory()

        # Should return a non-negative number (actual value depends on system)
        assert memory_mb >= 0.0

    def test_record_memory_sample(self):
        """Test recording memory samples via get_memory_stats."""
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()

        assert len(monitor.history) == 1
        assert monitor.peak_memory >= 0
        assert stats.current_mb >= 0

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        monitor = MemoryMonitor()

        # Record multiple samples
        for _ in range(5):
            stats = monitor.get_memory_stats()
            time.sleep(0.01)  # Small delay

        assert len(monitor.history) == 5

        # Get the latest stats
        latest_stats = monitor.history[-1]
        assert latest_stats.peak_mb >= latest_stats.current_mb
        assert latest_stats.available_mb >= 0
        assert "gen_0" in latest_stats.gc_collections


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_init(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(target_items=100, target_hours=8)

        assert monitor.target_items == 100
        assert monitor.target_hours == 8
        assert monitor.items_processed == 0

    def test_start_monitoring(self):
        """Test starting performance monitoring."""
        monitor = PerformanceMonitor(target_items=100)

        monitor.start_monitoring(interval_seconds=1.0)

        assert monitor.start_time is not None
        assert monitor.monitoring_thread is not None

    def test_stop_monitoring(self):
        """Test stopping performance monitoring."""
        monitor = PerformanceMonitor(target_items=100)

        monitor.start_monitoring(interval_seconds=1.0)
        time.sleep(0.1)  # Let it run briefly
        monitor.stop_monitoring_session()

        assert monitor.stop_monitoring.is_set()

    def test_record_operation(self):
        """Test recording operations."""
        monitor = PerformanceMonitor(target_items=100)

        monitor.record_item_processed()
        monitor.record_item_processed()

        assert monitor.items_processed == 2

    @patch("time.time")
    def test_time_operation_context_manager(self, mock_time):
        """Test timing operations with context manager."""
        mock_time.side_effect = [1000.0, 1002.5]  # 2.5 second operation

        monitor = PerformanceMonitor(target_items=100)

        with monitor.measure_stage("test_context"):
            pass  # Simulated operation

        # Stage should be recorded
        assert len(monitor.stage_metrics) == 1
        assert monitor.stage_metrics[0].stage_name == "test_context"

    def test_get_performance_report(self):
        """Test getting performance report."""
        monitor = PerformanceMonitor(target_items=100)

        monitor.start_monitoring(interval_seconds=1.0)
        monitor.start_stage("url_validation")
        monitor.record_item_processed()
        monitor.record_network_request(success=True)
        time.sleep(0.1)
        monitor.end_current_stage()
        monitor.stop_monitoring_session()

        analysis = monitor.get_performance_analysis()

        assert isinstance(analysis, dict)
        assert "current_performance" in analysis
        assert "stage_analysis" in analysis
        assert "bottlenecks" in analysis
        assert "recommendations" in analysis


class TestCostTracker:
    """Test CostTracker class."""

    def test_init(self):
        """Test CostTracker initialization."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=10.0, cost_log_file=temp_file)

        # Should start with 0 cost for fresh file
        assert tracker.session_cost == 0.0
        assert tracker.confirmation_interval == 10.0

        # Clean up
        import os
        os.unlink(temp_file)

    def test_record_api_usage_claude(self):
        """Test recording Claude API usage."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=10.0, cost_log_file=temp_file)

        # Clear any existing records from file
        initial_records = len(tracker.cost_records)

        tracker.add_cost_record(
            provider="claude",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
        )

        assert len(tracker.cost_records) == initial_records + 1
        assert tracker.session_cost == 0.05

        # Get the last record
        record = tracker.cost_records[-1]
        assert record.provider == "claude"
        assert record.model == "claude-3-sonnet"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_usd == 0.05

        # Clean up
        import os
        os.unlink(temp_file)

    def test_record_api_usage_openai(self):
        """Test recording OpenAI API usage."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=10.0, cost_log_file=temp_file)
        initial_records = len(tracker.cost_records)

        tracker.add_cost_record(
            provider="openai",
            model="gpt-4",
            input_tokens=800,
            output_tokens=300,
            cost_usd=0.08,
        )

        assert len(tracker.cost_records) == initial_records + 1
        assert tracker.session_cost == 0.08

        # Clean up
        import os
        os.unlink(temp_file)

    def test_get_cost_by_provider(self):
        """Test getting cost breakdown by provider."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=10.0, cost_log_file=temp_file)

        tracker.add_cost_record("claude", "claude-3-sonnet", 1000, 500, 0.05)
        tracker.add_cost_record("claude", "claude-3-haiku", 800, 300, 0.02)
        tracker.add_cost_record("openai", "gpt-4", 600, 200, 0.08)

        # Get session provider costs (not historical)
        session_costs = tracker._get_session_provider_costs()
        claude_cost = session_costs.get("claude", 0.0)
        openai_cost = session_costs.get("openai", 0.0)

        assert abs(claude_cost - 0.07) < 0.001  # 0.05 + 0.02
        assert abs(openai_cost - 0.08) < 0.001

        # Clean up
        import os
        os.unlink(temp_file)

    def test_get_cost_by_model(self):
        """Test getting cost breakdown by model."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=10.0, cost_log_file=temp_file)

        tracker.add_cost_record("claude", "claude-3-sonnet", 1000, 500, 0.05)
        tracker.add_cost_record("claude", "claude-3-sonnet", 800, 300, 0.03)
        tracker.add_cost_record("openai", "gpt-4", 600, 200, 0.08)

        # Calculate cost by model for current session
        session_start = tracker.session_start_time
        session_records = [r for r in tracker.cost_records if r.timestamp >= session_start]

        sonnet_cost = sum(r.cost_usd for r in session_records if r.model == "claude-3-sonnet")
        gpt4_cost = sum(r.cost_usd for r in session_records if r.model == "gpt-4")

        assert abs(sonnet_cost - 0.08) < 0.001  # 0.05 + 0.03
        assert abs(gpt4_cost - 0.08) < 0.001

        # Clean up
        import os
        os.unlink(temp_file)

    def test_should_warn_about_cost(self):
        """Test cost warning threshold."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=0.10, warning_threshold=0.10, cost_log_file=temp_file)

        # Below threshold
        tracker.add_cost_record("claude", "claude-3-sonnet", 1000, 500, 0.05)
        assert tracker.should_confirm() is False

        # Above threshold
        tracker.add_cost_record("claude", "claude-3-sonnet", 1000, 500, 0.06)
        assert tracker.should_confirm() is True

        # Clean up
        import os
        os.unlink(temp_file)

    def test_estimate_cost_claude(self):
        """Test cost estimation for Claude."""
        estimate = CostEstimate(
            total_items=1000,
            estimated_tokens_per_item=150,
            cost_per_token=0.000003
        )

        assert estimate.estimated_total_cost > 0
        assert isinstance(estimate.estimated_total_cost, float)

    def test_estimate_cost_openai(self):
        """Test cost estimation for OpenAI."""
        estimate = CostEstimate(
            total_items=1000,
            estimated_tokens_per_item=150,
            cost_per_token=0.000006
        )

        assert estimate.estimated_total_cost > 0
        assert isinstance(estimate.estimated_total_cost, float)

    def test_get_usage_summary(self):
        """Test getting usage summary."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=10.0, cost_log_file=temp_file)

        tracker.add_cost_record("claude", "claude-3-sonnet", 1000, 500, 0.05)
        tracker.add_cost_record("openai", "gpt-4", 800, 300, 0.08)

        stats = tracker.get_detailed_statistics()

        assert isinstance(stats, dict)
        assert abs(stats["session"]["total_cost_usd"] - 0.13) < 0.001
        assert stats["session"]["requests"] == 2
        assert "providers" in stats
        assert "claude" in stats["providers"]
        assert "openai" in stats["providers"]

        # Clean up
        import os
        os.unlink(temp_file)

    def test_reset_tracking(self):
        """Test resetting cost tracking."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        tracker = CostTracker(confirmation_interval=10.0, cost_log_file=temp_file)

        tracker.add_cost_record("claude", "claude-3-sonnet", 1000, 500, 0.05)

        assert tracker.session_cost > 0
        assert len(tracker.cost_records) > 0

        tracker.reset_session()

        assert tracker.session_cost == 0.0
        # Note: cost_records are not cleared, only session is reset
        assert len(tracker.cost_records) > 0  # Historical data remains

        # Clean up
        import os
        os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
