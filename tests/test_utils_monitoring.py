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
        estimator = TimeEstimator(total_items=100)

        assert estimator.total_items == 100
        assert estimator.completed_items == 0
        assert estimator.start_time is None
        assert len(estimator.progress_history) == 0

    def test_start(self):
        """Test starting time estimation."""
        estimator = TimeEstimator(total_items=100)

        with patch("time.time", return_value=1000.0):
            estimator.start()

        assert estimator.start_time == 1000.0

    def test_update_progress(self):
        """Test updating progress."""
        estimator = TimeEstimator(total_items=100)

        with patch("time.time", return_value=1000.0):
            estimator.start()

        with patch("time.time", return_value=1010.0):  # 10 seconds later
            estimator.update_progress(25)  # 25% complete

        assert estimator.completed_items == 25
        assert len(estimator.progress_history) == 1

        # Check progress history entry
        entry = estimator.progress_history[0]
        assert entry["completed"] == 25
        assert entry["timestamp"] == 1010.0

    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        estimator = TimeEstimator(total_items=100)

        with patch("time.time", return_value=1000.0):
            estimator.start()

        with patch("time.time", return_value=1030.0):  # 30 seconds later
            elapsed = estimator.get_elapsed_time()

        assert elapsed == 30.0

    def test_get_estimated_time_remaining_linear(self):
        """Test time remaining estimation with linear progress."""
        estimator = TimeEstimator(total_items=100)

        with patch("time.time", return_value=1000.0):
            estimator.start()

        with patch("time.time", return_value=1040.0):  # 40 seconds later
            estimator.update_progress(40)  # 40% complete

        remaining = estimator.get_estimated_time_remaining()

        # Should estimate 60 more seconds (40s for 40%, so 60s for remaining 60%)
        assert 55 <= remaining <= 65

    def test_get_estimated_time_remaining_no_progress(self):
        """Test time remaining estimation with no progress."""
        estimator = TimeEstimator(total_items=100)

        with patch("time.time", return_value=1000.0):
            estimator.start()

        with patch("time.time", return_value=1010.0):  # 10 seconds later
            # No progress update
            remaining = estimator.get_estimated_time_remaining()

        # Should return None or a very large number when no progress
        assert remaining is None or remaining > 10000

    def test_get_processing_rate(self):
        """Test processing rate calculation."""
        estimator = TimeEstimator(total_items=100)

        with patch("time.time", return_value=1000.0):
            estimator.start()

        with patch("time.time", return_value=1020.0):  # 20 seconds later
            estimator.update_progress(40)  # 40 items complete

        rate = estimator.get_processing_rate()

        # Should be 2 items per second (40 items in 20 seconds)
        assert rate == 2.0

    def test_get_eta_string(self):
        """Test ETA string formatting."""
        estimator = TimeEstimator(total_items=100)

        with patch("time.time", return_value=1000.0):
            estimator.start()

        with patch("time.time", return_value=1030.0):  # 30 seconds later
            estimator.update_progress(50)  # 50% complete

        eta_string = estimator.get_eta_string()

        # Should contain time format
        assert isinstance(eta_string, str)
        assert any(char in eta_string for char in [":", "s", "m", "h"])


class TestProgressBar:
    """Test ProgressBar class."""

    def test_init(self):
        """Test ProgressBar initialization."""
        progress_bar = ProgressBar(total=100, width=50)

        assert progress_bar.total == 100
        assert progress_bar.width == 50
        assert progress_bar.current == 0

    def test_format_bar_empty(self):
        """Test formatting empty progress bar."""
        progress_bar = ProgressBar(total=100, width=20)

        bar_string = progress_bar.format_bar(0)

        assert "[" in bar_string
        assert "]" in bar_string
        assert "0%" in bar_string
        assert bar_string.count("=") == 0

    def test_format_bar_partial(self):
        """Test formatting partial progress bar."""
        progress_bar = ProgressBar(total=100, width=20)

        bar_string = progress_bar.format_bar(50)

        assert "[" in bar_string
        assert "]" in bar_string
        assert "50%" in bar_string
        assert "=" in bar_string

    def test_format_bar_complete(self):
        """Test formatting complete progress bar."""
        progress_bar = ProgressBar(total=100, width=20)

        bar_string = progress_bar.format_bar(100)

        assert "[" in bar_string
        assert "]" in bar_string
        assert "100%" in bar_string
        assert bar_string.count("=") > 15  # Most characters should be filled

    def test_update_progress(self):
        """Test updating progress."""
        progress_bar = ProgressBar(total=100)

        with patch("sys.stdout") as mock_stdout:
            progress_bar.update(25)

        assert progress_bar.current == 25
        # Should have written to stdout
        mock_stdout.write.assert_called()

    def test_finish(self):
        """Test finishing progress bar."""
        progress_bar = ProgressBar(total=100)

        with patch("sys.stdout") as mock_stdout:
            progress_bar.finish()

        # Should write newline
        mock_stdout.write.assert_called()


class TestProgressLogger:
    """Test ProgressLogger class."""

    def test_init(self):
        """Test ProgressLogger initialization."""
        logger = ProgressLogger()
        assert logger is not None

    def test_log_progress(self):
        """Test logging progress."""
        logger = ProgressLogger()

        with patch(
            "bookmark_processor.utils.logging_setup.get_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            logger.log_progress(50, 100, "Processing bookmarks")

            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "50%" in call_args
            assert "Processing bookmarks" in call_args


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_init_default(self):
        """Test ProgressTracker initialization with defaults."""
        tracker = ProgressTracker(total_items=100)

        assert tracker.total_items == 100
        assert tracker.completed_items == 0
        assert tracker.show_bar is True
        assert tracker.show_logs is True

    def test_init_custom(self):
        """Test ProgressTracker initialization with custom settings."""
        tracker = ProgressTracker(
            total_items=200, description="Custom task", show_bar=False, show_logs=False
        )

        assert tracker.total_items == 200
        assert tracker.description == "Custom task"
        assert tracker.show_bar is False
        assert tracker.show_logs is False

    def test_start(self):
        """Test starting progress tracking."""
        tracker = ProgressTracker(total_items=100)

        tracker.start()

        assert tracker.time_estimator.start_time is not None
        assert tracker.started is True

    def test_update(self):
        """Test updating progress."""
        tracker = ProgressTracker(total_items=100, show_bar=False, show_logs=False)

        tracker.start()
        tracker.update(25)

        assert tracker.completed_items == 25
        assert tracker.time_estimator.completed_items == 25

    def test_increment(self):
        """Test incrementing progress."""
        tracker = ProgressTracker(total_items=100, show_bar=False, show_logs=False)

        tracker.start()
        tracker.increment()
        tracker.increment(3)

        assert tracker.completed_items == 4

    def test_finish(self):
        """Test finishing progress tracking."""
        tracker = ProgressTracker(total_items=100, show_bar=False, show_logs=False)

        tracker.start()
        tracker.update(100)
        tracker.finish()

        assert tracker.completed_items == 100
        assert tracker.finished is True

    def test_get_progress_percentage(self):
        """Test progress percentage calculation."""
        tracker = ProgressTracker(total_items=100)

        tracker.update(25)
        assert tracker.get_progress_percentage() == 25.0

        tracker.update(75)
        assert tracker.get_progress_percentage() == 75.0

    def test_get_status_summary(self):
        """Test status summary generation."""
        tracker = ProgressTracker(total_items=100, description="Test task")

        tracker.start()
        tracker.update(60)

        summary = tracker.get_status_summary()

        assert isinstance(summary, dict)
        assert summary["description"] == "Test task"
        assert summary["total_items"] == 100
        assert summary["completed_items"] == 60
        assert summary["progress_percentage"] == 60.0
        assert "elapsed_time" in summary
        assert "estimated_remaining" in summary


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""

    def test_init(self):
        """Test PerformanceMetrics initialization."""
        metrics = PerformanceMetrics()

        assert metrics.start_time is None
        assert metrics.end_time is None
        assert metrics.peak_memory_mb == 0
        assert metrics.total_operations == 0

    def test_record_operation(self):
        """Test recording operations."""
        metrics = PerformanceMetrics()

        metrics.record_operation("url_validation", 1.5)
        metrics.record_operation("ai_processing", 3.2)
        metrics.record_operation("url_validation", 1.8)

        assert metrics.total_operations == 3
        assert "url_validation" in metrics.operation_times
        assert "ai_processing" in metrics.operation_times
        assert len(metrics.operation_times["url_validation"]) == 2
        assert len(metrics.operation_times["ai_processing"]) == 1

    def test_get_average_time(self):
        """Test average time calculation."""
        metrics = PerformanceMetrics()

        metrics.record_operation("test", 1.0)
        metrics.record_operation("test", 2.0)
        metrics.record_operation("test", 3.0)

        avg_time = metrics.get_average_time("test")
        assert avg_time == 2.0

        # Non-existent operation
        avg_time = metrics.get_average_time("nonexistent")
        assert avg_time == 0.0

    def test_get_operation_stats(self):
        """Test operation statistics."""
        metrics = PerformanceMetrics()

        metrics.record_operation("test", 1.0)
        metrics.record_operation("test", 2.0)
        metrics.record_operation("test", 3.0)

        stats = metrics.get_operation_stats("test")

        assert stats["count"] == 3
        assert stats["total_time"] == 6.0
        assert stats["average_time"] == 2.0
        assert stats["min_time"] == 1.0
        assert stats["max_time"] == 3.0


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_init(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor()

        assert monitor.peak_memory_mb == 0
        assert len(monitor.memory_samples) == 0

    @patch("psutil.Process")
    def test_get_current_memory_mb(self, mock_process_class):
        """Test getting current memory usage."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        monitor = MemoryMonitor()
        memory_mb = monitor.get_current_memory_mb()

        assert memory_mb == 100.0

    @patch("psutil.Process")
    def test_record_memory_sample(self, mock_process_class):
        """Test recording memory samples."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 150 * 1024 * 1024  # 150 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        monitor = MemoryMonitor()
        monitor.record_memory_sample()

        assert len(monitor.memory_samples) == 1
        assert monitor.peak_memory_mb == 150.0

    @patch("psutil.Process")
    def test_get_memory_stats(self, mock_process_class):
        """Test getting memory statistics."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        # Mock different memory readings
        memory_readings = [100, 150, 120, 180, 90]  # MB
        mock_process.memory_info.side_effect = [
            Mock(rss=mb * 1024 * 1024) for mb in memory_readings
        ]

        monitor = MemoryMonitor()

        # Record samples
        for _ in memory_readings:
            monitor.record_memory_sample()

        stats = monitor.get_memory_stats()

        assert stats["peak_memory_mb"] == 180.0
        assert stats["current_memory_mb"] == 90.0
        assert stats["average_memory_mb"] == 128.0  # (100+150+120+180+90)/5
        assert stats["sample_count"] == 5


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_init(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()

        assert isinstance(monitor.metrics, PerformanceMetrics)
        assert isinstance(monitor.memory_monitor, MemoryMonitor)
        assert monitor.monitoring_active is False

    def test_start_monitoring(self):
        """Test starting performance monitoring."""
        monitor = PerformanceMonitor()

        monitor.start_monitoring()

        assert monitor.monitoring_active is True
        assert monitor.metrics.start_time is not None

    def test_stop_monitoring(self):
        """Test stopping performance monitoring."""
        monitor = PerformanceMonitor()

        monitor.start_monitoring()
        monitor.stop_monitoring()

        assert monitor.monitoring_active is False
        assert monitor.metrics.end_time is not None

    def test_record_operation(self):
        """Test recording operations."""
        monitor = PerformanceMonitor()

        monitor.record_operation("test_op", 2.5)

        assert monitor.metrics.total_operations == 1
        assert "test_op" in monitor.metrics.operation_times

    @patch("time.time")
    def test_time_operation_context_manager(self, mock_time):
        """Test timing operations with context manager."""
        mock_time.side_effect = [1000.0, 1002.5]  # 2.5 second operation

        monitor = PerformanceMonitor()

        with monitor.time_operation("test_context"):
            pass  # Simulated operation

        assert monitor.metrics.total_operations == 1
        avg_time = monitor.metrics.get_average_time("test_context")
        assert avg_time == 2.5

    def test_get_performance_report(self):
        """Test getting performance report."""
        monitor = PerformanceMonitor()

        monitor.start_monitoring()
        monitor.record_operation("url_validation", 1.0)
        monitor.record_operation("ai_processing", 3.0)
        monitor.stop_monitoring()

        report = monitor.get_performance_report()

        assert isinstance(report, dict)
        assert "total_operations" in report
        assert "total_time" in report
        assert "operations_per_second" in report
        assert "operation_breakdown" in report
        assert "memory_stats" in report


class TestCostTracker:
    """Test CostTracker class."""

    def test_init(self):
        """Test CostTracker initialization."""
        tracker = CostTracker()

        assert tracker.total_cost == 0.0
        assert len(tracker.api_usage) == 0
        assert tracker.cost_threshold == 10.0

    def test_record_api_usage_claude(self):
        """Test recording Claude API usage."""
        tracker = CostTracker()

        tracker.record_api_usage(
            provider="claude",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost=0.05,
        )

        assert len(tracker.api_usage) == 1
        assert tracker.total_cost == 0.05

        usage = tracker.api_usage[0]
        assert usage.provider == "claude"
        assert usage.model == "claude-3-sonnet"
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.cost == 0.05

    def test_record_api_usage_openai(self):
        """Test recording OpenAI API usage."""
        tracker = CostTracker()

        tracker.record_api_usage(
            provider="openai",
            model="gpt-4",
            input_tokens=800,
            output_tokens=300,
            cost=0.08,
        )

        assert len(tracker.api_usage) == 1
        assert tracker.total_cost == 0.08

    def test_get_cost_by_provider(self):
        """Test getting cost breakdown by provider."""
        tracker = CostTracker()

        tracker.record_api_usage("claude", "claude-3-sonnet", 1000, 500, 0.05)
        tracker.record_api_usage("claude", "claude-3-haiku", 800, 300, 0.02)
        tracker.record_api_usage("openai", "gpt-4", 600, 200, 0.08)

        claude_cost = tracker.get_cost_by_provider("claude")
        openai_cost = tracker.get_cost_by_provider("openai")

        assert claude_cost == 0.07  # 0.05 + 0.02
        assert openai_cost == 0.08

    def test_get_cost_by_model(self):
        """Test getting cost breakdown by model."""
        tracker = CostTracker()

        tracker.record_api_usage("claude", "claude-3-sonnet", 1000, 500, 0.05)
        tracker.record_api_usage("claude", "claude-3-sonnet", 800, 300, 0.03)
        tracker.record_api_usage("openai", "gpt-4", 600, 200, 0.08)

        sonnet_cost = tracker.get_cost_by_model("claude-3-sonnet")
        gpt4_cost = tracker.get_cost_by_model("gpt-4")

        assert sonnet_cost == 0.08  # 0.05 + 0.03
        assert gpt4_cost == 0.08

    def test_should_warn_about_cost(self):
        """Test cost warning threshold."""
        tracker = CostTracker(cost_threshold=0.10)

        # Below threshold
        tracker.record_api_usage("claude", "claude-3-sonnet", 1000, 500, 0.05)
        assert tracker.should_warn_about_cost() is False

        # Above threshold
        tracker.record_api_usage("claude", "claude-3-sonnet", 1000, 500, 0.06)
        assert tracker.should_warn_about_cost() is True

    def test_estimate_cost_claude(self):
        """Test cost estimation for Claude."""
        estimate = CostEstimate.estimate_claude_cost(
            model="claude-3-sonnet", input_tokens=1000, output_tokens=500
        )

        assert estimate > 0
        assert isinstance(estimate, float)

    def test_estimate_cost_openai(self):
        """Test cost estimation for OpenAI."""
        estimate = CostEstimate.estimate_openai_cost(
            model="gpt-4", input_tokens=1000, output_tokens=500
        )

        assert estimate > 0
        assert isinstance(estimate, float)

    def test_get_usage_summary(self):
        """Test getting usage summary."""
        tracker = CostTracker()

        tracker.record_api_usage("claude", "claude-3-sonnet", 1000, 500, 0.05)
        tracker.record_api_usage("openai", "gpt-4", 800, 300, 0.08)

        summary = tracker.get_usage_summary()

        assert isinstance(summary, dict)
        assert summary["total_cost"] == 0.13
        assert summary["total_requests"] == 2
        assert "provider_breakdown" in summary
        assert "model_breakdown" in summary
        assert summary["provider_breakdown"]["claude"] == 0.05
        assert summary["provider_breakdown"]["openai"] == 0.08

    def test_reset_tracking(self):
        """Test resetting cost tracking."""
        tracker = CostTracker()

        tracker.record_api_usage("claude", "claude-3-sonnet", 1000, 500, 0.05)

        assert tracker.total_cost > 0
        assert len(tracker.api_usage) > 0

        tracker.reset_tracking()

        assert tracker.total_cost == 0.0
        assert len(tracker.api_usage) == 0


if __name__ == "__main__":
    pytest.main([__file__])
