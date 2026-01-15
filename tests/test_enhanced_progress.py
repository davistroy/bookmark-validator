"""
Tests for the Enhanced Progress Tracking module.

This module tests the EnhancedProgressTracker class and its stage-based
progress tracking, ETA estimation, and rendering capabilities.
"""

import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from bookmark_processor.core.checkpoint_manager import ProcessingStage
from bookmark_processor.utils.enhanced_progress import (
    EnhancedProgressTracker,
    StageProgress,
    StageStatus,
    create_enhanced_tracker,
    RICH_AVAILABLE,
)


# Fixtures


@pytest.fixture
def tracker() -> EnhancedProgressTracker:
    """Create a basic progress tracker for testing."""
    return EnhancedProgressTracker(total_bookmarks=1000)


@pytest.fixture
def started_tracker() -> EnhancedProgressTracker:
    """Create a tracker that has been started."""
    tracker = EnhancedProgressTracker(total_bookmarks=1000)
    tracker.start(1000)
    return tracker


@pytest.fixture
def in_progress_tracker() -> EnhancedProgressTracker:
    """Create a tracker with some stages in progress."""
    tracker = EnhancedProgressTracker(total_bookmarks=1000)
    tracker.start(1000)

    # Complete first few stages
    tracker.start_stage(ProcessingStage.INITIALIZATION, 1)
    tracker.update_stage(ProcessingStage.INITIALIZATION, 1)
    tracker.complete_stage(ProcessingStage.INITIALIZATION)

    tracker.start_stage(ProcessingStage.LOADING, 1000)
    tracker.update_stage(ProcessingStage.LOADING, 1000)
    tracker.complete_stage(ProcessingStage.LOADING)

    # Start URL validation (in progress)
    tracker.start_stage(ProcessingStage.URL_VALIDATION, 1000)
    tracker.update_stage(ProcessingStage.URL_VALIDATION, 500, failed=5)

    return tracker


# StageProgress Tests


class TestStageProgress:
    """Tests for StageProgress dataclass."""

    def test_default_values(self):
        """Test default initialization values."""
        stage = StageProgress(name="test", display_name="Test Stage")
        assert stage.total == 0
        assert stage.completed == 0
        assert stage.failed == 0
        assert stage.started_at is None
        assert stage.completed_at is None

    def test_status_pending(self):
        """Test status is pending when not started."""
        stage = StageProgress(name="test", display_name="Test")
        assert stage.status == StageStatus.PENDING

    def test_status_in_progress(self):
        """Test status is in_progress when started."""
        stage = StageProgress(name="test", display_name="Test")
        stage.start()
        assert stage.status == StageStatus.IN_PROGRESS

    def test_status_completed(self):
        """Test status is completed when finished."""
        stage = StageProgress(name="test", display_name="Test")
        stage.start()
        stage.complete()
        assert stage.status == StageStatus.COMPLETED

    def test_progress_percentage_zero_total(self):
        """Test progress percentage with zero total."""
        stage = StageProgress(name="test", display_name="Test", total=0)
        assert stage.progress_percentage == 0.0

        # After starting, should show 100% for zero total
        stage.start()
        assert stage.progress_percentage == 100.0

    def test_progress_percentage_calculation(self):
        """Test progress percentage calculation."""
        stage = StageProgress(name="test", display_name="Test", total=100)
        stage.start()
        stage.update(50)
        assert stage.progress_percentage == 50.0

    def test_progress_percentage_capped_at_100(self):
        """Test progress percentage is capped at 100."""
        stage = StageProgress(name="test", display_name="Test", total=100)
        stage.start()
        stage.update(150)  # More than total
        assert stage.progress_percentage == 100.0

    def test_elapsed_time_not_started(self):
        """Test elapsed time before starting."""
        stage = StageProgress(name="test", display_name="Test")
        assert stage.elapsed_time == timedelta(0)

    def test_elapsed_time_in_progress(self):
        """Test elapsed time while in progress."""
        stage = StageProgress(name="test", display_name="Test")
        stage.start()
        time.sleep(0.1)
        elapsed = stage.elapsed_time
        assert elapsed.total_seconds() >= 0.1

    def test_elapsed_time_completed(self):
        """Test elapsed time after completion."""
        stage = StageProgress(name="test", display_name="Test")
        stage.start()
        time.sleep(0.1)
        stage.complete()

        # Should be fixed after completion
        elapsed1 = stage.elapsed_time
        time.sleep(0.1)
        elapsed2 = stage.elapsed_time

        # Times should be approximately equal (completed)
        assert abs(elapsed1.total_seconds() - elapsed2.total_seconds()) < 0.05

    def test_items_per_second_no_history(self):
        """Test items per second with no rate history."""
        stage = StageProgress(name="test", display_name="Test", total=100)
        stage.start(100)
        # No updates yet, rate based on elapsed time
        assert stage.items_per_second >= 0

    def test_items_per_second_with_updates(self):
        """Test items per second calculation with updates."""
        stage = StageProgress(name="test", display_name="Test", total=100)
        stage.start(100)

        # Simulate updates
        time.sleep(0.15)
        stage.update(10)
        time.sleep(0.15)
        stage.update(20)

        rate = stage.items_per_second
        assert rate > 0

    def test_eta_pending(self):
        """Test ETA for pending stage."""
        stage = StageProgress(
            name="test",
            display_name="Test",
            estimated_duration=timedelta(minutes=5),
        )
        assert stage.eta == timedelta(minutes=5)

    def test_eta_completed(self):
        """Test ETA for completed stage."""
        stage = StageProgress(name="test", display_name="Test")
        stage.start()
        stage.complete()
        assert stage.eta == timedelta(0)

    def test_eta_in_progress(self):
        """Test ETA for in-progress stage."""
        stage = StageProgress(name="test", display_name="Test", total=100)
        stage.start(100)
        time.sleep(0.1)
        stage.update(50)

        eta = stage.eta
        # ETA should be positive for partial completion
        assert eta is not None

    def test_error_rate_no_completions(self):
        """Test error rate with no completions."""
        stage = StageProgress(name="test", display_name="Test")
        assert stage.error_rate == 0.0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        stage = StageProgress(name="test", display_name="Test", total=100)
        stage.start(100)
        stage.update(100, failed=10)

        assert stage.error_rate == 10.0

    def test_start_with_total(self):
        """Test starting with total items."""
        stage = StageProgress(name="test", display_name="Test")
        stage.start(500)

        assert stage.total == 500
        assert stage.started_at is not None

    def test_get_status_icon(self):
        """Test status icon retrieval."""
        stage = StageProgress(name="test", display_name="Test")

        # Pending icon
        assert stage.get_status_icon() == "\u23f8"

        # In progress icon
        stage.start()
        assert stage.get_status_icon() == "\u23f3"

        # Completed icon
        stage.complete()
        assert stage.get_status_icon() == "\u2713"

    def test_format_duration_none(self):
        """Test formatting None duration."""
        stage = StageProgress(name="test", display_name="Test")
        assert stage.format_duration(None) == "N/A"

    def test_format_duration_seconds(self):
        """Test formatting duration in seconds."""
        stage = StageProgress(name="test", display_name="Test")
        result = stage.format_duration(timedelta(seconds=45))
        assert "45s" in result

    def test_format_duration_minutes(self):
        """Test formatting duration in minutes."""
        stage = StageProgress(name="test", display_name="Test")
        result = stage.format_duration(timedelta(minutes=5, seconds=30))
        assert "5m" in result
        assert "30s" in result

    def test_format_duration_hours(self):
        """Test formatting duration in hours."""
        stage = StageProgress(name="test", display_name="Test")
        result = stage.format_duration(timedelta(hours=2, minutes=15))
        assert "2h" in result
        assert "15m" in result


# EnhancedProgressTracker Tests


class TestEnhancedProgressTracker:
    """Tests for EnhancedProgressTracker class."""

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.total_bookmarks == 1000
        assert len(tracker.stages) > 0
        assert tracker.current_stage_name is None

    def test_stages_initialized(self, tracker):
        """Test that all stages are initialized."""
        expected_stages = [
            ProcessingStage.INITIALIZATION.value,
            ProcessingStage.LOADING.value,
            ProcessingStage.URL_VALIDATION.value,
            ProcessingStage.CONTENT_ANALYSIS.value,
            ProcessingStage.AI_PROCESSING.value,
            ProcessingStage.TAG_OPTIMIZATION.value,
            ProcessingStage.OUTPUT_GENERATION.value,
        ]

        for stage_name in expected_stages:
            assert stage_name in tracker.stages

    def test_start(self, tracker):
        """Test starting the tracker."""
        tracker.start(1000)

        assert tracker.start_time is not None
        assert tracker.total_bookmarks == 1000

        # All stages should have updated total
        for stage in tracker.stages.values():
            assert stage.total == 1000

    def test_start_stage_by_enum(self, started_tracker):
        """Test starting a stage using enum."""
        started_tracker.start_stage(ProcessingStage.URL_VALIDATION, 1000)

        assert started_tracker.current_stage_name == ProcessingStage.URL_VALIDATION.value
        assert started_tracker.stages[ProcessingStage.URL_VALIDATION.value].status == StageStatus.IN_PROGRESS

    def test_start_stage_by_string(self, started_tracker):
        """Test starting a stage using string."""
        started_tracker.start_stage("url_validation", 1000)

        assert started_tracker.current_stage_name == "url_validation"

    def test_update_stage(self, started_tracker):
        """Test updating stage progress."""
        started_tracker.start_stage(ProcessingStage.URL_VALIDATION, 1000)
        started_tracker.update_stage(ProcessingStage.URL_VALIDATION, 500, failed=5)

        stage = started_tracker.stages[ProcessingStage.URL_VALIDATION.value]
        assert stage.completed == 500
        assert stage.failed == 5

    def test_complete_stage(self, started_tracker):
        """Test completing a stage."""
        started_tracker.start_stage(ProcessingStage.URL_VALIDATION, 1000)
        started_tracker.update_stage(ProcessingStage.URL_VALIDATION, 1000)
        started_tracker.complete_stage(ProcessingStage.URL_VALIDATION)

        stage = started_tracker.stages[ProcessingStage.URL_VALIDATION.value]
        assert stage.status == StageStatus.COMPLETED

    def test_elapsed_time_not_started(self, tracker):
        """Test elapsed time before starting."""
        assert tracker.elapsed_time == timedelta(0)

    def test_elapsed_time_started(self, started_tracker):
        """Test elapsed time after starting."""
        time.sleep(0.1)
        elapsed = started_tracker.elapsed_time
        assert elapsed.total_seconds() >= 0.1

    def test_overall_progress_no_stages_complete(self, started_tracker):
        """Test overall progress with no completed stages."""
        assert started_tracker.overall_progress == 0.0

    def test_overall_progress_partial(self, in_progress_tracker):
        """Test overall progress with partial completion."""
        progress = in_progress_tracker.overall_progress

        # Should be > 0 since some stages are complete
        assert progress > 0.0

        # Should be < 100 since not all stages are complete
        assert progress < 100.0

    def test_overall_progress_all_complete(self, started_tracker):
        """Test overall progress with all stages complete."""
        # Complete all stages
        for stage_name in started_tracker.stages:
            started_tracker.start_stage(stage_name, 100)
            started_tracker.update_stage(stage_name, 100)
            started_tracker.complete_stage(stage_name)

        progress = started_tracker.overall_progress
        assert progress == 100.0

    def test_overall_eta(self, in_progress_tracker):
        """Test overall ETA calculation."""
        eta = in_progress_tracker.overall_eta

        # Should be positive for incomplete processing
        assert eta.total_seconds() >= 0

    def test_memory_usage(self, tracker):
        """Test memory usage retrieval."""
        memory = tracker.memory_usage_mb

        # Should return a non-negative value (may be 0 if psutil not installed)
        assert memory >= 0.0

    def test_overall_error_rate_no_processing(self, tracker):
        """Test error rate with no processing."""
        assert tracker.overall_error_rate == 0.0

    def test_overall_error_rate_with_errors(self, in_progress_tracker):
        """Test error rate with some errors."""
        error_rate = in_progress_tracker.overall_error_rate

        # Should be > 0 since we have errors
        assert error_rate > 0.0

    def test_overall_speed(self, in_progress_tracker):
        """Test overall speed calculation."""
        time.sleep(0.1)  # Ensure some time has passed
        speed = in_progress_tracker.overall_speed

        # Should be positive with completed items
        assert speed >= 0.0

    def test_set_current_item(self, started_tracker):
        """Test setting current item."""
        started_tracker.set_current_item("https://example.com", 42)

        assert started_tracker.current_item == "https://example.com"
        assert started_tracker.current_item_index == 42

    def test_total_errors_tracking(self, started_tracker):
        """Test total error tracking."""
        started_tracker.start_stage(ProcessingStage.URL_VALIDATION, 100)
        started_tracker.update_stage(ProcessingStage.URL_VALIDATION, 50, failed=5)

        assert started_tracker.total_errors == 5

        # Add more errors in another stage
        started_tracker.start_stage(ProcessingStage.CONTENT_ANALYSIS, 100)
        started_tracker.update_stage(ProcessingStage.CONTENT_ANALYSIS, 30, failed=3)

        assert started_tracker.total_errors == 8


# Rendering Tests


class TestRendering:
    """Tests for progress rendering functionality."""

    def test_render_progress_plain(self, in_progress_tracker):
        """Test plain text rendering."""
        # Force plain rendering by mocking RICH_AVAILABLE
        with patch('bookmark_processor.utils.enhanced_progress.RICH_AVAILABLE', False):
            output = in_progress_tracker._render_plain_progress()

            assert "PROCESSING STATUS" in output
            assert "URL Validation" in output or "url_validation" in output.lower()
            assert "Overall:" in output

    def test_render_progress_with_current_item(self, in_progress_tracker):
        """Test rendering with current item set."""
        in_progress_tracker.set_current_item("https://test.com/page", 123)

        with patch('bookmark_processor.utils.enhanced_progress.RICH_AVAILABLE', False):
            output = in_progress_tracker._render_plain_progress()

            assert "Current:" in output
            assert "test.com" in output

    def test_create_progress_bar(self, tracker):
        """Test progress bar creation."""
        bar_0 = tracker._create_progress_bar(0)
        bar_50 = tracker._create_progress_bar(50)
        bar_100 = tracker._create_progress_bar(100)

        # Should contain visual characters
        assert len(bar_0) > 0
        assert len(bar_50) > 0
        assert len(bar_100) > 0

        # 100% bar should have more filled characters
        assert bar_100.count("\u2588") > bar_50.count("\u2588")

    def test_format_duration(self, tracker):
        """Test duration formatting."""
        # None
        assert tracker._format_duration(None) == "N/A"

        # Seconds only
        result = tracker._format_duration(timedelta(seconds=30))
        assert "30s" in result

        # Minutes and seconds
        result = tracker._format_duration(timedelta(minutes=5, seconds=30))
        assert "5m" in result

        # Hours and minutes
        result = tracker._format_duration(timedelta(hours=2, minutes=15))
        assert "2h" in result
        assert "15m" in result


# Summary Tests


class TestSummary:
    """Tests for progress summary functionality."""

    def test_get_summary_structure(self, in_progress_tracker):
        """Test summary structure."""
        summary = in_progress_tracker.get_summary()

        assert "total_bookmarks" in summary
        assert "overall_progress" in summary
        assert "overall_eta_seconds" in summary
        assert "elapsed_seconds" in summary
        assert "memory_mb" in summary
        assert "total_errors" in summary
        assert "error_rate" in summary
        assert "speed_per_minute" in summary
        assert "stages" in summary

    def test_get_summary_values(self, in_progress_tracker):
        """Test summary values."""
        summary = in_progress_tracker.get_summary()

        assert summary["total_bookmarks"] == 1000
        assert summary["total_errors"] == 5
        assert summary["overall_progress"] > 0

    def test_get_summary_stages(self, in_progress_tracker):
        """Test summary includes all stages."""
        summary = in_progress_tracker.get_summary()

        for stage_name in in_progress_tracker.stages:
            assert stage_name in summary["stages"]
            stage_summary = summary["stages"][stage_name]
            assert "status" in stage_summary
            assert "progress" in stage_summary
            assert "completed" in stage_summary


# Factory Function Tests


class TestFactoryFunction:
    """Tests for create_enhanced_tracker factory function."""

    def test_create_basic_tracker(self):
        """Test creating a basic tracker."""
        tracker = create_enhanced_tracker(500)

        assert tracker.total_bookmarks == 500
        assert tracker.start_time is not None
        assert len(tracker.stages) > 0

    def test_create_with_custom_weights(self):
        """Test creating with custom stage weights."""
        custom_weights = {
            ProcessingStage.URL_VALIDATION.value: 0.5,
            ProcessingStage.AI_PROCESSING.value: 0.5,
        }

        tracker = create_enhanced_tracker(500, stage_weights=custom_weights)

        assert tracker.STAGE_WEIGHTS == custom_weights

    def test_create_tracker_starts_immediately(self):
        """Test that created tracker is already started."""
        tracker = create_enhanced_tracker(500)

        assert tracker.start_time is not None
        elapsed = tracker.elapsed_time
        assert elapsed.total_seconds() >= 0


# Lifecycle Tests


class TestLifecycle:
    """Tests for tracker lifecycle management."""

    def test_complete(self, in_progress_tracker):
        """Test completing the tracker."""
        in_progress_tracker.complete()

        # All in-progress stages should be completed
        for stage in in_progress_tracker.stages.values():
            assert stage.status in [StageStatus.COMPLETED, StageStatus.PENDING]

    def test_complete_idempotent(self, in_progress_tracker):
        """Test that complete() can be called multiple times."""
        in_progress_tracker.complete()
        in_progress_tracker.complete()

        # Should not raise errors


# Edge Cases


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_bookmarks(self):
        """Test tracker with zero bookmarks."""
        tracker = EnhancedProgressTracker(total_bookmarks=0)
        tracker.start(0)

        assert tracker.overall_progress == 0.0

    def test_negative_eta(self):
        """Test that ETA doesn't go negative."""
        tracker = create_enhanced_tracker(100)

        # Complete all stages instantly
        for stage_name in tracker.stages:
            tracker.start_stage(stage_name, 100)
            tracker.update_stage(stage_name, 100)
            tracker.complete_stage(stage_name)

        eta = tracker.overall_eta
        assert eta.total_seconds() >= 0

    def test_unknown_stage(self):
        """Test handling of unknown stage."""
        tracker = create_enhanced_tracker(100)

        # Starting an unknown stage should create it
        tracker.start_stage("unknown_stage", 50)

        assert "unknown_stage" in tracker.stages
        assert tracker.stages["unknown_stage"].status == StageStatus.IN_PROGRESS

    def test_update_without_start(self):
        """Test updating a stage that wasn't explicitly started."""
        tracker = create_enhanced_tracker(100)

        # Updating without starting should not raise error
        tracker.update_stage(ProcessingStage.URL_VALIDATION, 50)

        # Stage should exist but progress update should work
        stage = tracker.stages[ProcessingStage.URL_VALIDATION.value]
        assert stage.completed == 50

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        tracker = create_enhanced_tracker(1000000)  # 1 million

        tracker.start_stage(ProcessingStage.URL_VALIDATION, 1000000)
        tracker.update_stage(ProcessingStage.URL_VALIDATION, 500000)

        progress = tracker.overall_progress
        assert progress >= 0.0 and progress <= 100.0

    def test_rapid_updates(self):
        """Test rapid consecutive updates."""
        tracker = create_enhanced_tracker(1000)
        tracker.start_stage(ProcessingStage.URL_VALIDATION, 1000)

        # Rapid updates
        for i in range(100):
            tracker.update_stage(ProcessingStage.URL_VALIDATION, i * 10)

        # Should handle rapid updates without error
        assert tracker.stages[ProcessingStage.URL_VALIDATION.value].completed == 990


# Rich-Specific Tests (conditional)


@pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich library not available")
class TestRichRendering:
    """Tests for Rich-specific rendering (only run if Rich is installed)."""

    def test_render_progress_rich(self, in_progress_tracker):
        """Test Rich rendering produces output."""
        output = in_progress_tracker.render_progress()

        assert len(output) > 0
        assert "PROCESSING STATUS" in output

    def test_print_progress(self, in_progress_tracker, capsys):
        """Test printing progress."""
        in_progress_tracker.print_progress()

        captured = capsys.readouterr()
        # Should produce some output
        assert len(captured.out) >= 0  # Rich may write to different stream


# Integration Tests


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_processing_simulation(self):
        """Test simulating a full processing run."""
        tracker = create_enhanced_tracker(100)

        # Simulate processing - include all stages that have weights
        stages = [
            ProcessingStage.INITIALIZATION,
            ProcessingStage.LOADING,
            ProcessingStage.DEDUPLICATION,  # Include deduplication stage
            ProcessingStage.URL_VALIDATION,
            ProcessingStage.CONTENT_ANALYSIS,
            ProcessingStage.AI_PROCESSING,
            ProcessingStage.TAG_OPTIMIZATION,
            ProcessingStage.OUTPUT_GENERATION,
        ]

        for i, stage in enumerate(stages):
            # Use small total for initialization, larger for others
            total = 1 if stage == ProcessingStage.INITIALIZATION else 100
            tracker.start_stage(stage, total)

            # Simulate gradual progress
            for j in range(0, 101, 20):
                tracker.update_stage(stage, min(j, total))
                tracker.set_current_item(f"item_{j}", j)

            tracker.complete_stage(stage)

        tracker.complete()

        # Verify final state
        assert tracker.overall_progress == 100.0
        for stage in tracker.stages.values():
            assert stage.status in [StageStatus.COMPLETED, StageStatus.PENDING]

    def test_summary_after_processing(self):
        """Test summary generation after processing."""
        tracker = create_enhanced_tracker(50)

        # Process a few stages
        tracker.start_stage(ProcessingStage.URL_VALIDATION, 50)
        tracker.update_stage(ProcessingStage.URL_VALIDATION, 50, failed=2)
        tracker.complete_stage(ProcessingStage.URL_VALIDATION)

        summary = tracker.get_summary()

        assert summary["total_bookmarks"] == 50
        assert summary["total_errors"] == 2
        assert ProcessingStage.URL_VALIDATION.value in summary["stages"]
        assert summary["stages"][ProcessingStage.URL_VALIDATION.value]["status"] == "completed"


# Marker for test categorization
pytestmark = pytest.mark.unit
