"""
Enhanced Progress and Status Tracking System

This module provides comprehensive progress tracking, real-time status updates,
and performance monitoring for the bookmark processing pipeline.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as async_tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None
    async_tqdm = None


class ProcessingStage(Enum):
    """Stages of bookmark processing pipeline."""

    INITIALIZATION = "initialization"
    LOADING_DATA = "loading_data"
    VALIDATING_URLS = "validating_urls"
    EXTRACTING_CONTENT = "extracting_content"
    GENERATING_DESCRIPTIONS = "generating_descriptions"
    GENERATING_TAGS = "generating_tags"
    OPTIMIZING_TAGS = "optimizing_tags"
    SAVING_RESULTS = "saving_results"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


@dataclass
class StageMetrics:
    """Metrics for a processing stage."""

    stage: ProcessingStage
    start_time: float = 0.0
    end_time: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    items_total: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get stage duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.items_total == 0:
            return 100.0
        return ((self.items_total - self.items_failed) / self.items_total) * 100

    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        duration = self.duration
        if duration > 0:
            return self.items_processed / duration
        return 0.0


@dataclass
class ProgressSnapshot:
    """Snapshot of current progress state."""

    timestamp: float
    current_stage: ProcessingStage
    overall_progress: float
    stage_progress: float
    items_processed: int
    items_total: int
    elapsed_time: float
    estimated_time_remaining: float
    current_rate: float
    memory_usage_mb: float
    active_tasks: int
    health_status: str = "healthy"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "current_stage": self.current_stage.value,
            "overall_progress": self.overall_progress,
            "stage_progress": self.stage_progress,
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "elapsed_time": self.elapsed_time,
            "estimated_time_remaining": self.estimated_time_remaining,
            "current_rate": self.current_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "active_tasks": self.active_tasks,
            "health_status": self.health_status,
        }


class ProgressTracker:
    """Enhanced progress tracker with real-time updates and performance monitoring."""

    def __init__(
        self,
        total_items: int,
        update_interval: float = 1.0,
        show_progress_bar: bool = True,
        verbose: bool = False,
        checkpoint_callback: Optional[Callable] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            update_interval: Seconds between status updates
            show_progress_bar: Whether to show progress bars
            verbose: Enable verbose output
            checkpoint_callback: Optional callback for checkpoint saves
        """
        self.total_items = total_items
        self.update_interval = update_interval
        self.show_progress_bar = show_progress_bar and TQDM_AVAILABLE
        self.verbose = verbose
        self.checkpoint_callback = checkpoint_callback

        # Current state
        self.current_stage = ProcessingStage.INITIALIZATION
        self.start_time = time.time()
        self.items_processed = 0
        self.items_failed = 0

        # Stage tracking
        self.stage_metrics: Dict[ProcessingStage, StageMetrics] = {}
        self.stage_weights = {
            ProcessingStage.INITIALIZATION: 0.02,
            ProcessingStage.LOADING_DATA: 0.05,
            ProcessingStage.VALIDATING_URLS: 0.15,
            ProcessingStage.EXTRACTING_CONTENT: 0.20,
            ProcessingStage.GENERATING_DESCRIPTIONS: 0.35,
            ProcessingStage.GENERATING_TAGS: 0.15,
            ProcessingStage.OPTIMIZING_TAGS: 0.05,
            ProcessingStage.SAVING_RESULTS: 0.03,
            ProcessingStage.FINALIZING: 0.01,
        }

        # Performance tracking
        self.rate_history = deque(maxlen=60)  # Last 60 measurements
        self.last_update_time = time.time()
        self.last_items_processed = 0

        # Progress bars
        self.overall_pbar = None
        self.stage_pbar = None

        # Status display
        self.status_lines: List[str] = []
        self.last_status_update = 0.0

        # Resource monitoring
        self.memory_usage_history = deque(maxlen=60)
        self.active_tasks = 0
        self.max_active_tasks = 0

        self.logger = logging.getLogger(__name__)

    def start_stage(
        self, stage: ProcessingStage, total_items: Optional[int] = None
    ) -> None:
        """
        Start a new processing stage.

        Args:
            stage: The stage to start
            total_items: Optional total items for this stage
        """
        # Complete previous stage if any
        if self.current_stage in self.stage_metrics and self.current_stage != stage:
            self._complete_current_stage()

        self.current_stage = stage

        # Initialize stage metrics
        self.stage_metrics[stage] = StageMetrics(
            stage=stage,
            start_time=time.time(),
            items_total=total_items or self.total_items,
        )

        # Create stage progress bar if needed
        if self.show_progress_bar and stage != ProcessingStage.INITIALIZATION:
            if self.stage_pbar:
                self.stage_pbar.close()

            if TQDM_AVAILABLE:
                self.stage_pbar = tqdm(
                    total=self.stage_metrics[stage].items_total,
                    desc=f"{stage.value.replace('_', ' ').title()}",
                    position=1,
                    leave=False,
                )

        self.logger.info(f"Started stage: {stage.value}")

        if self.verbose:
            print(f"\n🔄 {stage.value.replace('_', ' ').title()} started...")

    def _complete_current_stage(self) -> None:
        """Complete the current stage."""
        if self.current_stage in self.stage_metrics:
            metrics = self.stage_metrics[self.current_stage]
            metrics.end_time = time.time()

            if self.stage_pbar:
                self.stage_pbar.close()
                self.stage_pbar = None

            if self.verbose:
                duration = metrics.duration
                rate = metrics.items_per_second
                print(
                    f"✅ {self.current_stage.value.replace('_', ' ').title()} completed"
                )
                print(f"   Duration: {duration:.1f}s, Rate: {rate:.1f} items/s")

    def update_progress(
        self,
        items_delta: int = 1,
        failed_delta: int = 0,
        stage_specific: bool = True,
    ) -> None:
        """
        Update progress counters.

        Args:
            items_delta: Number of items processed
            failed_delta: Number of items failed
            stage_specific: Whether to update stage-specific counters
        """
        # Update global counters
        self.items_processed += items_delta
        self.items_failed += failed_delta

        # Update stage counters
        if stage_specific and self.current_stage in self.stage_metrics:
            metrics = self.stage_metrics[self.current_stage]
            metrics.items_processed += items_delta
            metrics.items_failed += failed_delta

        # Update progress bars
        if self.show_progress_bar:
            if self.overall_pbar:
                self.overall_pbar.update(items_delta)
            if self.stage_pbar:
                self.stage_pbar.update(items_delta)

        # Calculate rate
        current_time = time.time()
        time_delta = current_time - self.last_update_time

        if time_delta >= self.update_interval:
            items_in_interval = self.items_processed - self.last_items_processed
            rate = items_in_interval / time_delta
            self.rate_history.append(rate)

            self.last_update_time = current_time
            self.last_items_processed = self.items_processed

            # Update status display
            self._update_status_display()

    def _update_status_display(self) -> None:
        """Update the status display with current information."""
        if not self.verbose:
            return

        current_time = time.time()
        if current_time - self.last_status_update < self.update_interval:
            return

        self.last_status_update = current_time

        # Get current snapshot
        snapshot = self.get_snapshot()

        # Format status lines
        status_lines = [
            f"\n📊 Processing Status Update:",
            f"  📍 Stage: {self.current_stage.value.replace('_', ' ').title()}",
            f"  📈 Overall: {snapshot.overall_progress:.1f}% ({self.items_processed}/{self.total_items})",
            f"  ⏱️  Elapsed: {self._format_duration(snapshot.elapsed_time)}",
            f"  ⏳ Remaining: {self._format_duration(snapshot.estimated_time_remaining)}",
            f"  🚀 Rate: {snapshot.current_rate:.1f} items/s",
            f"  💾 Memory: {snapshot.memory_usage_mb:.1f} MB",
        ]

        # Add stage-specific info
        if self.current_stage in self.stage_metrics:
            metrics = self.stage_metrics[self.current_stage]
            stage_progress = (
                metrics.items_processed / max(metrics.items_total, 1)
            ) * 100
            status_lines.append(
                f"  📊 Stage Progress: {stage_progress:.1f}% "
                f"({metrics.items_processed}/{metrics.items_total})"
            )

        # Add health status
        if snapshot.health_status != "healthy":
            status_lines.append(f"  ⚠️  Health: {snapshot.health_status}")

        # Print status update
        for line in status_lines:
            print(line)

    def get_snapshot(self) -> ProgressSnapshot:
        """
        Get current progress snapshot.

        Returns:
            ProgressSnapshot with current state
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Calculate overall progress
        overall_progress = self._calculate_overall_progress()

        # Calculate stage progress
        stage_progress = 0.0
        if self.current_stage in self.stage_metrics:
            metrics = self.stage_metrics[self.current_stage]
            if metrics.items_total > 0:
                stage_progress = (metrics.items_processed / metrics.items_total) * 100

        # Calculate current rate
        current_rate = 0.0
        if self.rate_history:
            current_rate = sum(self.rate_history) / len(self.rate_history)

        # Estimate time remaining
        if current_rate > 0 and self.items_processed < self.total_items:
            items_remaining = self.total_items - self.items_processed
            estimated_time_remaining = items_remaining / current_rate
        else:
            estimated_time_remaining = 0.0

        # Get memory usage
        memory_usage_mb = self._get_memory_usage()
        self.memory_usage_history.append(memory_usage_mb)

        # Determine health status
        health_status = self._determine_health_status()

        return ProgressSnapshot(
            timestamp=current_time,
            current_stage=self.current_stage,
            overall_progress=overall_progress,
            stage_progress=stage_progress,
            items_processed=self.items_processed,
            items_total=self.total_items,
            elapsed_time=elapsed_time,
            estimated_time_remaining=estimated_time_remaining,
            current_rate=current_rate,
            memory_usage_mb=memory_usage_mb,
            active_tasks=self.active_tasks,
            health_status=health_status,
        )

    def _calculate_overall_progress(self) -> float:
        """Calculate overall progress percentage across all stages."""
        total_progress = 0.0

        # Add completed stages
        for stage, weight in self.stage_weights.items():
            if stage in self.stage_metrics:
                metrics = self.stage_metrics[stage]
                if metrics.end_time > 0:
                    # Stage completed
                    total_progress += weight * 100
                elif stage == self.current_stage:
                    # Current stage
                    if metrics.items_total > 0:
                        stage_completion = metrics.items_processed / metrics.items_total
                        total_progress += weight * stage_completion * 100

        return min(total_progress, 100.0)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # psutil not available, return 0
            return 0.0

    def _determine_health_status(self) -> str:
        """Determine system health based on metrics."""
        # Check error rate
        if self.items_failed > 0:
            error_rate = self.items_failed / max(self.items_processed, 1)
            if error_rate > 0.2:
                return "critical"
            elif error_rate > 0.1:
                return "degraded"

        # Check processing rate
        if self.rate_history and len(self.rate_history) > 10:
            recent_rate = sum(list(self.rate_history)[-10:]) / 10
            if recent_rate < 0.1:
                return "slow"

        # Check memory usage trend
        if self.memory_usage_history and len(self.memory_usage_history) > 10:
            recent_memory = list(self.memory_usage_history)[-10:]
            memory_increase = recent_memory[-1] - recent_memory[0]
            if memory_increase > 500:  # 500MB increase
                return "memory_concern"

        return "healthy"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds <= 0:
            return "N/A"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def log_error(self, error: str, stage: Optional[ProcessingStage] = None) -> None:
        """
        Log an error for tracking.

        Args:
            error: Error message
            stage: Optional specific stage
        """
        target_stage = stage or self.current_stage

        if target_stage in self.stage_metrics:
            self.stage_metrics[target_stage].errors.append(error)

        self.logger.error(f"[{target_stage.value}] {error}")

    def set_active_tasks(self, count: int) -> None:
        """
        Set the number of active concurrent tasks.

        Args:
            count: Number of active tasks
        """
        self.active_tasks = count
        self.max_active_tasks = max(self.max_active_tasks, count)

    async def save_checkpoint(self, data: Dict[str, Any]) -> None:
        """
        Trigger checkpoint save with progress information.

        Args:
            data: Additional data to save
        """
        if self.checkpoint_callback:
            snapshot = self.get_snapshot()
            checkpoint_data = {
                "progress": snapshot.to_dict(),
                "stage_metrics": {
                    stage.value: {
                        "start_time": metrics.start_time,
                        "end_time": metrics.end_time,
                        "items_processed": metrics.items_processed,
                        "items_failed": metrics.items_failed,
                        "items_total": metrics.items_total,
                        "duration": metrics.duration,
                        "success_rate": metrics.success_rate,
                    }
                    for stage, metrics in self.stage_metrics.items()
                },
                "data": data,
            }

            await self.checkpoint_callback(checkpoint_data)

            if self.verbose:
                print(
                    f"💾 Checkpoint saved at {snapshot.overall_progress:.1f}% progress"
                )

    def get_stage_summary(self) -> Dict[str, Any]:
        """
        Get summary of all stages.

        Returns:
            Dictionary with stage summaries
        """
        summary = {}

        for stage, metrics in self.stage_metrics.items():
            summary[stage.value] = {
                "duration": metrics.duration,
                "items_processed": metrics.items_processed,
                "items_failed": metrics.items_failed,
                "success_rate": metrics.success_rate,
                "items_per_second": metrics.items_per_second,
                "errors": len(metrics.errors),
                "status": "completed" if metrics.end_time > 0 else "in_progress",
            }

        return summary

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns:
            Dictionary with performance metrics
        """
        total_duration = time.time() - self.start_time

        # Calculate stage efficiency
        stage_efficiency = {}
        for stage, weight in self.stage_weights.items():
            if stage in self.stage_metrics:
                metrics = self.stage_metrics[stage]
                expected_time = total_duration * weight
                actual_time = metrics.duration
                if expected_time > 0:
                    efficiency = (expected_time / max(actual_time, 0.01)) * 100
                    stage_efficiency[stage.value] = min(efficiency, 200)  # Cap at 200%

        return {
            "total_duration": total_duration,
            "total_items_processed": self.items_processed,
            "total_items_failed": self.items_failed,
            "overall_success_rate": (
                (self.items_processed - self.items_failed)
                / max(self.items_processed, 1)
            )
            * 100,
            "average_rate": self.items_processed / max(total_duration, 1),
            "peak_rate": max(self.rate_history) if self.rate_history else 0,
            "stage_summary": self.get_stage_summary(),
            "stage_efficiency": stage_efficiency,
            "max_concurrent_tasks": self.max_active_tasks,
            "peak_memory_mb": (
                max(self.memory_usage_history) if self.memory_usage_history else 0
            ),
            "final_status": self.current_stage.value,
        }

    def complete(self) -> None:
        """Complete progress tracking and clean up resources."""
        # Complete current stage
        if self.current_stage != ProcessingStage.COMPLETED:
            self._complete_current_stage()
            self.current_stage = ProcessingStage.COMPLETED

        # Close progress bars
        if self.overall_pbar:
            self.overall_pbar.close()
        if self.stage_pbar:
            self.stage_pbar.close()

        # Print final report if verbose
        if self.verbose:
            report = self.get_performance_report()
            self._print_final_report(report)

    def _print_final_report(self, report: Dict[str, Any]) -> None:
        """Print comprehensive final report."""
        print("\n" + "=" * 60)
        print("📊 PROCESSING COMPLETE - FINAL REPORT")
        print("=" * 60)

        print(f"\n⏱️  Total Duration: {self._format_duration(report['total_duration'])}")
        print(f"✅ Items Processed: {report['total_items_processed']}")
        print(f"❌ Items Failed: {report['total_items_failed']}")
        print(f"📈 Success Rate: {report['overall_success_rate']:.1f}%")
        print(f"🚀 Average Rate: {report['average_rate']:.1f} items/s")
        print(f"🏃 Peak Rate: {report['peak_rate']:.1f} items/s")
        print(f"💾 Peak Memory: {report['peak_memory_mb']:.1f} MB")
        print(f"🔀 Max Concurrent: {report['max_concurrent_tasks']}")

        print("\n📋 Stage Performance:")
        for stage_name, stage_data in report["stage_summary"].items():
            if stage_data["status"] == "completed":
                print(f"\n  {stage_name.replace('_', ' ').title()}:")
                print(f"    Duration: {self._format_duration(stage_data['duration'])}")
                print(f"    Processed: {stage_data['items_processed']}")
                print(f"    Rate: {stage_data['items_per_second']:.1f} items/s")
                print(f"    Success: {stage_data['success_rate']:.1f}%")

                if stage_name in report["stage_efficiency"]:
                    efficiency = report["stage_efficiency"][stage_name]
                    print(f"    Efficiency: {efficiency:.0f}%")

        print("\n" + "=" * 60)

    def __enter__(self) -> "ProgressTracker":
        """Context manager entry."""
        if self.show_progress_bar and TQDM_AVAILABLE:
            self.overall_pbar = tqdm(
                total=self.total_items,
                desc="Overall Progress",
                position=0,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.complete()


# Async context manager version
class AsyncProgressTracker(ProgressTracker):
    """Async version of progress tracker for async contexts."""

    async def __aenter__(self) -> "AsyncProgressTracker":
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.__exit__(exc_type, exc_val, exc_tb)


def create_progress_tracker(
    total_items: int,
    verbose: bool = False,
    show_progress_bar: bool = True,
    checkpoint_callback: Optional[Callable] = None,
) -> ProgressTracker:
    """
    Factory function to create appropriate progress tracker.

    Args:
        total_items: Total number of items to process
        verbose: Enable verbose output
        show_progress_bar: Whether to show progress bars
        checkpoint_callback: Optional checkpoint callback

    Returns:
        ProgressTracker instance
    """
    return ProgressTracker(
        total_items=total_items,
        verbose=verbose,
        show_progress_bar=show_progress_bar,
        checkpoint_callback=checkpoint_callback,
    )


# Legacy compatibility
class ProgressLevel(Enum):
    """Progress reporting levels for backward compatibility."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    VERBOSE = "verbose"


class AdvancedProgressTracker(ProgressTracker):
    """Legacy compatibility wrapper."""

    def __init__(
        self,
        total_items: int,
        description: str = "Processing",
        level: ProgressLevel = ProgressLevel.STANDARD,
        **kwargs,
    ):
        verbose = level in [ProgressLevel.DETAILED, ProgressLevel.VERBOSE]
        show_progress_bar = level != ProgressLevel.MINIMAL
        super().__init__(
            total_items, verbose=verbose, show_progress_bar=show_progress_bar, **kwargs
        )
        self.description = description

    def update(self, items_processed=None, increment=1, **kwargs):
        """Legacy update method."""
        if items_processed is not None:
            delta = items_processed - self.items_processed
            self.update_progress(items_delta=delta)
        else:
            self.update_progress(items_delta=increment)

    def finish(self, message=None):
        """Legacy finish method."""
        self.complete()


def track_progress(
    iterable,
    description: str = "Processing",
    level: ProgressLevel = ProgressLevel.STANDARD,
):
    """Legacy track_progress function."""
    verbose = level in [ProgressLevel.DETAILED, ProgressLevel.VERBOSE]
    show_progress_bar = level != ProgressLevel.MINIMAL

    tracker = ProgressTracker(
        total_items=len(iterable),
        verbose=verbose,
        show_progress_bar=show_progress_bar,
    )

    try:
        with tracker:
            for i, item in enumerate(iterable):
                yield item
                tracker.update_progress(items_delta=1)
    finally:
        tracker.complete()


# Backward compatibility aliases
ProgressBar = ProgressTracker


class ProgressLogger:
    """Simple progress logger for logging progress updates."""

    def __init__(self, name: str = "ProgressLogger"):
        self.name = name
        self.logger = logging.getLogger(name)

    def log_progress(self, message: str, level: str = "info"):
        """Log a progress message."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)


class TimeEstimator:
    """Simple time estimation utility."""

    def __init__(self):
        self.start_time = None
        self.items_processed = 0
        self.total_items = 0

    def start(self, total_items: int):
        """Start time estimation."""
        self.start_time = time.time()
        self.total_items = total_items
        self.items_processed = 0

    def update(self, items_processed: int):
        """Update progress."""
        self.items_processed = items_processed

    def get_eta(self) -> Optional[float]:
        """Get estimated time remaining."""
        if not self.start_time or self.items_processed == 0:
            return None

        elapsed = time.time() - self.start_time
        rate = self.items_processed / elapsed
        remaining_items = self.total_items - self.items_processed

        if rate > 0:
            return remaining_items / rate
        return None
