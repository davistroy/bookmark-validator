"""
Enhanced Progress Visibility for Bookmark Processing.

This module provides multi-stage progress tracking with per-stage ETA
calculation, memory monitoring, error rate tracking, and Rich console
rendering with live updates.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from io import StringIO
from typing import Any, Callable, Deque, Dict, List, Optional, Union

from ..core.checkpoint_manager import ProcessingStage

# Rich library imports with graceful fallback
try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Live = None
    Panel = None
    Progress = None
    TaskID = None


class StageStatus(Enum):
    """Status of a processing stage."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageProgress:
    """
    Track progress for a single processing stage.

    Provides per-stage metrics including progress, timing, and ETA.
    """

    name: str
    display_name: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    weight: float = 0.0  # Relative weight in overall progress

    # Rate tracking
    _rate_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _last_update_time: Optional[float] = None
    _last_completed: int = 0

    @property
    def status(self) -> StageStatus:
        """Get current status of the stage."""
        if self.completed_at is not None:
            return StageStatus.COMPLETED
        elif self.started_at is not None:
            return StageStatus.IN_PROGRESS
        return StageStatus.PENDING

    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total == 0:
            return 0.0 if self.status == StageStatus.PENDING else 100.0
        return min(100.0, (self.completed / self.total) * 100)

    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time since stage started."""
        if self.started_at is None:
            return timedelta(0)

        end_time = self.completed_at or datetime.now()
        return end_time - self.started_at

    @property
    def items_per_second(self) -> float:
        """Get current processing rate."""
        if not self._rate_history:
            elapsed = self.elapsed_time.total_seconds()
            if elapsed > 0:
                return self.completed / elapsed
            return 0.0
        return sum(self._rate_history) / len(self._rate_history)

    @property
    def eta(self) -> Optional[timedelta]:
        """Get estimated time remaining for this stage."""
        if self.status == StageStatus.COMPLETED:
            return timedelta(0)

        if self.status == StageStatus.PENDING:
            return self.estimated_duration

        if self.completed == 0 or self.total == 0:
            return self.estimated_duration

        rate = self.items_per_second
        if rate > 0:
            remaining = self.total - self.completed
            seconds_remaining = remaining / rate
            return timedelta(seconds=seconds_remaining)

        return self.estimated_duration

    @property
    def error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.completed == 0:
            return 0.0
        return (self.failed / self.completed) * 100

    def start(self, total_items: int = 0) -> None:
        """Start the stage."""
        self.started_at = datetime.now()
        if total_items > 0:
            self.total = total_items
        self._last_update_time = time.time()
        self._last_completed = 0

    def update(self, completed: int, failed: int = 0) -> None:
        """
        Update stage progress.

        Args:
            completed: Number of items completed (absolute, not delta)
            failed: Number of items failed (absolute, not delta)
        """
        # Calculate rate
        current_time = time.time()
        if self._last_update_time is not None:
            time_delta = current_time - self._last_update_time
            if time_delta > 0.1:  # Minimum interval for rate calculation
                items_delta = completed - self._last_completed
                if items_delta > 0:
                    rate = items_delta / time_delta
                    self._rate_history.append(rate)
                self._last_update_time = current_time
                self._last_completed = completed

        self.completed = completed
        self.failed = failed

    def complete(self) -> None:
        """Mark the stage as completed."""
        self.completed_at = datetime.now()
        self.completed = self.total

    def get_status_icon(self) -> str:
        """Get status icon for display."""
        icons = {
            StageStatus.PENDING: "\u23f8",  # Pause symbol
            StageStatus.IN_PROGRESS: "\u23f3",  # Hourglass
            StageStatus.COMPLETED: "\u2713",  # Checkmark
            StageStatus.SKIPPED: "\u23e9",  # Fast forward
            StageStatus.FAILED: "\u2717",  # X mark
        }
        return icons.get(self.status, "?")

    def format_duration(self, duration: Optional[timedelta]) -> str:
        """Format a duration for display."""
        if duration is None:
            return "N/A"

        total_seconds = int(duration.total_seconds())
        if total_seconds < 0:
            return "N/A"

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


@dataclass
class EnhancedProgressTracker:
    """
    Multi-stage progress tracking with ETA estimation and Rich rendering.

    Provides comprehensive progress visibility including:
    - Per-stage progress bars with ETA
    - Overall weighted progress
    - Memory usage monitoring
    - Error rate tracking
    - Live console updates
    """

    # Default stage weights (should sum to 1.0)
    STAGE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        ProcessingStage.INITIALIZATION.value: 0.02,
        ProcessingStage.LOADING.value: 0.03,
        ProcessingStage.DEDUPLICATION.value: 0.02,
        ProcessingStage.URL_VALIDATION.value: 0.15,
        ProcessingStage.CONTENT_ANALYSIS.value: 0.25,
        ProcessingStage.AI_PROCESSING.value: 0.38,
        ProcessingStage.TAG_OPTIMIZATION.value: 0.10,
        ProcessingStage.OUTPUT_GENERATION.value: 0.05,
    })

    # Stage display names
    STAGE_NAMES: Dict[str, str] = field(default_factory=lambda: {
        ProcessingStage.INITIALIZATION.value: "Initialization",
        ProcessingStage.LOADING.value: "Loading Data",
        ProcessingStage.DEDUPLICATION.value: "Deduplication",
        ProcessingStage.URL_VALIDATION.value: "URL Validation",
        ProcessingStage.CONTENT_ANALYSIS.value: "Content Analysis",
        ProcessingStage.AI_PROCESSING.value: "AI Processing",
        ProcessingStage.TAG_OPTIMIZATION.value: "Tag Optimization",
        ProcessingStage.OUTPUT_GENERATION.value: "Output Generation",
    })

    total_bookmarks: int = 0
    start_time: Optional[datetime] = None

    # Stage tracking
    stages: Dict[str, StageProgress] = field(default_factory=dict)
    current_stage_name: Optional[str] = None

    # Memory tracking
    _memory_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))

    # Error tracking
    total_errors: int = 0

    # Rich console
    _console: Optional["Console"] = None
    _live: Optional["Live"] = None

    # Current item being processed
    current_item: str = ""
    current_item_index: int = 0

    def __post_init__(self):
        """Initialize stages."""
        self._init_stages()

    def _init_stages(self) -> None:
        """Initialize all stage trackers."""
        for stage_value, weight in self.STAGE_WEIGHTS.items():
            display_name = self.STAGE_NAMES.get(stage_value, stage_value)
            self.stages[stage_value] = StageProgress(
                name=stage_value,
                display_name=display_name,
                weight=weight,
                total=self.total_bookmarks,
            )

    @property
    def console(self) -> Optional["Console"]:
        """Get or create Rich console."""
        if self._console is None and RICH_AVAILABLE:
            self._console = Console()
        return self._console

    @property
    def elapsed_time(self) -> timedelta:
        """Get total elapsed time."""
        if self.start_time is None:
            return timedelta(0)
        return datetime.now() - self.start_time

    @property
    def overall_progress(self) -> float:
        """
        Calculate overall progress as weighted sum of stage progress.

        Returns:
            Overall progress percentage (0-100)
        """
        total = 0.0
        for stage in self.stages.values():
            if stage.status == StageStatus.COMPLETED:
                total += stage.weight * 100
            elif stage.status == StageStatus.IN_PROGRESS:
                total += stage.weight * stage.progress_percentage
        return min(100.0, total)

    @property
    def overall_eta(self) -> timedelta:
        """
        Calculate overall ETA based on stage weights and progress.

        Returns:
            Estimated time remaining
        """
        # Find current and future stages
        remaining_time = timedelta(0)

        for stage in self.stages.values():
            if stage.status == StageStatus.COMPLETED:
                continue
            elif stage.status == StageStatus.IN_PROGRESS:
                eta = stage.eta
                if eta:
                    remaining_time += eta
            elif stage.status == StageStatus.PENDING:
                # Estimate based on weight and current rate
                eta = stage.estimated_duration
                if eta:
                    remaining_time += eta
                else:
                    # Estimate from completed stages
                    remaining_time += self._estimate_stage_duration(stage)

        return remaining_time

    def _estimate_stage_duration(self, stage: StageProgress) -> timedelta:
        """Estimate duration for a pending stage based on completed stages."""
        completed_stages = [
            s for s in self.stages.values()
            if s.status == StageStatus.COMPLETED and s.elapsed_time.total_seconds() > 0
        ]

        if not completed_stages:
            # Default estimate based on total items
            base_time = max(1, self.total_bookmarks / 10)  # 10 items/sec default
            return timedelta(seconds=base_time * stage.weight)

        # Calculate average processing time per weight unit
        total_time = sum(s.elapsed_time.total_seconds() for s in completed_stages)
        total_weight = sum(s.weight for s in completed_stages)

        if total_weight > 0:
            time_per_weight = total_time / total_weight
            return timedelta(seconds=time_per_weight * stage.weight)

        return timedelta(seconds=60 * stage.weight)  # Default 1 minute per weight unit

    @property
    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    @property
    def overall_error_rate(self) -> float:
        """Get overall error rate."""
        total_processed = sum(s.completed for s in self.stages.values())
        if total_processed == 0:
            return 0.0
        return (self.total_errors / total_processed) * 100

    @property
    def overall_speed(self) -> float:
        """Get overall processing speed (items/minute)."""
        elapsed_seconds = self.elapsed_time.total_seconds()
        if elapsed_seconds == 0:
            return 0.0

        total_processed = sum(s.completed for s in self.stages.values())
        # Avoid counting same items multiple times
        return (total_processed / elapsed_seconds) * 60

    def start(self, total_bookmarks: int = 0) -> None:
        """
        Start progress tracking.

        Args:
            total_bookmarks: Total number of bookmarks to process
        """
        self.start_time = datetime.now()
        if total_bookmarks > 0:
            self.total_bookmarks = total_bookmarks
            for stage in self.stages.values():
                stage.total = total_bookmarks

    def start_stage(
        self,
        stage: Union[str, ProcessingStage],
        total_items: Optional[int] = None,
    ) -> None:
        """
        Start a processing stage.

        Args:
            stage: Stage name or ProcessingStage enum
            total_items: Optional total items for this stage
        """
        if isinstance(stage, ProcessingStage):
            stage_name = stage.value
        else:
            stage_name = stage

        if stage_name not in self.stages:
            # Create new stage if not exists
            self.stages[stage_name] = StageProgress(
                name=stage_name,
                display_name=self.STAGE_NAMES.get(stage_name, stage_name),
                weight=self.STAGE_WEIGHTS.get(stage_name, 0.05),
                total=total_items or self.total_bookmarks,
            )

        stage_obj = self.stages[stage_name]
        stage_obj.start(total_items or self.total_bookmarks)
        self.current_stage_name = stage_name

    def update_stage(
        self,
        stage: Union[str, ProcessingStage],
        completed: int,
        failed: int = 0,
    ) -> None:
        """
        Update stage progress.

        Args:
            stage: Stage name or ProcessingStage enum
            completed: Number of items completed
            failed: Number of items failed
        """
        if isinstance(stage, ProcessingStage):
            stage_name = stage.value
        else:
            stage_name = stage

        if stage_name in self.stages:
            self.stages[stage_name].update(completed, failed)
            self.total_errors = sum(s.failed for s in self.stages.values())

    def complete_stage(self, stage: Union[str, ProcessingStage]) -> None:
        """
        Mark a stage as completed.

        Args:
            stage: Stage name or ProcessingStage enum
        """
        if isinstance(stage, ProcessingStage):
            stage_name = stage.value
        else:
            stage_name = stage

        if stage_name in self.stages:
            self.stages[stage_name].complete()

    def set_current_item(self, item: str, index: int = 0) -> None:
        """
        Set the currently processing item for display.

        Args:
            item: Description of current item
            index: Index of current item
        """
        self.current_item = item
        self.current_item_index = index

    def render_progress(self) -> str:
        """
        Render the progress display as a string.

        Returns:
            Formatted progress display string
        """
        if not RICH_AVAILABLE:
            return self._render_plain_progress()

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)

        # Header
        elapsed_str = self._format_duration(self.elapsed_time)
        console.print(
            Panel(
                f"[bold cyan]PROCESSING STATUS[/] - {elapsed_str} elapsed",
                expand=False,
            )
        )

        # Stage progress
        for stage_name, stage in self.stages.items():
            self._render_stage_line(console, stage)

        console.print()

        # Overall progress
        overall_pct = self.overall_progress
        eta = self._format_duration(self.overall_eta)
        memory = self.memory_usage_mb

        bar = self._create_progress_bar(overall_pct)
        console.print(
            f"Overall: {bar} {overall_pct:5.1f}% | "
            f"ETA: {eta} | Memory: {memory:.1f}GB"
        )

        # Separator
        console.print("[dim]" + "\u2500" * 60 + "[/]")

        # Current item
        if self.current_item:
            console.print(
                f"Current: [cyan]{self.current_item}[/] "
                f"({self.current_item_index:,}/{self.total_bookmarks:,})"
            )

        # Speed and errors
        speed = self.overall_speed
        error_rate = self.overall_error_rate
        console.print(
            f"Speed: [green]{speed:.1f}[/] URLs/min | "
            f"Errors: [{'red' if error_rate > 5 else 'yellow' if error_rate > 1 else 'green'}]"
            f"{self.total_errors:,}[/] ({error_rate:.1f}%)"
        )

        return output.getvalue()

    def _render_stage_line(self, console: "Console", stage: StageProgress) -> None:
        """Render a single stage progress line."""
        icon = stage.get_status_icon()
        name = stage.display_name

        if stage.status == StageStatus.COMPLETED:
            elapsed = stage.format_duration(stage.elapsed_time)
            bar = self._create_progress_bar(100)
            console.print(
                f"Stage: {name:20} {bar} 100% [green]{icon}[/] ({elapsed})"
            )
        elif stage.status == StageStatus.IN_PROGRESS:
            pct = stage.progress_percentage
            eta = stage.format_duration(stage.eta)
            elapsed = stage.format_duration(stage.elapsed_time)
            bar = self._create_progress_bar(pct)
            console.print(
                f"Stage: {name:20} {bar} {pct:3.0f}% [yellow]{icon}[/] "
                f"({elapsed} / ~{eta} left)"
            )
        else:  # PENDING
            eta = stage.format_duration(stage.estimated_duration or self._estimate_stage_duration(stage))
            bar = self._create_progress_bar(0)
            console.print(
                f"Stage: {name:20} {bar}   0% [dim]{icon}[/] (~{eta})"
            )

    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a text-based progress bar."""
        filled = int(width * percentage / 100)
        empty = width - filled
        return "[" + "\u2588" * filled + "\u2591" * empty + "]"

    def _format_duration(self, duration: Optional[timedelta]) -> str:
        """Format duration for display."""
        if duration is None:
            return "N/A"

        total_seconds = int(duration.total_seconds())
        if total_seconds < 0:
            return "N/A"

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _render_plain_progress(self) -> str:
        """Render progress in plain text (no Rich)."""
        lines = []

        # Header
        elapsed_str = self._format_duration(self.elapsed_time)
        lines.append("=" * 60)
        lines.append(f"PROCESSING STATUS - {elapsed_str} elapsed")
        lines.append("=" * 60)
        lines.append("")

        # Stages
        for stage in self.stages.values():
            icon = stage.get_status_icon()
            name = stage.display_name
            pct = stage.progress_percentage

            if stage.status == StageStatus.COMPLETED:
                elapsed = stage.format_duration(stage.elapsed_time)
                lines.append(f"Stage: {name:20} [{'#' * 20}] 100% {icon} ({elapsed})")
            elif stage.status == StageStatus.IN_PROGRESS:
                eta = stage.format_duration(stage.eta)
                elapsed = stage.format_duration(stage.elapsed_time)
                filled = int(20 * pct / 100)
                bar = "#" * filled + "." * (20 - filled)
                lines.append(
                    f"Stage: {name:20} [{bar}] {pct:3.0f}% {icon} "
                    f"({elapsed} / ~{eta} left)"
                )
            else:
                lines.append(f"Stage: {name:20} [{'.' * 20}]   0% {icon}")

        lines.append("")

        # Overall
        overall_pct = self.overall_progress
        eta = self._format_duration(self.overall_eta)
        memory = self.memory_usage_mb
        filled = int(20 * overall_pct / 100)
        bar = "#" * filled + "." * (20 - filled)
        lines.append(f"Overall: [{bar}] {overall_pct:5.1f}% | ETA: {eta} | Memory: {memory:.1f}MB")

        lines.append("-" * 60)

        # Current item
        if self.current_item:
            lines.append(
                f"Current: {self.current_item} ({self.current_item_index:,}/{self.total_bookmarks:,})"
            )

        # Speed and errors
        speed = self.overall_speed
        error_rate = self.overall_error_rate
        lines.append(f"Speed: {speed:.1f} URLs/min | Errors: {self.total_errors:,} ({error_rate:.1f}%)")

        return "\n".join(lines)

    def print_progress(self) -> None:
        """Print progress to console."""
        if RICH_AVAILABLE and self.console:
            self.console.print(self.render_progress())
        else:
            print(self.render_progress())

    def start_live_display(self) -> None:
        """Start live updating display (Rich only)."""
        if not RICH_AVAILABLE:
            return

        self._live = Live(
            self._create_live_display(),
            console=self.console,
            refresh_per_second=2,
        )
        self._live.start()

    def update_live_display(self) -> None:
        """Update the live display."""
        if self._live and RICH_AVAILABLE:
            self._live.update(self._create_live_display())

    def stop_live_display(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def _create_live_display(self) -> "Panel":
        """Create the live display panel."""
        if not RICH_AVAILABLE:
            return None

        # Build display content
        table = Table(show_header=False, box=None, expand=True)

        # Header row
        elapsed_str = self._format_duration(self.elapsed_time)
        table.add_row(
            Text(f"PROCESSING STATUS - {elapsed_str} elapsed", style="bold cyan")
        )
        table.add_row(Text(""))

        # Stage rows
        for stage in self.stages.values():
            status_text = self._format_stage_for_live(stage)
            table.add_row(status_text)

        table.add_row(Text(""))

        # Overall progress row
        overall_pct = self.overall_progress
        eta = self._format_duration(self.overall_eta)
        memory = self.memory_usage_mb

        overall_text = Text()
        overall_text.append("Overall: ")
        overall_text.append(self._create_progress_bar(overall_pct))
        overall_text.append(f" {overall_pct:5.1f}% | ETA: {eta} | Memory: {memory:.1f}MB")
        table.add_row(overall_text)

        table.add_row(Text("\u2500" * 60, style="dim"))

        # Current item row
        if self.current_item:
            current_text = Text()
            current_text.append("Current: ")
            current_text.append(self.current_item, style="cyan")
            current_text.append(f" ({self.current_item_index:,}/{self.total_bookmarks:,})")
            table.add_row(current_text)

        # Speed and errors row
        speed = self.overall_speed
        error_rate = self.overall_error_rate
        error_style = "red" if error_rate > 5 else "yellow" if error_rate > 1 else "green"

        stats_text = Text()
        stats_text.append(f"Speed: ")
        stats_text.append(f"{speed:.1f}", style="green")
        stats_text.append(" URLs/min | Errors: ")
        stats_text.append(f"{self.total_errors:,}", style=error_style)
        stats_text.append(f" ({error_rate:.1f}%)")
        table.add_row(stats_text)

        return Panel(table, title="Processing Progress", expand=False)

    def _format_stage_for_live(self, stage: StageProgress) -> "Text":
        """Format a stage for live display."""
        text = Text()
        icon = stage.get_status_icon()
        name = f"{stage.display_name:20}"

        if stage.status == StageStatus.COMPLETED:
            elapsed = stage.format_duration(stage.elapsed_time)
            text.append(f"Stage: {name} ")
            text.append(self._create_progress_bar(100), style="green")
            text.append(" 100% ")
            text.append(icon, style="green")
            text.append(f" ({elapsed})")
        elif stage.status == StageStatus.IN_PROGRESS:
            pct = stage.progress_percentage
            eta = stage.format_duration(stage.eta)
            elapsed = stage.format_duration(stage.elapsed_time)
            text.append(f"Stage: {name} ")
            text.append(self._create_progress_bar(pct), style="yellow")
            text.append(f" {pct:3.0f}% ")
            text.append(icon, style="yellow")
            text.append(f" ({elapsed} / ~{eta} left)")
        else:
            eta = stage.format_duration(
                stage.estimated_duration or self._estimate_stage_duration(stage)
            )
            text.append(f"Stage: {name} ")
            text.append(self._create_progress_bar(0), style="dim")
            text.append("   0% ")
            text.append(icon, style="dim")
            text.append(f" (~{eta})", style="dim")

        return text

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of progress tracking.

        Returns:
            Dictionary with progress summary
        """
        return {
            "total_bookmarks": self.total_bookmarks,
            "overall_progress": self.overall_progress,
            "overall_eta_seconds": self.overall_eta.total_seconds(),
            "elapsed_seconds": self.elapsed_time.total_seconds(),
            "memory_mb": self.memory_usage_mb,
            "total_errors": self.total_errors,
            "error_rate": self.overall_error_rate,
            "speed_per_minute": self.overall_speed,
            "stages": {
                name: {
                    "status": stage.status.value,
                    "progress": stage.progress_percentage,
                    "completed": stage.completed,
                    "failed": stage.failed,
                    "elapsed_seconds": stage.elapsed_time.total_seconds(),
                    "eta_seconds": stage.eta.total_seconds() if stage.eta else None,
                }
                for name, stage in self.stages.items()
            },
        }

    def complete(self) -> None:
        """Mark progress tracking as complete."""
        # Complete any in-progress stages
        for stage in self.stages.values():
            if stage.status == StageStatus.IN_PROGRESS:
                stage.complete()

        # Stop live display if running
        self.stop_live_display()


def create_enhanced_tracker(
    total_bookmarks: int,
    stage_weights: Optional[Dict[str, float]] = None,
) -> EnhancedProgressTracker:
    """
    Factory function to create an enhanced progress tracker.

    Args:
        total_bookmarks: Total number of bookmarks to process
        stage_weights: Optional custom stage weights

    Returns:
        Configured EnhancedProgressTracker
    """
    tracker = EnhancedProgressTracker(total_bookmarks=total_bookmarks)

    if stage_weights:
        tracker.STAGE_WEIGHTS = stage_weights
        tracker._init_stages()

    tracker.start(total_bookmarks)
    return tracker
