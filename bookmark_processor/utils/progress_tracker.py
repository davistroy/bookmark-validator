"""
Progress Tracking Module

Provides comprehensive progress tracking with real-time ETA calculation,
performance metrics, and detailed status reporting for long-running operations.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import threading
import sys

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.info("tqdm not available, using text-based progress display")


class ProgressLevel(Enum):
    """Progress reporting levels"""
    MINIMAL = "minimal"      # Basic percentage only
    STANDARD = "standard"    # Percentage + ETA + speed
    DETAILED = "detailed"    # All metrics + stage info
    VERBOSE = "verbose"      # Everything + debug info


@dataclass
class StageMetrics:
    """Metrics for a processing stage"""
    name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    items_processed: int = 0
    items_total: int = 0
    errors: int = 0
    warnings: int = 0
    
    @property
    def duration(self) -> timedelta:
        """Get stage duration"""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def items_per_second(self) -> float:
        """Get processing rate"""
        duration_seconds = self.duration.total_seconds()
        if duration_seconds > 0:
            return self.items_processed / duration_seconds
        return 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if stage is complete"""
        return self.end_time is not None
    
    @property
    def progress_percentage(self) -> float:
        """Get progress percentage for this stage"""
        if self.items_total > 0:
            return min(100.0, (self.items_processed / self.items_total) * 100)
        return 0.0


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot"""
    timestamp: datetime
    items_processed: int
    items_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    def __post_init__(self):
        self.timestamp = datetime.now()


class AdvancedProgressTracker:
    """Advanced progress tracking with ETA calculation and performance monitoring"""
    
    def __init__(self, 
                 total_items: int,
                 description: str = "Processing",
                 level: ProgressLevel = ProgressLevel.STANDARD,
                 eta_window_size: int = 20,
                 update_interval: float = 1.0):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the processing task
            level: Level of detail for progress reporting
            eta_window_size: Number of recent measurements for ETA calculation
            update_interval: Minimum seconds between display updates
        """
        self.total_items = total_items
        self.description = description
        self.level = level
        self.eta_window_size = eta_window_size
        self.update_interval = update_interval
        
        # Progress state
        self.current_items = 0
        self.start_time = datetime.now()
        self.last_update_time = 0.0
        self.is_finished = False
        
        # Stages
        self.stages: List[StageMetrics] = []
        self.current_stage: Optional[StageMetrics] = None
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=eta_window_size)
        self.rate_history: deque = deque(maxlen=eta_window_size)
        
        # Progress bar (if available)
        self.progress_bar: Optional[Any] = None
        self._init_progress_bar()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        
        logging.info(f"Progress tracker initialized: {total_items} items, level={level.value}")
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback function for progress updates"""
        self.progress_callbacks.append(callback)
    
    def start_stage(self, stage_name: str, items_total: int = 0) -> None:
        """
        Start a new processing stage.
        
        Args:
            stage_name: Name of the stage
            items_total: Total items for this stage (0 for unknown)
        """
        with self.lock:
            # End previous stage
            if self.current_stage and not self.current_stage.is_complete:
                self.current_stage.end_time = datetime.now()
            
            # Start new stage
            self.current_stage = StageMetrics(
                name=stage_name,
                items_total=items_total
            )
            self.stages.append(self.current_stage)
            
            logging.info(f"Started stage: {stage_name}")
            self._update_display(force=True)
    
    def update(self, 
               items_processed: Optional[int] = None,
               increment: int = 1,
               message: Optional[str] = None,
               stage_items: Optional[int] = None) -> None:
        """
        Update progress.
        
        Args:
            items_processed: Absolute number of items processed
            increment: Number of items to increment by
            message: Optional status message
            stage_items: Items processed in current stage
        """
        with self.lock:
            current_time = time.time()
            
            # Update totals
            if items_processed is not None:
                self.current_items = items_processed
            else:
                self.current_items += increment
            
            # Update current stage
            if self.current_stage:
                if stage_items is not None:
                    self.current_stage.items_processed = stage_items
                else:
                    self.current_stage.items_processed += increment
            
            # Record performance data
            self._record_performance()
            
            # Update display if enough time has passed
            if (current_time - self.last_update_time) >= self.update_interval:
                self._update_display(message=message)
                self.last_update_time = current_time
            
            # Call progress callbacks
            self._call_progress_callbacks()
    
    def add_error(self, error_message: Optional[str] = None) -> None:
        """Record an error"""
        with self.lock:
            if self.current_stage:
                self.current_stage.errors += 1
            
            if error_message and self.level == ProgressLevel.VERBOSE:
                logging.debug(f"Progress error: {error_message}")
    
    def add_warning(self, warning_message: Optional[str] = None) -> None:
        """Record a warning"""
        with self.lock:
            if self.current_stage:
                self.current_stage.warnings += 1
            
            if warning_message and self.level == ProgressLevel.VERBOSE:
                logging.debug(f"Progress warning: {warning_message}")
    
    def finish(self, message: Optional[str] = None) -> None:
        """Mark progress as finished"""
        with self.lock:
            if self.is_finished:
                return
            
            self.is_finished = True
            
            # End current stage
            if self.current_stage and not self.current_stage.is_complete:
                self.current_stage.end_time = datetime.now()
            
            # Final update
            self._update_display(message=message or "Complete", force=True)
            
            if self.progress_bar:
                self.progress_bar.close()
            
            # Print summary
            self._print_summary()
            
            logging.info("Progress tracking finished")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive progress statistics"""
        with self.lock:
            current_time = datetime.now()
            elapsed_time = current_time - self.start_time
            
            stats = {
                'total_items': self.total_items,
                'current_items': self.current_items,
                'progress_percentage': self.get_progress_percentage(),
                'elapsed_time_seconds': elapsed_time.total_seconds(),
                'items_per_second': self.get_current_rate(),
                'eta_seconds': self.get_eta_seconds(),
                'is_finished': self.is_finished,
                'current_stage': self.current_stage.name if self.current_stage else None,
                'total_stages': len(self.stages),
                'total_errors': sum(stage.errors for stage in self.stages),
                'total_warnings': sum(stage.warnings for stage in self.stages)
            }
            
            # Stage details
            stats['stages'] = []
            for stage in self.stages:
                stage_stats = {
                    'name': stage.name,
                    'items_processed': stage.items_processed,
                    'items_total': stage.items_total,
                    'progress_percentage': stage.progress_percentage,
                    'duration_seconds': stage.duration.total_seconds(),
                    'items_per_second': stage.items_per_second,
                    'errors': stage.errors,
                    'warnings': stage.warnings,
                    'is_complete': stage.is_complete
                }
                stats['stages'].append(stage_stats)
            
            return stats
    
    def get_progress_percentage(self) -> float:
        """Get overall progress percentage"""
        if self.total_items > 0:
            return min(100.0, (self.current_items / self.total_items) * 100)
        return 0.0
    
    def get_current_rate(self) -> float:
        """Get current processing rate (items per second)"""
        if len(self.rate_history) < 2:
            return 0.0
        
        # Calculate average rate from recent history
        return sum(self.rate_history) / len(self.rate_history)
    
    def get_eta_seconds(self) -> Optional[float]:
        """Get estimated time to completion in seconds"""
        if self.is_finished or self.total_items <= 0:
            return None
        
        current_rate = self.get_current_rate()
        if current_rate <= 0:
            return None
        
        remaining_items = max(0, self.total_items - self.current_items)
        return remaining_items / current_rate
    
    def get_eta_string(self) -> str:
        """Get ETA as formatted string"""
        eta_seconds = self.get_eta_seconds()
        
        if eta_seconds is None:
            return "Unknown"
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            minutes = eta_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = eta_seconds / 3600
            return f"{hours:.1f}h"
    
    def _init_progress_bar(self) -> None:
        """Initialize progress bar if tqdm is available"""
        if not TQDM_AVAILABLE or self.level == ProgressLevel.MINIMAL:
            return
        
        try:
            self.progress_bar = tqdm(
                total=self.total_items,
                desc=self.description,
                unit="items",
                dynamic_ncols=True,
                leave=True
            )
        except Exception as e:
            logging.debug(f"Could not initialize progress bar: {e}")
            self.progress_bar = None
    
    def _record_performance(self) -> None:
        """Record performance metrics"""
        current_time = time.time()
        
        # Calculate current rate
        if len(self.performance_history) > 0:
            last_snapshot = self.performance_history[-1]
            time_diff = current_time - last_snapshot.timestamp.timestamp()
            items_diff = self.current_items - last_snapshot.items_processed
            
            if time_diff > 0:
                current_rate = items_diff / time_diff
                self.rate_history.append(current_rate)
        
        # Record performance snapshot
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        except ImportError:
            memory_mb = 0.0
            cpu_percent = 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            items_processed=self.current_items,
            items_per_second=self.get_current_rate(),
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent
        )
        
        self.performance_history.append(snapshot)
    
    def _update_display(self, message: Optional[str] = None, force: bool = False) -> None:
        """Update progress display"""
        if self.level == ProgressLevel.MINIMAL and not force:
            return
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.n = self.current_items
            self.progress_bar.refresh()
        
        # Text display for non-tqdm or detailed levels
        if not TQDM_AVAILABLE or self.level in [ProgressLevel.DETAILED, ProgressLevel.VERBOSE]:
            self._print_text_progress(message)
    
    def _print_text_progress(self, message: Optional[str] = None) -> None:
        """Print text-based progress"""
        progress_pct = self.get_progress_percentage()
        rate = self.get_current_rate()
        eta = self.get_eta_string()
        
        # Build progress string
        progress_str = f"{self.description}: {progress_pct:.1f}% "
        progress_str += f"({self.current_items}/{self.total_items})"
        
        if self.level in [ProgressLevel.STANDARD, ProgressLevel.DETAILED, ProgressLevel.VERBOSE]:
            progress_str += f" | {rate:.1f} items/s | ETA: {eta}"
        
        if self.current_stage and self.level in [ProgressLevel.DETAILED, ProgressLevel.VERBOSE]:
            stage_pct = self.current_stage.progress_percentage
            progress_str += f" | Stage: {self.current_stage.name} ({stage_pct:.1f}%)"
        
        if message:
            progress_str += f" | {message}"
        
        # Print with carriage return for overwriting
        print(f"\r{progress_str}", end="", flush=True)
        
        # Add newline for detailed levels
        if self.level in [ProgressLevel.DETAILED, ProgressLevel.VERBOSE]:
            print()
    
    def _print_summary(self) -> None:
        """Print processing summary"""
        elapsed_time = datetime.now() - self.start_time
        avg_rate = self.current_items / elapsed_time.total_seconds() if elapsed_time.total_seconds() > 0 else 0
        
        print(f"\n{self.description} Summary:")
        print(f"  Total items: {self.current_items}/{self.total_items}")
        print(f"  Total time: {elapsed_time}")
        print(f"  Average rate: {avg_rate:.2f} items/second")
        
        if self.stages:
            print(f"  Stages completed: {len([s for s in self.stages if s.is_complete])}/{len(self.stages)}")
        
        total_errors = sum(stage.errors for stage in self.stages)
        total_warnings = sum(stage.warnings for stage in self.stages)
        
        if total_errors > 0:
            print(f"  Errors: {total_errors}")
        if total_warnings > 0:
            print(f"  Warnings: {total_warnings}")
    
    def _call_progress_callbacks(self) -> None:
        """Call registered progress callbacks"""
        if not self.progress_callbacks:
            return
        
        try:
            stats = self.get_statistics()
            for callback in self.progress_callbacks:
                callback(stats)
        except Exception as e:
            logging.debug(f"Progress callback error: {e}")
    
    def close(self) -> None:
        """Clean up resources"""
        if not self.is_finished:
            self.finish()


# Convenience functions for simple progress tracking
def create_progress_tracker(total_items: int, 
                          description: str = "Processing",
                          level: ProgressLevel = ProgressLevel.STANDARD) -> AdvancedProgressTracker:
    """Create a new progress tracker with standard settings"""
    return AdvancedProgressTracker(
        total_items=total_items,
        description=description,
        level=level
    )


def track_progress(iterable, 
                  description: str = "Processing",
                  level: ProgressLevel = ProgressLevel.STANDARD):
    """
    Convenience function to track progress over an iterable.
    
    Usage:
        for item in track_progress(items, "Processing items"):
            # process item
    """
    tracker = create_progress_tracker(
        total_items=len(iterable),
        description=description,
        level=level
    )
    
    try:
        for i, item in enumerate(iterable):
            yield item
            tracker.update(items_processed=i+1)
    finally:
        tracker.finish()