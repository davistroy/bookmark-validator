"""
Performance Monitoring Module

Provides comprehensive performance monitoring, profiling, and metrics collection
for the bookmark processor application.
"""

import gc
import json
import os
import threading
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Platform-specific imports
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    # Windows doesn't have the resource module
    HAS_RESOURCE = False
    resource = None  # type: ignore

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None  # type: ignore


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""

    timestamp: datetime
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_percent: float
    processing_rate_per_hour: float
    network_requests_count: int
    network_failures_count: int
    processing_time_seconds: float
    items_processed: int
    active_threads: int
    gc_collections: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class ProcessingStageMetrics:
    """Metrics for a specific processing stage"""

    stage_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    items_processed: int
    memory_delta_mb: float
    network_requests: int
    errors_count: int

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["start_time"] = self.start_time.isoformat()
        result["end_time"] = self.end_time.isoformat() if self.end_time else None
        return result


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for bookmark processing
    """

    def __init__(self, target_items: int = 3500, target_hours: int = 8):
        """
        Initialize performance monitor

        Args:
            target_items: Target number of items to process
            target_hours: Target processing time in hours
        """
        self.target_items = target_items
        self.target_hours = target_hours
        self.target_rate = target_items / target_hours  # items per hour

        # Monitoring state
        self.start_time: Optional[datetime] = None
        self.metrics_history: List[PerformanceMetrics] = []
        self.stage_metrics: List[ProcessingStageMetrics] = []
        self.current_stage: Optional[ProcessingStageMetrics] = None

        # Counters
        self.items_processed = 0
        self.network_requests = 0
        self.network_failures = 0
        self.memory_peak = 0.0

        # Memory tracking
        self.memory_tracking_enabled = False
        self.memory_snapshots: List[tuple] = []

        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.lock = threading.RLock()

        # System monitoring
        self.pid = os.getpid()

    def start_monitoring(self, interval_seconds: float = 30.0):
        """Start continuous performance monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.start_time = datetime.now()
        self.stop_monitoring.clear()

        # Start memory tracking
        tracemalloc.start()
        self.memory_tracking_enabled = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_seconds,), daemon=True
        )
        self.monitoring_thread.start()

    def stop_monitoring_session(self):
        """Stop performance monitoring"""
        self.stop_monitoring.set()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        if self.memory_tracking_enabled:
            tracemalloc.stop()
            self.memory_tracking_enabled = False

    def _monitoring_loop(self, interval: float):
        """Background monitoring loop"""
        while not self.stop_monitoring.wait(interval):
            try:
                self._collect_metrics()
            except Exception as e:
                print(f"Error collecting metrics: {e}")

    def _collect_metrics(self):
        """Collect current performance metrics"""
        with self.lock:
            # Memory metrics using psutil (cross-platform) or resource (Unix)
            memory_mb = 0.0
            try:
                if HAS_PSUTIL:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
                elif HAS_RESOURCE and resource is not None:
                    memory_usage = resource.getrusage(resource.RUSAGE_SELF)
                    memory_mb = (
                        memory_usage.ru_maxrss / 1024
                    )  # On Linux, ru_maxrss is in KB
                    if os.name != "posix":
                        memory_mb = (
                            memory_usage.ru_maxrss / 1024 / 1024
                        )  # On Windows, it might be in bytes
                self.memory_peak = max(self.memory_peak, memory_mb)
            except Exception:
                memory_mb = 0.0

            # CPU metrics (simplified)
            cpu_percent = 0.0  # Will be estimated from processing rate

            # Calculate processing rate
            elapsed_hours = self._get_elapsed_hours()
            rate_per_hour = (
                self.items_processed / elapsed_hours if elapsed_hours > 0 else 0
            )

            # GC stats
            gc_stats = {f"gen_{i}": gc.get_count()[i] for i in range(3)}

            # Thread count
            active_threads = threading.active_count()

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                memory_usage_mb=memory_mb,
                memory_peak_mb=self.memory_peak,
                cpu_percent=cpu_percent,
                processing_rate_per_hour=rate_per_hour,
                network_requests_count=self.network_requests,
                network_failures_count=self.network_failures,
                processing_time_seconds=elapsed_hours * 3600,
                items_processed=self.items_processed,
                active_threads=active_threads,
                gc_collections=gc_stats,
            )

            self.metrics_history.append(metrics)

            # Keep only recent metrics (last 24 hours)
            cutoff = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff
            ]

    def start_stage(self, stage_name: str):
        """Start a new processing stage"""
        with self.lock:
            # End current stage if active
            if self.current_stage:
                self.end_current_stage()

            # Start new stage
            self.current_stage = ProcessingStageMetrics(
                stage_name=stage_name,
                start_time=datetime.now(),
                end_time=None,
                duration_seconds=0.0,
                items_processed=0,
                memory_delta_mb=0.0,
                network_requests=0,
                errors_count=0,
            )

    def end_current_stage(self):
        """End the current processing stage"""
        with self.lock:
            if not self.current_stage:
                return

            end_time = datetime.now()
            duration = (end_time - self.current_stage.start_time).total_seconds()

            # Calculate memory delta
            try:
                current_memory = 0.0
                if HAS_PSUTIL:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    current_memory = memory_info.rss / (1024 * 1024)  # bytes to MB
                elif HAS_RESOURCE and resource is not None:
                    memory_usage = resource.getrusage(resource.RUSAGE_SELF)
                    current_memory = memory_usage.ru_maxrss / 1024  # KB to MB
                start_memory = self.memory_peak - current_memory  # Approximation
            except Exception:
                current_memory = 0.0
                start_memory = 0.0

            self.current_stage.end_time = end_time
            self.current_stage.duration_seconds = duration
            self.current_stage.memory_delta_mb = current_memory - start_memory

            self.stage_metrics.append(self.current_stage)
            self.current_stage = None

    def record_item_processed(self):
        """Record that an item has been processed"""
        with self.lock:
            self.items_processed += 1
            if self.current_stage:
                self.current_stage.items_processed += 1

    def record_network_request(self, success: bool = True):
        """Record a network request"""
        with self.lock:
            self.network_requests += 1
            if not success:
                self.network_failures += 1

            if self.current_stage:
                self.current_stage.network_requests += 1

    def record_error(self):
        """Record an error"""
        with self.lock:
            if self.current_stage:
                self.current_stage.errors_count += 1

    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance summary"""
        with self.lock:
            elapsed_hours = self._get_elapsed_hours()
            try:
                memory_mb = 0.0
                if HAS_PSUTIL:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)  # bytes to MB
                elif HAS_RESOURCE and resource is not None:
                    memory_usage = resource.getrusage(resource.RUSAGE_SELF)
                    memory_mb = memory_usage.ru_maxrss / 1024  # KB to MB
            except Exception:
                memory_mb = 0.0

            rate_per_hour = (
                self.items_processed / elapsed_hours if elapsed_hours > 0 else 0
            )
            estimated_completion = (
                (self.target_items - self.items_processed) / rate_per_hour
                if rate_per_hour > 0
                else float("inf")
            )

            progress_percent = (
                (self.items_processed / self.target_items * 100)
                if self.target_items > 0
                else 0
            )

            return {
                "elapsed_hours": elapsed_hours,
                "items_processed": self.items_processed,
                "target_items": self.target_items,
                "progress_percent": progress_percent,
                "current_rate_per_hour": rate_per_hour,
                "target_rate_per_hour": self.target_rate,
                "estimated_completion_hours": estimated_completion,
                "memory_usage_mb": memory_mb,
                "memory_peak_mb": self.memory_peak,
                "network_requests": self.network_requests,
                "network_failures": self.network_failures,
                "network_success_rate": (
                    (self.network_requests - self.network_failures)
                    / self.network_requests
                    * 100
                    if self.network_requests > 0
                    else 0
                ),
                "current_stage": (
                    self.current_stage.stage_name if self.current_stage else None
                ),
                "is_on_track": rate_per_hour >= self.target_rate * 0.9,  # 90% of target
                "memory_under_limit": memory_mb < 4000,  # 4GB limit
            }

    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get detailed performance analysis"""
        current = self.get_current_performance()

        # Stage analysis
        stage_summary = {}
        for stage in self.stage_metrics:
            if stage.stage_name not in stage_summary:
                stage_summary[stage.stage_name] = {
                    "total_duration": 0.0,
                    "total_items": 0,
                    "total_requests": 0,
                    "total_errors": 0,
                    "executions": 0,
                }

            summary = stage_summary[stage.stage_name]
            summary["total_duration"] += stage.duration_seconds
            summary["total_items"] += stage.items_processed
            summary["total_requests"] += stage.network_requests
            summary["total_errors"] += stage.errors_count
            summary["executions"] += 1

        # Calculate averages
        for stage_name, summary in stage_summary.items():
            if summary["executions"] > 0:
                summary["avg_duration"] = (
                    summary["total_duration"] / summary["executions"]
                )
                summary["avg_items_per_execution"] = (
                    summary["total_items"] / summary["executions"]
                )
                summary["items_per_second"] = (
                    summary["total_items"] / summary["total_duration"]
                    if summary["total_duration"] > 0
                    else 0
                )

        # Memory trend analysis
        recent_metrics = (
            self.metrics_history[-10:]
            if len(self.metrics_history) >= 10
            else self.metrics_history
        )
        memory_trend = "stable"
        if len(recent_metrics) >= 2:
            memory_start = recent_metrics[0].memory_usage_mb
            memory_end = recent_metrics[-1].memory_usage_mb
            memory_change = (memory_end - memory_start) / memory_start * 100

            if memory_change > 10:
                memory_trend = "increasing"
            elif memory_change < -10:
                memory_trend = "decreasing"

        return {
            "current_performance": current,
            "stage_analysis": stage_summary,
            "memory_trend": memory_trend,
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations(),
        }

    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        current = self.get_current_performance()

        # Check processing rate
        if current["current_rate_per_hour"] < current["target_rate_per_hour"] * 0.8:
            bottlenecks.append("Processing rate below target (slow processing)")

        # Check memory usage
        if current["memory_usage_mb"] > 3000:  # Warning at 3GB
            bottlenecks.append("High memory usage approaching 4GB limit")

        # Check network failure rate
        if current["network_success_rate"] < 90:
            bottlenecks.append("High network failure rate")

        # Check stage performance (avoid recursion by getting stage summary directly)
        stage_summary = {}
        for stage in self.stage_metrics:
            if stage.stage_name not in stage_summary:
                stage_summary[stage.stage_name] = {
                    "total_duration": 0.0,
                    "total_items": 0,
                    "total_requests": 0,
                    "total_errors": 0,
                    "executions": 0,
                }

            summary = stage_summary[stage.stage_name]
            summary["total_duration"] += stage.duration_seconds
            summary["total_items"] += stage.items_processed
            summary["total_requests"] += stage.network_requests
            summary["total_errors"] += stage.errors_count
            summary["executions"] += 1

        # Calculate averages and check for bottlenecks
        for stage_name, summary in stage_summary.items():
            if summary["executions"] > 0 and summary["total_duration"] > 0:
                items_per_second = summary["total_items"] / summary["total_duration"]
                if items_per_second < 1:  # Less than 1 item per second
                    bottlenecks.append(f"Slow {stage_name} stage processing")

        return bottlenecks

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        bottlenecks = self._identify_bottlenecks()

        if "slow processing" in str(bottlenecks).lower():
            recommendations.extend(
                [
                    "Consider increasing concurrent processing",
                    "Optimize CPU-intensive operations",
                    "Implement batch processing for similar operations",
                ]
            )

        if "memory" in str(bottlenecks).lower():
            recommendations.extend(
                [
                    "Implement data pagination to process in smaller chunks",
                    "Add garbage collection hints at processing boundaries",
                    "Consider using streaming for large data sets",
                ]
            )

        if "network" in str(bottlenecks).lower():
            recommendations.extend(
                [
                    "Implement connection pooling",
                    "Add retry logic with exponential backoff",
                    "Consider request batching",
                ]
            )

        return recommendations

    def _get_elapsed_hours(self) -> float:
        """Get elapsed time in hours"""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds() / 3600

    def save_report(self, file_path: Path):
        """Save performance report to file"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "analysis": self.get_performance_analysis(),
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "stage_metrics": [s.to_dict() for s in self.stage_metrics],
        }

        with open(file_path, "w") as f:
            json.dump(report, f, indent=2)

    @contextmanager
    def measure_stage(self, stage_name: str):
        """Context manager for measuring a processing stage"""
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_current_stage()

    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring_session()


# Global performance monitor instance
global_monitor = PerformanceMonitor()


def start_performance_monitoring(target_items: int = 3500, target_hours: int = 8):
    """Start global performance monitoring"""
    global global_monitor
    global_monitor = PerformanceMonitor(target_items, target_hours)
    global_monitor.start_monitoring()
    return global_monitor


def get_performance_summary() -> Dict[str, Any]:
    """Get current performance summary"""
    return global_monitor.get_current_performance()


def record_item_processed():
    """Record that an item was processed"""
    global_monitor.record_item_processed()


def record_network_request(success: bool = True):
    """Record a network request"""
    global_monitor.record_network_request(success)
