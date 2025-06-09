"""
URL Validation Module

Validates bookmark URLs with intelligent retry logic, rate limiting, and
browser simulation. Handles large datasets efficiently with concurrent
processing and comprehensive error reporting.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from queue import Queue, Empty
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from asyncio import Semaphore

import requests

from ..utils.browser_simulator import BrowserSimulator
from ..utils.intelligent_rate_limiter import IntelligentRateLimiter
from ..utils.retry_handler import RetryHandler
from ..utils.secure_logging import secure_logger
from ..utils.security_validator import SecurityValidationResult, SecurityValidator
from ..utils.progress_tracker import ProgressTracker, ProcessingStage


class ValidationError(Exception):
    """Exception raised when URL validation fails due to configuration or
    system errors."""

    pass


@dataclass
class ValidationResult:
    """Result of URL validation"""

    url: str
    is_valid: bool
    status_code: Optional[int] = None
    final_url: Optional[str] = None  # After redirects
    response_time: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    security_validation: Optional[SecurityValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "url": self.url,
            "is_valid": self.is_valid,
            "status_code": self.status_code,
            "final_url": self.final_url,
            "response_time": self.response_time,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "timestamp": self.timestamp.isoformat(),
            "security_validation": (
                self.security_validation.to_dict() if self.security_validation else None
            ),
        }


@dataclass
class ValidationStats:
    """Statistics for validation process"""

    total_urls: int = 0
    valid_urls: int = 0
    invalid_urls: int = 0
    timeout_urls: int = 0
    error_urls: int = 0
    redirected_urls: int = 0
    total_time: float = 0.0
    average_response_time: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)

    def update_from_result(self, result: ValidationResult) -> None:
        """Update statistics from a validation result"""
        self.total_urls += 1
        self.total_time += result.response_time

        if result.is_valid:
            self.valid_urls += 1
            if result.final_url and result.final_url != result.url:
                self.redirected_urls += 1
        else:
            self.invalid_urls += 1

            if result.error_type:
                self.error_distribution[result.error_type] = (
                    self.error_distribution.get(result.error_type, 0) + 1
                )

                if "timeout" in result.error_type.lower():
                    self.timeout_urls += 1
                else:
                    self.error_urls += 1

        # Update average response time
        if self.total_urls > 0:
            self.average_response_time = self.total_time / self.total_urls


@dataclass
class BatchConfig:
    """Configuration for batch processing"""

    min_batch_size: int = 10
    max_batch_size: int = 500
    optimal_batch_size: int = 100
    auto_tune_batch_size: bool = True
    max_concurrent_batches: int = 3
    batch_timeout: float = 300.0  # 5 minutes per batch
    retry_failed_batches: bool = True
    preserve_order: bool = True
    # New async/concurrent processing options
    enable_async_processing: bool = True
    async_concurrency_limit: int = 50  # Max concurrent async operations
    rate_limit_respect: bool = True
    adaptive_concurrency: bool = True  # Dynamically adjust based on performance
    # Cost tracking options
    enable_cost_tracking: bool = False
    cost_per_url_validation: float = 0.0001  # Default cost per URL validation
    cost_confirmation_threshold: float = 1.0  # USD threshold for user confirmation
    budget_limit: Optional[float] = None  # Optional budget limit in USD


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for batch processing"""

    operation_type: str
    batch_size: int
    estimated_cost_per_item: float
    total_estimated_cost: float
    cost_factors: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation_type": self.operation_type,
            "batch_size": self.batch_size,
            "estimated_cost_per_item": self.estimated_cost_per_item,
            "total_estimated_cost": self.total_estimated_cost,
            "cost_factors": self.cost_factors,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProgressUpdate:
    """Progress update for batch processing"""

    batch_id: str
    total_batches: int
    completed_batches: int
    current_batch_items: int
    current_batch_processed: int
    total_items_processed: int
    total_items_remaining: int
    processing_rate_per_hour: float
    estimated_time_remaining: float
    current_stage: str
    error_count: int
    success_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "batch_id": self.batch_id,
            "total_batches": self.total_batches,
            "completed_batches": self.completed_batches,
            "current_batch_items": self.current_batch_items,
            "current_batch_processed": self.current_batch_processed,
            "total_items_processed": self.total_items_processed,
            "total_items_remaining": self.total_items_remaining,
            "processing_rate_per_hour": self.processing_rate_per_hour,
            "estimated_time_remaining": self.estimated_time_remaining,
            "current_stage": self.current_stage,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "timestamp": self.timestamp.isoformat(),
        }

    def get_progress_percentage(self) -> float:
        """Get progress as percentage (0-100)"""
        if self.total_items_processed + self.total_items_remaining == 0:
            return 0.0
        total_items = self.total_items_processed + self.total_items_remaining
        return (self.total_items_processed / total_items) * 100.0

    def format_progress_text(self) -> str:
        """Format progress as human-readable text"""
        progress_pct = self.get_progress_percentage()
        time_remaining = self.format_time_remaining()

        return (
            f"Batch {self.completed_batches + 1}/{self.total_batches} "
            f"({progress_pct:.1f}% complete) - "
            f"{self.total_items_processed} processed, "
            f"{self.total_items_remaining} remaining - "
            f"Rate: {self.processing_rate_per_hour:.0f}/hr - "
            f"ETA: {time_remaining}"
        )

    def format_time_remaining(self) -> str:
        """Format estimated time remaining as human-readable string"""
        if self.estimated_time_remaining <= 0:
            return "0s"

        hours = int(self.estimated_time_remaining // 3600)
        minutes = int((self.estimated_time_remaining % 3600) // 60)
        seconds = int(self.estimated_time_remaining % 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


@dataclass
class BatchResult:
    """Result of batch processing"""

    batch_id: str
    items_processed: int
    items_successful: int
    items_failed: int
    processing_time: float
    average_item_time: float
    error_rate: float
    results: List[ValidationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    # Cost tracking fields
    actual_cost: Optional[float] = None
    cost_breakdown: Optional[CostBreakdown] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "batch_id": self.batch_id,
            "items_processed": self.items_processed,
            "items_successful": self.items_successful,
            "items_failed": self.items_failed,
            "processing_time": self.processing_time,
            "average_item_time": self.average_item_time,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
        }

        if self.actual_cost is not None:
            result["actual_cost"] = self.actual_cost

        if self.cost_breakdown is not None:
            result["cost_breakdown"] = self.cost_breakdown.to_dict()

        return result


class BatchProcessorInterface(ABC):
    """Abstract interface for batch processors"""

    @abstractmethod
    def process_batch(self, items: List[str], batch_id: str) -> BatchResult:
        """Process a batch of items"""
        pass

    @abstractmethod
    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size based on current performance"""
        pass

    @abstractmethod
    def estimate_processing_time(self, item_count: int) -> float:
        """Estimate processing time for given number of items"""
        pass


class EnhancedBatchProcessor:
    """Enhanced batch processor with configurable batch sizes and intelligent
    optimization"""

    def __init__(
        self,
        processor: BatchProcessorInterface,
        config: Optional[BatchConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_update_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ):
        """
        Initialize enhanced batch processor.

        Args:
            processor: The actual processor implementation
            config: Batch processing configuration
            progress_callback: Optional simple progress callback function (legacy)
            progress_update_callback: Optional detailed progress update callback
        """
        self.processor = processor
        self.config = config or BatchConfig()
        self.progress_callback = progress_callback
        self.progress_update_callback = progress_update_callback

        # Processing state
        self.processing_queue = Queue()
        self.results_queue = Queue()
        self.completed_batches: List[BatchResult] = []
        self.failed_batches: List[Tuple[str, List[str], Exception]] = []

        # Progress tracking state
        self.total_items = 0
        self.total_batches = 0
        self.processing_start_time = 0.0
        self.last_progress_update = 0.0
        self.total_errors = 0
        self.total_successes = 0

        # Performance tracking
        self.performance_history: List[Tuple[int, float]] = (
            []
        )  # (batch_size, avg_time_per_item)
        self.current_batch_size = self.config.optimal_batch_size

        # Threading and async coordination
        self.lock = threading.RLock()
        self.stop_event = threading.Event()

        # Async processing components
        self.async_semaphore: Optional[Semaphore] = None
        self.async_session: Optional[aiohttp.ClientSession] = None
        self.current_concurrency_limit = self.config.async_concurrency_limit

        # Rate limiting coordination
        self.rate_limit_tracker: Dict[str, float] = {}  # domain -> last_request_time
        self.domain_semaphores: Dict[str, Semaphore] = {}  # per-domain rate limiting

        # Cost tracking state
        self.total_session_cost = 0.0
        self.batch_cost_history: List[Tuple[str, float]] = []  # (batch_id, cost)
        self.cost_estimates: Dict[str, CostBreakdown] = {}  # batch_id -> cost_breakdown

        logging.info(f"Initialized enhanced batch processor with config: {self.config}")

        if self.config.enable_cost_tracking:
            logging.info(
                f"Cost tracking enabled: "
                f"cost_per_validation=${self.config.cost_per_url_validation:.6f}, "
                f"threshold=${self.config.cost_confirmation_threshold:.2f}, "
                f"budget_limit=${self.config.budget_limit}"
            )

    def estimate_batch_cost(
        self, item_count: int, operation_type: str = "url_validation"
    ) -> CostBreakdown:
        """
        Estimate cost for processing a batch of items.

        Args:
            item_count: Number of items in the batch
            operation_type: Type of operation being performed

        Returns:
            CostBreakdown with detailed cost estimation
        """
        if not self.config.enable_cost_tracking:
            return CostBreakdown(
                operation_type=operation_type,
                batch_size=item_count,
                estimated_cost_per_item=0.0,
                total_estimated_cost=0.0,
                cost_factors={"cost_tracking_disabled": 0.0},
            )

        # Base cost per item
        base_cost_per_item = self.config.cost_per_url_validation

        # Apply cost factors based on operation complexity
        cost_factors = {
            "base_url_validation": base_cost_per_item,
            "network_overhead": base_cost_per_item
            * 0.1,  # 10% overhead for network requests
            "processing_overhead": base_cost_per_item * 0.05,  # 5% for processing
        }

        # Adjust for batch size efficiency (larger batches are more efficient)
        if item_count > 100:
            cost_factors["bulk_discount"] = (
                -base_cost_per_item * 0.15
            )  # 15% discount for bulk
        elif item_count < 10:
            cost_factors["small_batch_premium"] = (
                base_cost_per_item * 0.2
            )  # 20% premium for small batches

        # Calculate total cost per item
        total_cost_per_item = sum(cost_factors.values())

        # Ensure minimum cost
        total_cost_per_item = max(total_cost_per_item, 0.00001)

        total_estimated_cost = item_count * total_cost_per_item

        return CostBreakdown(
            operation_type=operation_type,
            batch_size=item_count,
            estimated_cost_per_item=total_cost_per_item,
            total_estimated_cost=total_estimated_cost,
            cost_factors=cost_factors,
        )

    async def check_budget_and_confirm(self, estimated_cost: float) -> bool:
        """
        Check budget limits and get user confirmation if needed.

        Args:
            estimated_cost: Estimated cost for the operation

        Returns:
            True if processing should continue, False otherwise
        """
        if not self.config.enable_cost_tracking:
            return True

        # Check budget limit
        if self.config.budget_limit is not None:
            projected_total = self.total_session_cost + estimated_cost
            if projected_total > self.config.budget_limit:
                logging.warning(
                    f"Budget limit exceeded: ${projected_total:.4f} > "
                    f"${self.config.budget_limit:.2f}"
                )

                # Use cost tracker for detailed confirmation if available
                if self.cost_tracker:
                    prompt = f"""
🚨 Budget Limit Warning:
  💰 Current session cost: ${self.total_session_cost:.4f}
  📈 Estimated operation cost: ${estimated_cost:.4f}
  💳 Budget limit: ${self.config.budget_limit:.2f}
  ⚠️  Projected total: ${projected_total:.4f}

❓ Continue anyway? (y/n): """

                    try:
                        response = await asyncio.get_event_loop().run_in_executor(
                            None, input, prompt
                        )
                        return response.lower().strip() in ["y", "yes"]
                    except (KeyboardInterrupt, EOFError):
                        return False
                else:
                    return False

        # Check confirmation threshold
        if estimated_cost >= self.config.cost_confirmation_threshold:
            # Use cost tracker for detailed confirmation if available
            if self.cost_tracker:
                # Add temporary cost record for estimation
                self.cost_tracker.add_cost_record(
                    provider="url_validation",
                    model="batch_processor",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=estimated_cost,
                    operation_type="batch_estimation",
                    bookmark_count=1,
                    success=True,
                )

                confirmation_result = await self.cost_tracker.confirm_continuation()

                # Remove the temporary record (subtract the cost)
                self.cost_tracker.session_cost -= estimated_cost
                self.cost_tracker.total_cost -= estimated_cost
                if self.cost_tracker.cost_records:
                    self.cost_tracker.cost_records.pop()

                return confirmation_result
            else:
                # Simple confirmation without cost tracker
                prompt = f"""
💰 Cost Confirmation Required:
  📊 Estimated batch cost: ${estimated_cost:.4f}
  💵 Current session total: ${self.total_session_cost:.4f}
  🔮 Projected total: ${self.total_session_cost + estimated_cost:.4f}

❓ Continue with this batch? (y/n): """

                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, input, prompt
                    )
                    return response.lower().strip() in ["y", "yes"]
                except (KeyboardInterrupt, EOFError):
                    return False

        return True

    def record_batch_cost(self, batch_id: str, actual_cost: float) -> None:
        """
        Record the actual cost of a processed batch.

        Args:
            batch_id: Unique identifier for the batch
            actual_cost: Actual cost incurred
        """
        if not self.config.enable_cost_tracking:
            return

        with self.lock:
            self.total_session_cost += actual_cost
            self.batch_cost_history.append((batch_id, actual_cost))

            # Keep only recent history to manage memory
            if len(self.batch_cost_history) > 100:
                self.batch_cost_history = self.batch_cost_history[-100:]

            logging.debug(f"Recorded batch cost: {batch_id} = ${actual_cost:.6f}")

            # Record in cost tracker if available
            if self.cost_tracker:
                self.cost_tracker.add_cost_record(
                    provider="url_validation",
                    model="batch_processor",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=actual_cost,
                    operation_type="url_validation_batch",
                    bookmark_count=1,
                    success=True,
                )

    def get_cost_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cost statistics for the session.

        Returns:
            Dictionary with detailed cost statistics
        """
        if not self.config.enable_cost_tracking:
            return {"cost_tracking_enabled": False}

        with self.lock:
            stats = {
                "cost_tracking_enabled": True,
                "total_session_cost": self.total_session_cost,
                "batch_count": len(self.batch_cost_history),
                "average_batch_cost": (
                    sum(cost for _, cost in self.batch_cost_history)
                    / max(len(self.batch_cost_history), 1)
                ),
                "cost_per_url_validation": self.config.cost_per_url_validation,
                "confirmation_threshold": self.config.cost_confirmation_threshold,
                "budget_limit": self.config.budget_limit,
                "budget_remaining": (
                    self.config.budget_limit - self.total_session_cost
                    if self.config.budget_limit
                    else None
                ),
            }

            if self.batch_cost_history:
                recent_costs = [cost for _, cost in self.batch_cost_history[-10:]]
                stats.update(
                    {
                        "recent_average_cost": sum(recent_costs) / len(recent_costs),
                        "min_batch_cost": min(
                            cost for _, cost in self.batch_cost_history
                        ),
                        "max_batch_cost": max(
                            cost for _, cost in self.batch_cost_history
                        ),
                        "cost_trend": self._calculate_cost_trend(),
                    }
                )

            # Include cost tracker statistics if available
            if self.cost_tracker:
                stats["cost_tracker_stats"] = (
                    self.cost_tracker.get_detailed_statistics()
                )

            return stats

    def _calculate_cost_trend(self) -> str:
        """Calculate cost trend based on recent batches."""
        if len(self.batch_cost_history) < 3:
            return "insufficient_data"

        recent_costs = [cost for _, cost in self.batch_cost_history[-5:]]
        early_costs = (
            [cost for _, cost in self.batch_cost_history[-10:-5]]
            if len(self.batch_cost_history) >= 10
            else []
        )

        if not early_costs:
            return "insufficient_data"

        recent_avg = sum(recent_costs) / len(recent_costs)
        early_avg = sum(early_costs) / len(early_costs)

        change_percent = ((recent_avg - early_avg) / early_avg) * 100

        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"

    def _check_budget_and_confirm_sync(self, estimated_cost: float) -> bool:
        """
        Synchronous version of budget checking and user confirmation.

        Args:
            estimated_cost: Estimated cost for the operation

        Returns:
            True if processing should continue, False otherwise
        """
        if not self.config.enable_cost_tracking:
            return True

        # Check budget limit
        if self.config.budget_limit is not None:
            projected_total = self.total_session_cost + estimated_cost
            if projected_total > self.config.budget_limit:
                logging.warning(
                    f"Budget limit exceeded: ${projected_total:.4f} > "
                    f"${self.config.budget_limit:.2f}"
                )

                # Use cost tracker for detailed confirmation if available
                if self.cost_tracker:
                    prompt = f"""
🚨 Budget Limit Warning:
  💰 Current session cost: ${self.total_session_cost:.4f}
  📈 Estimated operation cost: ${estimated_cost:.4f}
  💳 Budget limit: ${self.config.budget_limit:.2f}
  ⚠️  Projected total: ${projected_total:.4f}

❓ Continue anyway? (y/n): """

                    try:
                        response = input(prompt)
                        return response.lower().strip() in ["y", "yes"]
                    except (KeyboardInterrupt, EOFError):
                        return False
                else:
                    return False

        # Check confirmation threshold
        if estimated_cost >= self.config.cost_confirmation_threshold:
            # Use cost tracker for detailed confirmation if available
            if self.cost_tracker:
                # Add temporary cost record for estimation
                self.cost_tracker.add_cost_record(
                    provider="url_validation",
                    model="batch_processor",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=estimated_cost,
                    operation_type="batch_estimation",
                    bookmark_count=1,
                    success=True,
                )

                # Get confirmation using cost tracker's prompt system
                try:
                    prompt = self.cost_tracker.get_confirmation_prompt()
                    response = input(prompt)
                    confirmation_result = response.lower().strip() in ["y", "yes", ""]

                    # Update confirmation tracking if user agrees
                    if confirmation_result:
                        self.cost_tracker.last_confirmation = (
                            self.cost_tracker.session_cost
                        )
                        self.cost_tracker.user_confirmed_cost = (
                            self.cost_tracker.session_cost
                        )

                except (KeyboardInterrupt, EOFError):
                    confirmation_result = False

                # Remove the temporary record (subtract the cost)
                self.cost_tracker.session_cost -= estimated_cost
                self.cost_tracker.total_cost -= estimated_cost
                if self.cost_tracker.cost_records:
                    self.cost_tracker.cost_records.pop()

                return confirmation_result
            else:
                # Simple confirmation without cost tracker
                prompt = f"""
💰 Cost Confirmation Required:
  📊 Estimated batch cost: ${estimated_cost:.4f}
  💵 Current session total: ${self.total_session_cost:.4f}
  🔮 Projected total: ${self.total_session_cost + estimated_cost:.4f}

❓ Continue with this batch? (y/n): """

                try:
                    response = input(prompt)
                    return response.lower().strip() in ["y", "yes"]
                except (KeyboardInterrupt, EOFError):
                    return False

        return True

    def add_items(self, items: List[str]) -> bool:
        """
        Add items to the processing queue with cost estimation and confirmation.

        Args:
            items: List of items to process

        Returns:
            True if items were added successfully, False if cancelled due to cost limits
        """
        with self.lock:
            # Estimate total cost if cost tracking is enabled
            if self.config.enable_cost_tracking:
                cost_breakdown = self.estimate_batch_cost(
                    len(items), "url_validation_batch"
                )
                logging.info(
                    f"Cost estimation for {len(items)} items: "
                    f"${cost_breakdown.total_estimated_cost:.6f}"
                )

                # Store cost estimate
                total_batch_id = f"total_processing_{int(time.time())}"
                self.cost_estimates[total_batch_id] = cost_breakdown

                # Check budget and get confirmation if needed (sync version)
                confirmation_result = self._check_budget_and_confirm_sync(
                    cost_breakdown.total_estimated_cost
                )
                if not confirmation_result:
                    logging.info("Processing cancelled by user due to cost concerns")
                    return False

            # Initialize progress tracking
            self.total_items = len(items)
            self.processing_start_time = time.time()
            self.last_progress_update = time.time()
            self.total_errors = 0
            self.total_successes = 0

            # Split items into batches
            batches = self._create_batches(items)
            self.total_batches = len(batches)

            # Estimate and store cost for each batch if tracking enabled
            if self.config.enable_cost_tracking:
                for batch_id, batch_items in batches:
                    batch_cost_breakdown = self.estimate_batch_cost(
                        len(batch_items), "url_validation"
                    )
                    self.cost_estimates[batch_id] = batch_cost_breakdown

            for batch_id, batch_items in batches:
                self.processing_queue.put((batch_id, batch_items))

            logging.info(
                f"Added {len(items)} items in {len(batches)} batches to "
                f"processing queue"
            )

            # Send initial progress update
            self._emit_progress_update("initialized", "batch_0", 0, 0)

            return True

    def _emit_progress_update(
        self,
        current_stage: str,
        batch_id: str,
        current_batch_processed: int,
        current_batch_total: int,
    ) -> None:
        """Emit a progress update if callback is provided"""
        if not self.progress_update_callback:
            return

        current_time = time.time()

        with self.lock:
            # Calculate current totals
            total_processed = (
                sum(batch.items_processed for batch in self.completed_batches)
                + current_batch_processed
            )
            total_remaining = self.total_items - total_processed

            # Calculate processing rate (items per hour)
            elapsed_time = current_time - self.processing_start_time
            if elapsed_time > 0:
                processing_rate_per_hour = (total_processed / elapsed_time) * 3600
            else:
                processing_rate_per_hour = 0.0

            # Calculate ETA
            if processing_rate_per_hour > 0 and total_remaining > 0:
                estimated_time_remaining = total_remaining / (
                    processing_rate_per_hour / 3600
                )
            else:
                estimated_time_remaining = 0.0

            # Calculate success rate
            if total_processed > 0:
                success_rate = self.total_successes / total_processed
            else:
                success_rate = 0.0

            # Create progress update
            progress_update = ProgressUpdate(
                batch_id=batch_id,
                total_batches=self.total_batches,
                completed_batches=len(self.completed_batches),
                current_batch_items=current_batch_total,
                current_batch_processed=current_batch_processed,
                total_items_processed=total_processed,
                total_items_remaining=total_remaining,
                processing_rate_per_hour=processing_rate_per_hour,
                estimated_time_remaining=estimated_time_remaining,
                current_stage=current_stage,
                error_count=self.total_errors,
                success_rate=success_rate,
                timestamp=datetime.now(),
            )

            # Emit update (don't fail if callback fails)
            try:
                self.progress_update_callback(progress_update)
                self.last_progress_update = current_time
            except Exception as e:
                logging.debug(f"Progress update callback failed: {e}")

    def _update_progress_counters(self, batch_result: BatchResult) -> None:
        """Update progress tracking counters from batch result"""
        with self.lock:
            self.total_successes += batch_result.items_successful
            self.total_errors += batch_result.items_failed

    def process_all(self) -> List[ValidationResult]:
        """
        Process all items in the queue with concurrent batch processing.
        Uses async processing if enabled and supported.

        Returns:
            List of all validation results
        """
        if self.config.enable_async_processing and hasattr(
            self.processor, "async_validate_batch"
        ):
            return self._process_all_async()
        else:
            return self._process_all_sync()

    def _process_all_sync(self) -> List[ValidationResult]:
        """
        Synchronous batch processing (original implementation).

        Returns:
            List of all validation results
        """
        start_time = time.time()
        all_results = []

        # Process batches concurrently
        with ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_batches
        ) as executor:
            # Submit all batches
            future_to_batch = {}

            while not self.processing_queue.empty():
                try:
                    batch_id, batch_items = self.processing_queue.get_nowait()
                    future = executor.submit(
                        self._process_single_batch, batch_id, batch_items
                    )
                    future_to_batch[future] = (batch_id, batch_items)
                except Empty:
                    break

            # Collect results
            processed_batches = 0
            total_batches = len(future_to_batch)

            for future in as_completed(future_to_batch):
                batch_id, batch_items = future_to_batch[future]

                try:
                    batch_result = future.result(timeout=self.config.batch_timeout)
                    self.completed_batches.append(batch_result)
                    all_results.extend(batch_result.results)

                    # Update performance tracking
                    self._update_performance_metrics(batch_result)

                    # Update progress tracking counters
                    self._update_progress_counters(batch_result)

                    processed_batches += 1

                    # Emit detailed progress update
                    self._emit_progress_update(
                        "processing_batch",
                        batch_result.batch_id,
                        batch_result.items_processed,
                        batch_result.items_processed,
                    )

                    if self.progress_callback:
                        self.progress_callback(
                            f"Completed batch {processed_batches}/{total_batches}: "
                            f"{batch_result.items_successful}/"
                            f"{batch_result.items_processed} successful"
                        )

                except Exception as e:
                    logging.error(f"Batch {batch_id} failed: {e}")
                    self.failed_batches.append((batch_id, batch_items, e))

                    # Add failed batch to retry queue if enabled
                    if self.config.retry_failed_batches:
                        retry_batch_id = f"{batch_id}_retry"
                        self.processing_queue.put((retry_batch_id, batch_items))

        # Handle any remaining failed batches
        if self.config.retry_failed_batches and self.failed_batches:
            retry_results = self._retry_failed_batches()
            all_results.extend(retry_results)

        total_time = time.time() - start_time

        # Auto-tune batch size if enabled
        if self.config.auto_tune_batch_size:
            self._auto_tune_batch_size()

        # Emit final progress update for sync processing completion
        self._emit_progress_update(
            "sync_processing_completed", "final", len(all_results), len(all_results)
        )

        logging.info(
            f"Sync batch processing completed in {total_time:.2f}s: "
            f"{len(all_results)} items processed across "
            f"{len(self.completed_batches)} batches"
        )

        return all_results

    def _process_all_async(self) -> List[ValidationResult]:
        """
        Asynchronous batch processing with rate limiting.

        Returns:
            List of all validation results
        """
        try:
            # Get or create an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run async processing
            return loop.run_until_complete(self._async_process_batches())

        except Exception as e:
            logging.error(f"Async processing failed, falling back to sync: {e}")
            return self._process_all_sync()

    async def _async_process_batches(self) -> List[ValidationResult]:
        """
        Core async batch processing logic.

        Returns:
            List of all validation results
        """
        start_time = time.time()
        all_results = []

        # Initialize async components
        await self._initialize_async_components()

        try:
            # Collect all batches
            batches = []
            while not self.processing_queue.empty():
                try:
                    batch_id, batch_items = self.processing_queue.get_nowait()
                    batches.append((batch_id, batch_items))
                except Empty:
                    break

            if not batches:
                return all_results

            # Create semaphore for concurrent batch processing
            batch_semaphore = Semaphore(self.config.max_concurrent_batches)

            # Process batches concurrently with rate limiting
            tasks = []
            for batch_id, batch_items in batches:
                task = self._async_process_single_batch(
                    batch_semaphore, batch_id, batch_items
                )
                tasks.append(task)

            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful_batches = 0
            for i, result in enumerate(batch_results):
                batch_id, batch_items = batches[i]

                if isinstance(result, Exception):
                    logging.error(f"Async batch {batch_id} failed: {result}")
                    self.failed_batches.append((batch_id, batch_items, result))
                else:
                    self.completed_batches.append(result)
                    all_results.extend(result.results)
                    self._update_performance_metrics(result)

                    # Update progress tracking counters
                    self._update_progress_counters(result)

                    successful_batches += 1

                    # Emit detailed progress update
                    self._emit_progress_update(
                        "async_processing_batch",
                        result.batch_id,
                        result.items_processed,
                        result.items_processed,
                    )

                    if self.progress_callback:
                        self.progress_callback(
                            f"Completed async batch {successful_batches}/"
                            f"{len(batches)}: {result.items_successful}/"
                            f"{result.items_processed} successful"
                        )

            # Handle retries for failed batches
            if self.config.retry_failed_batches and self.failed_batches:
                retry_results = await self._async_retry_failed_batches()
                all_results.extend(retry_results)

        finally:
            # Cleanup async components
            await self._cleanup_async_components()

        total_time = time.time() - start_time

        # Auto-tune batch size and concurrency
        if self.config.auto_tune_batch_size:
            self._auto_tune_batch_size()
        if self.config.adaptive_concurrency:
            self._adapt_concurrency_limits()

        # Emit final progress update for async processing completion
        self._emit_progress_update(
            "async_processing_completed", "final", len(all_results),
            len(all_results)
        )

        logging.info(
            f"Async batch processing completed in {total_time:.2f}s: "
            f"{len(all_results)} items processed across "
            f"{len(self.completed_batches)} batches"
        )

        return all_results

    def _create_batches(self, items: List[str]) -> List[Tuple[str, List[str]]]:
        """Create batches from items list"""
        batches = []
        current_batch_size = self.current_batch_size

        for i in range(0, len(items), current_batch_size):
            batch_items = items[i : i + current_batch_size]
            batch_id = f"batch_{len(batches):04d}_{int(time.time())}"
            batches.append((batch_id, batch_items))

        return batches

    def _process_single_batch(
        self, batch_id: str, batch_items: List[str]
    ) -> BatchResult:
        """Process a single batch of items"""
        start_time = time.time()

        try:
            # Emit progress update for batch start
            self._emit_progress_update(
                "processing_batch", batch_id, 0, len(batch_items)
            )

            # Use the processor to handle the batch
            result = self.processor.process_batch(batch_items, batch_id)

            processing_time = time.time() - start_time
            result.processing_time = processing_time

            # Calculate and record actual cost if tracking enabled
            if self.config.enable_cost_tracking:
                # Get cost estimate for this batch
                if batch_id in self.cost_estimates:
                    cost_breakdown = self.cost_estimates[batch_id]
                    # For URL validation, actual cost is typically the same as estimated
                    # In real implementations, this could be based on actual
                    # resource usage
                    actual_cost = cost_breakdown.total_estimated_cost

                    # Apply success rate adjustment to actual cost
                    if result.items_processed > 0:
                        success_rate = result.items_successful / result.items_processed
                        # Reduce cost based on success rate (failed operations
                        # might cost less)
                        actual_cost = actual_cost * (0.5 + 0.5 * success_rate)

                    result.actual_cost = actual_cost
                    result.cost_breakdown = cost_breakdown

                    # Record the cost
                    self.record_batch_cost(batch_id, actual_cost)
                else:
                    # Fallback cost calculation if no estimate available
                    actual_cost = len(batch_items) * self.config.cost_per_url_validation
                    result.actual_cost = actual_cost
                    self.record_batch_cost(batch_id, actual_cost)

            # Emit progress update for batch completion
            self._emit_progress_update(
                "completed_batch", batch_id, result.items_processed, len(batch_items)
            )

            logging.debug(
                f"Batch {batch_id} completed: {result.items_successful}/"
                f"{result.items_processed} successful in {processing_time:.2f}s"
                + (f", cost: ${result.actual_cost:.6f}" if result.actual_cost else "")
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Error processing batch {batch_id}: {e}")

            # Calculate partial cost for failed batch
            actual_cost = None
            if self.config.enable_cost_tracking:
                # Failed batches still incur some cost (partial processing)
                actual_cost = (
                    len(batch_items) * self.config.cost_per_url_validation * 0.1
                )  # 10% cost for failures
                self.record_batch_cost(batch_id, actual_cost)

            # Return error result
            return BatchResult(
                batch_id=batch_id,
                items_processed=len(batch_items),
                items_successful=0,
                items_failed=len(batch_items),
                processing_time=processing_time,
                average_item_time=(
                    processing_time / len(batch_items) if batch_items else 0
                ),
                error_rate=1.0,
                errors=[f"Batch processing failed: {str(e)}"],
                actual_cost=actual_cost,
            )

    def _update_performance_metrics(self, batch_result: BatchResult) -> None:
        """Update performance metrics based on batch result"""
        with self.lock:
            if batch_result.average_item_time > 0:
                self.performance_history.append(
                    (batch_result.items_processed, batch_result.average_item_time)
                )

                # Keep only recent history (last 20 batches)
                if len(self.performance_history) > 20:
                    self.performance_history = self.performance_history[-20:]

    def _auto_tune_batch_size(self) -> None:
        """Automatically tune batch size based on performance history"""
        if len(self.performance_history) < 3:
            return

        # Calculate average time per item for different batch sizes
        size_performance = {}
        for batch_size, avg_time in self.performance_history[-10:]:
            if batch_size not in size_performance:
                size_performance[batch_size] = []
            size_performance[batch_size].append(avg_time)

        # Find optimal batch size (lowest average time per item)
        best_size = self.current_batch_size
        best_time = float("inf")

        for size, times in size_performance.items():
            if len(times) >= 2:  # Need at least 2 samples
                avg_time = sum(times) / len(times)
                if avg_time < best_time:
                    best_time = avg_time
                    best_size = size

        # Adjust batch size within constraints
        new_size = max(
            self.config.min_batch_size, min(self.config.max_batch_size, best_size)
        )

        if new_size != self.current_batch_size:
            logging.info(
                f"Auto-tuning batch size from {self.current_batch_size} to {new_size}"
            )
            self.current_batch_size = new_size

    def _retry_failed_batches(self) -> List[ValidationResult]:
        """Retry failed batches with smaller batch sizes"""
        retry_results = []

        for batch_id, batch_items, error in self.failed_batches:
            logging.info(
                f"Retrying failed batch {batch_id} with {len(batch_items)} items"
            )

            # Emit progress update for retry start
            self._emit_progress_update("retrying_batch", batch_id, 0, len(batch_items))

            # Retry with smaller batch size
            smaller_batch_size = max(self.config.min_batch_size, len(batch_items) // 2)

            # Split into smaller batches
            for i in range(0, len(batch_items), smaller_batch_size):
                retry_items = batch_items[i : i + smaller_batch_size]
                retry_batch_id = f"{batch_id}_retry_{i//smaller_batch_size}"

                try:
                    # Emit progress update for retry batch start
                    self._emit_progress_update(
                        "processing_retry_batch", retry_batch_id, 0, len(retry_items)
                    )

                    retry_result = self.processor.process_batch(
                        retry_items, retry_batch_id
                    )
                    retry_results.extend(retry_result.results)

                    # Update progress tracking counters
                    self._update_progress_counters(retry_result)

                    # Emit progress update for retry batch completion
                    self._emit_progress_update(
                        "completed_retry_batch",
                        retry_batch_id,
                        retry_result.items_processed,
                        len(retry_items),
                    )

                except Exception as e:
                    logging.error(f"Retry batch {retry_batch_id} also failed: {e}")

        return retry_results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        with self.lock:
            total_items = sum(batch.items_processed for batch in self.completed_batches)
            total_successful = sum(
                batch.items_successful for batch in self.completed_batches
            )
            total_time = sum(batch.processing_time for batch in self.completed_batches)

            # Calculate total cost from completed batches
            total_batch_cost = sum(
                batch.actual_cost
                for batch in self.completed_batches
                if batch.actual_cost is not None
            )

            stats = {
                "total_batches": len(self.completed_batches),
                "failed_batches": len(self.failed_batches),
                "total_items": total_items,
                "successful_items": total_successful,
                "success_rate": (
                    total_successful / total_items if total_items > 0 else 0
                ),
                "total_processing_time": total_time,
                "average_batch_size": (
                    total_items / len(self.completed_batches)
                    if self.completed_batches
                    else 0
                ),
                "current_optimal_batch_size": self.current_batch_size,
                "performance_history_length": len(self.performance_history),
            }

            # Add cost statistics if cost tracking is enabled
            if self.config.enable_cost_tracking:
                cost_stats = self.get_cost_statistics()
                stats.update(
                    {
                        "cost_tracking": cost_stats,
                        "total_batch_cost": total_batch_cost,
                        "average_cost_per_item": (
                            total_batch_cost / total_items if total_items > 0 else 0
                        ),
                        "average_cost_per_batch": (
                            total_batch_cost / len(self.completed_batches)
                            if self.completed_batches
                            else 0
                        ),
                    }
                )

                # Add cost efficiency metrics
                if total_time > 0:
                    stats["cost_per_second"] = total_batch_cost / total_time
                if total_successful > 0:
                    stats["cost_per_successful_item"] = (
                        total_batch_cost / total_successful
                    )

            if self.performance_history:
                recent_times = [time for _, time in self.performance_history[-5:]]
                stats["recent_average_time_per_item"] = sum(recent_times) / len(
                    recent_times
                )

            return stats

    async def _initialize_async_components(self) -> None:
        """Initialize async processing components"""
        if self.async_semaphore is None:
            self.async_semaphore = Semaphore(self.current_concurrency_limit)

        if self.async_session is None:
            # Create aiohttp session with appropriate settings
            timeout = aiohttp.ClientTimeout(total=30.0, connect=10.0)
            connector = aiohttp.TCPConnector(
                limit=self.current_concurrency_limit,
                limit_per_host=10,  # Max connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )

            self.async_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "BookmarkProcessor/1.0"},
            )

        logging.debug(
            f"Initialized async components with concurrency limit: "
            f"{self.current_concurrency_limit}"
        )

    async def _cleanup_async_components(self) -> None:
        """Cleanup async processing components"""
        if self.async_session:
            await self.async_session.close()
            self.async_session = None

        # Clear domain semaphores
        self.domain_semaphores.clear()

        logging.debug("Cleaned up async components")

    async def _async_process_single_batch(
        self, batch_semaphore: Semaphore, batch_id: str, batch_items: List[str]
    ) -> BatchResult:
        """
        Process a single batch asynchronously with rate limiting.

        Args:
            batch_semaphore: Semaphore to limit concurrent batches
            batch_id: Unique identifier for this batch
            batch_items: List of items to process

        Returns:
            BatchResult with processing details
        """
        async with batch_semaphore:
            start_time = time.time()

            try:
                # Emit progress update for async batch start
                self._emit_progress_update(
                    "async_processing_batch", batch_id, 0, len(batch_items)
                )

                # Check if processor supports async processing
                if hasattr(self.processor, "async_validate_batch"):
                    # Use async method if available
                    results = await self.processor.async_validate_batch(
                        batch_items, batch_id
                    )
                else:
                    # Fall back to sync processing in thread pool
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None, self.processor.process_batch, batch_items, batch_id
                    )

                processing_time = time.time() - start_time
                results.processing_time = processing_time

                # Emit progress update for async batch completion
                self._emit_progress_update(
                    "async_completed_batch",
                    batch_id,
                    results.items_processed,
                    len(batch_items),
                )

                logging.debug(
                    f"Async batch {batch_id} completed: "
                    f"{results.items_successful}/{results.items_processed} "
                    f"successful in {processing_time:.2f}s"
                )

                return results

            except Exception as e:
                processing_time = time.time() - start_time
                logging.error(f"Error processing async batch {batch_id}: {e}")

                # Return error result
                return BatchResult(
                    batch_id=batch_id,
                    items_processed=len(batch_items),
                    items_successful=0,
                    items_failed=len(batch_items),
                    processing_time=processing_time,
                    average_item_time=(
                        processing_time / len(batch_items) if batch_items else 0
                    ),
                    error_rate=1.0,
                    errors=[f"Async batch processing failed: {str(e)}"],
                )

    async def _async_retry_failed_batches(self) -> List[ValidationResult]:
        """
        Retry failed batches asynchronously with smaller batch sizes.

        Returns:
            List of retry results
        """
        retry_results = []

        # Create semaphore for retry processing
        retry_semaphore = Semaphore(max(1, self.config.max_concurrent_batches // 2))

        tasks = []
        for batch_id, batch_items, error in self.failed_batches:
            logging.info(
                f"Retrying failed async batch {batch_id} with {len(batch_items)} items"
            )

            # Emit progress update for async retry start
            self._emit_progress_update(
                "async_retrying_batch", batch_id, 0, len(batch_items)
            )

            # Split into smaller batches for retry
            smaller_batch_size = max(self.config.min_batch_size, len(batch_items) // 2)

            for i in range(0, len(batch_items), smaller_batch_size):
                retry_items = batch_items[i : i + smaller_batch_size]
                retry_batch_id = f"{batch_id}_async_retry_{i//smaller_batch_size}"

                task = self._async_process_single_batch(
                    retry_semaphore, retry_batch_id, retry_items
                )
                tasks.append(task)

        if tasks:
            retry_batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in retry_batch_results:
                if isinstance(result, Exception):
                    logging.error(f"Async retry failed: {result}")
                else:
                    retry_results.extend(result.results)

                    # Update progress tracking counters for successful async retries
                    self._update_progress_counters(result)

                    # Emit progress update for async retry completion
                    self._emit_progress_update(
                        "async_completed_retry_batch",
                        result.batch_id,
                        result.items_processed,
                        len(result.results),
                    )

        return retry_results

    def _adapt_concurrency_limits(self) -> None:
        """Adapt concurrency limits based on performance metrics"""
        if len(self.performance_history) < 3:
            return

        # Calculate recent performance
        recent_performance = self.performance_history[-5:]
        avg_time = sum(time for _, time in recent_performance) / len(recent_performance)

        # Adjust concurrency based on performance
        if avg_time > 5.0:  # Slow performance
            new_limit = max(10, self.current_concurrency_limit - 10)
        elif avg_time < 1.0:  # Fast performance
            new_limit = min(100, self.current_concurrency_limit + 10)
        else:
            return  # No change needed

        if new_limit != self.current_concurrency_limit:
            logging.info(
                f"Adapting concurrency limit from "
                f"{self.current_concurrency_limit} to {new_limit}"
            )
            self.current_concurrency_limit = new_limit

            # Update semaphore if it exists
            if self.async_semaphore:
                # Note: Can't directly change semaphore limit,
                # will be updated on next initialization
                pass

    async def _apply_domain_rate_limiting(self, url: str) -> None:
        """Apply domain-specific rate limiting for async requests"""
        if not self.config.rate_limit_respect:
            return

        try:
            from urllib.parse import urlparse

            domain = urlparse(url).netloc.lower()

            # Get or create domain semaphore
            if domain not in self.domain_semaphores:
                # Limit concurrent requests per domain
                self.domain_semaphores[domain] = Semaphore(
                    2
                )  # Max 2 concurrent per domain

            # Apply rate limiting
            async with self.domain_semaphores[domain]:
                # Check if we need to wait based on last request time
                current_time = time.time()
                last_request = self.rate_limit_tracker.get(domain, 0)

                # Apply domain-specific delays (mirroring IntelligentRateLimiter logic)
                domain_delays = {
                    "google.com": 2.0,
                    "github.com": 1.5,
                    "youtube.com": 2.0,
                    "facebook.com": 3.0,
                    "linkedin.com": 2.0,
                    "twitter.com": 1.5,
                    "x.com": 1.5,
                    "reddit.com": 1.0,
                    "medium.com": 1.0,
                }

                required_delay = domain_delays.get(domain, 0.5)
                time_since_last = current_time - last_request

                if time_since_last < required_delay:
                    wait_time = required_delay - time_since_last
                    await asyncio.sleep(wait_time)

                # Update last request time
                self.rate_limit_tracker[domain] = time.time()

        except Exception as e:
            logging.debug(f"Rate limiting failed for {url}: {e}")
            # Don't fail the whole request for rate limiting issues
            await asyncio.sleep(0.1)  # Minimal fallback delay

    def reset(self) -> None:
        """Reset the batch processor state"""
        with self.lock:
            # Clear queues
            while not self.processing_queue.empty():
                try:
                    self.processing_queue.get_nowait()
                except Empty:
                    break

            while not self.results_queue.empty():
                try:
                    self.results_queue.get_nowait()
                except Empty:
                    break

            # Clear results
            self.completed_batches.clear()
            self.failed_batches.clear()
            self.performance_history.clear()

            # Clear async state
            self.rate_limit_tracker.clear()
            self.domain_semaphores.clear()

            # Clear cost tracking state
            self.total_session_cost = 0.0
            self.batch_cost_history.clear()
            self.cost_estimates.clear()

            # Reset batch size and concurrency to optimal
            self.current_batch_size = self.config.optimal_batch_size
            self.current_concurrency_limit = self.config.async_concurrency_limit

            logging.info("Batch processor state reset (including cost tracking)")

            # Reset cost tracker session if available
            if self.cost_tracker:
                self.cost_tracker.reset_session()


class URLValidator(BatchProcessorInterface):
    """Enhanced URL validation with retry logic and intelligent rate limiting"""

    # URLs to skip validation (known problematic patterns)
    SKIP_PATTERNS = [
        "javascript:",
        "mailto:",
        "tel:",
        "ftp:",
        "file:",
        "data:",
        "#",  # Fragment-only URLs
        "about:blank",
    ]

    # Valid HTTP status codes for success
    SUCCESS_CODES = {
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        300,
        301,
        302,
        303,
        304,
        307,
        308,
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_redirects: int = 5,
        max_concurrent: int = 10,
        user_agent_rotation: bool = True,
        verify_ssl: bool = True,
        rate_limiter: Optional[IntelligentRateLimiter] = None,
    ):
        """
        Initialize URL validator.

        Args:
            timeout: Request timeout in seconds
            max_redirects: Maximum number of redirects to follow
            max_concurrent: Maximum concurrent requests
            user_agent_rotation: Whether to rotate user agents
            verify_ssl: Whether to verify SSL certificates
            rate_limiter: Optional rate limiter instance
        """
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_concurrent = max_concurrent
        self.verify_ssl = verify_ssl

        # Initialize components
        self.rate_limiter = rate_limiter or IntelligentRateLimiter(
            max_concurrent=max_concurrent
        )
        self.browser_simulator = BrowserSimulator(rotate_agents=user_agent_rotation)
        self.retry_handler = RetryHandler()
        self.security_validator = SecurityValidator()

        # Create session
        self.session = self._create_session()

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = ValidationStats()

        logging.info(
            f"Initialized URL validator (timeout={timeout}s, "
            f"max_concurrent={max_concurrent}, ssl_verify={verify_ssl})"
        )

    def batch_validate_optimized(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> List[ValidationResult]:
        """
        Optimized batch validation with memory-efficient processing and progress
        tracking

        Args:
            urls: List of URLs to validate
            progress_callback: Optional progress callback function
            batch_size: Number of URLs to process in each batch
            max_workers: Number of concurrent workers
            progress_tracker: Optional progress tracker for detailed tracking

        Returns:
            List of ValidationResult objects
        """
        logging.info(
            f"Starting optimized batch validation of {len(urls)} URLs "
            f"in batches of {batch_size}"
        )
        start_time = time.time()

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        total_urls = len(unique_urls)

        # Initialize progress tracker stage if provided
        if progress_tracker:
            progress_tracker.start_stage(ProcessingStage.VALIDATING_URLS, total_urls)
            logging.info(f"Progress tracker initialized for {total_urls} URLs")

        if max_workers is None:
            max_workers = min(self.max_concurrent, 20)  # Cap at 20 for optimization

        all_results = []
        processed_count = 0
        failed_count = 0

        # Process in batches to manage memory
        for batch_start in range(0, total_urls, batch_size):
            batch_end = min(batch_start + batch_size, total_urls)
            batch_urls = unique_urls[batch_start:batch_end]
            batch_number = batch_start // batch_size + 1

            logging.info(f"Processing batch {batch_number}: {len(batch_urls)} URLs")

            # Validate batch with threading
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all validations
                future_to_url = {
                    executor.submit(self.validate_url, url): url for url in batch_urls
                }

                batch_results = []
                batch_failed = 0
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        processed_count += 1

                        # Track failures for progress tracker
                        if not result.is_valid:
                            batch_failed += 1
                            failed_count += 1

                        # Update progress tracker if available
                        if progress_tracker:
                            progress_tracker.update_progress(
                                items_delta=1,
                                failed_delta=1 if not result.is_valid else 0,
                                stage_specific=True,
                            )

                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                f"Validated {processed_count}/{total_urls}: {url}"
                            )

                    except Exception as e:
                        logging.error(f"Error validating {url}: {e}")
                        # Create error result
                        error_result = ValidationResult(
                            url=url,
                            is_valid=False,
                            error_message=f"Validation error: {str(e)}",
                            error_type="validation_error",
                        )
                        batch_results.append(error_result)
                        processed_count += 1
                        batch_failed += 1
                        failed_count += 1

                        # Update progress tracker for errors
                        if progress_tracker:
                            progress_tracker.update_progress(
                                items_delta=1, failed_delta=1, stage_specific=True
                            )
                            progress_tracker.log_error(
                                f"Validation error for {url}: {e}"
                            )

            # Add batch results to total
            all_results.extend(batch_results)

            # Log batch completion with progress tracker
            if progress_tracker:
                batch_success_rate = (
                    (len(batch_results) - batch_failed) / len(batch_results)
                ) * 100
                logging.info(
                    f"Batch {batch_number} completed: "
                    f"{len(batch_results) - batch_failed}/{len(batch_results)} valid "
                    f"({batch_success_rate:.1f}% success rate)"
                )

            # Force garbage collection between batches
            import gc

            gc.collect()

            # Brief pause between batches to allow rate limiting
            if batch_end < total_urls:
                time.sleep(0.1)

        total_time = time.time() - start_time
        valid_count = sum(1 for r in all_results if r.is_valid)
        overall_success_rate = (valid_count / len(all_results)) * 100

        # Log comprehensive completion statistics
        logging.info(
            f"Optimized batch validation completed: "
            f"{valid_count}/{len(all_results)} valid URLs "
            f"({overall_success_rate:.1f}% success rate) in {total_time:.2f}s"
        )

        if progress_tracker:
            logging.info(
                f"Progress tracker final state: {processed_count} processed, "
                f"{failed_count} failed"
            )

        return all_results

    def _get_mock_validation_result(self, url: str) -> ValidationResult:
        """
        Generate mock validation result for test mode.

        Args:
            url: URL being validated

        Returns:
            Mock ValidationResult
        """
        # Most URLs are valid in test mode, except a few specific test cases
        is_valid = not any(
            invalid in url.lower() for invalid in ["invalid", "404", "error", "timeout"]
        )

        return ValidationResult(
            url=url,
            is_valid=is_valid,
            status_code=200 if is_valid else 404,
            final_url=url,
            error_message=None if is_valid else "Mock test failure",
            error_type=None if is_valid else "test_error",
            response_time=0.1,  # Fast mock response
            content_type="text/html" if is_valid else None,
            security_validation=(
                SecurityValidationResult(
                    is_safe=True,
                    risk_level="low",
                    issues=[],
                    blocked_reason=None,
                )
                if is_valid
                else SecurityValidationResult(
                    is_safe=False,
                    risk_level="medium",
                    issues=["Test failure"],
                    blocked_reason="Mock test error",
                )
            ),
        )

    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate a single URL.

        Args:
            url: URL to validate

        Returns:
            ValidationResult object
        """
        import os

        # Test mode guard - return mock validation for testing
        if os.getenv("BOOKMARK_PROCESSOR_TEST_MODE") == "true":
            return self._get_mock_validation_result(url)

        start_time = time.time()

        # Security validation first
        security_result = self.security_validator.validate_url_security(url)
        if not security_result.is_safe:
            # Log security event
            if security_result.risk_level in ["high", "critical"]:
                secure_logger.log_url_validation_failure(
                    url,
                    security_result.blocked_reason or "Security validation failed",
                    security_result.issues,
                )

            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=security_result.blocked_reason
                or "Security validation failed",
                error_type="security_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Basic URL validation
        if not self._is_valid_url_format(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="Invalid URL format",
                error_type="format_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Check skip patterns
        if self._should_skip_url(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="URL type not supported for validation",
                error_type="unsupported_scheme",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Apply rate limiting
        self.rate_limiter.wait_if_needed(url)

        try:
            # Get headers for this request
            headers = self.browser_simulator.get_headers(url)

            # Make request
            response = self.session.head(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=self.verify_ssl,
            )

            response_time = time.time() - start_time

            # Check if successful
            is_valid = response.status_code in self.SUCCESS_CODES

            result = ValidationResult(
                url=url,
                is_valid=is_valid,
                status_code=response.status_code,
                final_url=response.url if response.url != url else None,
                response_time=response_time,
                content_type=response.headers.get("content-type"),
                content_length=self._parse_content_length(
                    response.headers.get("content-length")
                ),
                security_validation=security_result,
            )

            if not is_valid:
                result.error_message = f"HTTP {response.status_code}"
                result.error_type = self._classify_http_error(response.status_code)

            # Record success for rate limiter
            self.rate_limiter.record_success(url)

            return result

        except requests.exceptions.SSLError as e:
            # Try without SSL verification if allowed
            if self.verify_ssl:
                logging.debug(f"SSL error for {url}, trying without verification: {e}")
                try:
                    response = self.session.head(
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=True,
                        verify=False,
                    )

                    response_time = time.time() - start_time
                    is_valid = response.status_code in self.SUCCESS_CODES

                    result = ValidationResult(
                        url=url,
                        is_valid=is_valid,
                        status_code=response.status_code,
                        final_url=response.url if response.url != url else None,
                        response_time=response_time,
                        content_type=response.headers.get("content-type"),
                        content_length=self._parse_content_length(
                            response.headers.get("content-length")
                        ),
                        error_message=(
                            "SSL certificate warning"
                            if is_valid
                            else f"HTTP {response.status_code}"
                        ),
                        error_type=(
                            "ssl_warning"
                            if is_valid
                            else self._classify_http_error(response.status_code)
                        ),
                        security_validation=security_result,
                    )

                    self.rate_limiter.record_success(url)
                    return result

                except Exception:
                    pass  # Fall through to error handling

            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"SSL Error: {str(e)}",
                error_type="ssl_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        except requests.exceptions.Timeout:
            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Timeout after {self.timeout}s",
                error_type="timeout",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        except requests.exceptions.ConnectionError as e:
            self.rate_limiter.record_error(url)
            error_msg = str(e)
            error_type = "connection_error"

            if "dns" in error_msg.lower() or "name resolution" in error_msg.lower():
                error_type = "dns_error"
            elif "refused" in error_msg.lower():
                error_type = "connection_refused"

            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=error_msg,
                error_type=error_type,
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        except Exception as e:
            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type="unknown_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

    def batch_validate(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
        enable_retries: bool = True,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> List[ValidationResult]:
        """
        Validate multiple URLs with concurrent processing and retry logic.

        Args:
            urls: List of URLs to validate
            progress_callback: Optional progress callback function
            enable_retries: Whether to enable retry logic for failed URLs
            progress_tracker: Optional progress tracker for detailed per-batch
                monitoring

        Returns:
            List of ValidationResult objects
        """
        logging.info(f"Starting batch validation of {len(urls)} URLs")
        start_time = time.time()

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        if len(unique_urls) != len(urls):
            logging.info(f"Removed {len(urls) - len(unique_urls)} duplicate URLs")

        # Initialize progress tracking for batch processing
        if progress_tracker:
            progress_tracker.start_stage(
                ProcessingStage.VALIDATING_URLS, len(unique_urls)
            )
            progress_tracker.set_active_tasks(self.max_concurrent)

        results = []

        # First pass: concurrent validation
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all validation tasks
            future_to_url = {
                executor.submit(self.validate_url, url): url for url in unique_urls
            }

            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update statistics
                    with self.lock:
                        self.stats.update_from_result(result)

                    # Add failed URLs to retry queue
                    if enable_retries and not result.is_valid:
                        if result.error_type not in [
                            "format_error",
                            "unsupported_scheme",
                        ]:
                            error = Exception(
                                result.error_message or "Validation failed"
                            )
                            self.retry_handler.add_failed_url(url, error)

                    completed += 1

                    # Update progress tracking
                    if progress_tracker:
                        progress_tracker.update_progress(
                            items_delta=1,
                            failed_delta=1 if not result.is_valid else 0,
                            stage_specific=True,
                        )

                    if progress_callback:
                        progress_callback(
                            f"Validated {completed}/{len(unique_urls)}: {url}"
                        )

                except Exception as e:
                    logging.error(f"Error validating {url}: {e}")
                    error_result = ValidationResult(
                        url=url,
                        is_valid=False,
                        error_message=str(e),
                        error_type="validation_error",
                    )
                    results.append(error_result)

                    if enable_retries:
                        self.retry_handler.add_failed_url(url, e)

                    # Update progress tracking for errors
                    if progress_tracker:
                        progress_tracker.update_progress(
                            items_delta=1, failed_delta=1, stage_specific=True
                        )

        # Second pass: retry failed URLs
        if enable_retries and self.retry_handler.retry_queue:
            retry_count = len(self.retry_handler.retry_queue)
            logging.info(f"Retrying {retry_count} failed URLs")

            # Track retry progress if progress tracker available
            if progress_tracker:
                # Start a sub-stage for retries
                progress_tracker.log_error(
                    f"Starting retry of {retry_count} failed URLs"
                )

            retry_results = self.retry_handler.retry_failed_items(
                self.validate_url, progress_callback
            )

            # Update results with successful retries
            retry_urls = {r.url for r in retry_results}
            results = [r for r in results if r.url not in retry_urls]
            results.extend(retry_results)

            # Track retry completion
            if progress_tracker:
                successful_retries = sum(1 for r in retry_results if r.is_valid)
                progress_tracker.log_error(
                    f"Retry completed: {successful_retries}/{retry_count} successful"
                )

        total_time = time.time() - start_time

        # Final statistics
        valid_count = sum(1 for r in results if r.is_valid)
        logging.info(
            f"Batch validation completed in {total_time:.2f}s: "
            f"{valid_count}/{len(results)} URLs valid "
            f"({valid_count/len(results)*100:.1f}%)"
        )

        # Complete progress tracking
        if progress_tracker:
            progress_tracker.set_active_tasks(0)  # Clear active tasks
            # Log final batch statistics
            progress_tracker.log_error(
                f"Batch validation summary: {valid_count}/{len(results)} valid "
                f"({valid_count/len(results)*100:.1f}%) in {total_time:.2f}s"
            )

        return results

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        with self.lock:
            stats_dict = {
                "total_urls": self.stats.total_urls,
                "valid_urls": self.stats.valid_urls,
                "invalid_urls": self.stats.invalid_urls,
                "success_rate": self.stats.valid_urls / max(self.stats.total_urls, 1),
                "average_response_time": self.stats.average_response_time,
                "total_processing_time": self.stats.total_time,
                "redirected_urls": self.stats.redirected_urls,
                "error_distribution": dict(self.stats.error_distribution),
                "rate_limiter_stats": self.rate_limiter.get_all_stats(),
            }

            if self.retry_handler:
                stats_dict["retry_stats"] = self.retry_handler.get_retry_statistics()

            return stats_dict

    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()

        # Configure adapters for retries at the HTTP level
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=0,  # We handle retries at application level
            connect=1,  # Allow one connection retry
            read=1,  # Allow one read retry
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )

        # Enhanced adapter with larger connection pool
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # Number of connection pools
            pool_maxsize=50,  # Max connections per pool
            pool_block=False,  # Don't block when pool is full
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default timeout
        session.timeout = self.timeout

        return session

    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL has valid format"""
        if not url or not isinstance(url, str):
            return False

        url = url.strip()
        if not url:
            return False

        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped"""
        url_lower = url.lower().strip()

        for pattern in self.SKIP_PATTERNS:
            if url_lower.startswith(pattern):
                return True

        return False

    def _classify_http_error(self, status_code: int) -> str:
        """Classify HTTP error by status code"""
        if 400 <= status_code < 500:
            error_types = {
                400: "bad_request",
                401: "unauthorized",
                403: "forbidden",
                404: "not_found",
                405: "method_not_allowed",
                408: "request_timeout",
                429: "rate_limited",
            }
            return error_types.get(status_code, "client_error")
        elif 500 <= status_code < 600:
            error_types = {
                500: "internal_server_error",
                501: "not_implemented",
                502: "bad_gateway",
                503: "service_unavailable",
                504: "gateway_timeout",
            }
            return error_types.get(status_code, "server_error")
        else:
            return "http_error"

    def _parse_content_length(
        self, content_length_header: Optional[str]
    ) -> Optional[int]:
        """Parse content-length header"""
        if not content_length_header:
            return None

        try:
            return int(content_length_header)
        except (ValueError, TypeError):
            return None

    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        with self.lock:
            self.stats = ValidationStats()
            self.rate_limiter.reset_domain_stats()
            if self.retry_handler:
                self.retry_handler.clear_completed()

        logging.info("Reset validation statistics")

    def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, "session"):
            self.session.close()

        logging.info("URL validator closed")

    # BatchProcessorInterface implementation
    def process_batch(self, items: List[str], batch_id: str) -> BatchResult:
        """
        Process a batch of URLs and return BatchResult.

        Args:
            items: List of URLs to validate
            batch_id: Unique identifier for this batch

        Returns:
            BatchResult with processing details
        """
        start_time = time.time()

        logging.info(f"Processing batch {batch_id} with {len(items)} URLs")

        # Use existing batch validation method
        validation_results = self.batch_validate_optimized(
            items,
            progress_callback=None,  # Batch-level progress handled by
            # EnhancedBatchProcessor
            batch_size=len(items),  # Process all items in this batch at once
            max_workers=min(self.max_concurrent, len(items)),
        )

        # Calculate batch statistics
        successful_count = sum(1 for result in validation_results if result.is_valid)
        failed_count = len(validation_results) - successful_count
        processing_time = time.time() - start_time
        average_time_per_item = processing_time / len(items) if items else 0
        error_rate = failed_count / len(items) if items else 0

        # Collect error messages
        error_messages = []
        for result in validation_results:
            if not result.is_valid and result.error_message:
                error_messages.append(f"{result.url}: {result.error_message}")

        batch_result = BatchResult(
            batch_id=batch_id,
            items_processed=len(items),
            items_successful=successful_count,
            items_failed=failed_count,
            processing_time=processing_time,
            average_item_time=average_time_per_item,
            error_rate=error_rate,
            results=validation_results,
            errors=error_messages[
                :10
            ],  # Limit to first 10 errors to avoid memory issues
        )

        logging.info(
            f"Batch {batch_id} completed: {successful_count}/{len(items)} URLs valid "
            f"in {processing_time:.2f}s (avg {average_time_per_item:.3f}s per URL)"
        )

        return batch_result

    def get_optimal_batch_size(self) -> int:
        """
        Get the optimal batch size based on current performance metrics.

        Returns:
            Optimal batch size for URL validation
        """
        # Base batch size on concurrent workers and average response time
        base_size = self.max_concurrent * 10  # 10x concurrent workers as base

        # Adjust based on recent performance if available
        if hasattr(self, "stats") and self.stats.average_response_time > 0:
            if self.stats.average_response_time > 5.0:  # Slow responses
                return max(25, base_size // 2)  # Smaller batches
            elif self.stats.average_response_time < 1.0:  # Fast responses
                return min(500, base_size * 2)  # Larger batches

        return min(100, max(25, base_size))  # Default range 25-100

    def estimate_processing_time(self, item_count: int) -> float:
        """
        Estimate processing time for given number of URLs.

        Args:
            item_count: Number of URLs to process

        Returns:
            Estimated processing time in seconds
        """
        if not hasattr(self, "stats") or self.stats.average_response_time <= 0:
            # No historical data, use conservative estimate
            return item_count * 2.0  # 2 seconds per URL average

        # Use historical average response time
        base_time_per_url = self.stats.average_response_time

        # Add overhead for batch processing
        batch_overhead = 0.1  # 100ms overhead per URL for batch coordination

        # Account for concurrent processing efficiency
        concurrency_factor = min(self.max_concurrent, item_count) / item_count
        if concurrency_factor > 0:
            effective_time_per_url = (
                base_time_per_url * concurrency_factor
            ) + batch_overhead
        else:
            effective_time_per_url = base_time_per_url + batch_overhead

        estimated_time = item_count * effective_time_per_url

        # Add 20% buffer for safety
        return estimated_time * 1.2

    async def async_validate_url(self, url: str) -> ValidationResult:
        """
        Asynchronously validate a single URL.

        Args:
            url: URL to validate

        Returns:
            ValidationResult object
        """
        import os

        # Test mode guard - return mock validation for testing
        if os.getenv("BOOKMARK_PROCESSOR_TEST_MODE") == "true":
            return self._get_mock_validation_result(url)

        start_time = time.time()

        # Security validation first (sync, but fast)
        security_result = self.security_validator.validate_url_security(url)
        if not security_result.is_safe:
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=security_result.blocked_reason
                or "Security validation failed",
                error_type="security_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Basic URL validation
        if not self._is_valid_url_format(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="Invalid URL format",
                error_type="format_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        # Check skip patterns
        if self._should_skip_url(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="URL type not supported for validation",
                error_type="unsupported_scheme",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

        try:
            # Create async session if not exists
            if not hasattr(self, "_async_session") or self._async_session is None:
                await self._initialize_async_session()

            # Get headers for this request
            headers = self.browser_simulator.get_headers(url)

            # Apply async rate limiting
            await self._async_apply_rate_limiting(url)

            # Make async request
            async with self._async_session.head(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                allow_redirects=True,
                ssl=self.verify_ssl,
            ) as response:
                response_time = time.time() - start_time

                # Check if successful
                is_valid = response.status in self.SUCCESS_CODES

                result = ValidationResult(
                    url=url,
                    is_valid=is_valid,
                    status_code=response.status,
                    final_url=str(response.url) if str(response.url) != url else None,
                    response_time=response_time,
                    content_type=response.headers.get("content-type"),
                    content_length=self._parse_content_length(
                        response.headers.get("content-length")
                    ),
                    security_validation=security_result,
                )

                if not is_valid:
                    result.error_message = f"HTTP {response.status}"
                    result.error_type = self._classify_http_error(response.status)

                return result

        except asyncio.TimeoutError:
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Timeout after {self.timeout}s",
                error_type="timeout",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )
        except aiohttp.ClientError as e:
            error_msg = str(e)
            error_type = "connection_error"

            if "dns" in error_msg.lower() or "name resolution" in error_msg.lower():
                error_type = "dns_error"
            elif "refused" in error_msg.lower():
                error_type = "connection_refused"

            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=error_msg,
                error_type=error_type,
                response_time=time.time() - start_time,
                security_validation=security_result,
            )
        except Exception as e:
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type="unknown_error",
                response_time=time.time() - start_time,
                security_validation=security_result,
            )

    async def async_validate_batch(self, urls: List[str], batch_id: str) -> BatchResult:
        """
        Asynchronously validate a batch of URLs with rate limiting.

        Args:
            urls: List of URLs to validate
            batch_id: Unique identifier for this batch

        Returns:
            BatchResult with processing details
        """
        start_time = time.time()

        logging.info(f"Processing async batch {batch_id} with {len(urls)} URLs")

        # Create semaphore for concurrent validation within this batch
        validation_semaphore = Semaphore(min(self.max_concurrent, len(urls)))

        async def validate_with_semaphore(url: str) -> ValidationResult:
            async with validation_semaphore:
                return await self.async_validate_url(url)

        # Process all URLs concurrently with rate limiting
        tasks = [validate_with_semaphore(url) for url in urls]
        validation_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        final_results = []
        error_messages = []

        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                error_msg = f"{urls[i]}: {str(result)}"
                error_messages.append(error_msg)

                # Create error result
                error_result = ValidationResult(
                    url=urls[i],
                    is_valid=False,
                    error_message=str(result),
                    error_type="async_validation_error",
                    response_time=0.0,
                )
                final_results.append(error_result)
            else:
                final_results.append(result)

        # Calculate batch statistics
        successful_count = sum(1 for result in final_results if result.is_valid)
        failed_count = len(final_results) - successful_count
        processing_time = time.time() - start_time
        average_time_per_item = processing_time / len(urls) if urls else 0
        error_rate = failed_count / len(urls) if urls else 0

        batch_result = BatchResult(
            batch_id=batch_id,
            items_processed=len(urls),
            items_successful=successful_count,
            items_failed=failed_count,
            processing_time=processing_time,
            average_item_time=average_time_per_item,
            error_rate=error_rate,
            results=final_results,
            errors=error_messages[:10],  # Limit to first 10 errors
        )

        logging.info(
            f"Async batch {batch_id} completed: {successful_count}/{len(urls)} URLs "
            f"valid in {processing_time:.2f}s "
            f"(avg {average_time_per_item:.3f}s per URL)"
        )

        return batch_result

    async def _initialize_async_session(self) -> None:
        """Initialize aiohttp session for async validation"""
        if not hasattr(self, "_async_session") or self._async_session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout, connect=10.0)
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent,
                limit_per_host=5,  # Max connections per host
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=self.verify_ssl,
            )

            self._async_session = aiohttp.ClientSession(
                timeout=timeout, connector=connector
            )

    async def _async_apply_rate_limiting(self, url: str) -> None:
        """Apply rate limiting for async requests"""
        # Use the existing rate limiter's logic but with async sleep
        domain = self.rate_limiter._extract_domain(url)

        with self.rate_limiter.lock:
            stats = self.rate_limiter.domain_stats[domain]
            delay = self.rate_limiter._get_domain_delay(domain)

            # Calculate required wait time
            current_time = time.time()
            time_since_last = current_time - stats.last_request_time
            wait_time = max(0, delay - time_since_last)

            # Update stats immediately to prevent race conditions
            stats.last_request_time = current_time
            stats.request_count += 1
            self.rate_limiter.active_domains.add(domain)

        # Apply wait asynchronously
        if wait_time > 0:
            logging.debug(f"Async rate limiting {domain}: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

            with self.rate_limiter.lock:
                stats.total_wait_time += wait_time

    async def close_async_session(self) -> None:
        """Close the async session"""
        if hasattr(self, "_async_session") and self._async_session:
            await self._async_session.close()
            self._async_session = None

    def create_enhanced_batch_processor(
        self,
        config: Optional[BatchConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_update_callback: Optional[Callable[[ProgressUpdate], None]] = None,
        enable_cost_tracking: bool = False,
        cost_per_url_validation: float = 0.0001,
        cost_confirmation_threshold: float = 1.0,
        budget_limit: Optional[float] = None,
    ) -> EnhancedBatchProcessor:
        """
        Create an enhanced batch processor using this URLValidator.

        Args:
            config: Optional batch processing configuration
            progress_callback: Optional progress callback function
            progress_update_callback: Optional detailed progress update callback
            enable_cost_tracking: Whether to enable cost tracking
            cost_per_url_validation: Cost per URL validation operation
            cost_confirmation_threshold: USD threshold for user confirmation
            budget_limit: Optional budget limit in USD

        Returns:
            Configured EnhancedBatchProcessor instance
        """
        if config is None:
            # Create default config optimized for URL validation
            config = BatchConfig(
                min_batch_size=10,
                max_batch_size=min(500, self.max_concurrent * 25),
                optimal_batch_size=self.get_optimal_batch_size(),
                auto_tune_batch_size=True,
                max_concurrent_batches=max(
                    1, self.max_concurrent // 5
                ),  # Conservative concurrency
                batch_timeout=600.0,  # 10 minutes for URL validation batches
                retry_failed_batches=True,
                preserve_order=True,
                # Enable async processing with rate limiting
                enable_async_processing=True,
                async_concurrency_limit=min(50, self.max_concurrent * 5),
                rate_limit_respect=True,
                adaptive_concurrency=True,
                # Cost tracking configuration
                enable_cost_tracking=enable_cost_tracking,
                cost_per_url_validation=cost_per_url_validation,
                cost_confirmation_threshold=cost_confirmation_threshold,
                budget_limit=budget_limit,
            )

        return EnhancedBatchProcessor(
            processor=self,
            config=config,
            progress_callback=progress_callback,
            progress_update_callback=progress_update_callback,
        )
