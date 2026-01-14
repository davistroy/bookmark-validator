"""
Batch Processing Module

Provides enhanced batch processing capabilities with intelligent optimization,
concurrent processing, rate limiting, and cost tracking.

This module contains the EnhancedBatchProcessor which orchestrates batch
processing operations with features like:
- Configurable batch sizes with auto-tuning
- Async and sync processing modes
- Rate limiting and domain-specific delays
- Cost tracking and budget management
- Progress tracking and detailed statistics
- Automatic retry logic for failed batches
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

# Forward declarations for type checking
if TYPE_CHECKING:
    from .url_validator import (
        BatchConfig,
        BatchProcessorInterface,
        BatchResult,
        CostBreakdown,
        ProgressUpdate,
        ValidationResult,
    )


class EnhancedBatchProcessor:
    """Enhanced batch processor with configurable batch sizes and intelligent
    optimization"""

    def __init__(
        self,
        processor: "BatchProcessorInterface",
        config: Optional["BatchConfig"] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_update_callback: Optional[Callable[["ProgressUpdate"], None]] = None,
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

        # Optional cost tracker (can be set externally)
        self.cost_tracker = None

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


# Import data classes after class definition to avoid circular imports
from .url_validator import (
    BatchConfig,
    BatchProcessorInterface,
    BatchResult,
    CostBreakdown,
    ProgressUpdate,
    ValidationResult,
)

# Update type hints now that classes are imported
# This allows the EnhancedBatchProcessor to work with proper types
