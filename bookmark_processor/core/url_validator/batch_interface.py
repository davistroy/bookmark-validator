"""
Batch Processing Interface Implementation

Implements the BatchProcessorInterface for URL validation,
providing batch processing capabilities and performance estimation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional

from ..batch_types import BatchConfig, BatchResult

if TYPE_CHECKING:
    from .validator import URLValidator


class BatchProcessorMixin:
    """
    Mixin providing BatchProcessorInterface implementation for URLValidator.

    This mixin adds batch processing interface methods to URLValidator.
    """

    def process_batch(
        self: "URLValidator", items: List[str], batch_id: str
    ) -> BatchResult:
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

    def get_optimal_batch_size(self: "URLValidator") -> int:
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

    def estimate_processing_time(self: "URLValidator", item_count: int) -> float:
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

    def create_enhanced_batch_processor(
        self: "URLValidator",
        config: Optional[BatchConfig] = None,
        progress_callback=None,
        progress_update_callback=None,
        enable_cost_tracking: bool = False,
        cost_per_url_validation: float = 0.0001,
        cost_confirmation_threshold: float = 1.0,
        budget_limit: Optional[float] = None,
    ):
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
        # Import here to avoid circular dependency
        from ..batch_validator import EnhancedBatchProcessor

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


__all__ = ["BatchProcessorMixin"]
