"""
Batch Processing Module for Bookmark Processing

This module provides efficient batch processing capabilities for large bookmark
datasets, optimizing API usage through appropriate batch sizes and concurrent
processing within rate limits.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm

from bookmark_processor.core.ai_factory import AIManager
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.cost_tracker import CostTracker
from bookmark_processor.utils.rate_limiter import get_rate_limiter
from bookmark_processor.utils.error_handler import BatchProcessingError


class BatchProcessor:
    """
    Batch processor for efficient bookmark processing with rate limiting and
    cost tracking.
    """

    # Default batch sizes by provider
    DEFAULT_BATCH_SIZES = {
        "local": 50,  # Local processing can handle larger batches
        "claude": 10,  # Claude rate limit consideration
        "openai": 20,  # OpenAI can handle larger batches
    }

    def __init__(
        self,
        ai_manager: AIManager,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        max_concurrent: int = 10,
    ):
        """
        Initialize batch processor.

        Args:
            ai_manager: AI manager instance
            cost_tracker: Optional cost tracker (will create if not provided)
            verbose: Enable verbose output
            max_concurrent: Maximum concurrent operations
        """
        self.ai_manager = ai_manager
        self.cost_tracker = cost_tracker or CostTracker()
        self.verbose = verbose
        self.max_concurrent = max_concurrent

        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        self.end_time = None

        self.logger = logging.getLogger(__name__)

    def get_batch_size(self, provider: str) -> int:
        """
        Get appropriate batch size for a provider.

        Args:
            provider: AI provider name

        Returns:
            Recommended batch size
        """
        return self.DEFAULT_BATCH_SIZES.get(provider, 10)

    async def process_bookmarks(
        self,
        bookmarks: List[Bookmark],
        existing_content: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
        """
        Process a list of bookmarks with batch optimization and enhanced cost tracking.

        Args:
            bookmarks: List of bookmark objects to process
            existing_content: Optional list of existing content to enhance

        Returns:
            Tuple of (results, statistics)
        """
        self.start_time = time.time()
        self.processed_count = 0
        self.failed_count = 0

        provider = self.ai_manager.get_current_provider()
        batch_size = self.get_batch_size(provider)

        # Show cost estimate before starting
        if provider != "local" and self.verbose:
            cost_estimate = self.cost_tracker.get_cost_estimate(
                len(bookmarks), provider
            )
            print("\n💰 Cost Estimation:")
            print(f"  📊 Estimated cost: ${cost_estimate['estimated_cost_usd']:.2f}")
            print(f"  📈 Cost per bookmark: ${cost_estimate['cost_per_bookmark']:.4f}")
            print(f"  🎯 Confidence: {cost_estimate['confidence']}")
            print(f"  📝 Method: {cost_estimate['method']}")

        if self.verbose:
            print("\n🚀 Starting batch processing:")
            print(f"  Provider: {provider}")
            print(f"  Total bookmarks: {len(bookmarks)}")
            print(f"  Batch size: {batch_size}")
            print(f"  Max concurrent: {self.max_concurrent}")

        # Split into batches
        batches = []
        for i in range(0, len(bookmarks), batch_size):
            end_idx = min(i + batch_size, len(bookmarks))
            batch_bookmarks = bookmarks[i:end_idx]
            batch_content = existing_content[i:end_idx] if existing_content else None
            batches.append((batch_bookmarks, batch_content))

        # Process batches with progress tracking
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        if self.verbose:
            print(f"\n📦 Processing {len(batches)} batches...")

        async def process_single_batch(batch_data):
            async with semaphore:
                return await self._process_batch(*batch_data)

        # Use tqdm for progress tracking
        batch_tasks = [process_single_batch(batch_data) for batch_data in batches]

        try:
            if self.verbose:
                # Process with progress bar
                batch_results = []
                for task in tqdm.as_completed(batch_tasks, desc="Processing batches"):
                    batch_result = await task
                    batch_results.append(batch_result)

                    # Check cost confirmation after each batch
                    if not await self.cost_tracker.confirm_continuation():
                        # Cancel remaining tasks
                        for remaining_task in batch_tasks:
                            if not remaining_task.done():
                                remaining_task.cancel()
                        break
            else:
                # Process without progress bar
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

        except KeyboardInterrupt:
            self.logger.info("Batch processing interrupted by user")
            batch_results = []

        # Collect results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Batch processing error: {batch_result}")
                self.failed_count += len(batches[0][0])  # Estimate failed count
            else:
                results.extend(batch_result)
                self.processed_count += len(batch_result)

        self.end_time = time.time()

        # Generate statistics
        statistics = self._generate_statistics(len(bookmarks))

        if self.verbose:
            self._print_final_statistics(statistics)

        return results, statistics

    async def _process_batch(
        self,
        bookmarks: List[Bookmark],
        existing_content: Optional[List[str]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process a single batch of bookmarks with enhanced cost tracking.

        Args:
            bookmarks: Batch of bookmarks to process
            existing_content: Optional existing content for the batch

        Returns:
            List of results for the batch
        """
        # Get provider early so it's available in the except block
        provider = self.ai_manager.get_current_provider()

        try:
            # Process bookmarks in the batch
            results = await self.ai_manager.generate_descriptions_batch(
                bookmarks, existing_content
            )

            # Track detailed costs if using cloud AI
            if provider != "local":
                for i, (description, metadata) in enumerate(results):
                    if metadata.get("success", False) and "cost_usd" in metadata:
                        # Add detailed cost record
                        self.cost_tracker.add_cost_record(
                            provider=metadata.get("provider", provider),
                            model=metadata.get("model", "unknown"),
                            input_tokens=metadata.get("input_tokens", 0),
                            output_tokens=metadata.get("output_tokens", 0),
                            cost_usd=metadata["cost_usd"],
                            operation_type="description_generation",
                            bookmark_count=1,
                            success=metadata.get("success", False),
                        )

            return results

        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")

            # Return empty results for failed batch
            return [
                ("", {"provider": provider, "error": str(e), "success": False})
                for _ in bookmarks
            ]

    def _generate_statistics(self, total_bookmarks: int) -> Dict[str, Any]:
        """
        Generate comprehensive processing statistics.

        Args:
            total_bookmarks: Total number of bookmarks processed

        Returns:
            Statistics dictionary
        """
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        stats = {
            "total_bookmarks": total_bookmarks,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": self.processed_count / max(total_bookmarks, 1) * 100,
            "duration_seconds": duration,
            "bookmarks_per_minute": (self.processed_count / max(duration / 60, 0.01)),
            "provider": self.ai_manager.get_current_provider(),
            "provider_info": self.ai_manager.get_provider_info(),
        }

        # Add enhanced cost tracking statistics
        cost_stats = self.cost_tracker.get_detailed_statistics()
        stats["cost_tracking"] = cost_stats

        # Add AI usage statistics if available
        ai_stats = self.ai_manager.get_usage_statistics()
        if ai_stats:
            stats["ai_usage"] = ai_stats

        # Add error handling statistics
        error_stats = self.ai_manager.get_error_statistics()
        stats["error_handling"] = error_stats

        return stats

    def _print_final_statistics(self, statistics: Dict[str, Any]) -> None:
        """
        Print comprehensive final processing statistics.

        Args:
            statistics: Statistics dictionary
        """
        print("\n📊 Processing Complete!")
        print(
            f"  ✅ Processed: {statistics['processed_count']}/"
            f"{statistics['total_bookmarks']}"
        )
        print(f"  ❌ Failed: {statistics['failed_count']}")
        print(f"  📈 Success Rate: {statistics['success_rate']:.1f}%")
        print(f"  ⏱️  Duration: {statistics['duration_seconds']:.1f}s")
        print(f"  🚀 Speed: {statistics['bookmarks_per_minute']:.1f} bookmarks/min")
        print(f"  🤖 Provider: {statistics['provider']}")

        # Enhanced cost information
        cost_info = statistics.get("cost_tracking", {})
        session_info = cost_info.get("session", {})
        session_cost = session_info.get("total_cost_usd", 0.0)

        if session_cost > 0:
            print("\n💰 Cost Analysis:")
            print(f"  💵 Session Cost: ${session_cost:.4f}")
            print(
                f"  📊 Cost/Hour: "
                f"${session_info.get('cost_per_hour', 0.0):.4f}"
            )
            print(
                f"  📈 Success Rate: "
                f"{session_info.get('success_rate_percent', 0.0):.1f}%"
            )

            # Provider breakdown
            providers = cost_info.get("providers", {})
            if providers:
                print("  📋 Provider Breakdown:")
                for provider, provider_stats in providers.items():
                    print(
                        f"     {provider}: ${provider_stats['cost_usd']:.4f} "
                        f"({provider_stats['requests']} requests)"
                    )

        # Error handling information
        health_status = statistics.get("ai_usage", {}).get("health_status", {})

        if health_status:
            status = health_status.get("status", "unknown")
            print(f"\n🏥 System Health: {status.upper()}")
            if status not in ["healthy", "stable"]:
                print(f"  ⚠️  {health_status.get('message', 'No details available')}")

        # Token usage for cloud AI
        tokens_info = cost_info.get("tokens", {})
        if tokens_info.get("total_tokens", 0) > 0:
            print("\n🔤 Token Usage:")
            print(f"  📥 Input: {tokens_info['total_input_tokens']:,}")
            print(f"  📤 Output: {tokens_info['total_output_tokens']:,}")
            print(f"  📊 Total: {tokens_info['total_tokens']:,}")

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status for the active provider.

        Returns:
            Rate limit status dictionary
        """
        provider = self.ai_manager.get_current_provider()

        if provider == "local":
            return {"provider": "local", "status": "unlimited"}

        try:
            rate_limiter = get_rate_limiter(provider)
            return rate_limiter.get_status()
        except Exception as e:
            return {"provider": provider, "error": str(e)}

    async def estimate_processing_cost(
        self,
        bookmark_count: int,
        sample_bookmarks: Optional[List[Bookmark]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate the cost of processing a given number of bookmarks.

        Args:
            bookmark_count: Number of bookmarks to process
            sample_bookmarks: Optional sample bookmarks for more accurate estimation

        Returns:
            Cost estimation dictionary
        """
        provider = self.ai_manager.get_current_provider()

        return self.cost_tracker.get_cost_estimate(
            bookmark_count=bookmark_count,
            provider=provider,
            sample_size=min(20, len(sample_bookmarks)) if sample_bookmarks else 10,
        )

    def get_cost_statistics(self) -> Dict[str, Any]:
        """
        Get detailed cost statistics.

        Returns:
            Cost statistics dictionary
        """
        return self.cost_tracker.get_detailed_statistics()

    def export_cost_report(self, output_file: Optional[str] = None) -> str:
        """
        Export detailed cost report.

        Args:
            output_file: Optional output file path

        Returns:
            Path to exported report
        """
        return self.cost_tracker.export_cost_report(output_file)

    def reset_session(self) -> None:
        """Reset session statistics."""
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        self.end_time = None
        self.cost_tracker.reset_session()
        self.logger.info("Batch processor session reset")
