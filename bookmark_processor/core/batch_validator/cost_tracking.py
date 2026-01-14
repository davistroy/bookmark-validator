"""
Cost Tracking for Batch Processing

Provides cost estimation, tracking, and budget management for batch operations.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from ..batch_types import CostBreakdown


class CostTrackingMixin:
    """
    Mixin providing cost tracking capabilities for batch processing.

    This mixin adds cost estimation, recording, and budget management
    methods to the EnhancedBatchProcessor.
    """

    def estimate_batch_cost(
        self, item_count: int, operation_type: str = "url_validation"
    ) -> "CostBreakdown":
        """
        Estimate cost for processing a batch of items.

        Args:
            item_count: Number of items in the batch
            operation_type: Type of operation being performed

        Returns:
            CostBreakdown with detailed cost estimation
        """
        from ..batch_types import CostBreakdown

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
        Check budget limits and get user confirmation if needed (async).

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
                        import asyncio

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
                    import asyncio

                    response = await asyncio.get_event_loop().run_in_executor(
                        None, input, prompt
                    )
                    return response.lower().strip() in ["y", "yes"]
                except (KeyboardInterrupt, EOFError):
                    return False

        return True

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


__all__ = ["CostTrackingMixin"]
