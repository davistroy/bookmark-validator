"""
Enhanced Cost Tracking and User Control System

This module provides comprehensive cost tracking, user confirmation workflows,
and cost estimation capabilities for cloud AI services.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bookmark_processor.config.configuration import Configuration


@dataclass
class CostRecord:
    """Individual cost record for tracking API usage."""

    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    operation_type: str = "description_generation"
    bookmark_count: int = 1
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostRecord":
        """Create from dictionary for deserialization."""
        return cls(**data)


class CostTracker:
    """Enhanced cost tracker with persistence, user control, and detailed analytics."""

    def __init__(
        self,
        confirmation_interval: float = 10.0,
        cost_log_file: Optional[str] = None,
        auto_save: bool = True,
        warning_threshold: float = 5.0,
    ):
        """
        Initialize enhanced cost tracker.

        Args:
            confirmation_interval: USD amount at which to prompt for user confirmation
            cost_log_file: Optional file to persist cost records
            auto_save: Whether to automatically save cost records
            warning_threshold: USD amount for issuing warnings
        """
        self.confirmation_interval = confirmation_interval
        self.warning_threshold = warning_threshold
        self.auto_save = auto_save

        # Cost tracking
        self.total_cost = 0.0
        self.session_cost = 0.0
        self.provider_costs = {}
        self.last_confirmation = 0.0
        self.user_confirmed_cost = 0.0

        # Detailed records
        self.cost_records: List[CostRecord] = []
        self.session_start_time = time.time()

        # File persistence
        if cost_log_file:
            self.cost_log_file = Path(cost_log_file)
        else:
            self.cost_log_file = Path.cwd() / ".bookmark_costs" / "cost_log.json"

        # Create directory if needed
        self.cost_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing records
        self._load_cost_history()

        self.logger = logging.getLogger(__name__)

    def add_cost_record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        operation_type: str = "description_generation",
        bookmark_count: int = 1,
        success: bool = True,
    ) -> None:
        """
        Add a detailed cost record.

        Args:
            provider: AI provider name
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            operation_type: Type of operation performed
            bookmark_count: Number of bookmarks processed
            success: Whether the operation was successful
        """
        record = CostRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            operation_type=operation_type,
            bookmark_count=bookmark_count,
            success=success,
        )

        # Update tracking variables
        if provider not in self.provider_costs:
            self.provider_costs[provider] = 0.0

        self.provider_costs[provider] += cost_usd
        self.total_cost += cost_usd
        self.session_cost += cost_usd

        # Store detailed record
        self.cost_records.append(record)

        # Auto-save if enabled
        if self.auto_save:
            self._save_cost_history()

        # Log the cost addition
        self.logger.debug(
            f"Added cost record: {provider}/{model} - ${cost_usd:.4f} "
            f"({input_tokens}+{output_tokens} tokens, {bookmark_count} bookmarks)"
        )

        # Issue warning if threshold exceeded
        if (
            self.session_cost >= self.warning_threshold
            and self.session_cost - cost_usd < self.warning_threshold
        ):
            self.logger.warning(
                f"Cost warning: Session cost has reached ${self.session_cost:.2f}"
            )

    def should_confirm(self) -> bool:
        """
        Check if user confirmation is needed based on cost threshold.

        Returns:
            True if confirmation is needed
        """
        cost_since_last = self.session_cost - self.last_confirmation
        return cost_since_last >= self.confirmation_interval

    def get_confirmation_prompt(self) -> str:
        """
        Get detailed user confirmation prompt with cost breakdown.

        Returns:
            Formatted prompt string with comprehensive cost information
        """
        cost_since_last = self.session_cost - self.last_confirmation

        prompt = f"\\n💰 Cost Update - Session Analysis:\\n"
        prompt += f"  💵 Current session: ${self.session_cost:.2f}\\n"
        prompt += f"  📈 Since last confirmation: ${cost_since_last:.2f}\\n"
        prompt += f"  📊 Total historical: ${self.total_cost:.2f}\\n"

        # Provider breakdown for current session
        if self.provider_costs:
            prompt += f"\\n  📋 Session breakdown by provider:\\n"
            session_provider_costs = self._get_session_provider_costs()
            for provider, cost in session_provider_costs.items():
                prompt += f"    • {provider}: ${cost:.2f}\\n"

        # Recent usage statistics
        recent_stats = self._get_recent_usage_stats()
        if recent_stats:
            prompt += f"\\n  ⚡ Recent activity (last 10 min):\\n"
            prompt += f"    • Requests: {recent_stats['request_count']}\\n"
            prompt += f"    • Avg cost/request: ${recent_stats['avg_cost_per_request']:.4f}\\n"
            prompt += f"    • Success rate: {recent_stats['success_rate']:.1f}%\\n"

        # Cost projection
        if len(self.cost_records) > 5:
            projection = self._estimate_hourly_cost()
            prompt += f"\\n  🔮 Estimated hourly rate: ${projection:.2f}/hour\\n"

        prompt += f"\\n❓ Continue processing? (y/n): "
        return prompt

    async def confirm_continuation(self) -> bool:
        """
        Prompt user for continuation confirmation with enhanced information.

        Returns:
            True if user wants to continue
        """
        if not self.should_confirm():
            return True

        try:
            # Show detailed prompt
            prompt = self.get_confirmation_prompt()

            # For async environments, we need to handle input differently
            # This is a blocking operation, but necessary for user interaction
            response = await asyncio.get_event_loop().run_in_executor(
                None, input, prompt
            )

            response = response.lower().strip()

            if response in ["y", "yes", ""]:  # Empty response defaults to yes
                self.last_confirmation = self.session_cost
                self.user_confirmed_cost = self.session_cost
                self.logger.info(
                    f"User confirmed continuation at ${self.session_cost:.2f}"
                )
                return True
            else:
                self.logger.info(f"User stopped processing at ${self.session_cost:.2f}")
                return False

        except (KeyboardInterrupt, EOFError):
            self.logger.info("User interrupted confirmation prompt")
            return False
        except Exception as e:
            self.logger.error(f"Error during confirmation prompt: {e}")
            # Default to continue if there's an error with the prompt
            return True

    def _get_session_provider_costs(self) -> Dict[str, float]:
        """Get provider costs for the current session only."""
        session_costs = {}
        session_start = self.session_start_time

        for record in self.cost_records:
            if record.timestamp >= session_start:
                if record.provider not in session_costs:
                    session_costs[record.provider] = 0.0
                session_costs[record.provider] += record.cost_usd

        return session_costs

    def _get_recent_usage_stats(self, minutes: int = 10) -> Optional[Dict[str, Any]]:
        """Get usage statistics for recent activity."""
        cutoff_time = time.time() - (minutes * 60)
        recent_records = [r for r in self.cost_records if r.timestamp >= cutoff_time]

        if not recent_records:
            return None

        total_cost = sum(r.cost_usd for r in recent_records)
        successful_records = [r for r in recent_records if r.success]

        return {
            "request_count": len(recent_records),
            "successful_requests": len(successful_records),
            "total_cost": total_cost,
            "avg_cost_per_request": total_cost / len(recent_records),
            "success_rate": (len(successful_records) / len(recent_records)) * 100,
            "time_period_minutes": minutes,
        }

    def _estimate_hourly_cost(self) -> float:
        """Estimate hourly cost based on recent usage patterns."""
        # Use last 30 minutes of data for estimation
        cutoff_time = time.time() - (30 * 60)  # 30 minutes
        recent_records = [r for r in self.cost_records if r.timestamp >= cutoff_time]

        if len(recent_records) < 3:
            return 0.0

        total_recent_cost = sum(r.cost_usd for r in recent_records)
        time_span_hours = (time.time() - recent_records[0].timestamp) / 3600

        if time_span_hours <= 0:
            return 0.0

        return total_recent_cost / time_span_hours

    def get_cost_estimate(
        self,
        bookmark_count: int,
        provider: str,
        sample_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Estimate cost for processing a given number of bookmarks.

        Args:
            bookmark_count: Number of bookmarks to process
            provider: AI provider to use
            sample_size: Number of recent records to use for estimation

        Returns:
            Dictionary with cost estimation
        """
        # Get recent records for the provider
        provider_records = [
            r
            for r in self.cost_records[-100:]  # Last 100 records
            if r.provider == provider and r.success
        ]

        if not provider_records:
            # Fallback to default estimates
            default_costs = {
                "claude": 0.0006,  # ~$0.0006 per bookmark
                "openai": 0.0012,  # ~$0.0012 per bookmark
                "local": 0.0,  # Free
            }
            cost_per_bookmark = default_costs.get(provider, 0.001)
            estimated_cost = bookmark_count * cost_per_bookmark

            return {
                "provider": provider,
                "bookmark_count": bookmark_count,
                "estimated_cost_usd": estimated_cost,
                "cost_per_bookmark": cost_per_bookmark,
                "confidence": "low",
                "method": "default_estimate",
                "note": "No historical data available, using default estimates",
            }

        # Use recent data for more accurate estimation
        recent_records = provider_records[-sample_size:]
        total_cost = sum(r.cost_usd for r in recent_records)
        total_bookmarks = sum(r.bookmark_count for r in recent_records)

        if total_bookmarks == 0:
            cost_per_bookmark = 0.001  # Fallback
        else:
            cost_per_bookmark = total_cost / total_bookmarks

        estimated_cost = bookmark_count * cost_per_bookmark

        # Calculate confidence based on data quality
        confidence = "high" if len(recent_records) >= sample_size else "medium"

        return {
            "provider": provider,
            "bookmark_count": bookmark_count,
            "estimated_cost_usd": estimated_cost,
            "cost_per_bookmark": cost_per_bookmark,
            "confidence": confidence,
            "method": "historical_average",
            "sample_size": len(recent_records),
            "historical_bookmarks": total_bookmarks,
            "note": f"Based on last {len(recent_records)} successful operations",
        }

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cost tracking statistics."""
        session_records = [
            r for r in self.cost_records if r.timestamp >= self.session_start_time
        ]

        # Time-based analysis
        now = time.time()
        session_duration_hours = (now - self.session_start_time) / 3600

        # Success rate analysis
        total_requests = len(session_records)
        successful_requests = len([r for r in session_records if r.success])
        success_rate = (successful_requests / max(total_requests, 1)) * 100

        # Token analysis
        total_input_tokens = sum(r.input_tokens for r in session_records)
        total_output_tokens = sum(r.output_tokens for r in session_records)

        # Provider analysis
        provider_stats = {}
        for provider in set(r.provider for r in session_records):
            provider_records = [r for r in session_records if r.provider == provider]
            provider_cost = sum(r.cost_usd for r in provider_records)
            provider_requests = len(provider_records)

            provider_stats[provider] = {
                "requests": provider_requests,
                "cost_usd": provider_cost,
                "avg_cost_per_request": provider_cost / max(provider_requests, 1),
                "total_bookmarks": sum(r.bookmark_count for r in provider_records),
            }

        return {
            "session": {
                "total_cost_usd": self.session_cost,
                "duration_hours": session_duration_hours,
                "requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate_percent": success_rate,
                "cost_per_hour": self.session_cost / max(session_duration_hours, 0.01),
            },
            "tokens": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            "providers": provider_stats,
            "historical": {
                "total_historical_cost": self.total_cost,
                "total_records": len(self.cost_records),
                "confirmation_interval": self.confirmation_interval,
                "last_confirmation_cost": self.last_confirmation,
            },
            "thresholds": {
                "confirmation_interval": self.confirmation_interval,
                "warning_threshold": self.warning_threshold,
                "cost_since_confirmation": self.session_cost - self.last_confirmation,
                "needs_confirmation": self.should_confirm(),
            },
        }

    def _save_cost_history(self) -> None:
        """Save cost records to file."""
        try:
            data = {
                "records": [record.to_dict() for record in self.cost_records],
                "total_cost": self.total_cost,
                "last_updated": time.time(),
            }

            with open(self.cost_log_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cost history: {e}")

    def _load_cost_history(self) -> None:
        """Load cost records from file."""
        try:
            if not self.cost_log_file.exists():
                return

            with open(self.cost_log_file, "r") as f:
                data = json.load(f)

            # Load records
            self.cost_records = [
                CostRecord.from_dict(record_data)
                for record_data in data.get("records", [])
            ]

            # Calculate total cost from records
            self.total_cost = sum(r.cost_usd for r in self.cost_records)

            # Rebuild provider costs
            self.provider_costs = {}
            for record in self.cost_records:
                if record.provider not in self.provider_costs:
                    self.provider_costs[record.provider] = 0.0
                self.provider_costs[record.provider] += record.cost_usd

            self.logger.info(
                f"Loaded {len(self.cost_records)} cost records, total: ${self.total_cost:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load cost history: {e}")
            # Reset to clean state if loading fails
            self.cost_records = []
            self.total_cost = 0.0
            self.provider_costs = {}

    def export_cost_report(self, output_file: Optional[str] = None) -> str:
        """
        Export detailed cost report to file.

        Args:
            output_file: Optional output file path

        Returns:
            Path to the exported report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"cost_report_{timestamp}.json"

        output_path = Path(output_file)

        # Generate comprehensive report
        report = {
            "report_generated": datetime.now().isoformat(),
            "summary": self.get_detailed_statistics(),
            "all_records": [record.to_dict() for record in self.cost_records],
            "provider_analysis": self._generate_provider_analysis(),
            "usage_patterns": self._analyze_usage_patterns(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Cost report exported to {output_path}")
        return str(output_path)

    def _generate_provider_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis by provider."""
        analysis = {}

        for provider in set(r.provider for r in self.cost_records):
            provider_records = [r for r in self.cost_records if r.provider == provider]

            if not provider_records:
                continue

            total_cost = sum(r.cost_usd for r in provider_records)
            total_tokens = sum(
                r.input_tokens + r.output_tokens for r in provider_records
            )
            total_bookmarks = sum(r.bookmark_count for r in provider_records)

            analysis[provider] = {
                "total_requests": len(provider_records),
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "total_bookmarks": total_bookmarks,
                "avg_cost_per_request": total_cost / len(provider_records),
                "avg_cost_per_bookmark": total_cost / max(total_bookmarks, 1),
                "avg_tokens_per_request": total_tokens / len(provider_records),
                "cost_per_1k_tokens": (total_cost / max(total_tokens, 1)) * 1000,
                "first_used": min(r.timestamp for r in provider_records),
                "last_used": max(r.timestamp for r in provider_records),
            }

        return analysis

    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns over time."""
        if not self.cost_records:
            return {"note": "No usage data available"}

        # Group by hour
        hourly_costs = {}
        for record in self.cost_records:
            hour = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:00")
            if hour not in hourly_costs:
                hourly_costs[hour] = 0.0
            hourly_costs[hour] += record.cost_usd

        # Group by day
        daily_costs = {}
        for record in self.cost_records:
            day = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d")
            if day not in daily_costs:
                daily_costs[day] = 0.0
            daily_costs[day] += record.cost_usd

        return {
            "hourly_breakdown": hourly_costs,
            "daily_breakdown": daily_costs,
            "peak_hour": (
                max(hourly_costs.items(), key=lambda x: x[1]) if hourly_costs else None
            ),
            "total_days_active": len(daily_costs),
            "avg_daily_cost": sum(daily_costs.values()) / max(len(daily_costs), 1),
        }

    def reset_session(self) -> None:
        """Reset session tracking (but keep historical data)."""
        self.session_cost = 0.0
        self.last_confirmation = 0.0
        self.user_confirmed_cost = 0.0
        self.session_start_time = time.time()
        self.logger.info("Cost tracker session reset")


@dataclass
class APIUsage:
    """Track API usage statistics."""

    engine: str
    requests_made: int = 0
    tokens_consumed: int = 0
    estimated_cost: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "engine": self.engine,
            "requests_made": self.requests_made,
            "tokens_consumed": self.tokens_consumed,
            "estimated_cost": self.estimated_cost,
            "timestamp": self.timestamp,
        }


@dataclass
class CostEstimate:
    """Estimate costs for processing operations."""

    total_items: int
    estimated_tokens_per_item: int = 150
    cost_per_token: float = 0.0001  # Default rough estimate

    @property
    def estimated_total_tokens(self) -> int:
        """Calculate total estimated tokens."""
        return self.total_items * self.estimated_tokens_per_item

    @property
    def estimated_total_cost(self) -> float:
        """Calculate total estimated cost."""
        return self.estimated_total_tokens * self.cost_per_token

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "estimated_tokens_per_item": self.estimated_tokens_per_item,
            "cost_per_token": self.cost_per_token,
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_total_cost": self.estimated_total_cost,
        }
