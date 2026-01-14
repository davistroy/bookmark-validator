"""
Batch Processing Types and Interfaces

This module contains data classes and abstract interfaces for batch processing
operations, including validation results, statistics, and configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.error_handler import ValidationError


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
    security_validation: Optional[Any] = None  # SecurityValidationResult

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
                self.security_validation.to_dict()
                if self.security_validation and hasattr(self.security_validation, 'to_dict')
                else None
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
    # Async/concurrent processing options
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


__all__ = [
    "ValidationResult",
    "ValidationStats",
    "BatchConfig",
    "CostBreakdown",
    "ProgressUpdate",
    "BatchResult",
    "BatchProcessorInterface",
    "ValidationError",
]
