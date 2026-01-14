"""
Batch Validator Package

This package provides enhanced batch processing capabilities with:
- Cost tracking and budget management
- Sync and async processing modes
- Performance optimization and auto-tuning
- Progress tracking and statistics

The implementation is organized into mixins for maintainability:
- cost_tracking: Cost estimation and budget management
- performance: Auto-tuning and concurrency adaptation

The main EnhancedBatchProcessor class is currently in the parent module
but can be further decomposed in future refactoring.

Usage:
    from bookmark_processor.core.batch_validator import EnhancedBatchProcessor

    processor = EnhancedBatchProcessor(
        processor=validator,
        config=batch_config,
        progress_callback=callback_func
    )
    results = processor.process_all()
"""

# Import from the parent module (currently the main implementation)
from ..batch_validator import EnhancedBatchProcessor

# Import mixins for potential future use
from .cost_tracking import CostTrackingMixin
from .performance import PerformanceOptimizationMixin

__all__ = [
    "EnhancedBatchProcessor",
    "CostTrackingMixin",
    "PerformanceOptimizationMixin",
]
