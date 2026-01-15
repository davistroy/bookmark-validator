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

# Import mixins for potential future use
from .cost_tracking import CostTrackingMixin
from .performance import PerformanceOptimizationMixin


# Lazy import of EnhancedBatchProcessor to avoid circular imports
def _get_enhanced_batch_processor():
    """Lazy import of EnhancedBatchProcessor."""
    # Import at runtime to avoid circular dependency
    import importlib.util
    import sys
    import os

    # Get the path to batch_validator.py
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    batch_validator_path = os.path.join(core_dir, 'batch_validator.py')

    # Check if already loaded
    module_name = 'bookmark_processor.core._batch_validator'
    if module_name in sys.modules:
        return sys.modules[module_name].EnhancedBatchProcessor

    # Load from file path with proper package info for relative imports
    spec = importlib.util.spec_from_file_location(
        module_name,
        batch_validator_path,
        submodule_search_locations=[]
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        # Set up the module's package info so relative imports work
        module.__package__ = 'bookmark_processor.core'
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.EnhancedBatchProcessor

    raise ImportError("Could not load EnhancedBatchProcessor")


def __getattr__(name):
    if name == "EnhancedBatchProcessor":
        return _get_enhanced_batch_processor()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnhancedBatchProcessor",
    "CostTrackingMixin",
    "PerformanceOptimizationMixin",
]
