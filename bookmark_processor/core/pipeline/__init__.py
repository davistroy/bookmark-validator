"""
Pipeline Package

This package provides the main bookmark processing pipeline with:
- Pipeline configuration and results
- Stage-based processing (load, validate, analyze, AI, tags, folders, output)
- Checkpoint and resume functionality
- Factory methods for dependency injection

The package is organized into:
- config: PipelineConfig and PipelineResults dataclasses
- factory: PipelineFactory and helper functions
- pipeline: Main BookmarkProcessingPipeline class (in parent module)

Usage:
    from bookmark_processor.core.pipeline import (
        PipelineConfig,
        BookmarkProcessingPipeline,
        create_pipeline
    )

    config = PipelineConfig(
        input_file="input.csv",
        output_file="output.csv"
    )
    pipeline = create_pipeline(config.input_file, config.output_file)
    results = pipeline.execute()
"""

# Import config classes first (these have no circular dependencies)
from .config import PipelineConfig, PipelineResults

# Import factory (doesn't have circular dependency)
from .factory import PipelineFactory, create_pipeline


# Lazy import function to avoid circular imports with parent pipeline module
_cached_pipeline_class = None


def _get_pipeline_class():
    """Lazy import of BookmarkProcessingPipeline to avoid circular imports."""
    global _cached_pipeline_class
    if _cached_pipeline_class is not None:
        return _cached_pipeline_class

    # Import directly from the pipeline.py file using importlib
    import importlib.util
    import sys
    import os

    # Get the path to pipeline.py
    module_name = 'bookmark_processor.core._pipeline'
    if module_name in sys.modules:
        _cached_pipeline_class = sys.modules[module_name].BookmarkProcessingPipeline
        return _cached_pipeline_class

    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(core_dir, 'pipeline.py')

    # Create spec with proper submodule info so relative imports work
    spec = importlib.util.spec_from_file_location(
        module_name,
        pipeline_path,
        submodule_search_locations=[]
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        # Set up the module's package info for relative imports
        module.__package__ = 'bookmark_processor.core'
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _cached_pipeline_class = module.BookmarkProcessingPipeline
        return _cached_pipeline_class

    raise ImportError("Could not load BookmarkProcessingPipeline")


# Define __getattr__ for lazy loading
def __getattr__(name):
    if name == "BookmarkProcessingPipeline":
        return _get_pipeline_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configuration
    "PipelineConfig",
    "PipelineResults",
    # Main pipeline
    "BookmarkProcessingPipeline",
    # Factory
    "PipelineFactory",
    "create_pipeline",
]
