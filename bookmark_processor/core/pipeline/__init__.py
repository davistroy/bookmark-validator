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

# Import config classes
from .config import PipelineConfig, PipelineResults

# Import factory
from .factory import PipelineFactory, create_pipeline

# Import main pipeline class from parent module
from ..pipeline import BookmarkProcessingPipeline

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
