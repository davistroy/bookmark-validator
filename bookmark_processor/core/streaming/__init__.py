"""
Streaming/Incremental Processing Module.

This module provides streaming capabilities for processing large bookmark
collections without loading all data into memory at once.

Main components:
- StreamingBookmarkReader: Generator-based bookmark reading
- StreamingBookmarkWriter: Incremental bookmark writing
- StreamingPipeline: Streaming pipeline execution
"""

from .reader import StreamingBookmarkReader
from .writer import StreamingBookmarkWriter
from .pipeline import StreamingPipeline, StreamingPipelineConfig, StreamingPipelineResults

__all__ = [
    "StreamingBookmarkReader",
    "StreamingBookmarkWriter",
    "StreamingPipeline",
    "StreamingPipelineConfig",
    "StreamingPipelineResults",
]
