"""
Core bookmark processing modules.

This package contains the core functionality for bookmark processing,
including CSV handling, URL validation, AI processing, Chrome HTML parsing,
and AI-powered folder generation.
"""

from .chrome_html_parser import ChromeHTMLParser
from .folder_generator import AIFolderGenerator, FolderGenerationResult, FolderNode
from .pipeline import (
    BookmarkProcessingPipeline,
    PipelineConfig,
    PipelineFactory,
    PipelineResults,
    create_pipeline,
)
from ..utils.error_handler import ChromeHTMLError, ChromeHTMLStructureError

# URL validation and batch processing
from .url_validator import (
    BatchConfig,
    BatchProcessorInterface,
    BatchResult,
    CostBreakdown,
    EnhancedBatchProcessor,  # Re-exported from batch_validator
    ProgressUpdate,
    URLValidator,
    ValidationError,
    ValidationResult,
    ValidationStats,
    AsyncHttpClient,
)
from .batch_validator import EnhancedBatchProcessor as BatchProcessor

# Batch types module (extracted from url_validator)
from .batch_types import (
    BatchConfig as BatchConfigType,
    BatchResult as BatchResultType,
    CostBreakdown as CostBreakdownType,
    ProgressUpdate as ProgressUpdateType,
    ValidationResult as ValidationResultType,
    ValidationStats as ValidationStatsType,
)

# Async HTTP client (extracted from url_validator)
from .async_http_client import AsyncHttpClient as AsyncUrlValidator

__all__ = [
    "ChromeHTMLParser",
    "ChromeHTMLError",
    "ChromeHTMLStructureError",
    "AIFolderGenerator",
    "FolderNode",
    "FolderGenerationResult",
    "BookmarkProcessingPipeline",
    "PipelineConfig",
    "PipelineFactory",
    "PipelineResults",
    "create_pipeline",
    # URL Validation and Batch Processing
    "URLValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationStats",
    "BatchConfig",
    "CostBreakdown",
    "ProgressUpdate",
    "BatchResult",
    "BatchProcessorInterface",
    "EnhancedBatchProcessor",
    "BatchProcessor",
    # Extracted modules
    "AsyncHttpClient",
    "AsyncUrlValidator",
]
