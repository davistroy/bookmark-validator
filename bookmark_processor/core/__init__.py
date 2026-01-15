"""
Core bookmark processing modules.

This package contains the core functionality for bookmark processing,
including CSV handling, URL validation, AI processing, Chrome HTML parsing,
and AI-powered folder generation.
"""

from .chrome_html_parser import ChromeHTMLParser
from .folder_generator import AIFolderGenerator, FolderGenerationResult, FolderNode
from .quality_reporter import (
    QualityReporter,
    QualityMetrics,
    DescriptionMetrics,
    TagMetrics,
    FolderMetrics,
    AttentionItems,
    create_quality_report,
)

# Import pipeline config classes that don't have circular dependencies
from .pipeline.config import PipelineConfig, PipelineResults

# Import error handlers
from ..utils.error_handler import ChromeHTMLError, ChromeHTMLStructureError

# Import batch types (these are pure data classes with no circular deps)
from .batch_types import (
    BatchConfig,
    BatchResult,
    CostBreakdown,
    ProgressUpdate,
    ValidationResult,
    ValidationStats,
    BatchProcessorInterface,
)

# Import async HTTP client (no circular deps)
from .async_http_client import AsyncHttpClient


# Lazy imports to avoid circular dependencies
def _get_pipeline_class():
    from .pipeline import BookmarkProcessingPipeline
    return BookmarkProcessingPipeline


def _get_pipeline_factory():
    from .pipeline.factory import PipelineFactory
    return PipelineFactory


def _get_create_pipeline():
    from .pipeline.factory import create_pipeline
    return create_pipeline


def _get_url_validator():
    from .url_validator.validator import URLValidator as _BaseURLValidator
    from .url_validator.async_validator import AsyncValidatorMixin
    from .url_validator.batch_interface import BatchProcessorMixin

    class URLValidator(
        _BaseURLValidator,
        AsyncValidatorMixin,
        BatchProcessorMixin,
    ):
        pass

    return URLValidator


def _get_enhanced_batch_processor():
    # Import directly from the .py file, not the package
    from .batch_validator import EnhancedBatchProcessor
    return EnhancedBatchProcessor


def _get_validation_error():
    from ..utils.error_handler import ValidationError
    return ValidationError


# Module-level __getattr__ for lazy loading
_lazy_imports = {
    "BookmarkProcessingPipeline": _get_pipeline_class,
    "PipelineFactory": _get_pipeline_factory,
    "create_pipeline": _get_create_pipeline,
    "URLValidator": _get_url_validator,
    "EnhancedBatchProcessor": _get_enhanced_batch_processor,
    "BatchProcessor": _get_enhanced_batch_processor,
    "ValidationError": _get_validation_error,
    "AsyncUrlValidator": lambda: AsyncHttpClient,  # Alias
}

# Type aliases
BatchConfigType = BatchConfig
BatchResultType = BatchResult
CostBreakdownType = CostBreakdown
ProgressUpdateType = ProgressUpdate
ValidationResultType = ValidationResult
ValidationStatsType = ValidationStats


def __getattr__(name):
    if name in _lazy_imports:
        return _lazy_imports[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Type aliases
    "BatchConfigType",
    "BatchResultType",
    "CostBreakdownType",
    "ProgressUpdateType",
    "ValidationResultType",
    "ValidationStatsType",
    # Quality reporting (Phase 2)
    "QualityReporter",
    "QualityMetrics",
    "DescriptionMetrics",
    "TagMetrics",
    "FolderMetrics",
    "AttentionItems",
    "create_quality_report",
]
