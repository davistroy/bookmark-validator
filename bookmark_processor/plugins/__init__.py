"""
Bookmark Processor Plugin System

Provides an extensible plugin architecture for custom validators,
AI processors, and output formats.
"""

from .base import (
    BookmarkPlugin,
    ValidatorPlugin,
    AIProcessorPlugin,
    OutputPlugin,
    TagGeneratorPlugin,
    ContentEnhancerPlugin,
    PluginHook,
    PluginMetadata,
    ValidationResult,
)
from .loader import PluginLoader
from .registry import PluginRegistry

__all__ = [
    # Base classes
    "BookmarkPlugin",
    "ValidatorPlugin",
    "AIProcessorPlugin",
    "OutputPlugin",
    "TagGeneratorPlugin",
    "ContentEnhancerPlugin",
    "PluginHook",
    "PluginMetadata",
    "ValidationResult",
    # Infrastructure
    "PluginLoader",
    "PluginRegistry",
]
