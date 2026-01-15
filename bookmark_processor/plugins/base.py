"""
Plugin Base Classes

Defines abstract base classes for all plugin types in the bookmark processor.
Plugins can extend URL validation, AI processing, output formats, and more.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

# Import from parent package - handle relative import
try:
    from ..core.data_models import Bookmark
except ImportError:
    # For standalone testing
    Bookmark = Any


class PluginHook(str, Enum):
    """Available plugin hooks in the processing pipeline."""

    # Validation hooks
    PRE_VALIDATION = "pre_validation"
    POST_VALIDATION = "post_validation"
    VALIDATION_FILTER = "validation_filter"

    # Content hooks
    PRE_CONTENT_FETCH = "pre_content_fetch"
    POST_CONTENT_FETCH = "post_content_fetch"
    CONTENT_FILTER = "content_filter"

    # AI processing hooks
    PRE_AI_PROCESS = "pre_ai_process"
    POST_AI_PROCESS = "post_ai_process"
    AI_FALLBACK = "ai_fallback"

    # Tag hooks
    PRE_TAG_GENERATION = "pre_tag_generation"
    POST_TAG_GENERATION = "post_tag_generation"
    TAG_FILTER = "tag_filter"

    # Output hooks
    PRE_EXPORT = "pre_export"
    POST_EXPORT = "post_export"

    # Lifecycle hooks
    ON_START = "on_start"
    ON_COMPLETE = "on_complete"
    ON_ERROR = "on_error"


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    requires: List[str] = field(default_factory=list)  # Required plugin dependencies
    provides: List[str] = field(default_factory=list)  # Capabilities this plugin provides
    hooks: List[PluginHook] = field(default_factory=list)  # Hooks this plugin uses
    config_schema: Optional[Dict[str, Any]] = None  # JSON Schema for config validation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "requires": self.requires,
            "provides": self.provides,
            "hooks": [h.value for h in self.hooks],
        }


@dataclass
class ValidationResult:
    """Result of plugin validation."""

    is_valid: bool
    url: str
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    plugin_name: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "url": self.url,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "metadata": self.metadata,
            "plugin_name": self.plugin_name,
            "confidence": self.confidence,
        }


class BookmarkPlugin(ABC):
    """
    Base class for all bookmark processor plugins.

    Plugins must implement name and version properties, and can optionally
    implement lifecycle hooks (on_load, on_unload) and configuration handling.
    """

    def __init__(self):
        """Initialize the plugin."""
        self._config: Dict[str, Any] = {}
        self._enabled: bool = True
        self._logger = logging.getLogger(f"plugin.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this plugin."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version string of this plugin."""
        pass

    @property
    def description(self) -> str:
        """Return a description of the plugin."""
        return ""

    @property
    def author(self) -> str:
        """Return the author of the plugin."""
        return ""

    @property
    def requires(self) -> List[str]:
        """Return list of required plugin dependencies."""
        return []

    @property
    def provides(self) -> List[str]:
        """Return list of capabilities this plugin provides."""
        return []

    @property
    def hooks(self) -> List[PluginHook]:
        """Return list of hooks this plugin uses."""
        return []

    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set plugin enabled state."""
        self._enabled = value

    @property
    def config(self) -> Dict[str, Any]:
        """Get plugin configuration."""
        return self._config

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            requires=self.requires,
            provides=self.provides,
            hooks=self.hooks,
        )

    def on_load(self, config: Dict[str, Any]) -> None:
        """
        Called when the plugin is loaded.

        Override this method to perform initialization based on configuration.

        Args:
            config: Configuration dictionary for this plugin
        """
        self._config = config
        self._logger.info(f"Plugin {self.name} v{self.version} loaded")

    def on_unload(self) -> None:
        """
        Called when the plugin is unloaded.

        Override this method to perform cleanup.
        """
        self._logger.info(f"Plugin {self.name} unloaded")

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self._enabled,
            "config": self._config,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, version={self.version})>"


class ValidatorPlugin(BookmarkPlugin):
    """
    Plugin for custom URL validation.

    Extends the built-in URL validation with custom checks like
    paywall detection, content verification, etc.
    """

    @property
    def provides(self) -> List[str]:
        return ["validation"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [
            PluginHook.PRE_VALIDATION,
            PluginHook.POST_VALIDATION,
            PluginHook.VALIDATION_FILTER,
        ]

    @abstractmethod
    def validate(
        self, url: str, content: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a URL.

        Args:
            url: The URL to validate
            content: Optional fetched content for deeper validation

        Returns:
            ValidationResult with validation outcome
        """
        pass

    def should_validate(self, url: str) -> bool:
        """
        Check if this plugin should validate the given URL.

        Override to filter which URLs this validator handles.

        Args:
            url: URL to check

        Returns:
            True if this validator should process the URL
        """
        return True

    def get_priority(self) -> int:
        """
        Get validation priority.

        Lower numbers run first. Default is 100.
        """
        return 100


class AIProcessorPlugin(BookmarkPlugin):
    """
    Plugin for custom AI processing.

    Allows integration of custom AI models or services for
    description generation, summarization, etc.
    """

    @property
    def provides(self) -> List[str]:
        return ["ai_processing"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [
            PluginHook.PRE_AI_PROCESS,
            PluginHook.POST_AI_PROCESS,
            PluginHook.AI_FALLBACK,
        ]

    @abstractmethod
    def generate_description(
        self, bookmark: "Bookmark", content: str
    ) -> str:
        """
        Generate a description for a bookmark.

        Args:
            bookmark: The bookmark to process
            content: Fetched content from the URL

        Returns:
            Generated description string
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the AI processor is available.

        Returns:
            True if the processor can be used
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model being used."""
        return {
            "name": self.name,
            "version": self.version,
            "available": self.is_available(),
        }

    def estimate_cost(self, content_length: int) -> float:
        """
        Estimate the cost of processing content.

        Args:
            content_length: Length of content to process

        Returns:
            Estimated cost in USD (0.0 for free/local models)
        """
        return 0.0


class OutputPlugin(BookmarkPlugin):
    """
    Plugin for custom output formats.

    Allows exporting processed bookmarks in custom formats
    beyond the standard raindrop.io CSV.
    """

    @property
    def provides(self) -> List[str]:
        return ["output"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [PluginHook.PRE_EXPORT, PluginHook.POST_EXPORT]

    @abstractmethod
    def export(
        self, bookmarks: List["Bookmark"], output_path: Path
    ) -> None:
        """
        Export bookmarks to the output format.

        Args:
            bookmarks: List of processed bookmarks
            output_path: Path to write output file
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get the file extension for this output format.

        Returns:
            File extension (without dot), e.g., 'json', 'html'
        """
        pass

    def get_mime_type(self) -> str:
        """Get MIME type for this output format."""
        return "application/octet-stream"

    def supports_streaming(self) -> bool:
        """Check if this exporter supports streaming output."""
        return False


class TagGeneratorPlugin(BookmarkPlugin):
    """
    Plugin for custom tag generation.

    Allows custom logic for generating and optimizing tags.
    """

    @property
    def provides(self) -> List[str]:
        return ["tag_generation"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [
            PluginHook.PRE_TAG_GENERATION,
            PluginHook.POST_TAG_GENERATION,
            PluginHook.TAG_FILTER,
        ]

    @abstractmethod
    def generate_tags(
        self,
        bookmark: "Bookmark",
        content: str,
        existing_tags: List[str],
    ) -> List[str]:
        """
        Generate tags for a bookmark.

        Args:
            bookmark: The bookmark to generate tags for
            content: Fetched content from the URL
            existing_tags: Existing tags on the bookmark

        Returns:
            List of generated tags
        """
        pass

    def filter_tags(self, tags: List[str]) -> List[str]:
        """
        Filter and clean tags.

        Args:
            tags: Tags to filter

        Returns:
            Filtered tag list
        """
        return tags

    def get_max_tags(self) -> int:
        """Get maximum number of tags to generate."""
        return 5


class ContentEnhancerPlugin(BookmarkPlugin):
    """
    Plugin for content enhancement.

    Allows custom processing of fetched content before
    AI processing or tag generation.
    """

    @property
    def provides(self) -> List[str]:
        return ["content_enhancement"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [PluginHook.POST_CONTENT_FETCH, PluginHook.CONTENT_FILTER]

    @abstractmethod
    def enhance_content(
        self, bookmark: "Bookmark", content: str
    ) -> str:
        """
        Enhance or transform fetched content.

        Args:
            bookmark: The bookmark being processed
            content: Raw fetched content

        Returns:
            Enhanced content string
        """
        pass

    def should_process(self, bookmark: "Bookmark") -> bool:
        """Check if this enhancer should process the bookmark."""
        return True


# Type alias for plugin factories
PluginFactory = Callable[[], BookmarkPlugin]


__all__ = [
    "BookmarkPlugin",
    "ValidatorPlugin",
    "AIProcessorPlugin",
    "OutputPlugin",
    "TagGeneratorPlugin",
    "ContentEnhancerPlugin",
    "PluginHook",
    "PluginMetadata",
    "ValidationResult",
    "PluginFactory",
]
