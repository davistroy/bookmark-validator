"""
Comprehensive Tests for Plugin Registry

Tests the PluginRegistry class with full coverage including:
- All public methods
- Plugin registration and retrieval
- Error handling paths
- Plugin management operations
- Hook dispatching and chaining
- Plugin type indexing
- Dependency checking
"""

import logging
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

from bookmark_processor.plugins.base import (
    BookmarkPlugin,
    ValidatorPlugin,
    AIProcessorPlugin,
    OutputPlugin,
    TagGeneratorPlugin,
    ContentEnhancerPlugin,
    PluginHook,
    ValidationResult,
)
from bookmark_processor.plugins.loader import PluginLoader, PluginLoadError
from bookmark_processor.plugins.registry import (
    PluginRegistry,
    get_registry,
    reset_registry,
)


# ============================================================================
# Test Plugin Classes
# ============================================================================


class MockValidatorPlugin(ValidatorPlugin):
    """Mock validator plugin."""

    @property
    def name(self) -> str:
        return "test-validator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Test validator plugin"

    @property
    def author(self) -> str:
        return "Test Author"

    @property
    def requires(self) -> List[str]:
        return []

    def validate(self, url: str, content=None) -> ValidationResult:
        return ValidationResult(is_valid=True, url=url, plugin_name=self.name)

    def get_priority(self) -> int:
        return 50

    def on_pre_validation(self, *args, **kwargs):
        return "pre_validation_result"

    def on_post_validation(self, *args, **kwargs):
        return "post_validation_result"


class HighPriorityValidatorPlugin(ValidatorPlugin):
    """Higher priority validator plugin."""

    @property
    def name(self) -> str:
        return "high-priority-validator"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate(self, url: str, content=None) -> ValidationResult:
        return ValidationResult(is_valid=True, url=url)

    def get_priority(self) -> int:
        return 10


class LowPriorityValidatorPlugin(ValidatorPlugin):
    """Lower priority validator plugin."""

    @property
    def name(self) -> str:
        return "low-priority-validator"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate(self, url: str, content=None) -> ValidationResult:
        return ValidationResult(is_valid=True, url=url)

    def get_priority(self) -> int:
        return 200


class MockAIProcessorPlugin(AIProcessorPlugin):
    """Mock AI processor plugin."""

    @property
    def name(self) -> str:
        return "test-ai-processor"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_description(self, bookmark, content: str) -> str:
        return "Generated description"

    def is_available(self) -> bool:
        return True

    def on_pre_ai_process(self, *args, **kwargs):
        return "pre_ai_result"

    def on_post_ai_process(self, *args, **kwargs):
        return "post_ai_result"


class MockOutputPlugin(OutputPlugin):
    """Mock output plugin."""

    @property
    def name(self) -> str:
        return "test-output"

    @property
    def version(self) -> str:
        return "1.0.0"

    def export(self, bookmarks, output_path: Path) -> None:
        pass

    def get_file_extension(self) -> str:
        return "txt"

    def on_pre_export(self, *args, **kwargs):
        return "pre_export_result"


class MockTagGeneratorPlugin(TagGeneratorPlugin):
    """Mock tag generator plugin."""

    @property
    def name(self) -> str:
        return "test-tag-generator"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_tags(self, bookmark, content: str, existing_tags: List[str]) -> List[str]:
        return ["tag1", "tag2"]

    def on_pre_tag_generation(self, *args, **kwargs):
        return "pre_tag_result"


class MockContentEnhancerPlugin(ContentEnhancerPlugin):
    """Mock content enhancer plugin."""

    @property
    def name(self) -> str:
        return "test-content-enhancer"

    @property
    def version(self) -> str:
        return "1.0.0"

    def enhance_content(self, bookmark, content: str) -> str:
        return f"Enhanced: {content}"

    def on_post_content_fetch(self, *args, **kwargs):
        return "post_content_result"


class SimpleBookmarkPlugin(BookmarkPlugin):
    """Simple bookmark plugin for basic tests."""

    @property
    def name(self) -> str:
        return "simple-plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Simple test plugin"

    @property
    def author(self) -> str:
        return "Test Author"

    @property
    def requires(self) -> List[str]:
        return []

    @property
    def provides(self) -> List[str]:
        return ["simple-capability"]


class PluginWithDependencies(BookmarkPlugin):
    """Plugin with dependencies for testing."""

    @property
    def name(self) -> str:
        return "plugin-with-deps"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requires(self) -> List[str]:
        return ["simple-plugin", "missing-plugin"]


class PluginWithHooks(BookmarkPlugin):
    """Plugin with multiple hooks for testing."""

    @property
    def name(self) -> str:
        return "plugin-with-hooks"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def hooks(self) -> List[PluginHook]:
        return [PluginHook.ON_START, PluginHook.ON_COMPLETE, PluginHook.ON_ERROR]

    def on_start(self, *args, **kwargs):
        return "started"

    def on_complete(self, *args, **kwargs):
        return "completed"

    def on_error(self, error, *args, **kwargs):
        return f"error: {error}"


class ErrorPronePlugin(BookmarkPlugin):
    """Plugin that throws errors for testing error handling."""

    @property
    def name(self) -> str:
        return "error-prone-plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def hooks(self) -> List[PluginHook]:
        return [PluginHook.ON_START]

    def on_start(self, *args, **kwargs):
        raise RuntimeError("Plugin error")

    def on_unload(self) -> None:
        raise RuntimeError("Unload error")


class ChainablePlugin(BookmarkPlugin):
    """Plugin for testing hook chain."""

    def __init__(self, name_suffix: str = "1"):
        self._name_suffix = name_suffix  # Set before super().__init__() because name property is accessed
        super().__init__()

    @property
    def name(self) -> str:
        return f"chainable-{self._name_suffix}"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def hooks(self) -> List[PluginHook]:
        return [PluginHook.TAG_FILTER]

    def filter_tags(self, tags, *args, **kwargs):
        return tags + [f"added-by-{self._name_suffix}"]


# ============================================================================
# Registry Initialization Tests
# ============================================================================


class TestRegistryInitialization:
    """Tests for PluginRegistry initialization."""

    def test_registry_initialization_default_loader(self):
        """Test registry initializes with default loader."""
        registry = PluginRegistry()
        assert registry.loader is not None
        assert isinstance(registry.loader, PluginLoader)

    def test_registry_initialization_custom_loader(self):
        """Test registry initializes with custom loader."""
        custom_loader = PluginLoader()
        registry = PluginRegistry(loader=custom_loader)
        assert registry.loader is custom_loader

    def test_registry_initial_state(self):
        """Test registry has correct initial state."""
        registry = PluginRegistry()
        assert registry.list_plugins() == []
        assert len(registry.get_validators()) == 0
        assert len(registry.get_ai_processors()) == 0
        assert len(registry.get_output_plugins()) == 0
        assert len(registry.get_tag_generators()) == 0
        assert len(registry.get_content_enhancers()) == 0


# ============================================================================
# Plugin Registration Tests
# ============================================================================


class TestPluginRegistration:
    """Tests for plugin registration methods."""

    def test_register_plugin_class(self):
        """Test registering a plugin class."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)

        available = registry.list_available()
        assert "simple-plugin" in available

    def test_register_plugin_class_error(self):
        """Test registering a plugin class that fails instantiation."""

        class BadPlugin(BookmarkPlugin):
            def __init__(self):
                raise RuntimeError("Init failed")

            @property
            def name(self) -> str:
                return "bad-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

        registry = PluginRegistry()
        with pytest.raises(RuntimeError):
            registry.register(BadPlugin)

    def test_register_instance(self):
        """Test registering a plugin instance."""
        registry = PluginRegistry()
        plugin = SimpleBookmarkPlugin()
        registry.register_instance(plugin)

        assert registry.has_plugin("simple-plugin")
        assert registry.get("simple-plugin") is plugin

    def test_register_instance_replaces_existing(self):
        """Test registering an instance replaces existing plugin."""
        registry = PluginRegistry()
        plugin1 = SimpleBookmarkPlugin()
        plugin2 = SimpleBookmarkPlugin()

        registry.register_instance(plugin1)
        registry.register_instance(plugin2)

        assert registry.get("simple-plugin") is plugin2

    def test_register_instance_all_plugin_types(self):
        """Test registering all plugin types."""
        registry = PluginRegistry()

        validator = MockValidatorPlugin()
        ai_processor = MockAIProcessorPlugin()
        output_plugin = MockOutputPlugin()
        tag_generator = MockTagGeneratorPlugin()
        content_enhancer = MockContentEnhancerPlugin()

        registry.register_instance(validator)
        registry.register_instance(ai_processor)
        registry.register_instance(output_plugin)
        registry.register_instance(tag_generator)
        registry.register_instance(content_enhancer)

        assert len(registry.get_validators()) == 1
        assert len(registry.get_ai_processors()) == 1
        assert len(registry.get_output_plugins()) == 1
        assert len(registry.get_tag_generators()) == 1
        assert len(registry.get_content_enhancers()) == 1


# ============================================================================
# Plugin Indexing Tests
# ============================================================================


class TestPluginIndexing:
    """Tests for plugin type indexing."""

    def test_validators_sorted_by_priority(self):
        """Test validators are sorted by priority."""
        registry = PluginRegistry()

        low = LowPriorityValidatorPlugin()
        high = HighPriorityValidatorPlugin()
        medium = MockValidatorPlugin()

        # Register in random order
        registry.register_instance(low)
        registry.register_instance(high)
        registry.register_instance(medium)

        validators = registry.get_validators()
        priorities = [v.get_priority() for v in validators]
        assert priorities == sorted(priorities)

    def test_hook_subscribers_indexed(self):
        """Test hooks are properly indexed."""
        registry = PluginRegistry()
        plugin = PluginWithHooks()
        registry.register_instance(plugin)

        start_subs = registry.get_hook_subscribers(PluginHook.ON_START)
        complete_subs = registry.get_hook_subscribers(PluginHook.ON_COMPLETE)
        error_subs = registry.get_hook_subscribers(PluginHook.ON_ERROR)

        assert plugin in start_subs
        assert plugin in complete_subs
        assert plugin in error_subs

    def test_disabled_plugin_not_in_type_getters(self):
        """Test disabled plugins are excluded from type getters."""
        registry = PluginRegistry()
        validator = MockValidatorPlugin()
        validator.enabled = False
        registry.register_instance(validator)

        validators = registry.get_validators()
        assert len(validators) == 0

    def test_disabled_plugin_not_in_hook_subscribers(self):
        """Test disabled plugins are excluded from hook subscribers."""
        registry = PluginRegistry()
        plugin = PluginWithHooks()
        plugin.enabled = False
        registry.register_instance(plugin)

        subscribers = registry.get_hook_subscribers(PluginHook.ON_START)
        assert len(subscribers) == 0


# ============================================================================
# Plugin Loading Tests
# ============================================================================


class TestPluginLoading:
    """Tests for load_plugins method."""

    def test_load_plugins_success(self):
        """Test loading multiple plugins."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)
        registry.register(MockValidatorPlugin)

        loaded = registry.load_plugins(["simple-plugin", "test-validator"])
        assert len(loaded) == 2
        assert "simple-plugin" in loaded
        assert "test-validator" in loaded

    def test_load_plugins_with_config(self):
        """Test loading plugins with configuration."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)

        config = {"simple-plugin": {"key": "value"}}
        loaded = registry.load_plugins(["simple-plugin"], config)

        plugin = loaded["simple-plugin"]
        assert plugin.config == {"key": "value"}

    def test_load_plugins_partial_failure(self):
        """Test loading plugins with some failures."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)
        # "nonexistent" is not registered

        loaded = registry.load_plugins(["simple-plugin", "nonexistent"])
        assert "simple-plugin" in loaded
        assert "nonexistent" not in loaded

    def test_load_plugins_empty_list(self):
        """Test loading empty list of plugins."""
        registry = PluginRegistry()
        loaded = registry.load_plugins([])
        assert loaded == {}


# ============================================================================
# Plugin Unloading Tests
# ============================================================================


class TestPluginUnloading:
    """Tests for plugin unloading methods."""

    def test_unload_plugin_success(self):
        """Test unloading a registered plugin."""
        registry = PluginRegistry()
        registry.register_instance(SimpleBookmarkPlugin())

        result = registry.unload_plugin("simple-plugin")
        assert result is True
        assert not registry.has_plugin("simple-plugin")

    def test_unload_plugin_not_found(self):
        """Test unloading a non-existent plugin."""
        registry = PluginRegistry()
        result = registry.unload_plugin("nonexistent")
        assert result is False

    def test_unload_plugin_with_error(self):
        """Test unloading a plugin that throws on unload."""
        registry = PluginRegistry()
        registry.register_instance(ErrorPronePlugin())

        # Should not raise, just log warning
        result = registry.unload_plugin("error-prone-plugin")
        assert result is True
        assert not registry.has_plugin("error-prone-plugin")

    def test_unload_removes_from_indexes(self):
        """Test unloading removes plugin from all indexes."""
        registry = PluginRegistry()
        validator = MockValidatorPlugin()
        registry.register_instance(validator)

        registry.unload_plugin("test-validator")

        assert len(registry.get_validators()) == 0
        assert len(registry.get_hook_subscribers(PluginHook.PRE_VALIDATION)) == 0

    def test_unload_all(self):
        """Test unloading all plugins."""
        registry = PluginRegistry()
        registry.register_instance(SimpleBookmarkPlugin())
        registry.register_instance(MockValidatorPlugin())
        registry.register_instance(MockAIProcessorPlugin())

        registry.unload_all()

        assert len(registry.list_plugins()) == 0
        assert len(registry.get_validators()) == 0
        assert len(registry.get_ai_processors()) == 0


# ============================================================================
# Plugin Retrieval Tests
# ============================================================================


class TestPluginRetrieval:
    """Tests for plugin retrieval methods."""

    def test_get_existing_plugin(self):
        """Test getting an existing plugin."""
        registry = PluginRegistry()
        plugin = SimpleBookmarkPlugin()
        registry.register_instance(plugin)

        retrieved = registry.get("simple-plugin")
        assert retrieved is plugin

    def test_get_nonexistent_plugin(self):
        """Test getting a non-existent plugin."""
        registry = PluginRegistry()
        retrieved = registry.get("nonexistent")
        assert retrieved is None

    def test_get_by_type(self):
        """Test getting plugins by type."""
        registry = PluginRegistry()
        registry.register_instance(MockValidatorPlugin())
        registry.register_instance(MockAIProcessorPlugin())
        registry.register_instance(SimpleBookmarkPlugin())

        validators = registry.get_by_type(ValidatorPlugin)
        assert len(validators) == 1
        assert isinstance(validators[0], ValidatorPlugin)

        ai_processors = registry.get_by_type(AIProcessorPlugin)
        assert len(ai_processors) == 1

    def test_has_plugin(self):
        """Test checking if plugin exists."""
        registry = PluginRegistry()
        registry.register_instance(SimpleBookmarkPlugin())

        assert registry.has_plugin("simple-plugin") is True
        assert registry.has_plugin("nonexistent") is False

    def test_list_plugins(self):
        """Test listing loaded plugins."""
        registry = PluginRegistry()
        registry.register_instance(SimpleBookmarkPlugin())
        registry.register_instance(MockValidatorPlugin())

        plugins = registry.list_plugins()
        assert "simple-plugin" in plugins
        assert "test-validator" in plugins
        assert len(plugins) == 2

    def test_list_available(self):
        """Test listing available plugins."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)
        registry.register(MockValidatorPlugin)

        available = registry.list_available()
        assert "simple-plugin" in available
        assert "test-validator" in available


# ============================================================================
# Hook System Tests
# ============================================================================


class TestHookSystem:
    """Tests for hook dispatching system."""

    def test_dispatch_hook_basic(self):
        """Test basic hook dispatching."""
        registry = PluginRegistry()
        plugin = PluginWithHooks()
        registry.register_instance(plugin)

        results = registry.dispatch_hook(PluginHook.ON_START)
        assert "started" in results

    def test_dispatch_hook_with_args(self):
        """Test hook dispatching with arguments."""
        registry = PluginRegistry()
        plugin = PluginWithHooks()
        registry.register_instance(plugin)

        results = registry.dispatch_hook(PluginHook.ON_ERROR, "test error")
        assert "error: test error" in results

    def test_dispatch_hook_multiple_subscribers(self):
        """Test dispatching to multiple subscribers."""
        registry = PluginRegistry()
        registry.register_instance(MockValidatorPlugin())
        registry.register_instance(HighPriorityValidatorPlugin())

        # Both should be subscribed to PRE_VALIDATION
        subscribers = registry.get_hook_subscribers(PluginHook.PRE_VALIDATION)
        assert len(subscribers) == 2

    def test_dispatch_hook_no_subscribers(self):
        """Test dispatching hook with no subscribers."""
        registry = PluginRegistry()
        results = registry.dispatch_hook(PluginHook.ON_START)
        assert results == []

    def test_dispatch_hook_error_handling(self):
        """Test hook error handling continues to other plugins."""
        registry = PluginRegistry()
        registry.register_instance(ErrorPronePlugin())
        registry.register_instance(PluginWithHooks())

        # Should not raise, should continue to next plugin
        results = registry.dispatch_hook(PluginHook.ON_START)
        assert "started" in results

    def test_dispatch_hook_stop_on_failure(self):
        """Test hook stops on failure when configured."""
        registry = PluginRegistry()
        registry.register_instance(ErrorPronePlugin())
        registry.register_instance(PluginWithHooks())

        with pytest.raises(RuntimeError, match="Plugin error"):
            registry.dispatch_hook(PluginHook.ON_START, stop_on_failure=True)

    def test_dispatch_hook_no_handler(self):
        """Test dispatching hook when plugin has no handler method."""
        registry = PluginRegistry()
        # SimpleBookmarkPlugin has no hook handlers
        plugin = SimpleBookmarkPlugin()
        # Manually add to hook subscribers to test handler lookup
        registry._hook_subscribers[PluginHook.ON_START].append(plugin)

        results = registry.dispatch_hook(PluginHook.ON_START)
        assert results == []  # No results since no handler

    def test_dispatch_hook_chain(self):
        """Test hook chain dispatching."""
        registry = PluginRegistry()
        registry.register_instance(ChainablePlugin("1"))
        registry.register_instance(ChainablePlugin("2"))

        initial_tags = ["original"]
        result = registry.dispatch_hook_chain(PluginHook.TAG_FILTER, initial_tags)

        assert "original" in result
        assert "added-by-1" in result
        assert "added-by-2" in result

    def test_dispatch_hook_chain_error_continues(self):
        """Test hook chain continues after error."""

        class ErrorChainPlugin(BookmarkPlugin):
            @property
            def name(self) -> str:
                return "error-chain"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def hooks(self) -> List[PluginHook]:
                return [PluginHook.TAG_FILTER]

            def filter_tags(self, tags, *args, **kwargs):
                raise RuntimeError("Chain error")

        registry = PluginRegistry()
        registry.register_instance(ErrorChainPlugin())
        registry.register_instance(ChainablePlugin("after"))

        initial = ["original"]
        result = registry.dispatch_hook_chain(PluginHook.TAG_FILTER, initial)

        # Should continue with the value after error
        assert "original" in result
        assert "added-by-after" in result

    def test_dispatch_hook_chain_empty(self):
        """Test hook chain with no subscribers."""
        registry = PluginRegistry()
        initial = ["tag1", "tag2"]
        result = registry.dispatch_hook_chain(PluginHook.TAG_FILTER, initial)
        assert result == initial


# ============================================================================
# Hook Handler Mapping Tests
# ============================================================================


class TestHookHandlerMapping:
    """Tests for _get_hook_handler method."""

    def test_get_hook_handler_validation_hooks(self):
        """Test getting handlers for validation hooks."""
        registry = PluginRegistry()
        plugin = MockValidatorPlugin()

        handler = registry._get_hook_handler(plugin, PluginHook.PRE_VALIDATION)
        assert handler is not None
        assert handler() == "pre_validation_result"

        handler = registry._get_hook_handler(plugin, PluginHook.POST_VALIDATION)
        assert handler is not None
        assert handler() == "post_validation_result"

    def test_get_hook_handler_ai_hooks(self):
        """Test getting handlers for AI hooks."""
        registry = PluginRegistry()
        plugin = MockAIProcessorPlugin()

        handler = registry._get_hook_handler(plugin, PluginHook.PRE_AI_PROCESS)
        assert handler is not None
        assert handler() == "pre_ai_result"

    def test_get_hook_handler_export_hooks(self):
        """Test getting handlers for export hooks."""
        registry = PluginRegistry()
        plugin = MockOutputPlugin()

        handler = registry._get_hook_handler(plugin, PluginHook.PRE_EXPORT)
        assert handler is not None
        assert handler() == "pre_export_result"

    def test_get_hook_handler_tag_hooks(self):
        """Test getting handlers for tag hooks."""
        registry = PluginRegistry()
        plugin = MockTagGeneratorPlugin()

        handler = registry._get_hook_handler(plugin, PluginHook.PRE_TAG_GENERATION)
        assert handler is not None
        assert handler() == "pre_tag_result"

    def test_get_hook_handler_content_hooks(self):
        """Test getting handlers for content hooks."""
        registry = PluginRegistry()
        plugin = MockContentEnhancerPlugin()

        handler = registry._get_hook_handler(plugin, PluginHook.POST_CONTENT_FETCH)
        assert handler is not None
        assert handler() == "post_content_result"

    def test_get_hook_handler_lifecycle_hooks(self):
        """Test getting handlers for lifecycle hooks."""
        registry = PluginRegistry()
        plugin = PluginWithHooks()

        start_handler = registry._get_hook_handler(plugin, PluginHook.ON_START)
        complete_handler = registry._get_hook_handler(plugin, PluginHook.ON_COMPLETE)
        error_handler = registry._get_hook_handler(plugin, PluginHook.ON_ERROR)

        assert start_handler is not None
        assert complete_handler is not None
        assert error_handler is not None

    def test_get_hook_handler_missing_handler(self):
        """Test getting handler when plugin doesn't have method."""
        registry = PluginRegistry()
        plugin = SimpleBookmarkPlugin()

        handler = registry._get_hook_handler(plugin, PluginHook.ON_START)
        assert handler is None


# ============================================================================
# Plugin Information Tests
# ============================================================================


class TestPluginInformation:
    """Tests for plugin information methods."""

    def test_get_plugin_info_loaded(self):
        """Test getting info for loaded plugin."""
        registry = PluginRegistry()
        plugin = MockValidatorPlugin()
        registry.register_instance(plugin)

        info = registry.get_plugin_info("test-validator")

        assert info is not None
        assert info["name"] == "test-validator"
        assert info["version"] == "1.0.0"
        assert info["description"] == "Test validator plugin"
        assert info["author"] == "Test Author"
        assert info["loaded"] is True
        assert info["enabled"] is True
        assert "pre_validation" in info["hooks"]

    def test_get_plugin_info_not_loaded(self):
        """Test getting info for discovered but not loaded plugin."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)  # Discovered but not loaded

        info = registry.get_plugin_info("simple-plugin")

        # Should return info from loader
        assert info is not None
        assert info["name"] == "simple-plugin"

    def test_get_plugin_info_not_found(self):
        """Test getting info for non-existent plugin."""
        registry = PluginRegistry()
        info = registry.get_plugin_info("nonexistent")
        assert info is None

    def test_get_all_info(self):
        """Test getting info for all loaded plugins."""
        registry = PluginRegistry()
        registry.register_instance(SimpleBookmarkPlugin())
        registry.register_instance(MockValidatorPlugin())

        all_info = registry.get_all_info()

        assert "simple-plugin" in all_info
        assert "test-validator" in all_info
        assert all_info["simple-plugin"]["version"] == "1.0.0"

    def test_get_capabilities(self):
        """Test getting plugin capabilities."""
        registry = PluginRegistry()
        registry.register_instance(MockValidatorPlugin())
        registry.register_instance(MockAIProcessorPlugin())
        registry.register_instance(SimpleBookmarkPlugin())

        capabilities = registry.get_capabilities()

        assert "validation" in capabilities
        assert "test-validator" in capabilities["validation"]
        assert "ai_processing" in capabilities
        assert "test-ai-processor" in capabilities["ai_processing"]
        assert "simple-capability" in capabilities

    def test_get_capabilities_empty(self):
        """Test getting capabilities with no plugins."""
        registry = PluginRegistry()
        capabilities = registry.get_capabilities()
        assert capabilities == {}

    def test_check_dependencies_satisfied(self):
        """Test checking satisfied dependencies."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)
        registry.register_instance(SimpleBookmarkPlugin())  # Load it

        # Register plugin without dependencies
        registry._loader._discovered_plugins["simple-plugin"] = SimpleBookmarkPlugin

        missing = registry.check_dependencies("simple-plugin")
        assert missing == []

    def test_check_dependencies_missing(self):
        """Test checking missing dependencies."""
        registry = PluginRegistry()
        registry.register(PluginWithDependencies)

        missing = registry.check_dependencies("plugin-with-deps")
        assert "simple-plugin" in missing
        assert "missing-plugin" in missing

    def test_check_dependencies_plugin_not_found(self):
        """Test checking dependencies for non-existent plugin."""
        registry = PluginRegistry()
        missing = registry.check_dependencies("nonexistent")
        assert "Plugin nonexistent not found" in missing


# ============================================================================
# Unindex Plugin Tests
# ============================================================================


class TestUnindexPlugin:
    """Tests for _unindex_plugin method."""

    def test_unindex_validator_plugin(self):
        """Test unindexing a validator plugin."""
        registry = PluginRegistry()
        plugin = MockValidatorPlugin()
        registry.register_instance(plugin)

        registry._unindex_plugin(plugin)

        assert plugin not in registry._validators
        assert plugin not in registry._hook_subscribers[PluginHook.PRE_VALIDATION]

    def test_unindex_ai_processor_plugin(self):
        """Test unindexing an AI processor plugin."""
        registry = PluginRegistry()
        plugin = MockAIProcessorPlugin()
        registry.register_instance(plugin)

        registry._unindex_plugin(plugin)

        assert plugin not in registry._ai_processors

    def test_unindex_output_plugin(self):
        """Test unindexing an output plugin."""
        registry = PluginRegistry()
        plugin = MockOutputPlugin()
        registry.register_instance(plugin)

        registry._unindex_plugin(plugin)

        assert plugin not in registry._output_plugins

    def test_unindex_tag_generator_plugin(self):
        """Test unindexing a tag generator plugin."""
        registry = PluginRegistry()
        plugin = MockTagGeneratorPlugin()
        registry.register_instance(plugin)

        registry._unindex_plugin(plugin)

        assert plugin not in registry._tag_generators

    def test_unindex_content_enhancer_plugin(self):
        """Test unindexing a content enhancer plugin."""
        registry = PluginRegistry()
        plugin = MockContentEnhancerPlugin()
        registry.register_instance(plugin)

        registry._unindex_plugin(plugin)

        assert plugin not in registry._content_enhancers

    def test_unindex_plugin_not_in_lists(self):
        """Test unindexing plugin that isn't in lists (no error)."""
        registry = PluginRegistry()
        plugin = SimpleBookmarkPlugin()
        # Don't register, just try to unindex
        registry._unindex_plugin(plugin)  # Should not raise


# ============================================================================
# Global Registry Tests
# ============================================================================


class TestGlobalRegistryFunctions:
    """Tests for global registry functions."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_registry()

    def teardown_method(self):
        """Reset registry after each test."""
        reset_registry()

    def test_get_registry_creates_instance(self):
        """Test get_registry creates new instance."""
        registry = get_registry()
        assert registry is not None
        assert isinstance(registry, PluginRegistry)

    def test_get_registry_singleton(self):
        """Test get_registry returns same instance."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_registry_clears_instance(self):
        """Test reset_registry clears the global instance."""
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        assert r1 is not r2

    def test_reset_registry_unloads_plugins(self):
        """Test reset_registry unloads all plugins."""
        registry = get_registry()
        registry.register_instance(SimpleBookmarkPlugin())
        assert registry.has_plugin("simple-plugin")

        reset_registry()

        new_registry = get_registry()
        assert not new_registry.has_plugin("simple-plugin")

    def test_reset_registry_when_none(self):
        """Test reset_registry when no registry exists."""
        reset_registry()  # Clear any existing
        reset_registry()  # Should not raise


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_register_same_plugin_multiple_times(self):
        """Test registering same plugin class multiple times."""
        registry = PluginRegistry()
        registry.register(SimpleBookmarkPlugin)
        registry.register(SimpleBookmarkPlugin)

        available = registry.list_available()
        assert available.count("simple-plugin") == 1

    def test_index_plugin_hook_already_subscribed(self):
        """Test indexing plugin when already subscribed to hook."""
        registry = PluginRegistry()
        plugin = PluginWithHooks()

        # Manually add to subscribers first
        registry._hook_subscribers[PluginHook.ON_START].append(plugin)

        # Now index (should not duplicate)
        registry._index_plugin(plugin)

        subscribers = registry._hook_subscribers[PluginHook.ON_START]
        assert subscribers.count(plugin) == 1

    def test_dispatch_hook_with_kwargs(self):
        """Test dispatching hook with keyword arguments."""

        class KwargsPlugin(BookmarkPlugin):
            @property
            def name(self) -> str:
                return "kwargs-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def hooks(self) -> List[PluginHook]:
                return [PluginHook.ON_START]

            def on_start(self, key1=None, key2=None):
                return f"{key1}-{key2}"

        registry = PluginRegistry()
        registry.register_instance(KwargsPlugin())

        results = registry.dispatch_hook(PluginHook.ON_START, key1="val1", key2="val2")
        assert "val1-val2" in results

    def test_plugin_provides_multiple_capabilities(self):
        """Test plugin that provides multiple capabilities."""

        class MultiCapabilityPlugin(BookmarkPlugin):
            @property
            def name(self) -> str:
                return "multi-cap"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def provides(self) -> List[str]:
                return ["cap1", "cap2", "cap3"]

        registry = PluginRegistry()
        registry.register_instance(MultiCapabilityPlugin())

        capabilities = registry.get_capabilities()
        assert "cap1" in capabilities
        assert "cap2" in capabilities
        assert "cap3" in capabilities

    def test_empty_hooks_list(self):
        """Test plugin with empty hooks list."""
        registry = PluginRegistry()
        plugin = SimpleBookmarkPlugin()  # Has no hooks
        registry.register_instance(plugin)

        # Should not be in any hook subscribers
        for hook in PluginHook:
            assert plugin not in registry._hook_subscribers[hook]


# ============================================================================
# Loader Integration Tests
# ============================================================================


class TestLoaderIntegration:
    """Tests for registry integration with loader."""

    def test_loader_property(self):
        """Test accessing loader property."""
        custom_loader = PluginLoader()
        registry = PluginRegistry(loader=custom_loader)

        assert registry.loader is custom_loader

    def test_load_plugins_uses_loader(self):
        """Test load_plugins uses the loader."""
        registry = PluginRegistry()

        # Register through registry (which uses loader)
        registry.register(SimpleBookmarkPlugin)

        # Load through registry
        loaded = registry.load_plugins(["simple-plugin"])

        assert "simple-plugin" in loaded
        assert registry.has_plugin("simple-plugin")

    def test_unload_calls_loader_unload(self):
        """Test unload_plugin calls loader's unload."""
        mock_loader = MagicMock(spec=PluginLoader)
        registry = PluginRegistry(loader=mock_loader)

        plugin = SimpleBookmarkPlugin()
        registry.register_instance(plugin)

        registry.unload_plugin("simple-plugin")

        mock_loader.unload_plugin.assert_called_once_with("simple-plugin")


# Export test markers
__all__ = [
    "TestRegistryInitialization",
    "TestPluginRegistration",
    "TestPluginIndexing",
    "TestPluginLoading",
    "TestPluginUnloading",
    "TestPluginRetrieval",
    "TestHookSystem",
    "TestHookHandlerMapping",
    "TestPluginInformation",
    "TestUnindexPlugin",
    "TestGlobalRegistryFunctions",
    "TestEdgeCases",
    "TestLoaderIntegration",
]
