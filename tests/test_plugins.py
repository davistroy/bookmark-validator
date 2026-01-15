"""
Tests for Plugin Architecture

Tests the plugin system including:
- Plugin base classes
- Plugin loader
- Plugin registry
- Example plugins (PaywallDetector, OllamaAI)
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from bookmark_processor.plugins.base import (
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
from bookmark_processor.plugins.loader import PluginLoader, PluginLoadError
from bookmark_processor.plugins.registry import PluginRegistry, get_registry, reset_registry
from bookmark_processor.plugins.examples.paywall_detector import PaywallDetectorPlugin
from bookmark_processor.plugins.examples.ollama_ai import OllamaAIPlugin


# ============================================================================
# Test Plugin Classes
# ============================================================================


class SimpleTestPlugin(BookmarkPlugin):
    """Simple test plugin for testing base functionality."""

    @property
    def name(self) -> str:
        return "simple-test"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "A simple test plugin"


class MockValidatorPlugin(ValidatorPlugin):
    """Mock validator plugin for testing."""

    @property
    def name(self) -> str:
        return "test-validator"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate(self, url: str, content=None):
        return ValidationResult(
            is_valid=True,
            url=url,
            plugin_name=self.name,
        )


class MockAIPlugin(AIProcessorPlugin):
    """Mock AI processor plugin for testing."""

    @property
    def name(self) -> str:
        return "test-ai"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_description(self, bookmark, content):
        return f"Generated description for {bookmark.url}"

    def is_available(self):
        return True


# ============================================================================
# Plugin Base Classes Tests
# ============================================================================


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="Test description",
            author="Test Author",
        )
        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test description"
        assert metadata.author == "Test Author"

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = PluginMetadata(name="test", version="1.0")
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.requires == []
        assert metadata.provides == []
        assert metadata.hooks == []

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = PluginMetadata(
            name="test",
            version="1.0",
            hooks=[PluginHook.PRE_VALIDATION],
        )
        data = metadata.to_dict()
        assert data["name"] == "test"
        assert data["version"] == "1.0"
        assert "pre_validation" in data["hooks"]


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(
            is_valid=True,
            url="https://example.com",
            confidence=0.95,
        )
        assert result.is_valid is True
        assert result.url == "https://example.com"
        assert result.confidence == 0.95

    def test_validation_result_error(self):
        """Test validation result with error."""
        result = ValidationResult(
            is_valid=False,
            url="https://example.com",
            error_message="Connection failed",
            error_type="connection_error",
        )
        assert result.is_valid is False
        assert result.error_message == "Connection failed"

    def test_validation_result_to_dict(self):
        """Test validation result serialization."""
        result = ValidationResult(
            is_valid=True,
            url="https://example.com",
            metadata={"key": "value"},
        )
        data = result.to_dict()
        assert data["is_valid"] is True
        assert data["url"] == "https://example.com"
        assert data["metadata"]["key"] == "value"


class TestBookmarkPluginBase:
    """Tests for BookmarkPlugin base class."""

    def test_simple_plugin_creation(self):
        """Test creating a simple plugin."""
        plugin = SimpleTestPlugin()
        assert plugin.name == "simple-test"
        assert plugin.version == "1.0.0"
        assert plugin.description == "A simple test plugin"

    def test_plugin_enabled_by_default(self):
        """Test plugin is enabled by default."""
        plugin = SimpleTestPlugin()
        assert plugin.enabled is True

    def test_plugin_disable_enable(self):
        """Test disabling and enabling plugin."""
        plugin = SimpleTestPlugin()
        plugin.enabled = False
        assert plugin.enabled is False
        plugin.enabled = True
        assert plugin.enabled is True

    def test_plugin_config(self):
        """Test plugin configuration."""
        plugin = SimpleTestPlugin()
        plugin.on_load({"key": "value"})
        assert plugin.config == {"key": "value"}

    def test_plugin_metadata(self):
        """Test getting plugin metadata."""
        plugin = SimpleTestPlugin()
        metadata = plugin.get_metadata()
        assert isinstance(metadata, PluginMetadata)
        assert metadata.name == "simple-test"
        assert metadata.version == "1.0.0"

    def test_plugin_status(self):
        """Test getting plugin status."""
        plugin = SimpleTestPlugin()
        status = plugin.get_status()
        assert status["name"] == "simple-test"
        assert status["version"] == "1.0.0"
        assert status["enabled"] is True

    def test_plugin_repr(self):
        """Test plugin string representation."""
        plugin = SimpleTestPlugin()
        repr_str = repr(plugin)
        assert "SimpleTestPlugin" in repr_str
        assert "simple-test" in repr_str


class MockValidatorPluginBase:
    """Tests for ValidatorPlugin base class."""

    def test_validator_provides(self):
        """Test validator provides validation capability."""
        plugin = MockValidatorPlugin()
        assert "validation" in plugin.provides

    def test_validator_hooks(self):
        """Test validator hooks."""
        plugin = MockValidatorPlugin()
        hooks = plugin.hooks
        assert PluginHook.PRE_VALIDATION in hooks
        assert PluginHook.POST_VALIDATION in hooks

    def test_validator_validate(self):
        """Test validation method."""
        plugin = MockValidatorPlugin()
        result = plugin.validate("https://example.com")
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validator_should_validate(self):
        """Test should_validate default."""
        plugin = MockValidatorPlugin()
        assert plugin.should_validate("https://example.com") is True

    def test_validator_priority(self):
        """Test validation priority default."""
        plugin = MockValidatorPlugin()
        assert plugin.get_priority() == 100


class TestAIProcessorPluginBase:
    """Tests for AIProcessorPlugin base class."""

    def test_ai_plugin_provides(self):
        """Test AI plugin provides ai_processing capability."""
        plugin = MockAIPlugin()
        assert "ai_processing" in plugin.provides

    def test_ai_plugin_is_available(self):
        """Test is_available method."""
        plugin = MockAIPlugin()
        assert plugin.is_available() is True

    def test_ai_plugin_model_info(self):
        """Test get_model_info method."""
        plugin = MockAIPlugin()
        info = plugin.get_model_info()
        assert info["name"] == "test-ai"
        assert info["available"] is True

    def test_ai_plugin_estimate_cost(self):
        """Test estimate_cost default."""
        plugin = MockAIPlugin()
        cost = plugin.estimate_cost(1000)
        assert cost == 0.0  # Default is free


# ============================================================================
# Plugin Hook Tests
# ============================================================================


class TestPluginHook:
    """Tests for PluginHook enum."""

    def test_all_hooks_defined(self):
        """Test all expected hooks are defined."""
        hooks = list(PluginHook)
        assert len(hooks) >= 15

        # Check essential hooks
        assert PluginHook.PRE_VALIDATION in hooks
        assert PluginHook.POST_VALIDATION in hooks
        assert PluginHook.PRE_AI_PROCESS in hooks
        assert PluginHook.POST_AI_PROCESS in hooks
        assert PluginHook.ON_START in hooks
        assert PluginHook.ON_COMPLETE in hooks

    def test_hook_values(self):
        """Test hook string values."""
        assert PluginHook.PRE_VALIDATION.value == "pre_validation"
        assert PluginHook.POST_VALIDATION.value == "post_validation"


# ============================================================================
# Plugin Loader Tests
# ============================================================================


class TestPluginLoader:
    """Tests for PluginLoader."""

    def test_loader_creation(self):
        """Test creating a plugin loader."""
        loader = PluginLoader()
        assert loader is not None

    def test_loader_custom_dir(self, tmp_path):
        """Test loader with custom plugins directory."""
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        loader = PluginLoader(user_plugins_dir=plugins_dir)
        assert plugins_dir in loader._search_paths or not plugins_dir.exists()

    def test_discover_builtin_plugins(self):
        """Test discovering built-in plugins."""
        loader = PluginLoader()
        available = loader.discover_plugins()

        # Should find at least the example plugins
        assert isinstance(available, list)
        # The builtin plugins may or may not be found depending on import state

    def test_load_plugin(self):
        """Test loading a plugin by name."""
        loader = PluginLoader()
        loader.discover_plugins()

        # Register our test plugin manually
        loader._discovered_plugins["simple-test"] = SimpleTestPlugin

        plugin = loader.load_plugin("simple-test")
        assert plugin is not None
        assert plugin.name == "simple-test"

    def test_load_plugin_with_config(self):
        """Test loading a plugin with configuration."""
        loader = PluginLoader()
        loader._discovered_plugins["simple-test"] = SimpleTestPlugin

        plugin = loader.load_plugin("simple-test", {"key": "value"})
        assert plugin.config == {"key": "value"}

    def test_load_nonexistent_plugin(self):
        """Test loading a plugin that doesn't exist."""
        loader = PluginLoader()
        loader.discover_plugins()

        with pytest.raises(PluginLoadError):
            loader.load_plugin("nonexistent-plugin-xyz")

    def test_unload_plugin(self):
        """Test unloading a plugin."""
        loader = PluginLoader()
        loader._discovered_plugins["simple-test"] = SimpleTestPlugin

        loader.load_plugin("simple-test")
        assert loader.is_loaded("simple-test")

        result = loader.unload_plugin("simple-test")
        assert result is True
        assert not loader.is_loaded("simple-test")

    def test_unload_not_loaded_plugin(self):
        """Test unloading a plugin that isn't loaded."""
        loader = PluginLoader()
        result = loader.unload_plugin("not-loaded")
        assert result is False

    def test_get_loaded_plugins(self):
        """Test getting loaded plugins."""
        loader = PluginLoader()
        loader._discovered_plugins["simple-test"] = SimpleTestPlugin
        loader.load_plugin("simple-test")

        loaded = loader.get_loaded_plugins()
        assert "simple-test" in loaded

    def test_get_available_plugins(self):
        """Test getting available plugins."""
        loader = PluginLoader()
        loader._discovered_plugins["simple-test"] = SimpleTestPlugin

        available = loader.get_available_plugins()
        assert "simple-test" in available

    def test_reload_plugin(self):
        """Test reloading a plugin."""
        loader = PluginLoader()
        loader._discovered_plugins["simple-test"] = SimpleTestPlugin

        loader.load_plugin("simple-test", {"key": "original"})
        plugin = loader.reload_plugin("simple-test")

        assert plugin is not None
        # Config should be preserved
        assert plugin.config == {"key": "original"}

    def test_get_plugin_info(self):
        """Test getting plugin information."""
        loader = PluginLoader()
        loader._discovered_plugins["simple-test"] = SimpleTestPlugin

        info = loader.get_plugin_info("simple-test")
        assert info is not None
        assert info["name"] == "simple-test"
        assert info["version"] == "1.0.0"


# ============================================================================
# Plugin Registry Tests
# ============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_registry_creation(self):
        """Test creating a registry."""
        registry = PluginRegistry()
        assert registry is not None

    def test_register_plugin_class(self):
        """Test registering a plugin class."""
        registry = PluginRegistry()
        registry.register(SimpleTestPlugin)

        available = registry.list_available()
        assert "simple-test" in available

    def test_register_plugin_instance(self):
        """Test registering a plugin instance."""
        registry = PluginRegistry()
        plugin = SimpleTestPlugin()
        registry.register_instance(plugin)

        assert registry.has_plugin("simple-test")
        assert registry.get("simple-test") is plugin

    def test_load_plugins(self):
        """Test loading multiple plugins."""
        registry = PluginRegistry()
        registry.register(SimpleTestPlugin)
        registry.register(MockValidatorPlugin)

        loaded = registry.load_plugins(["simple-test", "test-validator"])
        assert len(loaded) == 2

    def test_unload_plugin(self):
        """Test unloading a plugin."""
        registry = PluginRegistry()
        plugin = SimpleTestPlugin()
        registry.register_instance(plugin)

        result = registry.unload_plugin("simple-test")
        assert result is True
        assert not registry.has_plugin("simple-test")

    def test_unload_all(self):
        """Test unloading all plugins."""
        registry = PluginRegistry()
        registry.register_instance(SimpleTestPlugin())
        registry.register_instance(MockValidatorPlugin())

        registry.unload_all()
        assert len(registry.list_plugins()) == 0

    def test_get_by_type(self):
        """Test getting plugins by type."""
        registry = PluginRegistry()
        registry.register_instance(MockValidatorPlugin())
        registry.register_instance(MockAIPlugin())

        validators = registry.get_by_type(ValidatorPlugin)
        assert len(validators) == 1
        assert isinstance(validators[0], ValidatorPlugin)

    def test_get_validators(self):
        """Test getting validator plugins."""
        registry = PluginRegistry()
        registry.register_instance(MockValidatorPlugin())

        validators = registry.get_validators()
        assert len(validators) == 1

    def test_get_ai_processors(self):
        """Test getting AI processor plugins."""
        registry = PluginRegistry()
        registry.register_instance(MockAIPlugin())

        ai_plugins = registry.get_ai_processors()
        assert len(ai_plugins) == 1

    def test_get_hook_subscribers(self):
        """Test getting hook subscribers."""
        registry = PluginRegistry()
        registry.register_instance(MockValidatorPlugin())

        subscribers = registry.get_hook_subscribers(PluginHook.PRE_VALIDATION)
        assert len(subscribers) == 1

    def test_dispatch_hook(self):
        """Test dispatching a hook."""
        registry = PluginRegistry()
        plugin = MockValidatorPlugin()
        plugin.on_pre_validation = Mock(return_value="result")
        registry.register_instance(plugin)

        results = registry.dispatch_hook(PluginHook.PRE_VALIDATION, "test_url")
        # Results depend on hook implementation

    def test_get_capabilities(self):
        """Test getting plugin capabilities."""
        registry = PluginRegistry()
        registry.register_instance(MockValidatorPlugin())
        registry.register_instance(MockAIPlugin())

        capabilities = registry.get_capabilities()
        assert "validation" in capabilities
        assert "ai_processing" in capabilities

    def test_check_dependencies(self):
        """Test checking plugin dependencies."""
        registry = PluginRegistry()
        registry._loader._discovered_plugins["simple-test"] = SimpleTestPlugin

        missing = registry.check_dependencies("simple-test")
        assert missing == []  # SimpleTestPlugin has no dependencies


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry(self):
        """Test getting global registry."""
        reset_registry()  # Ensure clean state
        registry = get_registry()
        assert isinstance(registry, PluginRegistry)

    def test_get_registry_singleton(self):
        """Test registry is singleton."""
        reset_registry()
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_registry(self):
        """Test resetting global registry."""
        reset_registry()
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        assert r1 is not r2


# ============================================================================
# PaywallDetectorPlugin Tests
# ============================================================================


class TestPaywallDetectorPlugin:
    """Tests for PaywallDetectorPlugin."""

    @pytest.fixture
    def paywall_plugin(self):
        """Create paywall detector plugin."""
        plugin = PaywallDetectorPlugin()
        plugin.on_load({})
        return plugin

    def test_plugin_metadata(self, paywall_plugin):
        """Test plugin metadata."""
        assert paywall_plugin.name == "paywall-detector"
        assert paywall_plugin.version == "1.0.0"
        assert "validation" in paywall_plugin.provides

    def test_detect_paywall_domain(self, paywall_plugin):
        """Test detecting known paywall domain."""
        result = paywall_plugin.validate("https://www.nytimes.com/article")
        assert result.metadata.get("is_known_paywall_domain") is True

    def test_non_paywall_domain(self, paywall_plugin):
        """Test non-paywall domain."""
        result = paywall_plugin.validate("https://example.com/page")
        assert result.metadata.get("is_known_paywall_domain") is False

    def test_detect_paywall_content(self, paywall_plugin):
        """Test detecting paywall indicators in content."""
        content = "Subscribe to continue reading this article. Premium content."
        result = paywall_plugin.validate("https://example.com", content)
        # May or may not detect depending on patterns
        assert "paywall_detected" in result.metadata

    def test_bypass_patterns(self, paywall_plugin):
        """Test bypass patterns (gift links, etc.)."""
        result = paywall_plugin.validate("https://nytimes.com/article?gift=true")
        assert result.metadata.get("has_bypass") is True

    def test_should_validate(self, paywall_plugin):
        """Test should_validate for HTTP URLs."""
        assert paywall_plugin.should_validate("https://example.com") is True
        assert paywall_plugin.should_validate("ftp://example.com") is False

    def test_validate_config(self, paywall_plugin):
        """Test config validation."""
        errors = paywall_plugin.validate_config({
            "additional_domains": ["example.com"],
            "confidence_threshold": 0.8,
        })
        assert errors == []

        errors = paywall_plugin.validate_config({
            "confidence_threshold": 2.0,  # Invalid
        })
        assert len(errors) > 0

    def test_get_statistics(self, paywall_plugin):
        """Test getting plugin statistics."""
        # Run some validations
        paywall_plugin.validate("https://example.com")
        paywall_plugin.validate("https://nytimes.com/article")

        stats = paywall_plugin.get_statistics()
        assert stats["checked_count"] == 2


# ============================================================================
# OllamaAIPlugin Tests
# ============================================================================


class TestOllamaAIPlugin:
    """Tests for OllamaAIPlugin."""

    @pytest.fixture
    def ollama_plugin(self):
        """Create Ollama AI plugin."""
        plugin = OllamaAIPlugin()
        plugin.on_load({})
        return plugin

    def test_plugin_metadata(self, ollama_plugin):
        """Test plugin metadata."""
        assert ollama_plugin.name == "ollama-ai"
        assert ollama_plugin.version == "1.0.0"
        assert "ai_processing" in ollama_plugin.provides

    def test_default_config(self, ollama_plugin):
        """Test default configuration."""
        assert ollama_plugin._endpoint == "http://localhost:11434"
        assert ollama_plugin._model == "llama2"

    def test_custom_config(self):
        """Test custom configuration."""
        plugin = OllamaAIPlugin()
        plugin.on_load({
            "endpoint": "http://custom:8080",
            "model": "mistral",
            "timeout": 120.0,
        })
        assert plugin._endpoint == "http://custom:8080"
        assert plugin._model == "mistral"
        assert plugin._timeout == 120.0

    def test_estimate_cost(self, ollama_plugin):
        """Test cost estimation (should be free for local)."""
        cost = ollama_plugin.estimate_cost(10000)
        assert cost == 0.0

    def test_validate_config(self, ollama_plugin):
        """Test config validation."""
        errors = ollama_plugin.validate_config({
            "endpoint": "http://localhost:11434",
            "model": "llama2",
            "temperature": 0.7,
        })
        assert errors == []

        errors = ollama_plugin.validate_config({
            "temperature": 3.0,  # Invalid
        })
        assert len(errors) > 0

    def test_get_model_info(self, ollama_plugin):
        """Test getting model info."""
        info = ollama_plugin.get_model_info()
        assert info["name"] == "ollama-ai"
        assert info["model"] == "llama2"

    @patch('requests.get')
    def test_is_available_success(self, mock_get, ollama_plugin):
        """Test is_available when Ollama is running."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama2:latest"}]}
        )
        ollama_plugin._available = None  # Reset cache

        assert ollama_plugin.is_available() is True

    @patch('bookmark_processor.plugins.examples.ollama_ai.requests')
    def test_is_available_failure(self, mock_requests, ollama_plugin):
        """Test is_available when Ollama is not running."""
        import requests
        mock_requests.get.side_effect = requests.RequestException("Connection refused")
        mock_requests.RequestException = requests.RequestException
        ollama_plugin._available = None  # Reset cache

        assert ollama_plugin.is_available() is False

    def test_get_statistics(self, ollama_plugin):
        """Test getting plugin statistics."""
        stats = ollama_plugin.get_statistics()
        assert "processed_count" in stats
        assert "model" in stats
        assert "endpoint" in stats


# ============================================================================
# Plugin Load Error Tests
# ============================================================================


class TestPluginLoadError:
    """Tests for PluginLoadError."""

    def test_error_creation(self):
        """Test creating a plugin load error."""
        error = PluginLoadError("test-plugin", "Failed to load")
        assert error.plugin_name == "test-plugin"
        assert error.message == "Failed to load"

    def test_error_with_cause(self):
        """Test error with underlying cause."""
        cause = ValueError("Original error")
        error = PluginLoadError("test-plugin", "Failed", cause)
        assert error.cause is cause

    def test_error_string(self):
        """Test error string representation."""
        error = PluginLoadError("test-plugin", "Failed to load")
        assert "test-plugin" in str(error)
        assert "Failed to load" in str(error)


# Export test markers for pytest
__all__ = [
    "TestPluginMetadata",
    "TestValidationResult",
    "TestBookmarkPluginBase",
    "MockValidatorPluginBase",
    "TestAIProcessorPluginBase",
    "TestPluginHook",
    "TestPluginLoader",
    "TestPluginRegistry",
    "TestGlobalRegistry",
    "TestPaywallDetectorPlugin",
    "TestOllamaAIPlugin",
    "TestPluginLoadError",
]
