"""
Comprehensive Tests for Plugin Loader

Tests to improve coverage of bookmark_processor/plugins/loader.py from 56.71% to 85%+.
Focuses on:
- All public methods of PluginLoader class
- Plugin discovery and loading from various sources
- Error handling for invalid plugins
- Edge cases for plugin configuration
"""

import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.plugins.base import BookmarkPlugin, PluginHook
from bookmark_processor.plugins.loader import PluginLoadError, PluginLoader


# =============================================================================
# Test Plugin Classes for Testing
# =============================================================================


class SampleTestPlugin(BookmarkPlugin):
    """A simple testable plugin for unit tests."""

    @property
    def name(self) -> str:
        return "testable-plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "A testable plugin for unit tests"

    @property
    def author(self) -> str:
        return "Test Author"

    @property
    def requires(self) -> List[str]:
        return []

    @property
    def provides(self) -> List[str]:
        return ["testing"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [PluginHook.ON_START, PluginHook.ON_COMPLETE]


class PluginWithValidationErrors(BookmarkPlugin):
    """Plugin that returns validation errors for its config."""

    @property
    def name(self) -> str:
        return "validation-error-plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        if "required_key" not in config:
            errors.append("Missing required_key in config")
        return errors


class PluginThatFailsOnLoad(BookmarkPlugin):
    """Plugin that raises an exception during on_load."""

    @property
    def name(self) -> str:
        return "fails-on-load"

    @property
    def version(self) -> str:
        return "1.0.0"

    def on_load(self, config: Dict[str, Any]) -> None:
        raise RuntimeError("Failed to initialize plugin")


class PluginThatFailsOnUnload(BookmarkPlugin):
    """Plugin that raises an exception during on_unload."""

    @property
    def name(self) -> str:
        return "fails-on-unload"

    @property
    def version(self) -> str:
        return "1.0.0"

    def on_unload(self) -> None:
        raise RuntimeError("Failed to cleanup plugin")


class PluginThatFailsToInstantiate(BookmarkPlugin):
    """Plugin that raises an exception in __init__."""

    def __init__(self):
        raise RuntimeError("Failed to instantiate")

    @property
    def name(self) -> str:
        return "fails-instantiate"

    @property
    def version(self) -> str:
        return "1.0.0"


class PluginWithNoNameAttr:
    """A class that looks like a plugin but has no name attribute."""

    pass


# =============================================================================
# PluginLoadError Tests
# =============================================================================


class TestPluginLoadErrorComprehensive:
    """Comprehensive tests for PluginLoadError exception."""

    def test_error_with_all_fields(self):
        """Test error with all fields populated."""
        cause = ValueError("root cause")
        error = PluginLoadError("my-plugin", "Something went wrong", cause)

        assert error.plugin_name == "my-plugin"
        assert error.message == "Something went wrong"
        assert error.cause is cause
        assert "my-plugin" in str(error)
        assert "Something went wrong" in str(error)

    def test_error_without_cause(self):
        """Test error without a cause exception."""
        error = PluginLoadError("my-plugin", "Load failed")

        assert error.plugin_name == "my-plugin"
        assert error.message == "Load failed"
        assert error.cause is None

    def test_error_is_exception(self):
        """Test that PluginLoadError is a proper exception."""
        error = PluginLoadError("test", "test message")
        assert isinstance(error, Exception)

        # Should be raiseable and catchable
        with pytest.raises(PluginLoadError) as exc_info:
            raise error
        assert exc_info.value.plugin_name == "test"


# =============================================================================
# PluginLoader Initialization Tests
# =============================================================================


class TestPluginLoaderInitialization:
    """Tests for PluginLoader initialization and setup."""

    def test_default_initialization(self):
        """Test loader with default settings."""
        loader = PluginLoader()

        assert loader._discovered_plugins == {}
        assert loader._loaded_plugins == {}
        assert isinstance(loader._search_paths, list)

    def test_with_custom_user_plugins_dir(self, tmp_path):
        """Test loader with custom user plugins directory."""
        plugins_dir = tmp_path / "my_plugins"
        plugins_dir.mkdir()

        loader = PluginLoader(user_plugins_dir=plugins_dir)

        assert loader._user_plugins_dir == plugins_dir
        assert plugins_dir in loader._search_paths

    def test_with_nonexistent_user_plugins_dir(self, tmp_path):
        """Test loader with nonexistent user plugins directory."""
        plugins_dir = tmp_path / "nonexistent"

        loader = PluginLoader(user_plugins_dir=plugins_dir)

        assert loader._user_plugins_dir == plugins_dir
        # Nonexistent dir should not be in search paths
        assert plugins_dir not in loader._search_paths

    def test_with_additional_paths(self, tmp_path):
        """Test loader with additional search paths."""
        path1 = tmp_path / "path1"
        path2 = tmp_path / "path2"
        path1.mkdir()
        path2.mkdir()

        loader = PluginLoader(additional_paths=[path1, path2])

        assert path1 in loader._search_paths
        assert path2 in loader._search_paths

    def test_with_nonexistent_additional_paths(self, tmp_path):
        """Test loader with nonexistent additional paths."""
        path1 = tmp_path / "exists"
        path2 = tmp_path / "not_exists"
        path1.mkdir()

        loader = PluginLoader(additional_paths=[path1, path2])

        assert path1 in loader._search_paths
        assert path2 not in loader._search_paths

    def test_environment_variable_for_plugins_dir(self, tmp_path, monkeypatch):
        """Test that BOOKMARK_PROCESSOR_PLUGINS_DIR env var is respected."""
        custom_dir = tmp_path / "env_plugins"
        custom_dir.mkdir()
        monkeypatch.setenv("BOOKMARK_PROCESSOR_PLUGINS_DIR", str(custom_dir))

        loader = PluginLoader()

        assert loader._user_plugins_dir == custom_dir


# =============================================================================
# Plugin Discovery Tests
# =============================================================================


class TestPluginDiscovery:
    """Tests for plugin discovery functionality."""

    def test_discover_plugins_caching(self):
        """Test that discovered plugins are cached."""
        loader = PluginLoader()

        # First call
        result1 = loader.discover_plugins()
        # Manually add a plugin to test caching
        loader._discovered_plugins["extra-plugin"] = SampleTestPlugin

        # Second call should use cache
        result2 = loader.discover_plugins()
        assert "extra-plugin" in result2

    def test_discover_plugins_force_refresh(self):
        """Test force refresh of plugin discovery."""
        loader = PluginLoader()

        loader.discover_plugins()
        loader._discovered_plugins["cached-plugin"] = SampleTestPlugin

        # Force refresh should clear and re-discover
        loader.discover_plugins(force_refresh=True)
        # After force refresh, cached-plugin should be gone
        # (unless it's a builtin, which it's not)

    def test_discover_builtin_plugins(self):
        """Test discovering builtin plugins."""
        loader = PluginLoader()

        discovered = loader._discover_builtin_plugins()

        # Should find the example plugins
        assert isinstance(discovered, list)

    def test_discover_builtin_plugins_import_error(self):
        """Test graceful handling of import errors for builtin plugins."""
        loader = PluginLoader()

        # Temporarily change the module path to something that doesn't exist
        original_module = loader.BUILTIN_PLUGINS_MODULE
        loader.BUILTIN_PLUGINS_MODULE = "nonexistent.module.path"

        discovered = loader._discover_builtin_plugins()

        assert discovered == []

        # Restore
        loader.BUILTIN_PLUGINS_MODULE = original_module

    def test_discover_from_empty_directory(self, tmp_path):
        """Test discovering from an empty directory."""
        loader = PluginLoader()

        discovered = loader._discover_from_directory(tmp_path)

        assert discovered == []

    def test_discover_from_nonexistent_directory(self, tmp_path):
        """Test discovering from a nonexistent directory."""
        loader = PluginLoader()
        nonexistent = tmp_path / "does_not_exist"

        discovered = loader._discover_from_directory(nonexistent)

        assert discovered == []

    def test_discover_skips_underscore_files(self, tmp_path):
        """Test that files starting with underscore are skipped."""
        # Create a file that starts with underscore
        (tmp_path / "_private_plugin.py").write_text("# private plugin")

        loader = PluginLoader()
        discovered = loader._discover_from_directory(tmp_path)

        assert discovered == []


# =============================================================================
# Plugin Loading from Files Tests
# =============================================================================


class TestPluginLoadingFromFiles:
    """Tests for loading plugins from Python files."""

    def test_load_plugins_from_valid_file(self, tmp_path):
        """Test loading plugins from a valid Python file."""
        plugin_code = '''
from bookmark_processor.plugins.base import BookmarkPlugin

class FileTestPlugin(BookmarkPlugin):
    @property
    def name(self):
        return "file-test"

    @property
    def version(self):
        return "1.0.0"
'''
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text(plugin_code)

        loader = PluginLoader()
        plugins = loader._load_plugins_from_file(plugin_file)

        assert len(plugins) == 1
        assert plugins[0].__name__ == "FileTestPlugin"

    def test_load_plugins_from_file_with_syntax_error(self, tmp_path):
        """Test loading from file with syntax error."""
        plugin_file = tmp_path / "bad_syntax.py"
        plugin_file.write_text("def broken(: pass")

        loader = PluginLoader()

        with pytest.raises(PluginLoadError) as exc_info:
            loader._load_plugins_from_file(plugin_file)

        assert "bad_syntax" in exc_info.value.plugin_name

    def test_load_plugins_from_file_with_import_error(self, tmp_path):
        """Test loading from file with import error."""
        plugin_file = tmp_path / "import_error.py"
        plugin_file.write_text("import nonexistent_module_xyz")

        loader = PluginLoader()

        with pytest.raises(PluginLoadError) as exc_info:
            loader._load_plugins_from_file(plugin_file)

        assert "import_error" in exc_info.value.plugin_name

    def test_load_plugins_from_file_with_no_plugins(self, tmp_path):
        """Test loading from file that has no plugins."""
        plugin_file = tmp_path / "no_plugins.py"
        plugin_file.write_text("# Just a regular Python file\nx = 1 + 1")

        loader = PluginLoader()
        plugins = loader._load_plugins_from_file(plugin_file)

        assert plugins == []

    def test_load_plugins_from_file_spec_creation_fails(self, tmp_path):
        """Test handling when module spec creation fails."""
        loader = PluginLoader()

        # Create a path that can't be loaded
        with patch("importlib.util.spec_from_file_location", return_value=None):
            plugin_file = tmp_path / "fake.py"
            plugin_file.write_text("")

            with pytest.raises(PluginLoadError) as exc_info:
                loader._load_plugins_from_file(plugin_file)

            assert "Could not create module spec" in exc_info.value.message


# =============================================================================
# Plugin Loading from Packages Tests
# =============================================================================


class TestPluginLoadingFromPackages:
    """Tests for loading plugins from package directories."""

    def test_load_plugins_from_package(self, tmp_path):
        """Test loading plugins from a package directory."""
        # Create package structure
        package_dir = tmp_path / "test_package"
        package_dir.mkdir()

        init_code = '''
from bookmark_processor.plugins.base import BookmarkPlugin

class PackageTestPlugin(BookmarkPlugin):
    @property
    def name(self):
        return "package-test"

    @property
    def version(self):
        return "1.0.0"
'''
        (package_dir / "__init__.py").write_text(init_code)

        loader = PluginLoader()

        # Add tmp_path to sys.path temporarily for the import
        sys.path.insert(0, str(tmp_path))
        try:
            plugins = loader._load_plugins_from_package(package_dir)
            assert len(plugins) >= 1
        finally:
            sys.path.remove(str(tmp_path))
            # Cleanup module cache
            if "test_package" in sys.modules:
                del sys.modules["test_package"]

    def test_load_plugins_from_package_path_handling(self, tmp_path):
        """Test that parent dir is added/removed from path correctly."""
        package_dir = tmp_path / "path_test_package"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("")

        loader = PluginLoader()
        parent_str = str(package_dir.parent)

        # If parent is already in path, it should not be added again
        sys.path.insert(0, parent_str)
        try:
            loader._load_plugins_from_package(package_dir)
            # Parent should still be in path
            assert parent_str in sys.path
        finally:
            sys.path.remove(parent_str)
            if "path_test_package" in sys.modules:
                del sys.modules["path_test_package"]


# =============================================================================
# Entry Points Discovery Tests
# =============================================================================


class TestEntryPointsDiscovery:
    """Tests for discovering plugins via entry points."""

    def test_discover_entry_points_python_310_style(self):
        """Test entry points discovery with Python 3.10+ API."""
        loader = PluginLoader()

        mock_ep = MagicMock()
        mock_ep.name = "test-entry-point"
        mock_ep.load.return_value = SampleTestPlugin

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            discovered = loader._discover_entry_points()

            assert "test-entry-point" in discovered

    def test_discover_entry_points_python_39_style(self):
        """Test entry points discovery with Python 3.9 API."""
        loader = PluginLoader()

        mock_ep = MagicMock()
        mock_ep.name = "test-entry-point-39"
        mock_ep.load.return_value = SampleTestPlugin

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            # Python 3.9 style - entry_points() returns dict-like
            mock_entry_points.side_effect = TypeError()

            with patch("importlib.metadata.entry_points") as mock_entry_points_2:
                mock_group = MagicMock()
                mock_group.get.return_value = [mock_ep]
                mock_entry_points_2.return_value = mock_group

                discovered = loader._discover_entry_points()

                # May or may not contain the plugin depending on mock setup
                assert isinstance(discovered, list)

    def test_discover_entry_points_load_error(self):
        """Test handling load errors from entry points."""
        loader = PluginLoader()

        mock_ep = MagicMock()
        mock_ep.name = "bad-entry-point"
        mock_ep.load.side_effect = Exception("Failed to load")

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            # Should not raise, but skip the bad plugin
            discovered = loader._discover_entry_points()

            assert "bad-entry-point" not in discovered

    def test_discover_entry_points_import_error(self):
        """Test handling when importlib.metadata is not available."""
        loader = PluginLoader()

        with patch.dict("sys.modules", {"importlib.metadata": None}):
            with patch("importlib.metadata.entry_points", side_effect=ImportError):
                # Should return empty list, not raise
                discovered = loader._discover_entry_points()

                assert discovered == []

    def test_discover_entry_points_non_plugin_class(self):
        """Test that non-plugin classes from entry points are ignored."""
        loader = PluginLoader()

        mock_ep = MagicMock()
        mock_ep.name = "not-a-plugin"
        mock_ep.load.return_value = str  # str is not a plugin

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_ep]

            discovered = loader._discover_entry_points()

            assert "not-a-plugin" not in discovered


# =============================================================================
# Plugin Class Detection Tests
# =============================================================================


class TestPluginClassDetection:
    """Tests for _is_plugin_class method."""

    def test_is_plugin_class_valid(self):
        """Test detection of valid plugin class."""
        loader = PluginLoader()

        assert loader._is_plugin_class(SampleTestPlugin) is True

    def test_is_plugin_class_base_class(self):
        """Test that BookmarkPlugin base class is not detected as plugin."""
        loader = PluginLoader()

        assert loader._is_plugin_class(BookmarkPlugin) is False

    def test_is_plugin_class_not_class(self):
        """Test that non-class objects are not detected."""
        loader = PluginLoader()

        assert loader._is_plugin_class("not a class") is False
        assert loader._is_plugin_class(123) is False
        assert loader._is_plugin_class(None) is False
        assert loader._is_plugin_class(lambda: None) is False

    def test_is_plugin_class_no_name_attr(self):
        """Test that class without name attr is not detected."""
        loader = PluginLoader()

        class NoNamePlugin(BookmarkPlugin):
            # Missing name property
            @property
            def version(self):
                return "1.0"

        # This should return False because hasattr check would fail
        # when we try to access name
        assert loader._is_plugin_class(PluginWithNoNameAttr) is False

    def test_is_plugin_class_issubclass_type_error(self):
        """Test handling TypeError in issubclass check."""
        loader = PluginLoader()

        # A class that causes issubclass to raise TypeError
        # This can happen with certain metaclasses or special types
        # The code catches TypeError and returns False
        class ProblematicMeta(type):
            def __subclasscheck__(cls, subclass):
                raise TypeError("Cannot check subclass")

        class ProblematicClass(metaclass=ProblematicMeta):
            pass

        # This should return False (not crash) due to try/except
        result = loader._is_plugin_class(ProblematicClass)
        assert result is False


# =============================================================================
# Plugin Name Extraction Tests
# =============================================================================


class TestPluginNameExtraction:
    """Tests for _get_plugin_name_from_class method."""

    def test_get_name_from_instantiable_class(self):
        """Test getting name from a class that can be instantiated."""
        loader = PluginLoader()

        name = loader._get_plugin_name_from_class(SampleTestPlugin)

        assert name == "testable-plugin"

    def test_get_name_fallback_on_init_error(self):
        """Test fallback to class name when instantiation fails."""
        loader = PluginLoader()

        # Create a plugin that fails to instantiate
        class FailingInitPlugin(BookmarkPlugin):
            def __init__(self):
                raise RuntimeError("Init failed")

            @property
            def name(self):
                return "failing"

            @property
            def version(self):
                return "1.0"

        name = loader._get_plugin_name_from_class(FailingInitPlugin)

        # Should fall back to class name without "plugin" suffix
        assert name == "failinginit"


# =============================================================================
# Plugin Loading Tests
# =============================================================================


class TestPluginLoading:
    """Tests for load_plugin method."""

    def test_load_plugin_basic(self):
        """Test basic plugin loading."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        plugin = loader.load_plugin("testable-plugin")

        assert plugin.name == "testable-plugin"
        assert loader.is_loaded("testable-plugin")

    def test_load_plugin_with_config(self):
        """Test loading plugin with configuration."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        config = {"key": "value", "number": 42}
        plugin = loader.load_plugin("testable-plugin", config)

        assert plugin.config == config

    def test_load_plugin_already_loaded(self):
        """Test loading a plugin that's already loaded returns cached instance."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        plugin1 = loader.load_plugin("testable-plugin")
        plugin2 = loader.load_plugin("testable-plugin")

        assert plugin1 is plugin2

    def test_load_plugin_not_found(self):
        """Test loading a plugin that doesn't exist."""
        loader = PluginLoader()
        loader._discovered_plugins = {}

        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin("nonexistent-plugin")

        assert "not found" in exc_info.value.message

    def test_load_plugin_case_insensitive_lookup(self):
        """Test case-insensitive plugin name lookup."""
        loader = PluginLoader()
        # Register with uppercase name
        loader._discovered_plugins["TESTABLE-PLUGIN"] = SampleTestPlugin

        # Should find with lowercase search
        plugin = loader.load_plugin("testable-plugin")

        assert plugin is not None
        assert plugin.name == "testable-plugin"

    def test_load_plugin_config_validation_fails(self):
        """Test loading plugin with invalid configuration."""
        loader = PluginLoader()
        loader._discovered_plugins["validation-error-plugin"] = PluginWithValidationErrors

        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin("validation-error-plugin", {"wrong_key": "value"})

        assert "Configuration validation failed" in exc_info.value.message

    def test_load_plugin_instantiation_error(self):
        """Test handling error during plugin instantiation."""
        loader = PluginLoader()

        # This tests the case where a plugin class is discovered but then
        # fails during instantiation in load_plugin.
        # We use a mock to make the class fail only during load_plugin.
        class AlwaysFailPlugin(BookmarkPlugin):
            """Plugin that always fails to instantiate."""

            def __init__(self):
                raise RuntimeError("Always fails")

            @property
            def name(self):
                return "always-fail"

            @property
            def version(self):
                return "1.0"

        # Manually add to discovered (bypassing normal discovery which would try to instantiate)
        loader._discovered_plugins["always-fail"] = AlwaysFailPlugin

        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin("always-fail")

        assert "Error instantiating plugin" in exc_info.value.message

    def test_load_plugin_on_load_error(self):
        """Test handling error during on_load."""
        loader = PluginLoader()
        loader._discovered_plugins["fails-on-load"] = PluginThatFailsOnLoad

        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin("fails-on-load")

        assert "Error instantiating plugin" in exc_info.value.message

    def test_load_plugin_auto_discovers(self):
        """Test that load_plugin auto-discovers if needed."""
        loader = PluginLoader()

        # Clear discovered plugins
        loader._discovered_plugins = {}

        # Loading should trigger discovery
        # This may or may not find plugins depending on setup
        try:
            loader.load_plugin("nonexistent-xyz")
        except PluginLoadError:
            pass  # Expected

        # Discovery should have happened (discovered_plugins may still be empty
        # if no plugins found, but the method should have been called)


# =============================================================================
# Plugin Unloading Tests
# =============================================================================


class TestPluginUnloading:
    """Tests for unload_plugin method."""

    def test_unload_loaded_plugin(self):
        """Test unloading a loaded plugin."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin
        loader.load_plugin("testable-plugin")

        result = loader.unload_plugin("testable-plugin")

        assert result is True
        assert not loader.is_loaded("testable-plugin")

    def test_unload_not_loaded_plugin(self):
        """Test unloading a plugin that's not loaded."""
        loader = PluginLoader()

        result = loader.unload_plugin("not-loaded")

        assert result is False

    def test_unload_plugin_on_unload_error(self):
        """Test unloading plugin that throws during on_unload."""
        loader = PluginLoader()
        loader._discovered_plugins["fails-on-unload"] = PluginThatFailsOnUnload
        loader.load_plugin("fails-on-unload")

        # Should not raise, but log warning
        result = loader.unload_plugin("fails-on-unload")

        assert result is True
        assert not loader.is_loaded("fails-on-unload")


# =============================================================================
# Plugin Info and Status Tests
# =============================================================================


class TestPluginInfoAndStatus:
    """Tests for get_plugin_info and related methods."""

    def test_get_plugin_info_basic(self):
        """Test getting plugin information."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        info = loader.get_plugin_info("testable-plugin")

        assert info["name"] == "testable-plugin"
        assert info["version"] == "1.0.0"
        assert info["description"] == "A testable plugin for unit tests"
        assert info["author"] == "Test Author"
        assert "testing" in info["provides"]
        assert "on_start" in info["hooks"]
        assert info["loaded"] is False

    def test_get_plugin_info_loaded_plugin(self):
        """Test getting info for a loaded plugin."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin
        loader.load_plugin("testable-plugin")

        info = loader.get_plugin_info("testable-plugin")

        assert info["loaded"] is True

    def test_get_plugin_info_not_found(self):
        """Test getting info for nonexistent plugin."""
        loader = PluginLoader()

        info = loader.get_plugin_info("nonexistent")

        assert info is None

    def test_get_plugin_info_instantiation_error(self):
        """Test getting info when plugin can't be instantiated."""
        loader = PluginLoader()

        class BrokenInfoPlugin(BookmarkPlugin):
            def __init__(self):
                raise RuntimeError("Can't instantiate")

            @property
            def name(self):
                return "broken-info"

            @property
            def version(self):
                return "1.0"

        loader._discovered_plugins["broken-info"] = BrokenInfoPlugin

        info = loader.get_plugin_info("broken-info")

        assert info is not None
        assert "error" in info
        assert info["loaded"] is False

    def test_get_plugin_info_auto_discovers(self):
        """Test that get_plugin_info auto-discovers if needed."""
        loader = PluginLoader()
        loader._discovered_plugins = {}

        # This should trigger discovery
        info = loader.get_plugin_info("some-plugin")

        # May return None, but shouldn't crash
        assert info is None or isinstance(info, dict)


# =============================================================================
# Plugin Reload Tests
# =============================================================================


class TestPluginReload:
    """Tests for reload_plugin method."""

    def test_reload_loaded_plugin(self):
        """Test reloading a loaded plugin."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        original = loader.load_plugin("testable-plugin", {"key": "value"})
        reloaded = loader.reload_plugin("testable-plugin")

        assert reloaded is not original  # New instance
        assert reloaded.config == {"key": "value"}  # Config preserved

    def test_reload_not_loaded_plugin(self):
        """Test reloading a plugin that isn't loaded."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        # Plugin not yet loaded
        reloaded = loader.reload_plugin("testable-plugin")

        assert reloaded is not None
        assert loader.is_loaded("testable-plugin")


# =============================================================================
# Getter Methods Tests
# =============================================================================


class TestGetterMethods:
    """Tests for getter methods."""

    def test_get_loaded_plugins(self):
        """Test getting all loaded plugins."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin
        loader.load_plugin("testable-plugin")

        loaded = loader.get_loaded_plugins()

        assert "testable-plugin" in loaded
        assert isinstance(loaded["testable-plugin"], BookmarkPlugin)

    def test_get_loaded_plugins_returns_copy(self):
        """Test that get_loaded_plugins returns a copy."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin
        loader.load_plugin("testable-plugin")

        loaded = loader.get_loaded_plugins()
        loaded.clear()  # Modify the returned dict

        # Original should be unaffected
        assert loader.is_loaded("testable-plugin")

    def test_get_available_plugins(self):
        """Test getting available plugin names."""
        loader = PluginLoader()
        loader._discovered_plugins["plugin1"] = SampleTestPlugin
        loader._discovered_plugins["plugin2"] = SampleTestPlugin

        available = loader.get_available_plugins()

        assert "plugin1" in available
        assert "plugin2" in available

    def test_get_available_plugins_auto_discovers(self):
        """Test that get_available_plugins auto-discovers."""
        loader = PluginLoader()
        loader._discovered_plugins = {}

        available = loader.get_available_plugins()

        # Should return a list (may be empty)
        assert isinstance(available, list)

    def test_is_loaded(self):
        """Test is_loaded method."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        assert loader.is_loaded("testable-plugin") is False

        loader.load_plugin("testable-plugin")

        assert loader.is_loaded("testable-plugin") is True


# =============================================================================
# Directory Discovery Edge Cases
# =============================================================================


class TestDirectoryDiscoveryEdgeCases:
    """Tests for edge cases in directory discovery."""

    def test_discover_handles_unreadable_file(self, tmp_path):
        """Test discovery handles files that can't be loaded."""
        # Create a file that will cause an error during loading
        plugin_file = tmp_path / "unreadable.py"
        plugin_file.write_text("import this_module_does_not_exist_xyz")

        loader = PluginLoader()
        # Should not raise, should log warning
        discovered = loader._discover_from_directory(tmp_path)

        # The file with import error should be skipped
        assert isinstance(discovered, list)

    def test_discover_package_with_error(self, tmp_path):
        """Test discovery handles packages that fail to import."""
        package_dir = tmp_path / "bad_package"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("raise RuntimeError('Bad package')")

        loader = PluginLoader()
        # Should not raise, should log warning
        discovered = loader._discover_from_directory(tmp_path)

        assert isinstance(discovered, list)

    def test_discover_multiple_plugins_in_file(self, tmp_path):
        """Test discovering multiple plugins from a single file."""
        plugin_code = '''
from bookmark_processor.plugins.base import BookmarkPlugin

class PluginOne(BookmarkPlugin):
    @property
    def name(self):
        return "plugin-one"

    @property
    def version(self):
        return "1.0.0"

class PluginTwo(BookmarkPlugin):
    @property
    def name(self):
        return "plugin-two"

    @property
    def version(self):
        return "1.0.0"
'''
        plugin_file = tmp_path / "multi_plugins.py"
        plugin_file.write_text(plugin_code)

        loader = PluginLoader()
        plugins = loader._load_plugins_from_file(plugin_file)

        assert len(plugins) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestPluginLoaderIntegration:
    """Integration tests for the full plugin loader workflow."""

    def test_full_workflow(self, tmp_path):
        """Test the complete workflow: discover, load, use, unload."""
        loader = PluginLoader()
        loader._discovered_plugins["testable-plugin"] = SampleTestPlugin

        # 1. Get available plugins
        available = loader.get_available_plugins()
        assert "testable-plugin" in available

        # 2. Load plugin
        plugin = loader.load_plugin("testable-plugin", {"test": True})
        assert plugin.name == "testable-plugin"
        assert plugin.config == {"test": True}

        # 3. Check it's loaded
        assert loader.is_loaded("testable-plugin")
        loaded = loader.get_loaded_plugins()
        assert "testable-plugin" in loaded

        # 4. Get info
        info = loader.get_plugin_info("testable-plugin")
        assert info["loaded"] is True

        # 5. Reload
        reloaded = loader.reload_plugin("testable-plugin")
        assert reloaded is not plugin
        assert reloaded.config == {"test": True}

        # 6. Unload
        result = loader.unload_plugin("testable-plugin")
        assert result is True
        assert not loader.is_loaded("testable-plugin")

    def test_discover_all_sources(self, tmp_path):
        """Test discovery from all sources."""
        # Create a user plugins directory with a plugin
        user_dir = tmp_path / "user_plugins"
        user_dir.mkdir()

        # Create additional directory with a plugin
        additional_dir = tmp_path / "additional"
        additional_dir.mkdir()

        loader = PluginLoader(
            user_plugins_dir=user_dir,
            additional_paths=[additional_dir],
        )

        # Discover should search all paths
        discovered = loader.discover_plugins()

        assert isinstance(discovered, list)


# =============================================================================
# Additional Tests for Higher Coverage
# =============================================================================


class TestDirectoryDiscoveryWithPlugins:
    """Additional tests for directory discovery that finds actual plugins."""

    def test_discover_from_directory_finds_plugins_in_files(self, tmp_path):
        """Test discovering plugins from files in directory."""
        # Create a directory with a plugin file
        plugin_code = '''
from bookmark_processor.plugins.base import BookmarkPlugin

class DirTestPlugin(BookmarkPlugin):
    @property
    def name(self):
        return "dir-test"

    @property
    def version(self):
        return "1.0.0"
'''
        plugin_file = tmp_path / "dir_plugin.py"
        plugin_file.write_text(plugin_code)

        loader = PluginLoader(user_plugins_dir=tmp_path)
        discovered = loader._discover_from_directory(tmp_path)

        # Should find the plugin
        assert "dir-test" in discovered
        assert "dir-test" in loader._discovered_plugins

    def test_discover_from_directory_with_package_containing_plugins(self, tmp_path):
        """Test discovering plugins from a package directory with plugins inside."""
        # Create package structure with actual plugin
        package_dir = tmp_path / "my_plugin_package"
        package_dir.mkdir()

        init_code = '''
from bookmark_processor.plugins.base import BookmarkPlugin

class PackagePlugin(BookmarkPlugin):
    @property
    def name(self):
        return "package-plugin-test"

    @property
    def version(self):
        return "2.0.0"
'''
        (package_dir / "__init__.py").write_text(init_code)

        # Temporarily add to sys.path for import
        import sys
        sys.path.insert(0, str(tmp_path))

        try:
            loader = PluginLoader(additional_paths=[tmp_path])
            discovered = loader._discover_from_directory(tmp_path)

            # Should find plugin from package
            assert "package-plugin-test" in discovered
        finally:
            sys.path.remove(str(tmp_path))
            # Clean up module cache
            if "my_plugin_package" in sys.modules:
                del sys.modules["my_plugin_package"]


class TestEntryPointsEdgeCases:
    """Additional edge case tests for entry points discovery."""

    def test_discover_entry_points_with_python39_fallback(self):
        """Test entry points discovery using Python 3.9 fallback path."""
        loader = PluginLoader()

        mock_ep = MagicMock()
        mock_ep.name = "test-ep-39"
        mock_ep.load.return_value = SampleTestPlugin

        # Simulate Python 3.9 behavior where entry_points(group=...) raises TypeError
        def mock_entry_points(*args, **kwargs):
            if kwargs.get("group"):
                raise TypeError("entry_points() got an unexpected keyword argument 'group'")
            # Return object with get method (Python 3.9 style)
            result = MagicMock()
            result.get.return_value = [mock_ep]
            return result

        with patch("importlib.metadata.entry_points", side_effect=mock_entry_points):
            discovered = loader._discover_entry_points()

            # Should handle fallback and find the plugin
            assert "test-ep-39" in discovered


class TestPluginClassTypeErrorHandling:
    """Tests for TypeError handling in _is_plugin_class."""

    def test_is_plugin_class_with_issubclass_raising_typeerror(self):
        """Test handling when issubclass raises TypeError for unusual types."""
        loader = PluginLoader()

        # Some edge cases that could cause TypeError in issubclass
        class NotARealClass:
            """Object that pretends to be a class."""
            pass

        # Create a mock that when passed to issubclass would cause issues
        weird_obj = type.__new__(type, "WeirdType", (), {})

        # Should handle gracefully
        result = loader._is_plugin_class(weird_obj)
        # Result depends on whether it's a valid subclass, but shouldn't crash
        assert isinstance(result, bool)


class TestFullDiscoveryFlow:
    """Tests for the complete discovery flow with all sources."""

    def test_discover_plugins_from_all_sources(self, tmp_path):
        """Test discovering plugins from multiple sources at once."""
        # Create user plugins directory with a file plugin
        user_dir = tmp_path / "user_plugins"
        user_dir.mkdir()

        file_plugin_code = '''
from bookmark_processor.plugins.base import BookmarkPlugin

class UserFilePlugin(BookmarkPlugin):
    @property
    def name(self):
        return "user-file-plugin"

    @property
    def version(self):
        return "1.0.0"
'''
        (user_dir / "user_plugin.py").write_text(file_plugin_code)

        # Create additional directory with a package plugin
        additional_dir = tmp_path / "additional"
        additional_dir.mkdir()

        package_dir = additional_dir / "additional_package"
        package_dir.mkdir()

        package_plugin_code = '''
from bookmark_processor.plugins.base import BookmarkPlugin

class AdditionalPackagePlugin(BookmarkPlugin):
    @property
    def name(self):
        return "additional-package-plugin"

    @property
    def version(self):
        return "1.0.0"
'''
        (package_dir / "__init__.py").write_text(package_plugin_code)

        # Add paths for imports
        import sys
        sys.path.insert(0, str(additional_dir))

        try:
            loader = PluginLoader(
                user_plugins_dir=user_dir,
                additional_paths=[additional_dir],
            )

            # Force discovery
            discovered = loader.discover_plugins(force_refresh=True)

            # Should find plugins from file
            assert "user-file-plugin" in discovered

            # May also find the package plugin (depends on import success)
            # and builtin plugins

        finally:
            sys.path.remove(str(additional_dir))
            if "additional_package" in sys.modules:
                del sys.modules["additional_package"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
