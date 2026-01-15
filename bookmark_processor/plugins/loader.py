"""
Plugin Loader

Discovers and loads plugins from various sources including:
- Built-in plugins
- User plugins directory
- Installed packages (entry points)
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

from .base import BookmarkPlugin, PluginFactory


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""

    def __init__(self, plugin_name: str, message: str, cause: Optional[Exception] = None):
        self.plugin_name = plugin_name
        self.message = message
        self.cause = cause
        super().__init__(f"Failed to load plugin '{plugin_name}': {message}")


class PluginLoader:
    """
    Discovers and loads bookmark processor plugins.

    Supports loading from:
    - Built-in example plugins
    - User plugins directory (~/.bookmark_processor/plugins)
    - Installed packages via entry points
    - Explicit plugin paths
    """

    # Entry point group for installed plugins
    ENTRY_POINT_GROUP = "bookmark_processor.plugins"

    # Built-in plugins module path
    BUILTIN_PLUGINS_MODULE = "bookmark_processor.plugins.examples"

    def __init__(
        self,
        user_plugins_dir: Optional[Path] = None,
        additional_paths: Optional[List[Path]] = None,
    ):
        """
        Initialize the plugin loader.

        Args:
            user_plugins_dir: Custom user plugins directory
            additional_paths: Additional paths to search for plugins
        """
        self._logger = logging.getLogger("plugin.loader")

        # Set up search paths
        self._user_plugins_dir = user_plugins_dir or self._get_default_user_plugins_dir()
        self._additional_paths = additional_paths or []
        self._search_paths: List[Path] = []
        self._setup_search_paths()

        # Cache of discovered plugins
        self._discovered_plugins: Dict[str, Type[BookmarkPlugin]] = {}
        self._loaded_plugins: Dict[str, BookmarkPlugin] = {}

        self._logger.info(f"Plugin loader initialized with search paths: {self._search_paths}")

    def _get_default_user_plugins_dir(self) -> Path:
        """Get the default user plugins directory."""
        import os

        # Check for custom directory in environment
        if custom_dir := os.environ.get("BOOKMARK_PROCESSOR_PLUGINS_DIR"):
            return Path(custom_dir)

        # Default to ~/.bookmark_processor/plugins
        return Path.home() / ".bookmark_processor" / "plugins"

    def _setup_search_paths(self) -> None:
        """Set up plugin search paths."""
        self._search_paths = []

        # User plugins directory
        if self._user_plugins_dir.exists():
            self._search_paths.append(self._user_plugins_dir)

        # Additional configured paths
        for path in self._additional_paths:
            if path.exists():
                self._search_paths.append(path)

    def discover_plugins(self, force_refresh: bool = False) -> List[str]:
        """
        Discover all available plugins.

        Args:
            force_refresh: Force re-discovery even if cached

        Returns:
            List of discovered plugin names
        """
        if self._discovered_plugins and not force_refresh:
            return list(self._discovered_plugins.keys())

        self._discovered_plugins.clear()
        discovered: List[str] = []

        # 1. Discover built-in plugins
        builtin = self._discover_builtin_plugins()
        discovered.extend(builtin)

        # 2. Discover from user plugins directory
        user = self._discover_from_directory(self._user_plugins_dir)
        discovered.extend(user)

        # 3. Discover from additional paths
        for path in self._additional_paths:
            additional = self._discover_from_directory(path)
            discovered.extend(additional)

        # 4. Discover from entry points (installed packages)
        entry_points = self._discover_entry_points()
        discovered.extend(entry_points)

        self._logger.info(f"Discovered {len(discovered)} plugins: {discovered}")
        return discovered

    def _discover_builtin_plugins(self) -> List[str]:
        """Discover built-in example plugins."""
        discovered = []

        try:
            # Import the examples module
            examples_module = importlib.import_module(self.BUILTIN_PLUGINS_MODULE)

            # Look for plugin classes
            for name in dir(examples_module):
                obj = getattr(examples_module, name)
                if self._is_plugin_class(obj):
                    plugin_name = self._get_plugin_name_from_class(obj)
                    self._discovered_plugins[plugin_name] = obj
                    discovered.append(plugin_name)
                    self._logger.debug(f"Discovered builtin plugin: {plugin_name}")

        except ImportError as e:
            self._logger.debug(f"Could not load builtin plugins module: {e}")

        return discovered

    def _discover_from_directory(self, directory: Path) -> List[str]:
        """
        Discover plugins from a directory.

        Args:
            directory: Directory to search

        Returns:
            List of discovered plugin names
        """
        discovered = []

        if not directory.exists():
            return discovered

        # Look for Python files
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                plugin_classes = self._load_plugins_from_file(file_path)
                for plugin_class in plugin_classes:
                    plugin_name = self._get_plugin_name_from_class(plugin_class)
                    self._discovered_plugins[plugin_name] = plugin_class
                    discovered.append(plugin_name)
                    self._logger.debug(f"Discovered plugin from file: {plugin_name}")

            except Exception as e:
                self._logger.warning(f"Error loading plugins from {file_path}: {e}")

        # Look for plugin packages (directories with __init__.py)
        for subdir in directory.iterdir():
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                try:
                    plugin_classes = self._load_plugins_from_package(subdir)
                    for plugin_class in plugin_classes:
                        plugin_name = self._get_plugin_name_from_class(plugin_class)
                        self._discovered_plugins[plugin_name] = plugin_class
                        discovered.append(plugin_name)
                        self._logger.debug(f"Discovered plugin from package: {plugin_name}")

                except Exception as e:
                    self._logger.warning(f"Error loading plugins from {subdir}: {e}")

        return discovered

    def _discover_entry_points(self) -> List[str]:
        """
        Discover plugins from installed packages via entry points.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        try:
            # Python 3.10+ importlib.metadata
            from importlib.metadata import entry_points

            # Get entry points for our group
            try:
                # Python 3.10+
                eps = entry_points(group=self.ENTRY_POINT_GROUP)
            except TypeError:
                # Python 3.9
                eps = entry_points().get(self.ENTRY_POINT_GROUP, [])

            for ep in eps:
                try:
                    plugin_class = ep.load()
                    if self._is_plugin_class(plugin_class):
                        plugin_name = ep.name
                        self._discovered_plugins[plugin_name] = plugin_class
                        discovered.append(plugin_name)
                        self._logger.debug(f"Discovered plugin from entry point: {plugin_name}")
                except Exception as e:
                    self._logger.warning(f"Error loading plugin entry point {ep.name}: {e}")

        except ImportError:
            self._logger.debug("importlib.metadata not available")

        return discovered

    def _load_plugins_from_file(self, file_path: Path) -> List[Type[BookmarkPlugin]]:
        """
        Load plugin classes from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of plugin classes found in the file
        """
        module_name = f"bookmark_plugin_{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(
                file_path.stem,
                f"Could not create module spec for {file_path}",
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_name]
            raise PluginLoadError(file_path.stem, f"Error executing module: {e}", e)

        # Find plugin classes
        plugins = []
        for name in dir(module):
            obj = getattr(module, name)
            if self._is_plugin_class(obj):
                plugins.append(obj)

        return plugins

    def _load_plugins_from_package(self, package_dir: Path) -> List[Type[BookmarkPlugin]]:
        """
        Load plugin classes from a package directory.

        Args:
            package_dir: Path to package directory

        Returns:
            List of plugin classes found in the package
        """
        module_name = f"bookmark_plugin_{package_dir.name}"

        # Add parent directory to path temporarily
        parent_dir = str(package_dir.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            added_to_path = True
        else:
            added_to_path = False

        try:
            # Import the package
            module = importlib.import_module(package_dir.name)

            # Find plugin classes
            plugins = []
            for name in dir(module):
                obj = getattr(module, name)
                if self._is_plugin_class(obj):
                    plugins.append(obj)

            return plugins

        finally:
            if added_to_path:
                sys.path.remove(parent_dir)

    def _is_plugin_class(self, obj: Any) -> bool:
        """Check if an object is a valid plugin class."""
        if not isinstance(obj, type):
            return False

        if obj is BookmarkPlugin:
            return False

        try:
            return issubclass(obj, BookmarkPlugin) and hasattr(obj, "name")
        except TypeError:
            return False

    def _get_plugin_name_from_class(self, plugin_class: Type[BookmarkPlugin]) -> str:
        """Get the plugin name from a class."""
        try:
            # Try to instantiate to get the name
            instance = plugin_class()
            return instance.name
        except Exception:
            # Fall back to class name
            return plugin_class.__name__.lower().replace("plugin", "")

    def load_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BookmarkPlugin:
        """
        Load and initialize a plugin by name.

        Args:
            name: Plugin name
            config: Optional configuration for the plugin

        Returns:
            Initialized plugin instance

        Raises:
            PluginLoadError: If plugin cannot be loaded
        """
        # Check if already loaded
        if name in self._loaded_plugins:
            self._logger.debug(f"Plugin {name} already loaded, returning cached instance")
            return self._loaded_plugins[name]

        # Make sure plugins are discovered
        if not self._discovered_plugins:
            self.discover_plugins()

        # Find the plugin class
        if name not in self._discovered_plugins:
            # Try case-insensitive lookup
            name_lower = name.lower()
            for discovered_name in self._discovered_plugins:
                if discovered_name.lower() == name_lower:
                    name = discovered_name
                    break
            else:
                raise PluginLoadError(
                    name,
                    f"Plugin not found. Available plugins: {list(self._discovered_plugins.keys())}",
                )

        plugin_class = self._discovered_plugins[name]

        try:
            # Instantiate the plugin
            plugin = plugin_class()

            # Validate configuration if provided
            if config:
                errors = plugin.validate_config(config)
                if errors:
                    raise PluginLoadError(
                        name,
                        f"Configuration validation failed: {errors}",
                    )

            # Initialize the plugin
            plugin.on_load(config or {})

            # Cache the loaded plugin
            self._loaded_plugins[name] = plugin

            self._logger.info(f"Loaded plugin: {name} v{plugin.version}")
            return plugin

        except PluginLoadError:
            raise
        except Exception as e:
            raise PluginLoadError(name, f"Error instantiating plugin: {e}", e)

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if plugin was unloaded, False if not loaded
        """
        if name not in self._loaded_plugins:
            return False

        plugin = self._loaded_plugins[name]

        try:
            plugin.on_unload()
        except Exception as e:
            self._logger.warning(f"Error during plugin unload: {e}")

        del self._loaded_plugins[name]
        self._logger.info(f"Unloaded plugin: {name}")
        return True

    def get_loaded_plugins(self) -> Dict[str, BookmarkPlugin]:
        """Get all loaded plugins."""
        return self._loaded_plugins.copy()

    def get_available_plugins(self) -> List[str]:
        """Get list of available (discovered) plugin names."""
        if not self._discovered_plugins:
            self.discover_plugins()
        return list(self._discovered_plugins.keys())

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded."""
        return name in self._loaded_plugins

    def reload_plugin(self, name: str) -> BookmarkPlugin:
        """
        Reload a plugin (unload and load again).

        Args:
            name: Plugin name

        Returns:
            Reloaded plugin instance
        """
        config = None
        if name in self._loaded_plugins:
            config = self._loaded_plugins[name].config
            self.unload_plugin(name)

        return self.load_plugin(name, config)

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin information dict or None if not found
        """
        if not self._discovered_plugins:
            self.discover_plugins()

        if name not in self._discovered_plugins:
            return None

        plugin_class = self._discovered_plugins[name]

        try:
            instance = plugin_class()
            return {
                "name": instance.name,
                "version": instance.version,
                "description": instance.description,
                "author": instance.author,
                "requires": instance.requires,
                "provides": instance.provides,
                "hooks": [h.value for h in instance.hooks],
                "loaded": name in self._loaded_plugins,
            }
        except Exception as e:
            return {
                "name": name,
                "error": str(e),
                "loaded": False,
            }


__all__ = ["PluginLoader", "PluginLoadError"]
