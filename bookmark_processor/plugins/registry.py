"""
Plugin Registry

Central registry for managing plugin instances and dispatching
hook calls to appropriate plugins.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from .base import (
    AIProcessorPlugin,
    BookmarkPlugin,
    ContentEnhancerPlugin,
    OutputPlugin,
    PluginHook,
    TagGeneratorPlugin,
    ValidatorPlugin,
)
from .loader import PluginLoader, PluginLoadError


T = TypeVar("T", bound=BookmarkPlugin)


class PluginRegistry:
    """
    Central registry for bookmark processor plugins.

    Manages plugin registration, lifecycle, and hook dispatching.
    Provides a convenient interface for accessing plugins by type
    and executing hook callbacks.
    """

    def __init__(self, loader: Optional[PluginLoader] = None):
        """
        Initialize the plugin registry.

        Args:
            loader: Optional PluginLoader instance (creates one if not provided)
        """
        self._loader = loader or PluginLoader()
        self._logger = logging.getLogger("plugin.registry")

        # Plugin storage by name
        self._plugins: Dict[str, BookmarkPlugin] = {}

        # Plugin indexing by type
        self._validators: List[ValidatorPlugin] = []
        self._ai_processors: List[AIProcessorPlugin] = []
        self._output_plugins: List[OutputPlugin] = []
        self._tag_generators: List[TagGeneratorPlugin] = []
        self._content_enhancers: List[ContentEnhancerPlugin] = []

        # Hook subscriptions
        self._hook_subscribers: Dict[PluginHook, List[BookmarkPlugin]] = defaultdict(list)

        # Plugin execution order (for hooks)
        self._execution_order: Dict[str, int] = {}

        self._logger.info("Plugin registry initialized")

    @property
    def loader(self) -> PluginLoader:
        """Get the plugin loader."""
        return self._loader

    def register(self, plugin_class: Type[BookmarkPlugin]) -> None:
        """
        Register a plugin class.

        This adds the plugin to the discovered plugins but does not load it.

        Args:
            plugin_class: Plugin class to register
        """
        try:
            instance = plugin_class()
            name = instance.name
            self._loader._discovered_plugins[name] = plugin_class
            self._logger.info(f"Registered plugin class: {name}")
        except Exception as e:
            self._logger.error(f"Error registering plugin class: {e}")
            raise

    def register_instance(self, plugin: BookmarkPlugin) -> None:
        """
        Register an already-instantiated plugin.

        Args:
            plugin: Plugin instance to register
        """
        name = plugin.name

        if name in self._plugins:
            self._logger.warning(f"Plugin {name} already registered, replacing")
            self._unindex_plugin(self._plugins[name])

        self._plugins[name] = plugin
        self._index_plugin(plugin)

        self._logger.info(f"Registered plugin instance: {name} v{plugin.version}")

    def _index_plugin(self, plugin: BookmarkPlugin) -> None:
        """Index a plugin by type and hooks."""
        # Index by type
        if isinstance(plugin, ValidatorPlugin):
            self._validators.append(plugin)
            self._validators.sort(key=lambda p: p.get_priority())

        if isinstance(plugin, AIProcessorPlugin):
            self._ai_processors.append(plugin)

        if isinstance(plugin, OutputPlugin):
            self._output_plugins.append(plugin)

        if isinstance(plugin, TagGeneratorPlugin):
            self._tag_generators.append(plugin)

        if isinstance(plugin, ContentEnhancerPlugin):
            self._content_enhancers.append(plugin)

        # Index by hooks
        for hook in plugin.hooks:
            if plugin not in self._hook_subscribers[hook]:
                self._hook_subscribers[hook].append(plugin)

    def _unindex_plugin(self, plugin: BookmarkPlugin) -> None:
        """Remove a plugin from indexes."""
        if isinstance(plugin, ValidatorPlugin) and plugin in self._validators:
            self._validators.remove(plugin)

        if isinstance(plugin, AIProcessorPlugin) and plugin in self._ai_processors:
            self._ai_processors.remove(plugin)

        if isinstance(plugin, OutputPlugin) and plugin in self._output_plugins:
            self._output_plugins.remove(plugin)

        if isinstance(plugin, TagGeneratorPlugin) and plugin in self._tag_generators:
            self._tag_generators.remove(plugin)

        if isinstance(plugin, ContentEnhancerPlugin) and plugin in self._content_enhancers:
            self._content_enhancers.remove(plugin)

        for hook in plugin.hooks:
            if plugin in self._hook_subscribers[hook]:
                self._hook_subscribers[hook].remove(plugin)

    def load_plugins(
        self,
        plugin_names: List[str],
        config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, BookmarkPlugin]:
        """
        Load multiple plugins by name.

        Args:
            plugin_names: List of plugin names to load
            config: Optional configuration dict keyed by plugin name

        Returns:
            Dict of loaded plugins keyed by name
        """
        config = config or {}
        loaded = {}

        for name in plugin_names:
            try:
                plugin_config = config.get(name, {})
                plugin = self._loader.load_plugin(name, plugin_config)
                self.register_instance(plugin)
                loaded[name] = plugin
            except PluginLoadError as e:
                self._logger.error(f"Failed to load plugin {name}: {e}")

        return loaded

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin by name.

        Args:
            name: Plugin name

        Returns:
            True if plugin was unloaded
        """
        if name not in self._plugins:
            return False

        plugin = self._plugins[name]
        self._unindex_plugin(plugin)

        try:
            plugin.on_unload()
        except Exception as e:
            self._logger.warning(f"Error during plugin unload: {e}")

        del self._plugins[name]
        self._loader.unload_plugin(name)

        self._logger.info(f"Unloaded plugin: {name}")
        return True

    def unload_all(self) -> None:
        """Unload all plugins."""
        for name in list(self._plugins.keys()):
            self.unload_plugin(name)

    def get(self, name: str) -> Optional[BookmarkPlugin]:
        """
        Get a loaded plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None if not loaded
        """
        return self._plugins.get(name)

    def get_by_type(self, plugin_type: Type[T]) -> List[T]:
        """
        Get all loaded plugins of a specific type.

        Args:
            plugin_type: Plugin class to filter by

        Returns:
            List of plugins of the specified type
        """
        return [p for p in self._plugins.values() if isinstance(p, plugin_type)]

    def get_validators(self) -> List[ValidatorPlugin]:
        """Get all loaded validator plugins (sorted by priority)."""
        return [p for p in self._validators if p.enabled]

    def get_ai_processors(self) -> List[AIProcessorPlugin]:
        """Get all loaded AI processor plugins."""
        return [p for p in self._ai_processors if p.enabled]

    def get_output_plugins(self) -> List[OutputPlugin]:
        """Get all loaded output plugins."""
        return [p for p in self._output_plugins if p.enabled]

    def get_tag_generators(self) -> List[TagGeneratorPlugin]:
        """Get all loaded tag generator plugins."""
        return [p for p in self._tag_generators if p.enabled]

    def get_content_enhancers(self) -> List[ContentEnhancerPlugin]:
        """Get all loaded content enhancer plugins."""
        return [p for p in self._content_enhancers if p.enabled]

    def has_plugin(self, name: str) -> bool:
        """Check if a plugin is loaded."""
        return name in self._plugins

    def list_plugins(self) -> List[str]:
        """Get list of loaded plugin names."""
        return list(self._plugins.keys())

    def list_available(self) -> List[str]:
        """Get list of available (discovered) plugin names."""
        return self._loader.get_available_plugins()

    # =========================================================================
    # Hook System
    # =========================================================================

    def get_hook_subscribers(self, hook: PluginHook) -> List[BookmarkPlugin]:
        """
        Get plugins subscribed to a hook.

        Args:
            hook: The hook to query

        Returns:
            List of plugins subscribed to the hook
        """
        return [p for p in self._hook_subscribers[hook] if p.enabled]

    def dispatch_hook(
        self,
        hook: PluginHook,
        *args: Any,
        stop_on_failure: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Dispatch a hook to all subscribed plugins.

        Args:
            hook: The hook to dispatch
            *args: Positional arguments to pass to handlers
            stop_on_failure: Stop on first failure
            **kwargs: Keyword arguments to pass to handlers

        Returns:
            List of results from each handler
        """
        results = []
        subscribers = self.get_hook_subscribers(hook)

        for plugin in subscribers:
            try:
                handler = self._get_hook_handler(plugin, hook)
                if handler:
                    result = handler(*args, **kwargs)
                    results.append(result)
            except Exception as e:
                self._logger.error(
                    f"Error in hook {hook.value} for plugin {plugin.name}: {e}"
                )
                if stop_on_failure:
                    raise

        return results

    def dispatch_hook_chain(
        self,
        hook: PluginHook,
        initial_value: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Dispatch a hook as a chain, passing result to next handler.

        Args:
            hook: The hook to dispatch
            initial_value: Initial value to pass through chain
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Final value after all handlers
        """
        value = initial_value
        subscribers = self.get_hook_subscribers(hook)

        for plugin in subscribers:
            try:
                handler = self._get_hook_handler(plugin, hook)
                if handler:
                    value = handler(value, *args, **kwargs)
            except Exception as e:
                self._logger.error(
                    f"Error in hook chain {hook.value} for plugin {plugin.name}: {e}"
                )
                # Continue with current value

        return value

    def _get_hook_handler(
        self, plugin: BookmarkPlugin, hook: PluginHook
    ) -> Optional[Callable]:
        """Get the handler method for a hook from a plugin."""
        # Map hooks to method names
        hook_methods = {
            PluginHook.PRE_VALIDATION: "on_pre_validation",
            PluginHook.POST_VALIDATION: "on_post_validation",
            PluginHook.VALIDATION_FILTER: "filter_validation",
            PluginHook.PRE_CONTENT_FETCH: "on_pre_content_fetch",
            PluginHook.POST_CONTENT_FETCH: "on_post_content_fetch",
            PluginHook.CONTENT_FILTER: "filter_content",
            PluginHook.PRE_AI_PROCESS: "on_pre_ai_process",
            PluginHook.POST_AI_PROCESS: "on_post_ai_process",
            PluginHook.AI_FALLBACK: "on_ai_fallback",
            PluginHook.PRE_TAG_GENERATION: "on_pre_tag_generation",
            PluginHook.POST_TAG_GENERATION: "on_post_tag_generation",
            PluginHook.TAG_FILTER: "filter_tags",
            PluginHook.PRE_EXPORT: "on_pre_export",
            PluginHook.POST_EXPORT: "on_post_export",
            PluginHook.ON_START: "on_start",
            PluginHook.ON_COMPLETE: "on_complete",
            PluginHook.ON_ERROR: "on_error",
        }

        method_name = hook_methods.get(hook)
        if method_name and hasattr(plugin, method_name):
            return getattr(plugin, method_name)

        return None

    # =========================================================================
    # Plugin Information
    # =========================================================================

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin information dict or None
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            return {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "author": plugin.author,
                "requires": plugin.requires,
                "provides": plugin.provides,
                "hooks": [h.value for h in plugin.hooks],
                "enabled": plugin.enabled,
                "loaded": True,
                "config": plugin.config,
            }

        return self._loader.get_plugin_info(name)

    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded plugins."""
        return {name: self.get_plugin_info(name) for name in self._plugins}

    def get_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities provided by loaded plugins.

        Returns:
            Dict mapping capability name to list of plugins providing it
        """
        capabilities: Dict[str, List[str]] = defaultdict(list)

        for plugin in self._plugins.values():
            for capability in plugin.provides:
                capabilities[capability].append(plugin.name)

        return dict(capabilities)

    def check_dependencies(self, plugin_name: str) -> List[str]:
        """
        Check if a plugin's dependencies are satisfied.

        Args:
            plugin_name: Plugin to check

        Returns:
            List of missing dependencies
        """
        info = self._loader.get_plugin_info(plugin_name)
        if not info:
            return [f"Plugin {plugin_name} not found"]

        requires = info.get("requires", [])
        missing = []

        for dep in requires:
            if dep not in self._plugins:
                missing.append(dep)

        return missing


# Global registry instance (optional singleton pattern)
_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global plugin registry."""
    global _global_registry
    if _global_registry:
        _global_registry.unload_all()
    _global_registry = None


__all__ = [
    "PluginRegistry",
    "get_registry",
    "reset_registry",
]
