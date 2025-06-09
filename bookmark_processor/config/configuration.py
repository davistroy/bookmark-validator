"""
New Pydantic-based configuration management for the Bookmark Processor.

This module replaces the old ConfigParser-based system with a modern Pydantic
configuration system that reduces 42 options to 15 essential ones with built-in
validation and type safety.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .pydantic_config import BookmarkConfig, ConfigurationManager


class Configuration:
    """
    Modern configuration manager that wraps the Pydantic-based system.

    This class provides a compatibility layer that maintains the same interface
    as the old ConfigParser-based Configuration class while using the new
    Pydantic system underneath.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to user configuration file (TOML/JSON)
        """
        self._manager = ConfigurationManager(config_path)
        self._config = self._manager.config

    @property
    def config(self) -> BookmarkConfig:
        """Get the underlying Pydantic configuration."""
        return self._config

    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.

        Args:
            args: Dictionary of validated arguments
        """
        self._manager.update_from_cli_args(args)
        self._config = self._manager.config

    def get(self, section: str, option: str, fallback: Any = None) -> str:
        """Get configuration value as string (for compatibility)."""
        try:
            if section == "network":
                if option == "timeout":
                    return str(self._config.network.timeout)
                elif option == "max_retries":
                    return str(self._config.network.max_retries)
                elif option == "max_concurrent_requests":
                    return str(self._config.network.concurrent_requests)
                elif option == "default_delay":
                    return "0.5"  # Fixed value in new system
                elif option == "user_agent_rotation":
                    return "true"  # Always enabled in new system
                elif option.endswith("_delay"):
                    return "1.5"  # Simplified delay handling

            elif section == "processing":
                if option == "batch_size":
                    return str(self._config.processing.batch_size)
                elif option == "max_description_length":
                    return str(self._config.processing.max_description_length)
                elif option == "ai_model":
                    return "facebook/bart-large-cnn"  # Fixed in new system
                elif option == "use_existing_content":
                    return "true"  # Always true in new system
                elif option == "max_tags_per_bookmark":
                    return "5"  # Fixed value
                elif option == "target_unique_tags":
                    return "150"  # Fixed value

            elif section == "checkpoint":
                if option == "enabled":
                    return str(self._config.checkpoint_enabled).lower()
                elif option == "save_interval":
                    return str(self._config.checkpoint_interval)
                elif option == "checkpoint_dir":
                    return str(self._config.checkpoint_dir)
                elif option == "auto_cleanup":
                    return "true"  # Always true in new system

            elif section == "output":
                if option == "output_format":
                    return self._config.output.format
                elif option == "error_log_detailed":
                    return str(self._config.output.detailed_errors).lower()
                elif option == "preserve_folder_structure":
                    return "true"  # Always true in new system
                elif option == "include_timestamps":
                    return "true"  # Always true in new system

            elif section == "logging":
                if option == "log_level":
                    return "INFO"  # Simplified logging
                elif option == "log_file":
                    return "bookmark_processor.log"
                elif option == "console_output":
                    return "true"
                elif option == "performance_logging":
                    return "true"

            elif section == "ai":
                if option == "default_engine":
                    return self._config.processing.ai_engine
                elif option == "claude_rpm":
                    return str(self._config.ai.claude_rpm)
                elif option == "openai_rpm":
                    return str(self._config.ai.openai_rpm)
                elif option == "claude_batch_size":
                    return "10"  # Fixed value
                elif option == "openai_batch_size":
                    return "20"  # Fixed value
                elif option == "show_running_costs":
                    return "true"  # Always true
                elif option == "cost_confirmation_interval":
                    return str(self._config.ai.cost_confirmation_interval)
                elif option == "max_cost_per_run":
                    return "0.0"  # No limit by default
                elif option == "pause_at_cost":
                    return "true"

            elif section == "executable":
                if option == "model_cache_dir":
                    return "~/.cache/bookmark-processor/models"
                elif option == "temp_dir":
                    return "/tmp/bookmark-processor"
                elif option == "cleanup_on_exit":
                    return "true"

            return str(fallback) if fallback is not None else ""

        except Exception:
            return str(fallback) if fallback is not None else ""

    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """Get configuration value as integer."""
        try:
            value = self.get(section, option, str(fallback))
            return int(value)
        except (ValueError, TypeError):
            return fallback

    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """Get configuration value as float."""
        try:
            value = self.get(section, option, str(fallback))
            return float(value)
        except (ValueError, TypeError):
            return fallback

    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """Get configuration value as boolean."""
        try:
            value = self.get(section, option, str(fallback)).lower()
            return value in ("true", "1", "yes", "on", "enabled")
        except (ValueError, TypeError, AttributeError):
            return fallback

    def get_model_cache_dir(self) -> Path:
        """Get AI model cache directory with environment variable expansion."""
        cache_dir = "~/.cache/bookmark-processor/models"
        cache_dir = os.path.expanduser(cache_dir)
        cache_dir = os.path.expandvars(cache_dir)
        return Path(cache_dir)

    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return self._config.checkpoint_dir

    def get_ai_engine(self) -> str:
        """Get the configured AI engine."""
        return self._config.processing.ai_engine

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider. Returns None if not configured.

        Args:
            provider: API provider name (claude or openai)

        Returns:
            API key string or None
        """
        return self._manager.get_api_key(provider)

    def has_api_key(self, provider: str) -> bool:
        """Check if API key is configured for a provider."""
        return self._manager.has_api_key(provider)

    def get_rate_limit(self, provider: str) -> int:
        """Get rate limit (requests per minute) for a provider."""
        if provider == "claude":
            return self._config.ai.claude_rpm
        elif provider == "openai":
            return self._config.ai.openai_rpm
        return 60

    def get_batch_size(self, provider: str) -> int:
        """Get batch size for a provider."""
        # Fixed batch sizes in new system
        if provider == "claude":
            return 10
        elif provider == "openai":
            return 20
        return 10

    def get_cost_tracking_settings(self) -> Dict[str, Any]:
        """Get cost tracking configuration."""
        return {
            "show_running_costs": True,  # Always enabled
            "cost_confirmation_interval": self._config.ai.cost_confirmation_interval,
            "max_cost_per_run": 0.0,  # No limit by default
            "pause_at_cost": True,
        }

    def validate_ai_configuration(self) -> Tuple[bool, Optional[str]]:
        """
        Validate AI configuration including API keys.

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self._manager.validate_ai_configuration()


# Factory function for easy migration
def create_configuration(config_path: Optional[Path] = None) -> Configuration:
    """
    Create a new Configuration instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configuration instance
    """
    return Configuration(config_path)


# Alias for backward compatibility during migration
AppConfig = Configuration
