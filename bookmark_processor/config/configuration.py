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
        """
        Get configuration value as string.

        DEPRECATED: This method exists for backward compatibility only.
        Use direct Pydantic attribute access instead:
        - config.network.timeout instead of config.get('network', 'timeout')
        - config.processing.batch_size instead of config.get('processing', 'batch_size')

        Args:
            section: Configuration section (ignored)
            option: Configuration option (ignored)
            fallback: Fallback value to return

        Returns:
            Fallback value as string
        """
        import warnings
        warnings.warn(
            f"config.get('{section}', '{option}') is deprecated. "
            "Use direct Pydantic attribute access instead (e.g., config.network.timeout)",
            DeprecationWarning,
            stacklevel=2
        )
        return str(fallback) if fallback is not None else ""

    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """
        Get configuration value as integer.

        DEPRECATED: Use direct Pydantic attribute access instead.
        """
        import warnings
        warnings.warn(
            f"config.getint('{section}', '{option}') is deprecated. "
            "Use direct Pydantic attribute access instead",
            DeprecationWarning,
            stacklevel=2
        )
        return fallback

    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """
        Get configuration value as float.

        DEPRECATED: Use direct Pydantic attribute access instead.
        """
        import warnings
        warnings.warn(
            f"config.getfloat('{section}', '{option}') is deprecated. "
            "Use direct Pydantic attribute access instead",
            DeprecationWarning,
            stacklevel=2
        )
        return fallback

    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """
        Get configuration value as boolean.

        DEPRECATED: Use direct Pydantic attribute access instead.
        """
        import warnings
        warnings.warn(
            f"config.getboolean('{section}', '{option}') is deprecated. "
            "Use direct Pydantic attribute access instead",
            DeprecationWarning,
            stacklevel=2
        )
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
