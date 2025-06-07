"""
Configuration management for the Bookmark Processor.

This module handles loading and merging configuration from multiple sources:
1. Default configuration
2. User configuration file
3. Command-line arguments
"""

import configparser
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bookmark_processor.utils.api_key_validator import APIKeyValidator


class Configuration:
    """Manages application configuration with multiple sources."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to user configuration file
        """
        self.config = configparser.ConfigParser()
        self._load_default_config()

        # Try to load user config from standard location if not specified
        if config_path:
            self._load_user_config(config_path)
        else:
            # Check for user_config.ini in the same directory as default config
            user_config_path = (
                self._get_default_config_path().parent / "user_config.ini"
            )
            if user_config_path.exists():
                self._load_user_config(user_config_path)

    def _get_default_config_path(self) -> Path:
        """Get path to default configuration file."""
        if getattr(sys, "frozen", False):
            # Running as PyInstaller executable
            app_dir = Path(sys.executable).parent
            config_path = app_dir / "config" / "default_config.ini"
        else:
            # Running as Python script
            config_path = Path(__file__).parent / "default_config.ini"

        return config_path

    def _load_default_config(self) -> None:
        """Load default configuration."""
        default_path = self._get_default_config_path()
        if default_path.exists():
            self.config.read(default_path)
        else:
            # Fallback to hardcoded defaults if file not found
            self._set_hardcoded_defaults()

    def _set_hardcoded_defaults(self) -> None:
        """Set hardcoded default values."""
        # Network settings
        self.config["network"] = {
            "timeout": "30",
            "max_retries": "3",
            "default_delay": "0.5",
            "max_concurrent_requests": "10",
            "user_agent_rotation": "true",
            "google_delay": "2.0",
            "github_delay": "1.5",
            "youtube_delay": "2.0",
            "linkedin_delay": "2.0",
        }

        # Processing settings
        self.config["processing"] = {
            "batch_size": "100",
            "max_tags_per_bookmark": "5",
            "target_unique_tags": "150",
            "ai_model": "facebook/bart-large-cnn",
            "max_description_length": "150",
            "use_existing_content": "true",
        }

        # Checkpoint settings
        self.config["checkpoint"] = {
            "enabled": "true",
            "save_interval": "50",
            "checkpoint_dir": ".bookmark_checkpoints",
            "auto_cleanup": "true",
        }

        # Output settings
        self.config["output"] = {
            "output_format": "raindrop_import",
            "preserve_folder_structure": "true",
            "include_timestamps": "true",
            "error_log_detailed": "true",
        }

        # Logging settings
        self.config["logging"] = {
            "log_level": "INFO",
            "log_file": "bookmark_processor.log",
            "console_output": "true",
            "performance_logging": "true",
        }

        # AI settings
        self.config["ai"] = {
            "default_engine": "local",
            "claude_rpm": "50",
            "openai_rpm": "60",
            "claude_batch_size": "10",
            "openai_batch_size": "20",
            "show_running_costs": "true",
            "cost_confirmation_interval": "10.0",
        }

        # Executable settings
        self.config["executable"] = {
            "model_cache_dir": "~/.cache/bookmark-processor/models",
            "temp_dir": "/tmp/bookmark-processor",
            "cleanup_on_exit": "true",
        }

    def _load_user_config(self, config_path: Path) -> None:
        """Load user configuration file."""
        if config_path.exists():
            self.config.read(config_path)

    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.

        Args:
            args: Dictionary of validated arguments
        """
        # Update processing settings
        if "batch_size" in args:
            self.config.set("processing", "batch_size", str(args["batch_size"]))

        if "max_retries" in args:
            self.config.set("network", "max_retries", str(args["max_retries"]))

        # Update checkpoint settings
        if args.get("clear_checkpoints"):
            self.config.set("checkpoint", "enabled", "false")
        elif args.get("resume"):
            self.config.set("checkpoint", "enabled", "true")

        # Update logging settings
        if args.get("verbose"):
            self.config.set("logging", "log_level", "DEBUG")
            self.config.set("logging", "console_output", "true")

        # Update AI engine setting
        if "ai_engine" in args and args["ai_engine"]:
            self.config.set("ai", "default_engine", args["ai_engine"])

    def get(self, section: str, option: str, fallback: Any = None) -> str:
        """Get configuration value."""
        return self.config.get(section, option, fallback=fallback)

    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """Get configuration value as integer."""
        return self.config.getint(section, option, fallback=fallback)

    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """Get configuration value as float."""
        return self.config.getfloat(section, option, fallback=fallback)

    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """Get configuration value as boolean."""
        return self.config.getboolean(section, option, fallback=fallback)

    def get_model_cache_dir(self) -> Path:
        """Get AI model cache directory with environment variable expansion."""
        cache_dir = self.get("executable", "model_cache_dir")
        cache_dir = os.path.expandvars(cache_dir)
        return Path(cache_dir)

    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return Path(self.get("checkpoint", "checkpoint_dir", ".bookmark_checkpoints"))

    def get_ai_engine(self) -> str:
        """Get the configured AI engine."""
        return self.get("ai", "default_engine", fallback="local")

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider. Returns None if not configured.

        Args:
            provider: API provider name (claude or openai)

        Returns:
            API key string or None
        """
        key_name = f"{provider}_api_key"
        try:
            key = self.get("ai", key_name)
            # Don't return empty strings as valid keys
            return key if key and key.strip() else None
        except (configparser.NoSectionError, configparser.NoOptionError):
            return None

    def has_api_key(self, provider: str) -> bool:
        """Check if API key is configured for a provider."""
        return self.get_api_key(provider) is not None

    def get_rate_limit(self, provider: str) -> int:
        """Get rate limit (requests per minute) for a provider."""
        return self.getint("ai", f"{provider}_rpm", fallback=60)

    def get_batch_size(self, provider: str) -> int:
        """Get batch size for a provider."""
        return self.getint("ai", f"{provider}_batch_size", fallback=10)

    def get_cost_tracking_settings(self) -> Dict[str, Any]:
        """Get cost tracking configuration."""
        return {
            "show_running_costs": self.getboolean(
                "ai", "show_running_costs", fallback=True
            ),
            "cost_confirmation_interval": self.getfloat(
                "ai", "cost_confirmation_interval", fallback=10.0
            ),
            "max_cost_per_run": self.getfloat(
                "ai", "max_cost_per_run", fallback=0.0
            ),  # 0 = no limit
            "pause_at_cost": self.getboolean("ai", "pause_at_cost", fallback=True),
        }

    def validate_ai_configuration(self) -> Tuple[bool, Optional[str]]:
        """
        Validate AI configuration including API keys.

        Returns:
            Tuple of (is_valid, error_message)
        """
        return APIKeyValidator.validate_configuration(self)
