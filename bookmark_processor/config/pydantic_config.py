"""
Pydantic-based configuration system for Bookmark Processor.

This module replaces the complex INI-based configuration system with a
simplified Pydantic configuration that reduces 42 options to 15 essential ones.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Literal, Optional
from pydantic import (
    BaseModel,
    SecretStr,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)
import json
import toml


class NetworkConfig(BaseModel):
    """Network and URL validation settings."""

    timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds",
        json_schema_extra={
            "error_msg": "Timeout must be between 5 and 300 seconds. "
            "Recommended: 30 seconds for balanced performance."
        },
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
        json_schema_extra={
            "error_msg": "Max retries must be between 0 and 10. "
            "Values above 5 may cause slow processing."
        },
    )
    concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent requests",
        json_schema_extra={
            "error_msg": "Concurrent requests must be between 1 and 50. "
            "Higher values may cause rate limiting."
        },
    )

    @field_validator("concurrent_requests")
    @classmethod
    def validate_concurrent_requests(cls, v):
        """Provide performance warnings for extreme values."""
        if v > 25:
            import warnings

            warnings.warn(
                f"High concurrent requests ({v}) may trigger rate limiting "
                f"from websites. Consider using 10-15 for optimal results.",
                UserWarning,
            )
        elif v < 5:
            import warnings

            warnings.warn(
                f"Low concurrent requests ({v}) may result in very slow processing. "
                "Consider using 5-10 for better performance.",
                UserWarning,
            )
        return v


class ProcessingConfig(BaseModel):
    """Core processing settings."""

    batch_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Processing batch size",
        json_schema_extra={
            "error_msg": "Batch size must be between 10 and 1000. "
            "Recommended: 50-200 for optimal memory usage."
        },
    )
    max_description_length: int = Field(
        default=150,
        ge=50,
        le=500,
        description="Maximum AI description length",
        json_schema_extra={
            "error_msg": "Description length must be between 50 and 500 "
            "characters. Recommended: 100-200 for readability."
        },
    )
    ai_engine: Literal["local", "claude", "openai"] = Field(
        default="local",
        description="AI engine to use",
        json_schema_extra={
            "error_msg": "AI engine must be 'local', 'claude', or 'openai'. "
            "Note: cloud engines require API keys."
        },
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        """Provide performance warnings for extreme batch sizes."""
        if v > 500:
            import warnings

            warnings.warn(
                f"Large batch size ({v}) may use excessive memory for large datasets. "
                "Consider using 100-200 for optimal performance.",
                UserWarning,
            )
        elif v < 25:
            import warnings

            warnings.warn(
                f"Small batch size ({v}) may slow overall processing. "
                "Consider using 50-100 for better throughput.",
                UserWarning,
            )
        return v


class AIConfig(BaseModel):
    """AI service configuration with secure API key handling."""

    claude_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Claude API key",
        json_schema_extra={
            "error_msg": "Claude API key should be provided when using "
            "'claude' AI engine. Keep this secure and never commit "
            "to version control."
        },
    )
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key",
        json_schema_extra={
            "error_msg": "OpenAI API key should be provided when using "
            "'openai' AI engine. Keep this secure and never commit "
            "to version control."
        },
    )
    claude_rpm: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Claude requests per minute",
        json_schema_extra={
            "error_msg": "Claude RPM must be between 1 and 1000. "
            "Check your API tier limits to avoid rate limiting."
        },
    )
    openai_rpm: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="OpenAI requests per minute",
        json_schema_extra={
            "error_msg": "OpenAI RPM must be between 1 and 1000. "
            "Check your API tier limits to avoid rate limiting."
        },
    )
    cost_confirmation_interval: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Cost confirmation threshold in USD",
        json_schema_extra={
            "error_msg": "Cost confirmation interval must be between "
            "$0.00 and $100.00. Set to 0 to disable cost confirmations."
        },
    )

    @field_validator("claude_api_key", "openai_api_key", mode="before")
    @classmethod
    def validate_api_key_format(cls, v, info):
        """Validate API key format and provide security warnings."""
        if v is None or v == "":
            return None

        # Convert to string for validation
        key_str = str(v)

        # Check for placeholder values
        if key_str in [
            "your-claude-api-key-here",
            "your-openai-api-key-here",
            "sk-placeholder",
        ]:
            raise ValueError(
                f"Please replace the placeholder API key with your actual "
                f"{info.field_name.replace('_api_key', '').title()} API key. "
                f"You can get your API key from the respective "
                f"provider's dashboard."
            )

        # Basic format validation
        if len(key_str) < 10:
            import warnings

            warnings.warn(
                f"{info.field_name.replace('_api_key', '').title()} API key "
                f"appears to be very short. Please verify this is a valid "
                f"API key.",
                UserWarning,
            )

        # Check for OpenAI key format (starts with sk-)
        if info.field_name == "openai_api_key" and not key_str.startswith("sk-"):
            import warnings

            warnings.warn(
                "OpenAI API keys typically start with 'sk-'. "
                "Please verify this is a valid OpenAI API key.",
                UserWarning,
            )

        return SecretStr(key_str)


class OutputConfig(BaseModel):
    """Output format and file settings."""

    format: Literal["raindrop_import"] = Field(
        default="raindrop_import",
        description="Output format",
        json_schema_extra={
            "error_msg": "Output format must be 'raindrop_import'. "
            "This ensures compatibility with Raindrop.io bookmark imports."
        },
    )
    detailed_errors: bool = Field(
        default=True,
        description="Include detailed error logging",
        json_schema_extra={
            "error_msg": "Detailed errors setting must be true or false. "
            "Recommended: true for better debugging."
        },
    )


class BookmarkConfig(BaseModel):
    """Main configuration model with 15 essential options."""

    # Core configuration groups
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Direct essential options
    checkpoint_enabled: bool = Field(
        default=True,
        description="Enable checkpoint/resume functionality",
        json_schema_extra={
            "error_msg": "Checkpoint enabled must be true or false. "
            "Recommended: true for long-running processes to prevent "
            "data loss."
        },
    )
    checkpoint_interval: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Save checkpoint every N items",
        json_schema_extra={
            "error_msg": "Checkpoint interval must be between 1 and 1000 "
            "items. Recommended: 50-100 for optimal balance between "
            "safety and performance."
        },
    )
    checkpoint_dir: Path = Field(
        default=Path(".bookmark_checkpoints"),
        description="Checkpoint directory",
        json_schema_extra={
            "error_msg": "Checkpoint directory must be a valid directory "
            "path. Will be created if it doesn't exist."
        },
    )

    @field_validator("checkpoint_interval")
    @classmethod
    def validate_checkpoint_interval(cls, v):
        """Provide performance warnings for extreme checkpoint intervals."""
        if v > 500:
            import warnings

            warnings.warn(
                f"Large checkpoint interval ({v}) may result in "
                f"significant data loss on interruption. Consider using "
                f"50-100 for better safety.",
                UserWarning,
            )
        elif v < 10:
            import warnings

            warnings.warn(
                f"Small checkpoint interval ({v}) may slow overall "
                f"processing due to frequent saves. Consider using 25-50 "
                f"for better performance.",
                UserWarning,
            )
        return v

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def validate_checkpoint_dir(cls, v):
        """Ensure checkpoint directory is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v

    @model_validator(mode="after")
    def validate_ai_engine_requires_key(self):
        """Validate that cloud AI engines have corresponding API keys."""
        ai_engine = self.processing.ai_engine

        if ai_engine == "claude" and not self.ai.claude_api_key:
            raise ValueError("Claude AI engine selected but no API key provided")
        elif ai_engine == "openai" and not self.ai.openai_api_key:
            raise ValueError("OpenAI AI engine selected but no API key provided")

        return self


class ConfigurationManager:
    """Manages loading and validation of configuration from multiple sources."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file (TOML or JSON)
        """
        self._config: Optional[BookmarkConfig] = None
        self._load_configuration(config_path)

    def _get_default_config_paths(self) -> list[Path]:
        """Get list of default configuration file paths to try."""
        if getattr(sys, "frozen", False):
            # Running as PyInstaller executable
            app_dir = Path(sys.executable).parent
            base_paths = [
                app_dir / "config" / "user_config.toml",
                app_dir / "config" / "user_config.json",
                app_dir / "bookmark_config.toml",
                app_dir / "bookmark_config.json",
            ]
        else:
            # Running as Python script
            config_dir = Path(__file__).parent
            project_root = config_dir.parent.parent
            base_paths = [
                config_dir / "user_config.toml",
                config_dir / "user_config.json",
                project_root / "bookmark_config.toml",
                project_root / "bookmark_config.json",
            ]

        return base_paths

    def _load_configuration(self, config_path: Optional[Path] = None) -> None:
        """Load configuration from file or use defaults."""
        config_data = {}

        # Try to load from specified path or default locations
        if config_path:
            config_data = self._load_config_file(config_path)
        else:
            for path in self._get_default_config_paths():
                if path.exists():
                    config_data = self._load_config_file(path)
                    break

        # Load API keys from environment variables if not in config
        self._load_api_keys_from_env(config_data)

        # Create and validate configuration
        try:
            self._config = BookmarkConfig(**config_data)
        except ValidationError as e:
            raise ValueError(format_config_error(e))
        except Exception as e:
            raise ValueError(format_config_error(e))

    def _load_config_file(self, config_path: Path) -> Dict:
        """Load configuration from TOML or JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            if config_path.suffix.lower() == ".toml":
                return toml.load(config_path)
            elif config_path.suffix.lower() == ".json":
                with open(config_path, "r") as f:
                    return json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def _load_api_keys_from_env(self, config_data: Dict) -> None:
        """Load API keys from environment variables as fallback."""
        if "ai" not in config_data:
            config_data["ai"] = {}

        # Check for API keys in environment variables
        claude_key = os.getenv("CLAUDE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if claude_key and "claude_api_key" not in config_data["ai"]:
            config_data["ai"]["claude_api_key"] = claude_key

        if openai_key and "openai_api_key" not in config_data["ai"]:
            config_data["ai"]["openai_api_key"] = openai_key

    def update_from_cli_args(self, args: Dict) -> None:
        """Update configuration from command-line arguments."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        # Create new config data with CLI overrides
        config_dict = self._config.dict()

        # Update processing settings
        if "batch_size" in args:
            config_dict["processing"]["batch_size"] = args["batch_size"]

        if "ai_engine" in args and args["ai_engine"]:
            config_dict["processing"]["ai_engine"] = args["ai_engine"]

        # Update network settings
        if "max_retries" in args:
            config_dict["network"]["max_retries"] = args["max_retries"]

        # Update checkpoint settings
        if args.get("clear_checkpoints"):
            config_dict["checkpoint_enabled"] = False
        elif args.get("resume"):
            config_dict["checkpoint_enabled"] = True

        # Recreate configuration with updates
        self._config = BookmarkConfig(**config_dict)

    @property
    def config(self) -> BookmarkConfig:
        """Get the current configuration."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider, returning the actual secret value."""
        if provider == "claude" and self.config.ai.claude_api_key:
            return self.config.ai.claude_api_key.get_secret_value()
        elif provider == "openai" and self.config.ai.openai_api_key:
            return self.config.ai.openai_api_key.get_secret_value()
        return None

    def has_api_key(self, provider: str) -> bool:
        """Check if API key is configured for provider."""
        return self.get_api_key(provider) is not None

    def validate_ai_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate AI configuration."""
        ai_engine = self.config.processing.ai_engine

        if ai_engine == "local":
            return True, None
        elif ai_engine == "claude":
            if self.has_api_key("claude"):
                return True, None
            else:
                return False, "Claude AI engine selected but no API key provided"
        elif ai_engine == "openai":
            if self.has_api_key("openai"):
                return True, None
            else:
                return False, "OpenAI AI engine selected but no API key provided"
        else:
            return False, f"Unknown AI engine: {ai_engine}"

    def create_sample_config(self, output_path: Path, format: str = "toml") -> None:
        """Create a sample configuration file."""
        sample_config = {
            "processing": {
                "ai_engine": "local",
                "batch_size": 100,
                "max_description_length": 150,
            },
            "network": {"timeout": 30, "max_retries": 3, "concurrent_requests": 10},
            "ai": {
                "claude_rpm": 50,
                "openai_rpm": 60,
                "cost_confirmation_interval": 10.0,
                # Note: API keys should be added manually and not
                # committed to version control
                "claude_api_key": "your-claude-api-key-here",
                "openai_api_key": "your-openai-api-key-here",
            },
            "output": {"format": "raindrop_import", "detailed_errors": True},
            "checkpoint_enabled": True,
            "checkpoint_interval": 50,
            "checkpoint_dir": ".bookmark_checkpoints",
        }

        if format.lower() == "toml":
            with open(output_path, "w") as f:
                toml.dump(sample_config, f)
        elif format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(sample_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


class ConfigurationErrorFormatter:
    """Formats Pydantic validation errors into user-friendly messages."""

    @staticmethod
    def format_validation_error(error: ValidationError) -> str:
        """
        Convert Pydantic ValidationError into a user-friendly error message.

        Args:
            error: Pydantic ValidationError instance

        Returns:
            Formatted error message with helpful guidance
        """
        error_messages = []

        for error_detail in error.errors():
            location = ConfigurationErrorFormatter._format_error_location(
                error_detail["loc"]
            )
            error_type = error_detail["type"]
            input_value = error_detail.get("input", "N/A")

            # Try to get custom error message from field schema
            custom_msg = ConfigurationErrorFormatter._get_custom_error_message(
                error_detail, error_type
            )

            if custom_msg:
                formatted_msg = f"❌ {location}: {custom_msg}"
            else:
                # Generate user-friendly message based on error type
                formatted_msg = ConfigurationErrorFormatter._format_by_error_type(
                    location, error_type, error_detail, input_value
                )

            error_messages.append(formatted_msg)

        header = "🔧 Configuration Validation Failed:\n"
        separator = "\n" + "─" * 60 + "\n"

        formatted_errors = "\n".join(error_messages)

        footer = (
            "\n\n💡 Tips:\n"
            "• Check the configuration file format (TOML or JSON)\n"
            "• Verify API keys are not placeholder values\n"
            "• Ensure numeric values are within recommended ranges\n"
            "• Use 'bookmark-processor --create-config' to generate a sample file"
        )

        return header + separator + formatted_errors + footer

    @staticmethod
    def _format_error_location(location: tuple) -> str:
        """Format the error location path."""
        if not location:
            return "Configuration"

        path_parts = []
        for part in location:
            if isinstance(part, str):
                path_parts.append(part)
            else:
                path_parts.append(f"[{part}]")

        return " → ".join(path_parts)

    @staticmethod
    def _get_custom_error_message(error_detail: dict, error_type: str) -> Optional[str]:
        """Try to extract custom error message from field schema."""
        # This would need to be enhanced to access the field schema
        # For now, return None to use default formatting
        return None

    @staticmethod
    def _format_by_error_type(
        location: str, error_type: str, error_detail: dict, input_value
    ) -> str:
        """Format error message based on Pydantic error type."""

        if error_type == "missing":
            return f"❌ {location}: Required field is missing"

        elif error_type == "value_error":
            msg = error_detail.get("msg", "Invalid value")
            return f"❌ {location}: {msg} (got: {input_value})"

        elif error_type == "type_error":
            expected_type = error_detail.get("msg", "correct type")
            return (
                f"❌ {location}: Expected {expected_type} "
                f"(got: {type(input_value).__name__})"
            )

        elif error_type in [
            "greater_than_equal",
            "less_than_equal",
            "greater_than",
            "less_than",
        ]:
            limit = error_detail.get("ctx", {}).get("limit_value", "limit")
            operator = {
                "greater_than_equal": "≥",
                "less_than_equal": "≤",
                "greater_than": ">",
                "less_than": "<",
            }.get(error_type, "?")
            return (
                f"❌ {location}: Value must be {operator} {limit} (got: {input_value})"
            )

        elif error_type == "literal_error":
            expected = error_detail.get("ctx", {}).get("expected", "valid option")
            return f"❌ {location}: Must be one of {expected} (got: {input_value})"

        elif error_type == "string_too_short":
            min_length = error_detail.get("ctx", {}).get("min_length", "minimum")
            return (
                f"❌ {location}: String too short, minimum {min_length} "
                f"characters (got: {len(str(input_value))})"
            )

        elif error_type == "string_too_long":
            max_length = error_detail.get("ctx", {}).get("max_length", "maximum")
            return (
                f"❌ {location}: String too long, maximum {max_length} "
                f"characters (got: {len(str(input_value))})"
            )

        elif (
            "api_key" in location.lower()
            and "placeholder" in str(input_value).lower()
        ):
            provider = "Claude" if "claude" in location.lower() else "OpenAI"
            return (
                f"❌ {location}: Please replace placeholder with your "
                f"actual {provider} API key\n"
                f"   💡 Get your API key from the {provider} dashboard and "
                f"update your configuration file"
            )

        else:
            # Generic fallback
            msg = error_detail.get("msg", "Invalid configuration value")
            return f"❌ {location}: {msg} (got: {input_value})"


def format_config_error(error: Exception) -> str:
    """
    Format any configuration-related error into a user-friendly message.

    Args:
        error: Exception that occurred during configuration

    Returns:
        Formatted error message
    """
    if isinstance(error, ValidationError):
        return ConfigurationErrorFormatter.format_validation_error(error)

    elif isinstance(error, FileNotFoundError):
        return (
            f"🔧 Configuration File Not Found:\n"
            f"❌ Could not find configuration file: {error.filename}\n\n"
            f"💡 Solutions:\n"
            f"• Create a configuration file using: bookmark-processor --create-config\n"
            f"• Use default configuration by omitting the --config parameter\n"
            f"• Check the file path is correct and accessible"
        )

    elif isinstance(error, ValueError) and "configuration" in str(error).lower():
        return (
            f"🔧 Configuration Error:\n"
            f"❌ {str(error)}\n\n"
            f"💡 Tips:\n"
            f"• Check your configuration file syntax\n"
            f"• Verify all required values are provided\n"
            f"• Use sample configuration as reference"
        )

    else:
        return f"🔧 Unexpected Configuration Error:\n❌ {str(error)}"
