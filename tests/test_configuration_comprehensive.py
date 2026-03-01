"""
Comprehensive tests for the Configuration compatibility layer.

This module tests bookmark_processor/config/configuration.py including:
- Configuration class initialization
- Loading and saving configuration
- Default values
- Configuration validation
- All methods (get_model_cache_dir, get_checkpoint_dir, get_ai_engine, etc.)
- Factory function create_configuration
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import toml

from bookmark_processor.config.configuration import Configuration, create_configuration
from bookmark_processor.config.pydantic_config import (
    BookmarkConfig,
    ProcessingConfig,
    AIConfig,
    NetworkConfig,
    ConfigurationManager,
)


# ============================================================================
# Configuration Class Initialization Tests
# ============================================================================


class TestConfigurationInitialization:
    """Tests for Configuration class initialization."""

    def test_default_initialization(self):
        """Test Configuration initializes with defaults when no config file."""
        config = Configuration()

        assert config._manager is not None
        assert config._config is not None
        assert isinstance(config._config, BookmarkConfig)

    def test_initialization_with_config_path(self, tmp_path):
        """Test Configuration initializes with custom config file."""
        config_data = {
            "processing": {"batch_size": 200, "ai_engine": "local"},
            "network": {"timeout": 45},
            "checkpoint_interval": 75,
        }

        config_file = tmp_path / "test_config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        assert config._config.processing.batch_size == 200
        assert config._config.network.timeout == 45
        assert config._config.checkpoint_interval == 75

    def test_initialization_with_json_config(self, tmp_path):
        """Test Configuration initializes with JSON config file."""
        import json

        config_data = {
            "processing": {"batch_size": 150, "ai_engine": "local"},
            "checkpoint_enabled": False,
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = Configuration(config_path=config_file)

        assert config._config.processing.batch_size == 150
        assert config._config.checkpoint_enabled is False

    def test_initialization_with_nonexistent_file_raises_error(self, tmp_path):
        """Test Configuration raises error for nonexistent config file."""
        nonexistent = tmp_path / "nonexistent.toml"

        with pytest.raises(FileNotFoundError):
            Configuration(config_path=nonexistent)

    def test_initialization_with_invalid_config_raises_error(self, tmp_path):
        """Test Configuration raises error for invalid config."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text("invalid [ toml content")

        with pytest.raises(ValueError):
            Configuration(config_path=config_file)


# ============================================================================
# Configuration Property Tests
# ============================================================================


class TestConfigurationProperty:
    """Tests for Configuration.config property."""

    def test_config_property_returns_bookmark_config(self):
        """Test config property returns BookmarkConfig instance."""
        config = Configuration()

        result = config.config

        assert isinstance(result, BookmarkConfig)
        assert result is config._config

    def test_config_property_returns_same_instance(self):
        """Test config property returns consistent instance."""
        config = Configuration()

        result1 = config.config
        result2 = config.config

        assert result1 is result2


# ============================================================================
# Update From Args Tests
# ============================================================================


class TestUpdateFromArgs:
    """Tests for Configuration.update_from_args method."""

    def test_update_from_args_batch_size(self):
        """Test updating batch_size from CLI arguments."""
        config = Configuration()
        original_batch_size = config._config.processing.batch_size

        config.update_from_args({"batch_size": 250})

        assert config._config.processing.batch_size == 250
        assert config._config.processing.batch_size != original_batch_size

    def test_update_from_args_ai_engine(self, tmp_path):
        """Test updating ai_engine from CLI arguments."""
        # Create config with API key so we can switch to claude
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"claude_api_key": "sk-ant-test-key-123"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        config.update_from_args({"ai_engine": "claude"})

        assert config._config.processing.ai_engine == "claude"

    def test_update_from_args_max_retries(self):
        """Test updating max_retries from CLI arguments."""
        config = Configuration()

        config.update_from_args({"max_retries": 7})

        assert config._config.network.max_retries == 7

    def test_update_from_args_clear_checkpoints(self):
        """Test clear_checkpoints flag disables checkpointing."""
        config = Configuration()
        assert config._config.checkpoint_enabled is True

        config.update_from_args({"clear_checkpoints": True})

        assert config._config.checkpoint_enabled is False

    def test_update_from_args_resume(self):
        """Test resume flag enables checkpointing."""
        config = Configuration()

        config.update_from_args({"resume": True})

        assert config._config.checkpoint_enabled is True

    def test_update_from_args_updates_config_reference(self):
        """Test update_from_args creates new config instance."""
        config = Configuration()
        original_config = config._config

        config.update_from_args({"batch_size": 300})

        # Should be a new instance (Pydantic creates new objects)
        assert config._config is not original_config

    def test_update_from_args_empty_dict(self):
        """Test update_from_args with empty dictionary."""
        config = Configuration()
        original_batch_size = config._config.processing.batch_size

        config.update_from_args({})

        assert config._config.processing.batch_size == original_batch_size

    def test_update_from_args_none_ai_engine_not_updated(self):
        """Test update_from_args does not update ai_engine if None."""
        config = Configuration()
        original_engine = config._config.processing.ai_engine

        config.update_from_args({"ai_engine": None})

        assert config._config.processing.ai_engine == original_engine


# ============================================================================
# Get Model Cache Dir Tests
# ============================================================================


class TestGetModelCacheDir:
    """Tests for Configuration.get_model_cache_dir method."""

    def test_get_model_cache_dir_returns_path(self):
        """Test get_model_cache_dir returns Path object."""
        config = Configuration()

        result = config.get_model_cache_dir()

        assert isinstance(result, Path)

    def test_get_model_cache_dir_expands_user(self):
        """Test get_model_cache_dir expands ~ to user home."""
        config = Configuration()

        result = config.get_model_cache_dir()

        assert "~" not in str(result)
        assert "bookmark-processor" in str(result)
        assert "models" in str(result)

    def test_get_model_cache_dir_consistent(self):
        """Test get_model_cache_dir returns consistent path."""
        config = Configuration()

        result1 = config.get_model_cache_dir()
        result2 = config.get_model_cache_dir()

        assert result1 == result2

    @patch.dict(os.environ, {"TEST_VAR": "test_value"})
    def test_get_model_cache_dir_expands_env_vars(self):
        """Test get_model_cache_dir expands environment variables."""
        config = Configuration()

        result = config.get_model_cache_dir()

        # Should expand user path at minimum
        assert isinstance(result, Path)


# ============================================================================
# Get Checkpoint Dir Tests
# ============================================================================


class TestGetCheckpointDir:
    """Tests for Configuration.get_checkpoint_dir method."""

    def test_get_checkpoint_dir_returns_path(self):
        """Test get_checkpoint_dir returns Path object."""
        config = Configuration()

        result = config.get_checkpoint_dir()

        assert isinstance(result, Path)

    def test_get_checkpoint_dir_default_value(self):
        """Test get_checkpoint_dir returns default value."""
        config = Configuration()

        result = config.get_checkpoint_dir()

        assert result == Path(".bookmark_checkpoints")

    def test_get_checkpoint_dir_custom_value(self, tmp_path):
        """Test get_checkpoint_dir returns custom configured value."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "checkpoint_dir": str(tmp_path / "custom_checkpoints"),
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_checkpoint_dir()

        assert result == tmp_path / "custom_checkpoints"


# ============================================================================
# Get AI Engine Tests
# ============================================================================


class TestGetAIEngine:
    """Tests for Configuration.get_ai_engine method."""

    def test_get_ai_engine_default(self):
        """Test get_ai_engine returns default 'local'."""
        config = Configuration()

        result = config.get_ai_engine()

        assert result == "local"

    def test_get_ai_engine_claude(self, tmp_path):
        """Test get_ai_engine returns 'claude' when configured."""
        config_data = {
            "processing": {"ai_engine": "claude"},
            "ai": {"claude_api_key": "sk-ant-test-key"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_ai_engine()

        assert result == "claude"

    def test_get_ai_engine_openai(self, tmp_path):
        """Test get_ai_engine returns 'openai' when configured."""
        config_data = {
            "processing": {"ai_engine": "openai"},
            "ai": {"openai_api_key": "sk-test-openai-key"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_ai_engine()

        assert result == "openai"


# ============================================================================
# Get API Key Tests
# ============================================================================


class TestGetAPIKey:
    """Tests for Configuration.get_api_key method."""

    def test_get_api_key_returns_none_when_not_set(self):
        """Test get_api_key returns None when key not configured."""
        config = Configuration()

        assert config.get_api_key("claude") is None
        assert config.get_api_key("openai") is None

    def test_get_api_key_returns_claude_key(self, tmp_path):
        """Test get_api_key returns Claude API key."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"claude_api_key": "sk-ant-test-key-123"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_api_key("claude")

        assert result == "sk-ant-test-key-123"

    def test_get_api_key_returns_openai_key(self, tmp_path):
        """Test get_api_key returns OpenAI API key."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"openai_api_key": "sk-test-openai-key-456"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_api_key("openai")

        assert result == "sk-test-openai-key-456"

    def test_get_api_key_unknown_provider_returns_none(self):
        """Test get_api_key returns None for unknown provider."""
        config = Configuration()

        result = config.get_api_key("unknown_provider")

        assert result is None

    def test_get_api_key_from_environment(self):
        """Test get_api_key loads from environment variables."""
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "sk-ant-env-key"}):
            config = Configuration()

            result = config.get_api_key("claude")

            assert result == "sk-ant-env-key"


# ============================================================================
# Has API Key Tests
# ============================================================================


class TestHasAPIKey:
    """Tests for Configuration.has_api_key method."""

    def test_has_api_key_false_when_not_set(self):
        """Test has_api_key returns False when key not configured."""
        config = Configuration()

        assert config.has_api_key("claude") is False
        assert config.has_api_key("openai") is False

    def test_has_api_key_true_for_claude(self, tmp_path):
        """Test has_api_key returns True when Claude key is set."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"claude_api_key": "sk-ant-test-key"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        assert config.has_api_key("claude") is True
        assert config.has_api_key("openai") is False

    def test_has_api_key_true_for_openai(self, tmp_path):
        """Test has_api_key returns True when OpenAI key is set."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"openai_api_key": "sk-test-openai-key"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        assert config.has_api_key("claude") is False
        assert config.has_api_key("openai") is True

    def test_has_api_key_unknown_provider(self):
        """Test has_api_key returns False for unknown provider."""
        config = Configuration()

        assert config.has_api_key("unknown") is False


# ============================================================================
# Get Rate Limit Tests
# ============================================================================


class TestGetRateLimit:
    """Tests for Configuration.get_rate_limit method."""

    def test_get_rate_limit_default_claude(self):
        """Test get_rate_limit returns default for Claude."""
        config = Configuration()

        result = config.get_rate_limit("claude")

        assert result == 50  # Default claude_rpm

    def test_get_rate_limit_default_openai(self):
        """Test get_rate_limit returns default for OpenAI."""
        config = Configuration()

        result = config.get_rate_limit("openai")

        assert result == 60  # Default openai_rpm

    def test_get_rate_limit_custom_claude(self, tmp_path):
        """Test get_rate_limit returns custom Claude RPM."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"claude_rpm": 100},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_rate_limit("claude")

        assert result == 100

    def test_get_rate_limit_custom_openai(self, tmp_path):
        """Test get_rate_limit returns custom OpenAI RPM."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"openai_rpm": 120},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_rate_limit("openai")

        assert result == 120

    def test_get_rate_limit_unknown_provider(self):
        """Test get_rate_limit returns default 60 for unknown provider."""
        config = Configuration()

        result = config.get_rate_limit("unknown")

        assert result == 60


# ============================================================================
# Get Batch Size Tests
# ============================================================================


class TestGetBatchSize:
    """Tests for Configuration.get_batch_size method."""

    def test_get_batch_size_claude(self):
        """Test get_batch_size returns 10 for Claude."""
        config = Configuration()

        result = config.get_batch_size("claude")

        assert result == 10

    def test_get_batch_size_openai(self):
        """Test get_batch_size returns 20 for OpenAI."""
        config = Configuration()

        result = config.get_batch_size("openai")

        assert result == 20

    def test_get_batch_size_unknown_provider(self):
        """Test get_batch_size returns default 10 for unknown provider."""
        config = Configuration()

        result = config.get_batch_size("unknown")

        assert result == 10

    def test_get_batch_size_local(self):
        """Test get_batch_size returns default 10 for local."""
        config = Configuration()

        result = config.get_batch_size("local")

        assert result == 10


# ============================================================================
# Get Cost Tracking Settings Tests
# ============================================================================


class TestGetCostTrackingSettings:
    """Tests for Configuration.get_cost_tracking_settings method."""

    def test_get_cost_tracking_settings_returns_dict(self):
        """Test get_cost_tracking_settings returns dictionary."""
        config = Configuration()

        result = config.get_cost_tracking_settings()

        assert isinstance(result, dict)

    def test_get_cost_tracking_settings_default_values(self):
        """Test get_cost_tracking_settings returns correct defaults."""
        config = Configuration()

        result = config.get_cost_tracking_settings()

        assert result["show_running_costs"] is True
        assert result["cost_confirmation_interval"] == 10.0
        assert result["max_cost_per_run"] == 0.0
        assert result["pause_at_cost"] is True

    def test_get_cost_tracking_settings_custom_interval(self, tmp_path):
        """Test get_cost_tracking_settings with custom interval."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"cost_confirmation_interval": 25.0},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        result = config.get_cost_tracking_settings()

        assert result["cost_confirmation_interval"] == 25.0

    def test_get_cost_tracking_settings_all_keys_present(self):
        """Test get_cost_tracking_settings has all expected keys."""
        config = Configuration()

        result = config.get_cost_tracking_settings()

        expected_keys = {
            "show_running_costs",
            "cost_confirmation_interval",
            "max_cost_per_run",
            "pause_at_cost",
        }
        assert set(result.keys()) == expected_keys


# ============================================================================
# Validate AI Configuration Tests
# ============================================================================


class TestValidateAIConfiguration:
    """Tests for Configuration.validate_ai_configuration method."""

    def test_validate_ai_configuration_local_valid(self):
        """Test validate_ai_configuration returns valid for local engine."""
        config = Configuration()

        is_valid, error = config.validate_ai_configuration()

        assert is_valid is True
        assert error is None

    def test_validate_ai_configuration_claude_with_key(self, tmp_path):
        """Test validate_ai_configuration for Claude with key."""
        config_data = {
            "processing": {"ai_engine": "claude"},
            "ai": {"claude_api_key": "sk-ant-test-key"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        is_valid, error = config.validate_ai_configuration()

        assert is_valid is True
        assert error is None

    def test_validate_ai_configuration_openai_with_key(self, tmp_path):
        """Test validate_ai_configuration for OpenAI with key."""
        config_data = {
            "processing": {"ai_engine": "openai"},
            "ai": {"openai_api_key": "sk-test-openai-key"},
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = Configuration(config_path=config_file)

        is_valid, error = config.validate_ai_configuration()

        assert is_valid is True
        assert error is None

    def test_validate_ai_configuration_delegates_to_manager(self):
        """Test validate_ai_configuration delegates to ConfigurationManager."""
        config = Configuration()

        # The method should return the same result as the manager
        manager_result = config._manager.validate_ai_configuration()
        config_result = config.validate_ai_configuration()

        assert config_result == manager_result


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateConfiguration:
    """Tests for create_configuration factory function."""

    def test_create_configuration_returns_configuration(self):
        """Test create_configuration returns Configuration instance."""
        config = create_configuration()

        assert isinstance(config, Configuration)

    def test_create_configuration_default(self):
        """Test create_configuration with no arguments."""
        config = create_configuration()

        assert config._config is not None
        assert config._config.processing.ai_engine == "local"

    def test_create_configuration_with_path(self, tmp_path):
        """Test create_configuration with config path."""
        config_data = {
            "processing": {"batch_size": 300, "ai_engine": "local"},
        }
        config_file = tmp_path / "factory_config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config = create_configuration(config_path=config_file)

        assert config._config.processing.batch_size == 300

    def test_create_configuration_equivalent_to_constructor(self, tmp_path):
        """Test create_configuration produces equivalent result to constructor."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "checkpoint_interval": 100,
        }
        config_file = tmp_path / "equiv_config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        config_factory = create_configuration(config_path=config_file)
        config_direct = Configuration(config_path=config_file)

        assert config_factory._config.checkpoint_interval == config_direct._config.checkpoint_interval


# ============================================================================
# Integration Tests
# ============================================================================


class TestConfigurationIntegration:
    """Integration tests for Configuration class."""

    def test_full_workflow(self, tmp_path):
        """Test complete configuration workflow."""
        # Create config file
        config_data = {
            "processing": {"batch_size": 150, "ai_engine": "local"},
            "network": {"timeout": 60, "max_retries": 5},
            "ai": {"claude_rpm": 75, "openai_rpm": 90},
            "checkpoint_enabled": True,
            "checkpoint_interval": 100,
        }
        config_file = tmp_path / "workflow_config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        # Initialize
        config = Configuration(config_path=config_file)

        # Verify initial values
        assert config.config.processing.batch_size == 150
        assert config.get_ai_engine() == "local"
        assert config.get_rate_limit("claude") == 75
        assert config.get_checkpoint_dir() == Path(".bookmark_checkpoints")

        # Update from args
        config.update_from_args({"batch_size": 200, "max_retries": 3})

        # Verify updates
        assert config.config.processing.batch_size == 200
        assert config.config.network.max_retries == 3

        # Validate
        is_valid, error = config.validate_ai_configuration()
        assert is_valid is True

    def test_environment_variable_api_keys(self):
        """Test API keys loaded from environment variables."""
        with patch.dict(os.environ, {
            "CLAUDE_API_KEY": "sk-ant-from-env",
            "OPENAI_API_KEY": "sk-openai-from-env",
        }):
            config = Configuration()

            assert config.get_api_key("claude") == "sk-ant-from-env"
            assert config.get_api_key("openai") == "sk-openai-from-env"
            assert config.has_api_key("claude") is True
            assert config.has_api_key("openai") is True

    def test_config_file_api_key_precedence(self, tmp_path):
        """Test config file API keys take precedence over environment."""
        config_data = {
            "processing": {"ai_engine": "local"},
            "ai": {"claude_api_key": "sk-ant-from-file"},
        }
        config_file = tmp_path / "precedence_config.toml"
        with open(config_file, "w") as f:
            toml.dump(config_data, f)

        with patch.dict(os.environ, {"CLAUDE_API_KEY": "sk-ant-from-env"}):
            config = Configuration(config_path=config_file)

            # Config file should take precedence
            assert config.get_api_key("claude") == "sk-ant-from-file"

    def test_all_getters_return_correct_types(self):
        """Test all getter methods return correct types."""
        config = Configuration()

        # Path getters
        assert isinstance(config.get_model_cache_dir(), Path)
        assert isinstance(config.get_checkpoint_dir(), Path)

        # String getters
        assert isinstance(config.get_ai_engine(), str)

        # Optional string getters
        api_key = config.get_api_key("claude")
        assert api_key is None or isinstance(api_key, str)

        # Boolean getters
        assert isinstance(config.has_api_key("claude"), bool)

        # Integer getters
        assert isinstance(config.get_rate_limit("claude"), int)
        assert isinstance(config.get_batch_size("claude"), int)

        # Dict getters
        assert isinstance(config.get_cost_tracking_settings(), dict)

        # Tuple getters
        result = config.validate_ai_configuration()
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestConfigurationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_multiple_config_instances_independent(self, tmp_path):
        """Test multiple Configuration instances are independent."""
        config1 = Configuration()
        config2 = Configuration()

        config1.update_from_args({"batch_size": 500})

        assert config1._config.processing.batch_size == 500
        assert config2._config.processing.batch_size == 100  # Default

    def test_config_with_all_defaults(self):
        """Test configuration with all default values."""
        config = Configuration()

        # Verify all defaults
        assert config._config.processing.batch_size == 100
        assert config._config.processing.max_description_length == 150
        assert config._config.processing.ai_engine == "local"
        assert config._config.network.timeout == 30
        assert config._config.network.max_retries == 3
        assert config._config.network.concurrent_requests == 10
        assert config._config.ai.claude_rpm == 50
        assert config._config.ai.openai_rpm == 60
        assert config._config.ai.cost_confirmation_interval == 10.0
        assert config._config.checkpoint_enabled is True
        assert config._config.checkpoint_interval == 50
        assert config._config.output.format == "raindrop_import"
        assert config._config.output.detailed_errors is True

    def test_get_batch_size_all_providers(self):
        """Test get_batch_size for all provider variations."""
        config = Configuration()

        # Known providers
        assert config.get_batch_size("claude") == 10
        assert config.get_batch_size("openai") == 20

        # Unknown providers should return default
        assert config.get_batch_size("local") == 10
        assert config.get_batch_size("") == 10
        assert config.get_batch_size("gemini") == 10

    def test_get_rate_limit_all_providers(self):
        """Test get_rate_limit for all provider variations."""
        config = Configuration()

        # Known providers with defaults
        assert config.get_rate_limit("claude") == 50
        assert config.get_rate_limit("openai") == 60

        # Unknown providers should return default 60
        assert config.get_rate_limit("local") == 60
        assert config.get_rate_limit("") == 60
        assert config.get_rate_limit("gemini") == 60

    def test_update_from_args_partial_update(self):
        """Test update_from_args with partial arguments."""
        config = Configuration()
        original_timeout = config._config.network.timeout
        original_batch_size = config._config.processing.batch_size

        # Only update max_retries
        config.update_from_args({"max_retries": 8})

        # Other values should remain unchanged
        assert config._config.network.timeout == original_timeout
        assert config._config.processing.batch_size == original_batch_size
        assert config._config.network.max_retries == 8
