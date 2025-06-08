"""
Configuration migration utility for converting INI files to TOML/JSON format.

This utility helps users migrate from the old INI-based configuration system
to the new Pydantic-based TOML/JSON configuration system.
"""

import configparser
import json
import toml
from pathlib import Path
from typing import Dict, Any


class ConfigurationMigrator:
    """Utility for migrating INI configuration files to TOML/JSON format."""

    def __init__(self):
        """Initialize the configuration migrator."""
        self.ini_to_new_mapping = {
            # Network section mappings
            "network.timeout": "network.timeout",
            "network.max_retries": "network.max_retries",
            "network.max_concurrent_requests": "network.concurrent_requests",
            # Processing section mappings
            "processing.batch_size": "processing.batch_size",
            "processing.max_description_length": "processing.max_description_length",
            "processing.ai_model": None,  # Removed in new system
            "processing.use_existing_content": None,  # Now always true
            "processing.max_tags_per_bookmark": None,  # Simplified
            "processing.target_unique_tags": None,  # Simplified
            # AI section mappings
            "ai.default_engine": "processing.ai_engine",
            "ai.claude_api_key": "ai.claude_api_key",
            "ai.openai_api_key": "ai.openai_api_key",
            "ai.claude_rpm": "ai.claude_rpm",
            "ai.openai_rpm": "ai.openai_rpm",
            "ai.cost_confirmation_interval": "ai.cost_confirmation_interval",
            # Checkpoint section mappings
            "checkpoint.enabled": "checkpoint_enabled",
            "checkpoint.save_interval": "checkpoint_interval",
            "checkpoint.checkpoint_dir": "checkpoint_dir",
            "checkpoint.auto_cleanup": None,  # Always true now
            # Output section mappings
            "output.output_format": "output.format",
            "output.error_log_detailed": "output.detailed_errors",
            "output.preserve_folder_structure": None,  # Always true now
            "output.include_timestamps": None,  # Always true now
            # Logging section - not needed in new system (simplified)
            "logging.log_level": None,
            "logging.log_file": None,
            "logging.console_output": None,
            "logging.performance_logging": None,
            # Executable section - not needed in new system
            "executable.model_cache_dir": None,
            "executable.temp_dir": None,
            "executable.cleanup_on_exit": None,
        }

    def migrate_ini_to_dict(self, ini_path: Path) -> Dict[str, Any]:
        """
        Migrate INI configuration file to dictionary format.

        Args:
            ini_path: Path to the INI configuration file

        Returns:
            Dictionary with new configuration structure
        """
        if not ini_path.exists():
            raise FileNotFoundError(f"INI file not found: {ini_path}")

        # Load INI file
        config = configparser.ConfigParser()
        config.read(ini_path)

        # Initialize new configuration structure
        new_config = {"processing": {}, "network": {}, "ai": {}, "output": {}}

        # Migrate each section and option
        for section in config.sections():
            for option in config.options(section):
                old_key = f"{section}.{option}"
                new_key = self.ini_to_new_mapping.get(old_key)

                if new_key is None:
                    # Skip options that are no longer needed
                    continue

                value = config.get(section, option)

                # Convert value to appropriate type
                converted_value = self._convert_value(value, old_key)

                # Set value in new structure
                self._set_nested_value(new_config, new_key, converted_value)

        # Set default values for required fields not in INI
        self._set_defaults(new_config)

        return new_config

    def _convert_value(self, value: str, key: str) -> Any:
        """Convert string value to appropriate type."""
        # Boolean conversions
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer conversions
        if key in [
            "network.timeout",
            "network.max_retries",
            "network.concurrent_requests",
            "processing.batch_size",
            "processing.max_description_length",
            "ai.claude_rpm",
            "ai.openai_rpm",
            "checkpoint_interval",
        ]:
            try:
                return int(value)
            except ValueError:
                pass

        # Float conversions
        if key in ["ai.cost_confirmation_interval"]:
            try:
                return float(value)
            except ValueError:
                pass

        # String conversions (default)
        return value

    def _set_nested_value(self, config: Dict, key: str, value: Any) -> None:
        """Set a nested value in the configuration dictionary."""
        if "." in key:
            section, option = key.split(".", 1)
            if section not in config:
                config[section] = {}
            config[section][option] = value
        else:
            config[key] = value

    def _set_defaults(self, config: Dict) -> None:
        """Set default values for required fields."""
        # Ensure all required sections exist
        for section in ["processing", "network", "ai", "output"]:
            if section not in config:
                config[section] = {}

        # Set processing defaults
        if "ai_engine" not in config["processing"]:
            config["processing"]["ai_engine"] = "local"
        if "batch_size" not in config["processing"]:
            config["processing"]["batch_size"] = 100
        if "max_description_length" not in config["processing"]:
            config["processing"]["max_description_length"] = 150

        # Set network defaults
        if "timeout" not in config["network"]:
            config["network"]["timeout"] = 30
        if "max_retries" not in config["network"]:
            config["network"]["max_retries"] = 3
        if "concurrent_requests" not in config["network"]:
            config["network"]["concurrent_requests"] = 10

        # Set AI defaults
        if "claude_rpm" not in config["ai"]:
            config["ai"]["claude_rpm"] = 50
        if "openai_rpm" not in config["ai"]:
            config["ai"]["openai_rpm"] = 60
        if "cost_confirmation_interval" not in config["ai"]:
            config["ai"]["cost_confirmation_interval"] = 10.0

        # Set output defaults
        if "format" not in config["output"]:
            config["output"]["format"] = "raindrop_import"
        if "detailed_errors" not in config["output"]:
            config["output"]["detailed_errors"] = True

        # Set checkpoint defaults
        if "checkpoint_enabled" not in config:
            config["checkpoint_enabled"] = True
        if "checkpoint_interval" not in config:
            config["checkpoint_interval"] = 50
        if "checkpoint_dir" not in config:
            config["checkpoint_dir"] = ".bookmark_checkpoints"

    def migrate_to_toml(self, ini_path: Path, toml_path: Path) -> None:
        """
        Migrate INI file to TOML format.

        Args:
            ini_path: Path to source INI file
            toml_path: Path to destination TOML file
        """
        config_dict = self.migrate_ini_to_dict(ini_path)

        with open(toml_path, "w") as f:
            f.write("# Migrated from INI configuration\n")
            f.write("# Please review and update API keys as needed\n\n")
            toml.dump(config_dict, f)

        print(f"✓ Successfully migrated {ini_path} to {toml_path}")

    def migrate_to_json(self, ini_path: Path, json_path: Path) -> None:
        """
        Migrate INI file to JSON format.

        Args:
            ini_path: Path to source INI file
            json_path: Path to destination JSON file
        """
        config_dict = self.migrate_ini_to_dict(ini_path)

        # Add metadata
        config_dict["_migration_info"] = {
            "migrated_from": str(ini_path),
            "note": "Please review and update API keys as needed",
        }

        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"✓ Successfully migrated {ini_path} to {json_path}")

    def create_migration_report(self, ini_path: Path) -> str:
        """
        Create a migration report showing what will be migrated.

        Args:
            ini_path: Path to INI file to analyze

        Returns:
            Migration report as string
        """
        if not ini_path.exists():
            return f"❌ INI file not found: {ini_path}"

        config = configparser.ConfigParser()
        config.read(ini_path)

        report = [f"Migration Report for: {ini_path}", "=" * 50, ""]

        migrated_count = 0
        removed_count = 0

        for section in config.sections():
            report.append(f"[{section}]")
            for option in config.options(section):
                old_key = f"{section}.{option}"
                new_key = self.ini_to_new_mapping.get(old_key)

                if new_key is None:
                    report.append(
                        f"  ❌ {option} = {config.get(section, option)} (REMOVED - no longer needed)"
                    )
                    removed_count += 1
                else:
                    report.append(
                        f"  ✓ {option} = {config.get(section, option)} → {new_key}"
                    )
                    migrated_count += 1
            report.append("")

        report.extend(
            [
                "Summary:",
                f"  ✓ {migrated_count} options will be migrated",
                f"  ❌ {removed_count} options will be removed (no longer needed)",
                "",
                "Note: Removed options are part of the simplification to 15 essential options.",
            ]
        )

        return "\n".join(report)


def main():
    """Command-line interface for migration utility."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate INI configuration files to TOML/JSON format"
    )
    parser.add_argument("ini_file", help="Path to INI configuration file")
    parser.add_argument(
        "--format",
        choices=["toml", "json"],
        default="toml",
        help="Output format (default: toml)",
    )
    parser.add_argument(
        "--output", help="Output file path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Show migration report without performing migration",
    )

    args = parser.parse_args()

    ini_path = Path(args.ini_file)
    migrator = ConfigurationMigrator()

    if args.report_only:
        print(migrator.create_migration_report(ini_path))
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ini_path.with_suffix(f".{args.format}")

    # Perform migration
    try:
        if args.format == "toml":
            migrator.migrate_to_toml(ini_path, output_path)
        else:
            migrator.migrate_to_json(ini_path, output_path)

        print(f"\n✓ Migration completed successfully!")
        print(f"Please review {output_path} and add your API keys as needed.")

    except Exception as e:
        print(f"❌ Migration failed: {e}")


if __name__ == "__main__":
    main()
