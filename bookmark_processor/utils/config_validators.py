"""
Configuration File Validators

This module provides validators for configuration file format and content.
It includes schema validation, type checking, range validation, and hierarchical
validation for nested configuration objects.
"""

import configparser
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .input_validator import (
    Validator,
    ValidationResult,
    ValidationSeverity,
    StringValidator,
    NumberValidator,
    create_required_string,
    create_optional_string
)


class ConfigurationSchema:
    """Defines the expected schema for configuration files"""
    
    def __init__(self):
        self.schema = {
            'network': {
                'timeout': {'type': 'int', 'min': 1, 'max': 300, 'default': 30},
                'max_retries': {'type': 'int', 'min': 0, 'max': 10, 'default': 3},
                'default_delay': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 0.5},
                'max_concurrent_requests': {'type': 'int', 'min': 1, 'max': 50, 'default': 10},
                'user_agent_rotation': {'type': 'bool', 'default': True},
                'google_delay': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 2.0},
                'github_delay': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 1.5},
                'youtube_delay': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 2.0},
                'linkedin_delay': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 2.0},
            },
            'processing': {
                'batch_size': {'type': 'int', 'min': 1, 'max': 1000, 'default': 100},
                'max_tags_per_bookmark': {'type': 'int', 'min': 1, 'max': 20, 'default': 5},
                'target_unique_tags': {'type': 'int', 'min': 10, 'max': 1000, 'default': 150},
                'ai_model': {'type': 'str', 'default': 'facebook/bart-large-cnn'},
                'max_description_length': {'type': 'int', 'min': 50, 'max': 1000, 'default': 150},
                'use_existing_content': {'type': 'bool', 'default': True},
            },
            'checkpoint': {
                'enabled': {'type': 'bool', 'default': True},
                'save_interval': {'type': 'int', 'min': 1, 'max': 1000, 'default': 50},
                'checkpoint_dir': {'type': 'str', 'default': '.bookmark_checkpoints'},
                'auto_cleanup': {'type': 'bool', 'default': True},
            },
            'output': {
                'output_format': {'type': 'str', 'allowed': ['raindrop_import'], 'default': 'raindrop_import'},
                'preserve_folder_structure': {'type': 'bool', 'default': True},
                'include_timestamps': {'type': 'bool', 'default': True},
                'error_log_detailed': {'type': 'bool', 'default': True},
            },
            'logging': {
                'log_level': {'type': 'str', 'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 'default': 'INFO'},
                'log_file': {'type': 'str', 'default': 'bookmark_processor.log'},
                'console_output': {'type': 'bool', 'default': True},
                'performance_logging': {'type': 'bool', 'default': True},
            },
            'ai': {
                'default_engine': {'type': 'str', 'allowed': ['local', 'claude', 'openai'], 'default': 'local'},
                'claude_api_key': {'type': 'str', 'optional': True, 'sensitive': True},
                'openai_api_key': {'type': 'str', 'optional': True, 'sensitive': True},
                'claude_rpm': {'type': 'int', 'min': 1, 'max': 1000, 'default': 50},
                'openai_rpm': {'type': 'int', 'min': 1, 'max': 1000, 'default': 60},
                'claude_batch_size': {'type': 'int', 'min': 1, 'max': 100, 'default': 10},
                'openai_batch_size': {'type': 'int', 'min': 1, 'max': 100, 'default': 20},
                'show_running_costs': {'type': 'bool', 'default': True},
                'cost_confirmation_interval': {'type': 'float', 'min': 0.0, 'max': 1000.0, 'default': 10.0},
                'max_cost_per_run': {'type': 'float', 'min': 0.0, 'max': 10000.0, 'default': 0.0, 'optional': True},
                'pause_at_cost': {'type': 'bool', 'default': True, 'optional': True},
            },
            'executable': {
                'model_cache_dir': {'type': 'str', 'default': '~/.cache/bookmark-processor/models'},
                'temp_dir': {'type': 'str', 'default': '/tmp/bookmark-processor'},
                'cleanup_on_exit': {'type': 'bool', 'default': True},
            }
        }
        
        self.required_sections = {'network', 'processing', 'checkpoint', 'output', 'logging', 'ai'}
        self.optional_sections = {'executable'}
    
    def get_section_schema(self, section_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific section"""
        return self.schema.get(section_name)
    
    def get_option_schema(self, section_name: str, option_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific option"""
        section_schema = self.get_section_schema(section_name)
        if section_schema:
            return section_schema.get(option_name)
        return None
    
    def is_section_required(self, section_name: str) -> bool:
        """Check if a section is required"""
        return section_name in self.required_sections
    
    def is_option_required(self, section_name: str, option_name: str) -> bool:
        """Check if an option is required"""
        option_schema = self.get_option_schema(section_name, option_name)
        return option_schema is not None and not option_schema.get('optional', False)


class ConfigValueValidator(Validator):
    """Validator for individual configuration values"""
    
    def __init__(self, option_schema: Dict[str, Any], section_name: str, option_name: str):
        """
        Initialize config value validator
        
        Args:
            option_schema: Schema definition for the option
            section_name: Name of the configuration section
            option_name: Name of the configuration option
        """
        super().__init__(field_name=f"{section_name}.{option_name}")
        self.option_schema = option_schema
        self.section_name = section_name
        self.option_name = option_name
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate configuration value according to schema"""
        result = ValidationResult(is_valid=True)
        
        # Handle missing values
        if value is None or (isinstance(value, str) and not value.strip()):
            if self.option_schema.get('optional', False):
                result.sanitized_value = self.option_schema.get('default')
                return result
            else:
                result.add_error(f"Required option is missing or empty", self.field_name)
                return result
        
        # Get expected type
        expected_type = self.option_schema.get('type', 'str')
        
        # Type-specific validation
        if expected_type == 'str':
            result = self._validate_string(value, result)
        elif expected_type == 'int':
            result = self._validate_integer(value, result)
        elif expected_type == 'float':
            result = self._validate_float(value, result)
        elif expected_type == 'bool':
            result = self._validate_boolean(value, result)
        else:
            result.add_error(f"Unknown type '{expected_type}' in schema", self.field_name)
        
        return result
    
    def _validate_string(self, value: Any, result: ValidationResult) -> ValidationResult:
        """Validate string configuration value"""
        if not isinstance(value, str):
            value = str(value)
        
        value = value.strip()
        result.sanitized_value = value
        
        # Check allowed values
        allowed_values = self.option_schema.get('allowed')
        if allowed_values and value not in allowed_values:
            result.add_error(f"Value '{value}' not in allowed values: {allowed_values}", self.field_name)
        
        # Check for sensitive values (API keys)
        if self.option_schema.get('sensitive', False):
            if value and len(value) < 10:
                result.add_warning("API key appears to be very short", self.field_name)
            elif value and not re.match(r'^[a-zA-Z0-9\-_]+$', value):
                result.add_warning("API key contains unusual characters", self.field_name)
        
        # Check for path values
        if 'dir' in self.option_name.lower() and value:
            expanded_path = os.path.expanduser(os.path.expandvars(value))
            result.sanitized_value = expanded_path
            
            # Validate path format
            try:
                Path(expanded_path)
            except Exception as e:
                result.add_error(f"Invalid path format: {e}", self.field_name)
        
        return result
    
    def _validate_integer(self, value: Any, result: ValidationResult) -> ValidationResult:
        """Validate integer configuration value"""
        try:
            if isinstance(value, str):
                int_value = int(value)
            elif isinstance(value, (int, float)):
                int_value = int(value)
            else:
                result.add_error(f"Cannot convert {type(value).__name__} to integer", self.field_name)
                return result
        except ValueError as e:
            result.add_error(f"Invalid integer value: {e}", self.field_name)
            return result
        
        result.sanitized_value = int_value
        
        # Check range constraints
        min_val = self.option_schema.get('min')
        max_val = self.option_schema.get('max')
        
        if min_val is not None and int_value < min_val:
            result.add_error(f"Value {int_value} is below minimum {min_val}", self.field_name)
        
        if max_val is not None and int_value > max_val:
            result.add_error(f"Value {int_value} is above maximum {max_val}", self.field_name)
        
        # Add performance warnings
        if self.option_name == 'batch_size':
            if int_value < 10:
                result.add_warning("Very small batch size may slow processing", self.field_name)
            elif int_value > 500:
                result.add_warning("Large batch size may use excessive memory", self.field_name)
        
        return result
    
    def _validate_float(self, value: Any, result: ValidationResult) -> ValidationResult:
        """Validate float configuration value"""
        try:
            if isinstance(value, str):
                float_value = float(value)
            elif isinstance(value, (int, float)):
                float_value = float(value)
            else:
                result.add_error(f"Cannot convert {type(value).__name__} to float", self.field_name)
                return result
        except ValueError as e:
            result.add_error(f"Invalid float value: {e}", self.field_name)
            return result
        
        result.sanitized_value = float_value
        
        # Check range constraints
        min_val = self.option_schema.get('min')
        max_val = self.option_schema.get('max')
        
        if min_val is not None and float_value < min_val:
            result.add_error(f"Value {float_value} is below minimum {min_val}", self.field_name)
        
        if max_val is not None and float_value > max_val:
            result.add_error(f"Value {float_value} is above maximum {max_val}", self.field_name)
        
        return result
    
    def _validate_boolean(self, value: Any, result: ValidationResult) -> ValidationResult:
        """Validate boolean configuration value"""
        if isinstance(value, bool):
            result.sanitized_value = value
        elif isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ('true', '1', 'yes', 'on', 'enabled', 'y'):
                result.sanitized_value = True
            elif value_lower in ('false', '0', 'no', 'off', 'disabled', 'n'):
                result.sanitized_value = False
            else:
                result.add_error(f"Invalid boolean value '{value}'", self.field_name)
        elif isinstance(value, (int, float)):
            result.sanitized_value = bool(value)
        else:
            result.add_error(f"Cannot convert {type(value).__name__} to boolean", self.field_name)
        
        return result


class ConfigurationFileValidator(Validator):
    """Validator for complete configuration files"""
    
    def __init__(self):
        super().__init__(field_name="configuration_file")
        self.schema = ConfigurationSchema()
        self.logger = logging.getLogger(__name__)
    
    def validate(self, config_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a configuration file
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True)
        
        if config_path is None:
            result.add_info("No configuration file provided, using defaults", self.field_name)
            result.sanitized_value = None
            return result
        
        path = Path(config_path)
        
        # Basic file validation
        if not path.exists():
            result.add_error(f"Configuration file does not exist: {path}", self.field_name)
            return result
        
        if not path.is_file():
            result.add_error(f"Configuration path is not a file: {path}", self.field_name)
            return result
        
        # Try to parse the configuration file
        config = configparser.ConfigParser()
        try:
            config.read(path, encoding='utf-8')
        except configparser.Error as e:
            result.add_error(f"Configuration file parsing error: {e}", self.field_name)
            return result
        except Exception as e:
            result.add_error(f"Cannot read configuration file: {e}", self.field_name)
            return result
        
        # Validate structure and content
        structure_result = self._validate_structure(config)
        result = result.merge(structure_result)
        
        # Validate individual values
        values_result = self._validate_values(config)
        result = result.merge(values_result)
        
        # Cross-section validation
        cross_validation_result = self._validate_cross_sections(config)
        result = result.merge(cross_validation_result)
        
        result.sanitized_value = str(path.resolve())
        return result
    
    def _validate_structure(self, config: configparser.ConfigParser) -> ValidationResult:
        """Validate configuration file structure"""
        result = ValidationResult(is_valid=True)
        
        # Check for required sections
        for section_name in self.schema.required_sections:
            if not config.has_section(section_name):
                result.add_error(f"Missing required section: [{section_name}]", self.field_name)
        
        # Check for unknown sections
        known_sections = self.schema.required_sections | self.schema.optional_sections
        for section_name in config.sections():
            if section_name not in known_sections:
                result.add_warning(f"Unknown section: [{section_name}]", self.field_name)
        
        # Check for empty sections
        for section_name in config.sections():
            if not config.options(section_name):
                result.add_warning(f"Empty section: [{section_name}]", self.field_name)
        
        return result
    
    def _validate_values(self, config: configparser.ConfigParser) -> ValidationResult:
        """Validate individual configuration values"""
        result = ValidationResult(is_valid=True)
        sanitized_config = {}
        
        # Validate each section and option
        for section_name in config.sections():
            section_schema = self.schema.get_section_schema(section_name)
            if not section_schema:
                continue  # Already warned about unknown sections
            
            sanitized_section = {}
            
            for option_name, value in config.items(section_name):
                option_schema = section_schema.get(option_name)
                
                if not option_schema:
                    result.add_warning(f"Unknown option: {section_name}.{option_name}", self.field_name)
                    sanitized_section[option_name] = value
                    continue
                
                # Validate the value
                validator = ConfigValueValidator(option_schema, section_name, option_name)
                value_result = validator.validate(value)
                result = result.merge(value_result)
                
                sanitized_section[option_name] = value_result.sanitized_value
            
            # Check for missing required options
            for option_name, option_schema in section_schema.items():
                if not option_schema.get('optional', False) and option_name not in sanitized_section:
                    result.add_error(f"Missing required option: {section_name}.{option_name}", self.field_name)
                    sanitized_section[option_name] = option_schema.get('default')
            
            sanitized_config[section_name] = sanitized_section
        
        result.metadata['sanitized_config'] = sanitized_config
        return result
    
    def _validate_cross_sections(self, config: configparser.ConfigParser) -> ValidationResult:
        """Validate relationships between different sections"""
        result = ValidationResult(is_valid=True)
        
        # AI engine and API key validation
        if config.has_section('ai'):
            default_engine = config.get('ai', 'default_engine', fallback='local')
            
            if default_engine in ['claude', 'openai']:
                api_key_option = f'{default_engine}_api_key'
                if not config.has_option('ai', api_key_option):
                    result.add_warning(f"AI engine '{default_engine}' selected but no API key configured", self.field_name)
                else:
                    api_key = config.get('ai', api_key_option)
                    if not api_key or not api_key.strip():
                        result.add_warning(f"AI engine '{default_engine}' selected but API key is empty", self.field_name)
        
        # Batch size and AI engine rate limits
        if config.has_section('processing') and config.has_section('ai'):
            batch_size = config.getint('processing', 'batch_size', fallback=100)
            default_engine = config.get('ai', 'default_engine', fallback='local')
            
            if default_engine in ['claude', 'openai']:
                rpm = config.getint('ai', f'{default_engine}_rpm', fallback=60)
                ai_batch_size = config.getint('ai', f'{default_engine}_batch_size', fallback=10)
                
                if batch_size > ai_batch_size * 5:
                    result.add_warning(
                        f"Processing batch size ({batch_size}) is much larger than AI batch size ({ai_batch_size})",
                        self.field_name
                    )
        
        # Checkpoint directory and temp directory conflicts
        if config.has_section('checkpoint') and config.has_section('executable'):
            checkpoint_dir = config.get('checkpoint', 'checkpoint_dir', fallback='.bookmark_checkpoints')
            temp_dir = config.get('executable', 'temp_dir', fallback='/tmp/bookmark-processor')
            
            checkpoint_path = os.path.expanduser(os.path.expandvars(checkpoint_dir))
            temp_path = os.path.expanduser(os.path.expandvars(temp_dir))
            
            if checkpoint_path == temp_path:
                result.add_warning("Checkpoint directory and temp directory are the same", self.field_name)
        
        # Logging level and performance settings
        if config.has_section('logging'):
            log_level = config.get('logging', 'log_level', fallback='INFO')
            performance_logging = config.getboolean('logging', 'performance_logging', fallback=True)
            
            if log_level == 'DEBUG' and not performance_logging:
                result.add_info("DEBUG logging enabled but performance logging disabled", self.field_name)
        
        return result


class ConfigurationValidator:
    """Main validator for configuration management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_validator = ConfigurationFileValidator()
        self.schema = ConfigurationSchema()
    
    def validate_config_file(self, config_path: Optional[Union[str, Path]]) -> ValidationResult:
        """
        Validate a configuration file
        
        Args:
            config_path: Path to configuration file or None
            
        Returns:
            ValidationResult with validation details
        """
        return self.file_validator.validate(config_path)
    
    def validate_config_object(self, config: configparser.ConfigParser) -> ValidationResult:
        """
        Validate a ConfigParser object
        
        Args:
            config: ConfigParser object to validate
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True)
        
        # Validate structure
        structure_result = self.file_validator._validate_structure(config)
        result = result.merge(structure_result)
        
        # Validate values
        values_result = self.file_validator._validate_values(config)
        result = result.merge(values_result)
        
        # Cross-section validation
        cross_validation_result = self.file_validator._validate_cross_sections(config)
        result = result.merge(cross_validation_result)
        
        return result
    
    def create_default_config(self, output_path: Union[str, Path]) -> None:
        """
        Create a default configuration file
        
        Args:
            output_path: Path where to save the default config
        """
        config = configparser.ConfigParser()
        
        # Add all sections with their default values
        for section_name, section_schema in self.schema.schema.items():
            config.add_section(section_name)
            
            for option_name, option_schema in section_schema.items():
                # Skip sensitive options
                if option_schema.get('sensitive', False):
                    continue
                
                # Skip optional options without defaults
                if option_schema.get('optional', False) and 'default' not in option_schema:
                    continue
                
                default_value = option_schema.get('default', '')
                config.set(section_name, option_name, str(default_value))
        
        # Write the configuration file
        with open(output_path, 'w', encoding='utf-8') as f:
            config.write(f)
    
    def create_template_config(self, output_path: Union[str, Path]) -> None:
        """
        Create a template configuration file with comments
        
        Args:
            output_path: Path where to save the template config
        """
        template_content = """# Bookmark Processor Configuration Template
# Copy this file to user_config.ini and customize as needed

[network]
# Network timeout in seconds (1-300)
timeout = 30

# Maximum retry attempts for failed requests (0-10)
max_retries = 3

# Default delay between requests in seconds (0.0-10.0)
default_delay = 0.5

# Maximum concurrent requests (1-50)
max_concurrent_requests = 10

# Enable user agent rotation for better compatibility
user_agent_rotation = true

# Site-specific delays in seconds
google_delay = 2.0
github_delay = 1.5
youtube_delay = 2.0
linkedin_delay = 2.0

[processing]
# Processing batch size (1-1000)
batch_size = 100

# Maximum tags per bookmark (1-20)
max_tags_per_bookmark = 5

# Target number of unique tags across all bookmarks (10-1000)
target_unique_tags = 150

# AI model for local processing
ai_model = facebook/bart-large-cnn

# Maximum description length (50-1000)
max_description_length = 150

# Use existing content as input for AI processing
use_existing_content = true

[checkpoint]
# Enable checkpoint/resume functionality
enabled = true

# Save checkpoint every N processed items (1-1000)
save_interval = 50

# Directory for checkpoint files
checkpoint_dir = .bookmark_checkpoints

# Automatically clean up old checkpoints
auto_cleanup = true

[output]
# Output format (currently only raindrop_import supported)
output_format = raindrop_import

# Preserve folder structure from input
preserve_folder_structure = true

# Include timestamps in output
include_timestamps = true

# Enable detailed error logging
error_log_detailed = true

[logging]
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level = INFO

# Log file name
log_file = bookmark_processor.log

# Enable console output
console_output = true

# Enable performance logging
performance_logging = true

[ai]
# Default AI engine: local, claude, openai
default_engine = local

# Cloud AI API keys (uncomment and add your keys)
# claude_api_key = your-claude-api-key-here
# openai_api_key = your-openai-api-key-here

# Rate limits (requests per minute)
claude_rpm = 50
openai_rpm = 60

# Batch sizes for cloud AI
claude_batch_size = 10
openai_batch_size = 20

# Cost tracking settings
show_running_costs = true
cost_confirmation_interval = 10.0

[executable]
# Directory for cached AI models
model_cache_dir = ~/.cache/bookmark-processor/models

# Temporary directory for processing
temp_dir = /tmp/bookmark-processor

# Clean up temporary files on exit
cleanup_on_exit = true
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
    
    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """
        Generate a detailed validation report
        
        Args:
            validation_result: Result from configuration validation
            
        Returns:
            Formatted validation report
        """
        if validation_result.is_valid and not validation_result.issues:
            return "✅ Configuration validation passed with no issues."
        
        report_parts = ["Configuration Validation Report", "=" * 40]
        
        if validation_result.is_valid:
            report_parts.append("✅ Overall Status: VALID (with warnings/info)")
        else:
            report_parts.append("❌ Overall Status: INVALID")
        
        # Group issues by severity
        errors = validation_result.get_errors()
        warnings = validation_result.get_warnings()
        info_issues = [issue for issue in validation_result.issues 
                      if issue.severity == ValidationSeverity.INFO]
        
        if errors:
            report_parts.extend(["", "❌ ERRORS:"])
            for issue in errors:
                report_parts.append(f"  • {issue.message}")
        
        if warnings:
            report_parts.extend(["", "⚠️  WARNINGS:"])
            for issue in warnings:
                report_parts.append(f"  • {issue.message}")
        
        if info_issues:
            report_parts.extend(["", "ℹ️  INFORMATION:"])
            for issue in info_issues:
                report_parts.append(f"  • {issue.message}")
        
        # Add suggestions
        if errors or warnings:
            report_parts.extend([
                "",
                "💡 SUGGESTIONS:",
                "  • Check configuration file syntax (INI format)",
                "  • Verify all required sections and options are present",
                "  • Ensure numeric values are within valid ranges",
                "  • Add API keys for cloud AI engines if needed",
                "  • Use absolute paths for directories",
                ""
            ])
        
        return "\n".join(report_parts)


# Factory functions for common validation scenarios
def validate_user_config(config_path: Optional[Union[str, Path]]) -> ValidationResult:
    """
    Validate a user configuration file
    
    Args:
        config_path: Path to user configuration file
        
    Returns:
        ValidationResult with validation details
    """
    validator = ConfigurationValidator()
    return validator.validate_config_file(config_path)


def create_config_template(output_path: Union[str, Path]) -> None:
    """
    Create a configuration template file
    
    Args:
        output_path: Path where to save the template
    """
    validator = ConfigurationValidator()
    validator.create_template_config(output_path)


def get_config_validation_report(config_path: Optional[Union[str, Path]]) -> str:
    """
    Get a detailed validation report for a configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Formatted validation report
    """
    validator = ConfigurationValidator()
    result = validator.validate_config_file(config_path)
    return validator.generate_validation_report(result)