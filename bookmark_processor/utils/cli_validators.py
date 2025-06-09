"""
Command-line Argument Validators

This module provides specialized validators for all command-line arguments and options.
It includes validation for argument combinations, mutual exclusivity, and contextual
help messages for invalid arguments.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .input_validator import (
    NumberValidator,
    StringValidator,
    ValidationResult,
    Validator,
)


class PathValidator(Validator):
    """Validator for file and directory paths"""

    def __init__(
        self,
        field_name: str,
        path_type: str = "file",  # 'file', 'dir', 'file_or_dir'
        must_exist: bool = True,
        must_be_readable: bool = True,
        must_be_writable: bool = False,
        create_parent_dirs: bool = False,
        allowed_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize path validator

        Args:
            field_name: Name of the field being validated
            path_type: Type of path ('file', 'dir', 'file_or_dir')
            must_exist: Whether path must already exist
            must_be_readable: Whether path must be readable
            must_be_writable: Whether path must be writable
            create_parent_dirs: Whether to create parent directories
            allowed_extensions: List of allowed file extensions
        """
        super().__init__(field_name, required=True, allow_none=False)
        self.path_type = path_type
        self.must_exist = must_exist
        self.must_be_readable = must_be_readable
        self.must_be_writable = must_be_writable
        self.create_parent_dirs = create_parent_dirs
        self.allowed_extensions = set(ext.lower() for ext in (allowed_extensions or []))

    def validate(self, value: Any) -> ValidationResult:
        """Validate path value"""
        result = ValidationResult(is_valid=True)

        if not value:
            result.add_error("Path cannot be empty", self.field_name)
            return result

        try:
            path = Path(value).resolve()
        except Exception as e:
            result.add_error(f"Invalid path format: {e}", self.field_name)
            return result

        result.sanitized_value = str(path)

        # Check if path exists
        if self.must_exist and not path.exists():
            result.add_error(f"Path does not exist: {path}", self.field_name)
            return result

        # Check path type
        if path.exists():
            if self.path_type == "file" and not path.is_file():
                result.add_error(f"Path is not a file: {path}", self.field_name)
            elif self.path_type == "dir" and not path.is_dir():
                result.add_error(f"Path is not a directory: {path}", self.field_name)

        # Check file extension
        if (
            self.allowed_extensions
            and path.suffix.lower() not in self.allowed_extensions
        ):
            result.add_error(
                f"File extension '{path.suffix}' not allowed. "
                f"Allowed: {list(self.allowed_extensions)}",
                self.field_name,
            )

        # Check permissions
        if path.exists():
            if self.must_be_readable and not os.access(path, os.R_OK):
                result.add_error(f"Path is not readable: {path}", self.field_name)

            if self.must_be_writable and not os.access(path, os.W_OK):
                result.add_error(f"Path is not writable: {path}", self.field_name)
        else:
            # Check parent directory permissions for new files
            parent = path.parent
            if not parent.exists():
                if self.create_parent_dirs:
                    try:
                        parent.mkdir(parents=True, exist_ok=True)
                        result.add_info(
                            f"Created parent directories: {parent}", self.field_name
                        )
                    except Exception as e:
                        result.add_error(
                            f"Cannot create parent directories: {e}", self.field_name
                        )
                        return result
                else:
                    result.add_error(
                        f"Parent directory does not exist: {parent}", self.field_name
                    )
                    return result

            if self.must_be_writable and not os.access(parent, os.W_OK):
                result.add_error(
                    f"Parent directory is not writable: {parent}", self.field_name
                )

        return result


class InputFileValidator(PathValidator):
    """Validator specifically for input files"""

    def __init__(self, field_name: str = "input"):
        super().__init__(
            field_name=field_name,
            path_type="file",
            must_exist=True,
            must_be_readable=True,
            must_be_writable=False,
            allowed_extensions=[".csv"],
        )

    def validate(self, value: Any) -> ValidationResult:
        """Validate input file with additional checks"""
        result = super().validate(value)

        if result.is_valid and result.sanitized_value:
            path = Path(result.sanitized_value)

            # Check file size
            try:
                file_size = path.stat().st_size
                if file_size == 0:
                    result.add_error("Input file is empty", self.field_name)
                elif file_size > 500 * 1024 * 1024:  # 500 MB
                    result.add_warning(
                        f"Input file is very large ({file_size // (1024*1024)} MB)",
                        self.field_name,
                    )
            except Exception as e:
                result.add_warning(f"Cannot check file size: {e}", self.field_name)

            # Basic CSV format check
            try:
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        result.add_error(
                            "Input file appears to be empty", self.field_name
                        )
                    elif (
                        "," not in first_line
                        and ";" not in first_line
                        and "\t" not in first_line
                    ):
                        result.add_warning(
                            "File may not be in CSV format (no delimiters found)",
                            self.field_name,
                        )
            except Exception as e:
                result.add_warning(
                    f"Cannot read file for format check: {e}", self.field_name
                )

        return result


class OutputFileValidator(PathValidator):
    """Validator specifically for output files"""

    def __init__(self, field_name: str = "output"):
        super().__init__(
            field_name=field_name,
            path_type="file",
            must_exist=False,
            must_be_readable=False,
            must_be_writable=True,
            create_parent_dirs=True,
            allowed_extensions=[".csv"],
        )

    def validate(self, value: Any) -> ValidationResult:
        """Validate output file with additional checks"""
        result = super().validate(value)

        if result.is_valid and result.sanitized_value:
            path = Path(result.sanitized_value)

            # Check if file exists and warn about overwriting
            if path.exists():
                result.add_warning(
                    f"Output file exists and will be overwritten: {path}",
                    self.field_name,
                )

            # Check if output path is same as input
            # This requires additional context from other arguments
            result.metadata["output_path"] = str(path)

        return result


class ConfigFileValidator(PathValidator):
    """Validator specifically for configuration files"""

    def __init__(self, field_name: str = "config"):
        super().__init__(
            field_name=field_name,
            path_type="file",
            must_exist=True,
            must_be_readable=True,
            must_be_writable=False,
            allowed_extensions=[".ini", ".cfg", ".conf"],
        )

    def validate(self, value: Any) -> ValidationResult:
        """Validate config file with additional checks"""
        if value is None:
            # Config file is optional
            result = ValidationResult(is_valid=True, sanitized_value=None)
            return result

        result = super().validate(value)

        if result.is_valid and result.sanitized_value:
            path = Path(result.sanitized_value)

            # Basic config file format check
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "[" not in content or "]" not in content:
                        result.add_warning(
                            "Configuration file may not be in INI format",
                            self.field_name,
                        )
            except Exception as e:
                result.add_warning(
                    f"Cannot read config file for format check: {e}", self.field_name
                )

        return result


class BatchSizeValidator(NumberValidator):
    """Validator for batch size argument"""

    def __init__(self):
        super().__init__(
            field_name="batch_size",
            required=True,
            allow_none=False,
            min_value=1,
            max_value=1000,
            integer_only=True,
            positive_only=True,
        )

    def validate(self, value: Any) -> ValidationResult:
        """Validate batch size with additional recommendations"""
        result = super().validate(value)

        if result.is_valid and result.sanitized_value is not None:
            batch_size = result.sanitized_value

            # Provide recommendations
            if batch_size < 10:
                result.add_warning(
                    "Very small batch size may slow processing", self.field_name
                )
            elif batch_size > 500:
                result.add_warning(
                    "Large batch size may use excessive memory", self.field_name
                )
            elif 50 <= batch_size <= 200:
                result.add_info("Batch size is in recommended range", self.field_name)

        return result


class MaxRetriesValidator(NumberValidator):
    """Validator for max retries argument"""

    def __init__(self):
        super().__init__(
            field_name="max_retries",
            required=True,
            allow_none=False,
            min_value=0,
            max_value=10,
            integer_only=True,
        )

    def validate(self, value: Any) -> ValidationResult:
        """Validate max retries with additional recommendations"""
        result = super().validate(value)

        if result.is_valid and result.sanitized_value is not None:
            max_retries = result.sanitized_value

            # Provide recommendations
            if max_retries == 0:
                result.add_warning(
                    "Zero retries may result in many failed URLs", self.field_name
                )
            elif max_retries > 5:
                result.add_warning(
                    "High retry count may significantly slow processing",
                    self.field_name,
                )

        return result


class AIEngineValidator(StringValidator):
    """Validator for AI engine selection"""

    def __init__(self):
        super().__init__(
            field_name="ai_engine",
            required=True,
            allow_none=False,
            allowed_values=["local", "claude", "openai"],
        )

    def validate(self, value: Any) -> ValidationResult:
        """Validate AI engine selection"""
        result = super().validate(value)

        if result.is_valid and result.sanitized_value:
            engine = result.sanitized_value

            # Add engine-specific information
            if engine == "local":
                result.add_info(
                    "Using local AI models (no API key required)", self.field_name
                )
            elif engine in ["claude", "openai"]:
                result.add_info(
                    f"Using {engine} cloud API (requires API key in config)",
                    self.field_name,
                )

        return result


class DuplicateStrategyValidator(StringValidator):
    """Validator for duplicate resolution strategy"""

    def __init__(self):
        super().__init__(
            field_name="duplicate_strategy",
            required=True,
            allow_none=False,
            allowed_values=["newest", "oldest", "most_complete", "highest_quality"],
        )

    def validate(self, value: Any) -> ValidationResult:
        """Validate duplicate strategy with descriptions"""
        result = super().validate(value)

        if result.is_valid and result.sanitized_value:
            strategy = result.sanitized_value

            descriptions = {
                "newest": "Keep bookmark with most recent creation date",
                "oldest": "Keep bookmark with oldest creation date",
                "most_complete": "Keep bookmark with most filled fields",
                "highest_quality": "Keep bookmark with best overall data quality",
            }

            description = descriptions.get(strategy, "Unknown strategy")
            result.add_info(f"Duplicate strategy: {description}", self.field_name)

        return result


class ArgumentCombinationValidator(Validator):
    """Validator for checking argument combinations and conflicts"""

    def __init__(self):
        super().__init__(field_name="argument_combination")

    def validate(self, args_dict: Dict[str, Any]) -> ValidationResult:
        """Validate argument combinations"""
        result = ValidationResult(is_valid=True)

        # Check conflicting flags
        if args_dict.get("resume") and args_dict.get("clear_checkpoints"):
            result.add_error(
                "Cannot use --resume and --clear-checkpoints together. "
                "Choose one or the other.",
                self.field_name,
            )

        # Check input/output path conflict
        input_path = args_dict.get("input_path")
        output_path = args_dict.get("output_path")
        if (
            input_path
            and output_path
            and Path(input_path).resolve() == Path(output_path).resolve()
        ):
            result.add_error(
                "Input and output files cannot be the same", self.field_name
            )

        # Check AI engine and duplicate strategy combination
        ai_engine = args_dict.get("ai_engine")
        detect_duplicates = args_dict.get("detect_duplicates", True)
        if ai_engine != "local" and not detect_duplicates:
            result.add_warning(
                "Using cloud AI without duplicate detection may process "
                "the same URLs multiple times",
                self.field_name,
            )

        # Check batch size with AI engine
        batch_size = args_dict.get("batch_size", 100)
        if ai_engine in ["claude", "openai"] and batch_size > 100:
            result.add_warning(
                f"Large batch size ({batch_size}) with cloud AI may hit rate limits",
                self.field_name,
            )

        return result


class CLIArgumentValidator:
    """Main validator for all CLI arguments"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validators = {
            "input": InputFileValidator(),
            "output": OutputFileValidator(),
            "config": ConfigFileValidator(),
            "batch_size": BatchSizeValidator(),
            "max_retries": MaxRetriesValidator(),
            "ai_engine": AIEngineValidator(),
            "duplicate_strategy": DuplicateStrategyValidator(),
        }
        self.combination_validator = ArgumentCombinationValidator()

    def validate_all_arguments(self, args: argparse.Namespace) -> ValidationResult:
        """
        Validate all command-line arguments

        Args:
            args: Parsed arguments from argparse

        Returns:
            ValidationResult with all validation details
        """
        result = ValidationResult(is_valid=True)
        validated_args = {}

        # Validate individual arguments
        for arg_name, validator in self.validators.items():
            if hasattr(args, arg_name):
                arg_value = getattr(args, arg_name)
                arg_result = validator.validate(arg_value)

                # Merge results
                result = result.merge(arg_result)

                # Store validated value
                validated_args[
                    (
                        f"{arg_name}_path"
                        if arg_name in ["input", "output", "config"]
                        else arg_name
                    )
                ] = arg_result.sanitized_value

        # Add other boolean/simple arguments
        validated_args.update(
            {
                "resume": getattr(args, "resume", False),
                "verbose": getattr(args, "verbose", False),
                "clear_checkpoints": getattr(args, "clear_checkpoints", False),
                "detect_duplicates": not getattr(args, "no_duplicates", False),
            }
        )

        # Validate argument combinations
        combination_result = self.combination_validator.validate(validated_args)
        result = result.merge(combination_result)

        # Store all validated arguments
        result.sanitized_value = validated_args

        return result

    def generate_help_message(self, validation_result: ValidationResult) -> str:
        """
        Generate helpful error message with suggestions

        Args:
            validation_result: Result from validation

        Returns:
            Formatted help message
        """
        if validation_result.is_valid:
            return "All arguments are valid."

        message_parts = ["Argument validation failed:"]

        # Group issues by severity
        errors = validation_result.get_errors()
        warnings = validation_result.get_warnings()

        if errors:
            message_parts.append("\nErrors:")
            for issue in errors:
                message_parts.append(f"  ❌ {issue}")

        if warnings:
            message_parts.append("\nWarnings:")
            for issue in warnings:
                message_parts.append(f"  ⚠️  {issue}")

        # Add general suggestions
        message_parts.extend(
            [
                "\nSuggestions:",
                "  • Check file paths for typos and ensure files exist",
                "  • Verify numeric arguments are within valid ranges",
                "  • Ensure conflicting arguments are not used together",
                "  • Use --help for detailed usage information",
            ]
        )

        return "\n".join(message_parts)


def validate_cli_arguments(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Validate CLI arguments and return validated values or error message

    Args:
        args: Parsed arguments from argparse

    Returns:
        Tuple of (validated_args_dict, error_message)
        If validation succeeds: (validated_args, None)
        If validation fails: (None, error_message)
    """
    validator = CLIArgumentValidator()
    result = validator.validate_all_arguments(args)

    if result.is_valid:
        return result.sanitized_value, None
    else:
        error_message = validator.generate_help_message(result)
        return None, error_message


def create_enhanced_parser() -> argparse.ArgumentParser:
    """
    Create an enhanced argument parser with validation hints

    Returns:
        Enhanced ArgumentParser with detailed help
    """
    parser = argparse.ArgumentParser(
        prog="bookmark-processor",
        description=(
            "Bookmark Validation and Enhancement Tool - "
            "Process raindrop.io bookmark exports"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  bookmark-processor --input bookmarks.csv --output enhanced.csv

  # Resume from checkpoint
  bookmark-processor --input bookmarks.csv --output enhanced.csv --resume

  # Use cloud AI with custom batch size
  bookmark-processor --input bookmarks.csv --output enhanced.csv \\
    --ai-engine claude --batch-size 50 --verbose

  # Disable duplicate detection
  bookmark-processor --input bookmarks.csv --output enhanced.csv --no-duplicates
  # Custom duplicate strategy
  bookmark-processor --input bookmarks.csv --output enhanced.csv \\
    --duplicate-strategy newest

Cloud AI Setup:
  1. Copy config/user_config.ini.template to config/user_config.ini
  2. Add your API keys:
     [ai]
     claude_api_key = your-claude-api-key-here
     openai_api_key = your-openai-api-key-here

File Requirements:
  • Input: CSV file in raindrop.io export format (11 columns)
  • Output: Will be created in raindrop.io import format (6 columns)
  • Config: Optional INI format configuration file

Performance Tips:
  • Use batch size 50-200 for optimal performance
  • Enable --verbose for progress monitoring
  • Use --resume for large datasets that may be interrupted

For more information: https://github.com/davistroy/bookmark-validator
        """,
    )

    # Version
    parser.add_argument("--version", "-V", action="version", version="%(prog)s 1.0.0")

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="FILE",
        help=(
            "Input CSV file (raindrop.io export format). "
            "Must exist and be readable."
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="FILE",
        help=(
            "Output CSV file (raindrop.io import format). "
            "Parent directory must be writable."
        ),
    )

    # Optional file arguments
    parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        help="Custom configuration file (INI format). Optional.",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        metavar="N",
        help="Processing batch size (1-1000, default: 100). Recommended: 50-200.",
    )

    parser.add_argument(
        "--max-retries",
        "-m",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Maximum retry attempts for failed URLs (0-10, default: 3)."
        ),
    )

    # AI engine selection
    parser.add_argument(
        "--ai-engine",
        choices=["local", "claude", "openai"],
        default="local",
        help=(
            "AI engine for description generation (default: local). "
            "Cloud engines require API keys."
        ),
    )

    # Duplicate handling
    parser.add_argument(
        "--duplicate-strategy",
        choices=["newest", "oldest", "most_complete", "highest_quality"],
        default="highest_quality",
        help="Strategy for resolving duplicate URLs (default: highest_quality).",
    )

    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Disable duplicate URL detection and removal.",
    )

    # Checkpoint and resume
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume processing from existing checkpoint.",
    )

    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear existing checkpoints and start fresh. Conflicts with --resume.",
    )

    # Output control
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output and progress information.",
    )

    return parser


# Alias for backward compatibility
CLIValidator = CLIArgumentValidator
