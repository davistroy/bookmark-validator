"""
Input validation utilities for the Bookmark Processor.

This module provides validation functions for command-line arguments
and other user inputs.
"""

import os
from pathlib import Path
from typing import Union

from bookmark_processor.config.configuration import Configuration


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_input_file(file_path: Union[str, Path]) -> Path:
    """
    Validate that input file exists and is readable.

    Args:
        file_path: Path to the input file

    Returns:
        Validated Path object

    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"Input file does not exist: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Input path is not a file: {file_path}")

    if not os.access(path, os.R_OK):
        raise ValidationError(f"Input file is not readable: {file_path}")

    if path.suffix.lower() != ".csv":
        raise ValidationError(f"Input file must be a CSV file, got: {path.suffix}")

    return path.absolute()


def validate_output_file(file_path: Union[str, Path]) -> Path:
    """
    Validate that output file path is writable.

    Args:
        file_path: Path to the output file

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path isn't writable or parent doesn't exist
    """
    path = Path(file_path)

    # Check if parent directory exists
    parent = path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory: {parent}: {e}")

    # Check if we can write to the directory
    if not os.access(parent, os.W_OK):
        raise ValidationError(f"Output directory is not writable: {parent}")

    # Check if file exists and is writable
    if path.exists() and not os.access(path, os.W_OK):
        raise ValidationError(f"Output file exists and is not writable: {file_path}")

    if path.suffix.lower() != ".csv":
        raise ValidationError(f"Output file must be a CSV file, got: {path.suffix}")

    return path.absolute()


def validate_config_file(file_path: Union[str, Path, None]) -> Union[Path, None]:
    """
    Validate configuration file if provided.

    Args:
        file_path: Path to the config file or None

    Returns:
        Validated Path object or None

    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    if file_path is None:
        return None

    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"Configuration file does not exist: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Configuration path is not a file: {file_path}")

    if not os.access(path, os.R_OK):
        raise ValidationError(f"Configuration file is not readable: {file_path}")

    if path.suffix.lower() not in [".ini", ".cfg", ".conf"]:
        raise ValidationError(
            f"Configuration file must be .ini, .cfg, or .conf file, got: {path.suffix}"
        )

    return path.absolute()


def validate_batch_size(batch_size: int) -> int:
    """
    Validate batch size is within reasonable bounds.

    Args:
        batch_size: Batch size value

    Returns:
        Validated batch size

    Raises:
        ValidationError: If batch size is invalid
    """
    if batch_size < 1:
        raise ValidationError("Batch size must be at least 1")

    if batch_size > 1000:
        raise ValidationError("Batch size cannot exceed 1000")

    return batch_size


def validate_max_retries(max_retries: int) -> int:
    """
    Validate max retries is within reasonable bounds.

    Args:
        max_retries: Maximum retry attempts

    Returns:
        Validated max retries

    Raises:
        ValidationError: If max retries is invalid
    """
    if max_retries < 0:
        raise ValidationError("Max retries cannot be negative")

    if max_retries > 10:
        raise ValidationError("Max retries cannot exceed 10")

    return max_retries


def validate_conflicting_arguments(resume: bool, clear_checkpoints: bool) -> None:
    """
    Validate that conflicting arguments aren't both set.

    Args:
        resume: Whether resume flag is set
        clear_checkpoints: Whether clear checkpoints flag is set

    Raises:
        ValidationError: If conflicting arguments are set
    """
    if resume and clear_checkpoints:
        raise ValidationError(
            "Cannot use --resume and --clear-checkpoints together. "
            "Choose one or the other."
        )


def validate_ai_engine(ai_engine: str, config: Configuration) -> str:
    """
    Validate AI engine selection and associated configuration.

    Args:
        ai_engine: The selected AI engine (local, claude, or openai)
        config: Configuration object to check for API keys

    Returns:
        Validated AI engine string

    Raises:
        ValidationError: If AI engine is invalid or missing API keys
    """
    valid_engines = ["local", "claude", "openai"]
    
    if ai_engine not in valid_engines:
        raise ValidationError(
            f"Invalid AI engine '{ai_engine}'. "
            f"Must be one of: {', '.join(valid_engines)}"
        )
    
    # Local AI doesn't need API key validation
    if ai_engine == "local":
        return ai_engine
    
    # Check if API key is configured for cloud AI
    if not config.has_api_key(ai_engine):
        raise ValidationError(
            f"Missing API key for {ai_engine}. "
            f"Please add '{ai_engine}_api_key' to your user_config.ini file."
        )
    
    # Validate API key format
    is_valid, error_msg = config.validate_ai_configuration()
    if not is_valid:
        raise ValidationError(f"AI configuration validation failed: {error_msg}")
    
    return ai_engine
