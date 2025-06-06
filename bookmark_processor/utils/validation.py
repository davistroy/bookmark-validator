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


def validate_input_file(file_path: Union[str, Path, None]) -> Union[Path, None]:
    """
    Validate that input file exists and is readable, or validate auto-detection mode.

    Args:
        file_path: Path to the input file, or None for auto-detection

    Returns:
        Validated Path object, or None for auto-detection mode

    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    # Handle auto-detection mode
    if file_path is None:
        return None

    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"Input file does not exist: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Input path is not a file: {file_path}")

    if not os.access(path, os.R_OK):
        raise ValidationError(f"Input file is not readable: {file_path}")

    # Accept both CSV and HTML files
    allowed_extensions = [".csv", ".html", ".htm"]
    if path.suffix.lower() not in allowed_extensions:
        raise ValidationError(f"Input file must be CSV or HTML, got: {path.suffix}")
        
    # Additional validation for file format
    try:
        from bookmark_processor.core.import_module import MultiFormatImporter
        importer = MultiFormatImporter()
        file_info = importer.get_file_info(path)
        if not file_info['is_supported']:
            raise ValidationError(f"Unsupported file format or invalid file content")
    except ImportError:
        # If import fails, just check extension
        pass

    return path.absolute()


def validate_auto_detection_mode() -> None:
    """
    Validate that auto-detection mode can be used in the current directory.

    Raises:
        ValidationError: If auto-detection is not possible
    """
    try:
        from bookmark_processor.core.multi_file_processor import MultiFileProcessor
        processor = MultiFileProcessor()
        report = processor.validate_directory_for_auto_detection()
        
        if not report["can_auto_detect"]:
            if "error" in report:
                raise ValidationError(f"Auto-detection failed: {report['error']}")
            else:
                raise ValidationError("No valid bookmark files found in current directory")
                
    except ImportError:
        raise ValidationError("Auto-detection functionality not available")


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


def validate_csv_structure(csv_data, expected_columns=None):
    """
    Validate CSV structure and format.
    
    Args:
        csv_data: CSV data to validate (DataFrame or file path)
        expected_columns: Optional list of expected column names
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If CSV structure is invalid
    """
    try:
        import pandas as pd
        
        if isinstance(csv_data, str) or isinstance(csv_data, Path):
            # Load CSV file
            df = pd.read_csv(csv_data)
        else:
            # Assume it's already a DataFrame
            df = csv_data
            
        if df.empty:
            raise ValidationError("CSV file is empty")
            
        if expected_columns:
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                raise ValidationError(f"Missing columns: {', '.join(missing_columns)}")
                
        return True
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"CSV validation failed: {str(e)}")


def validate_bookmark_data(bookmark_data: dict) -> bool:
    """
    Validate bookmark data structure and content.
    
    Args:
        bookmark_data: Dictionary containing bookmark data
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If bookmark data is invalid
    """
    required_fields = ["url", "title"]
    
    for field in required_fields:
        if field not in bookmark_data or not bookmark_data[field]:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate URL format
    validate_url_format(bookmark_data["url"])
    
    return True


def validate_url_format(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid, False if invalid
        
    Note: 
        Only accepts HTTP(S) and FTP URLs for bookmark processing.
        Rejects javascript:, mailto:, and malformed URLs.
    """
    import re
    
    # Handle None, empty, or whitespace-only strings
    if not url or not isinstance(url, str) or not url.strip():
        return False
    
    url = url.strip()
    
    # Reject dangerous schemes
    dangerous_schemes = ['javascript:', 'data:', 'vbscript:']
    for scheme in dangerous_schemes:
        if url.lower().startswith(scheme):
            return False
    
    # Accept HTTP(S) and FTP URLs
    valid_url_pattern = re.compile(
        r'^(https?|ftp)://'  # http://, https://, or ftp://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    # Reject mailto: URLs (not suitable for bookmarks)
    if url.lower().startswith('mailto:'):
        return False
    
    # Reject malformed URLs
    if url.count('://') != 1:
        return False
    
    if '://' in url and not url.split('://')[1]:
        return False
    
    return bool(valid_url_pattern.match(url))


def sanitize_input(input_data) -> str:
    """
    Sanitize input data to prevent injection attacks.
    
    Args:
        input_data: Input data to sanitize (any type)
        
    Returns:
        Sanitized string
    """
    import re
    
    # Handle None
    if input_data is None:
        return ""
    
    # Convert to string
    if not isinstance(input_data, str):
        input_data = str(input_data)
    
    # Remove HTML tags
    sanitized = re.sub(r'<[^>]*>', '', input_data)
    
    # Replace newlines and carriage returns with spaces, but preserve the fact they were there
    has_trailing_newlines = re.search(r'[\r\n]\s*$', sanitized)
    
    # Normalize whitespace (replace newlines, tabs, carriage returns with spaces)
    sanitized = re.sub(r'[\r\n\t]+', ' ', sanitized)
    
    # Collapse multiple spaces into single spaces  
    sanitized = re.sub(r' +', ' ', sanitized)
    
    # Trim leading and trailing, but preserve trailing space if original had trailing newlines
    sanitized = sanitized.strip()
    if has_trailing_newlines:
        sanitized += ' '
    
    return sanitized
