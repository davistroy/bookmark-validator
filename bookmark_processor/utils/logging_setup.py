"""
Logging configuration for the Bookmark Processor.

This module sets up logging based on configuration settings.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(config=None, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.

    Args:
        config: Configuration object (unused, kept for backward compatibility)
        log_file: Optional log file path override
    """
    # Logging settings are now fixed values (simplified in new config system)
    log_level = "INFO"
    console_output = True

    if log_file is None:
        log_file = "bookmark_processor.log"

    # Create logs directory if needed
    if getattr(sys, "frozen", False):
        # Running as executable
        app_dir = Path(sys.executable).parent
    else:
        # Running as script
        app_dir = Path.cwd()

    log_dir = app_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{Path(log_file).stem}_{timestamp}.log"

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure handlers
    handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(level=getattr(logging, log_level.upper()), handlers=handlers)

    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Bookmark Processor starting - Log file: {log_path}")
    logger.info(f"Log level: {log_level}")

    # Set up specific loggers
    # Reduce noise from some libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
