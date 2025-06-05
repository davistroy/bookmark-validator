#!/usr/bin/env python3
"""
Main entry point for the Bookmark Validation and Enhancement Tool.

This module serves as the primary entry point for both the development
environment and the Windows executable.
"""

import sys
from bookmark_processor.cli import main


if __name__ == "__main__":
    sys.exit(main())