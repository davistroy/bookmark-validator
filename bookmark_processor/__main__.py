#!/usr/bin/env python3
"""
Package entry point for the Bookmark Validation and Enhancement Tool.

This allows the package to be executed with: python -m bookmark_processor
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
