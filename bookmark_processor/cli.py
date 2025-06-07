"""
Command-line interface for the Bookmark Validation and Enhancement Tool.

This module provides the CLI for processing raindrop.io bookmark exports,
validating URLs, generating AI-enhanced descriptions, and creating
optimized tagging systems.
"""

import argparse
import logging
import sys

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.bookmark_processor import BookmarkProcessor
from bookmark_processor.utils.logging_setup import setup_logging
from bookmark_processor.utils.validation import (
    ValidationError,
    validate_ai_engine,
    validate_auto_detection_mode,
    validate_batch_size,
    validate_config_file,
    validate_conflicting_arguments,
    validate_input_file,
    validate_max_retries,
    validate_output_file,
)


class CLIInterface:
    """Enhanced command line interface for Windows executable."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser."""
        parser = argparse.ArgumentParser(
            prog="bookmark-processor",
            description=(
                "Bookmark Validation and Enhancement Tool - "
                "Process raindrop.io bookmark exports"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv
  bookmark-processor.exe --input chrome_bookmarks.html --output enhanced.csv
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv --resume
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --batch-size 50 --verbose
  bookmark-processor.exe --input chrome_bookmarks.html --output enhanced.csv \\
    --ai-engine claude --verbose
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --duplicate-strategy newest --verbose
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --no-duplicates
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --chrome-html --html-title "My Bookmarks"
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --chrome-html --html-output custom_bookmarks.html
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --no-folders --ai-engine openai
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --max-bookmarks-per-folder 15 --verbose

Output Formats:
  By default, only CSV output (raindrop.io import format) is generated.
  Use --chrome-html to also generate Chrome-compatible HTML bookmark files.
  HTML files include timestamped filenames unless --html-output is specified.

AI Folder Generation:
  By default, AI-powered semantic folder structures are generated (max 20 bookmarks/folder).
  Use --no-folders to disable and preserve original folder structure.
  Use --max-bookmarks-per-folder to adjust folder size limits.

Cloud AI Setup:
  Copy user_config.ini.template to user_config.ini and add your API keys:
  [ai]
  claude_api_key = your-claude-api-key-here
  openai_api_key = your-openai-api-key-here

Duplicate Detection:
  By default, duplicate URLs are detected and removed using the 'highest_quality'
  strategy. Use --no-duplicates to disable or --duplicate-strategy to change.

For more information, visit: https://github.com/davistroy/bookmark-validator
            """,
        )

        # Add version argument
        parser.add_argument(
            "--version", "-V", action="version", version="%(prog)s 1.0.0"
        )

        # Input arguments
        parser.add_argument(
            "--input",
            "-i",
            help="Input file (raindrop.io CSV export or Chrome HTML bookmarks). If not specified, auto-detects all CSV and HTML files in current directory.",
        )
        parser.add_argument(
            "--output",
            "-o",
            required=True,
            help="Output CSV file (raindrop.io import format)",
        )

        # Optional arguments
        parser.add_argument("--config", "-c", help="Custom configuration file path")
        parser.add_argument(
            "--resume",
            "-r",
            action="store_true",
            help="Resume from existing checkpoint",
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )
        parser.add_argument(
            "--batch-size",
            "-b",
            type=int,
            default=100,
            help="Processing batch size (default: 100)",
        )
        parser.add_argument(
            "--max-retries",
            "-m",
            type=int,
            default=3,
            help="Maximum retry attempts (default: 3)",
        )
        parser.add_argument(
            "--clear-checkpoints",
            action="store_true",
            help="Clear existing checkpoints and start fresh",
        )
        parser.add_argument(
            "--ai-engine",
            choices=["local", "claude", "openai"],
            default="local",
            help="Select AI engine for processing (default: local)",
        )
        parser.add_argument(
            "--no-duplicates",
            action="store_true",
            help="Disable duplicate URL detection and removal",
        )
        parser.add_argument(
            "--duplicate-strategy",
            choices=["newest", "oldest", "most_complete", "highest_quality"],
            default="highest_quality",
            help="Strategy for resolving duplicates (default: highest_quality)",
        )
        
        # Folder generation options
        parser.add_argument(
            "--generate-folders",
            action="store_true",
            default=True,
            help="Generate AI-powered semantic folder structure (default: enabled)",
        )
        parser.add_argument(
            "--no-folders",
            action="store_true",
            help="Disable AI folder generation and use original folder structure",
        )
        parser.add_argument(
            "--max-bookmarks-per-folder",
            type=int,
            default=20,
            help="Maximum bookmarks per folder (default: 20)",
        )
        
        # Output format options
        parser.add_argument(
            "--chrome-html",
            action="store_true",
            help="Generate Chrome HTML bookmark file in addition to CSV",
        )
        parser.add_argument(
            "--html-output",
            help="Custom path for Chrome HTML output (auto-generated with timestamp if not specified)",
        )
        parser.add_argument(
            "--html-title",
            default="Enhanced Bookmarks",
            help="Title for Chrome HTML bookmark file (default: Enhanced Bookmarks)",
        )

        return parser

    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse command line arguments."""
        return self.parser.parse_args(args)

    def validate_args(self, args: argparse.Namespace) -> dict:
        """
        Validate all arguments and return processed values.

        Args:
            args: Parsed arguments from argparse

        Returns:
            Dictionary of validated and processed arguments

        Raises:
            ValidationError: If any validation fails
        """
        # Validate file paths
        input_path = validate_input_file(args.input)
        
        # If no input file specified, validate auto-detection mode
        if input_path is None:
            validate_auto_detection_mode()
            
        output_path = validate_output_file(args.output)
        config_path = validate_config_file(args.config)

        # Validate numeric arguments
        batch_size = validate_batch_size(args.batch_size)
        max_retries = validate_max_retries(args.max_retries)

        # Validate conflicting arguments
        validate_conflicting_arguments(args.resume, args.clear_checkpoints)

        return {
            "input_path": input_path,
            "output_path": output_path,
            "config_path": config_path,
            "resume": args.resume,
            "verbose": args.verbose,
            "batch_size": batch_size,
            "max_retries": max_retries,
            "clear_checkpoints": args.clear_checkpoints,
            "ai_engine": args.ai_engine,
            "detect_duplicates": not args.no_duplicates,
            "duplicate_strategy": args.duplicate_strategy,
            "generate_folders": args.generate_folders and not args.no_folders,
            "max_bookmarks_per_folder": args.max_bookmarks_per_folder,
            "generate_chrome_html": args.chrome_html,
            "chrome_html_output": args.html_output,
            "html_title": args.html_title,
        }

    def process_arguments(self, validated_args: dict) -> Configuration:
        """
        Process validated arguments and set up configuration.

        Args:
            validated_args: Dictionary of validated arguments

        Returns:
            Configured Configuration object
            
        Raises:
            ValidationError: If AI engine validation fails
        """
        # Initialize configuration
        config = Configuration(validated_args["config_path"])

        # Update configuration with command-line arguments
        config.update_from_args(validated_args)

        # Validate AI engine with configuration (after loading config)
        validated_args["ai_engine"] = validate_ai_engine(
            validated_args["ai_engine"], config
        )

        # Set up logging
        setup_logging(config)

        return config

    def run(self, args=None) -> int:
        """Execute CLI interface."""
        try:
            # Parse and validate arguments
            parsed_args = self.parse_args(args)
            validated_args = self.validate_args(parsed_args)

            # Process arguments and set up configuration
            config = self.process_arguments(validated_args)

            # Set up logger
            logger = logging.getLogger(__name__)
            logger.info("Bookmark Processor CLI starting")
            logger.info(f"Input file: {validated_args['input_path']}")
            logger.info(f"Output file: {validated_args['output_path']}")
            logger.info(f"AI engine: {validated_args['ai_engine']}")
            logger.info(f"Batch size: {validated_args['batch_size']}")
            logger.info(f"Max retries: {validated_args['max_retries']}")

            if validated_args["verbose"]:
                print("✓ Arguments validated and configuration loaded successfully!")
                
                # Handle input display based on mode
                if validated_args['input_path'] is None:
                    print(f"  Input: Auto-detection mode (current directory)")
                    # Show auto-detection details
                    try:
                        from bookmark_processor.core.multi_file_processor import MultiFileProcessor
                        processor = MultiFileProcessor()
                        report = processor.validate_directory_for_auto_detection()
                        print(f"  Detected files: {len(report['valid_files'])}")
                        print(f"  Total estimated bookmarks: {report['total_estimated_bookmarks']}")
                        for file_info in report['valid_files'][:3]:  # Show first 3 files
                            print(f"    - {file_info['name']} ({file_info['format']}, ~{file_info['estimated_bookmarks']} bookmarks)")
                        if len(report['valid_files']) > 3:
                            print(f"    ... and {len(report['valid_files']) - 3} more files")
                    except Exception:
                        pass
                else:
                    print(f"  Input: {validated_args['input_path']}")
                    # Show single file format information
                    try:
                        from bookmark_processor.core.import_module import MultiFormatImporter
                        importer = MultiFormatImporter()
                        file_info = importer.get_file_info(validated_args['input_path'])
                        print(f"  Input format: {file_info['format']}")
                        print(f"  File size: {file_info['size_bytes'] / 1024 / 1024:.2f} MB")
                        if file_info['estimated_bookmarks'] > 0:
                            print(f"  Estimated bookmarks: {file_info['estimated_bookmarks']}")
                    except Exception:
                        pass
                
                print(f"  Output: {validated_args['output_path']}")
                if validated_args["config_path"]:
                    print(f"  Config: {validated_args['config_path']}")
                print(f"  AI engine: {validated_args['ai_engine']}")
                print(f"  Batch size: {validated_args['batch_size']}")
                print(f"  Max retries: {validated_args['max_retries']}")
                print(f"  Resume: {validated_args['resume']}")
                print(f"  Clear checkpoints: {validated_args['clear_checkpoints']}")
                print(f"  Duplicate detection: {validated_args['detect_duplicates']}")
                if validated_args['detect_duplicates']:
                    print(f"  Duplicate strategy: {validated_args['duplicate_strategy']}")

            # Initialize and run the bookmark processor
            processor = BookmarkProcessor(config)
            return processor.run_cli(validated_args)

        except ValidationError as e:
            print(f"Validation Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            # Log the full exception if logger is available
            try:
                logger = logging.getLogger(__name__)
                logger.exception("Unexpected error in CLI")
            except Exception:
                pass
            return 1


def main(args=None):
    """Main entry point for the CLI."""
    cli = CLIInterface()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
