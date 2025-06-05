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
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv --resume
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --batch-size 50 --verbose
  bookmark-processor.exe --input bookmarks.csv --output enhanced.csv \\
    --ai-engine claude --verbose

Cloud AI Setup:
  Copy user_config.ini.template to user_config.ini and add your API keys:
  [ai]
  claude_api_key = your-claude-api-key-here
  openai_api_key = your-openai-api-key-here

For more information, visit: https://github.com/davistroy/bookmark-validator
            """,
        )

        # Add version argument
        parser.add_argument(
            "--version", "-V", action="version", version="%(prog)s 1.0.0"
        )

        # Required arguments
        parser.add_argument(
            "--input",
            "-i",
            required=True,
            help="Input CSV file (raindrop.io export format)",
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
                print(f"  Input: {validated_args['input_path']}")
                print(f"  Output: {validated_args['output_path']}")
                if validated_args["config_path"]:
                    print(f"  Config: {validated_args['config_path']}")
                print(f"  AI engine: {validated_args['ai_engine']}")
                print(f"  Batch size: {validated_args['batch_size']}")
                print(f"  Max retries: {validated_args['max_retries']}")
                print(f"  Resume: {validated_args['resume']}")
                print(f"  Clear checkpoints: {validated_args['clear_checkpoints']}")

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
