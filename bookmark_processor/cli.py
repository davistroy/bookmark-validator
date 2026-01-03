"""
Command-line interface for the Bookmark Validation and Enhancement Tool.

This module provides a modern CLI using Typer and Rich for processing
raindrop.io bookmark exports, validating URLs, generating AI-enhanced
descriptions, and creating optimized tagging systems.
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback for environments without typer/rich
    typer = None

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


# Create console for rich output
console = Console() if RICH_AVAILABLE else None


class AIEngine(str, Enum):
    """Available AI engines for description generation."""

    local = "local"
    claude = "claude"
    openai = "openai"


class DuplicateStrategy(str, Enum):
    """Strategy for resolving duplicate bookmarks."""

    newest = "newest"
    oldest = "oldest"
    most_complete = "most_complete"
    highest_quality = "highest_quality"


class ConfigTemplate(str, Enum):
    """Available configuration templates."""

    basic = "basic"
    claude = "claude"
    openai = "openai"
    performance = "performance"
    large_dataset = "large-dataset"


# Create Typer app if available
if RICH_AVAILABLE:
    app = typer.Typer(
        name="bookmark-processor",
        help="Bookmark Validation and Enhancement Tool - Process raindrop.io exports",
        add_completion=True,
        rich_markup_mode="rich",
    )


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print("[bold]bookmark-processor[/bold] version 2.0.0")
        raise typer.Exit()


def print_config_details(validated_args: dict, config: Configuration):
    """Print detailed configuration information using Rich."""
    if not RICH_AVAILABLE:
        return

    table = Table(title="Configuration Details", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Input/Output
    if validated_args["input_path"]:
        table.add_row("Input", str(validated_args["input_path"]))
    else:
        table.add_row("Input", "Auto-detection mode")
    table.add_row("Output", str(validated_args["output_path"]))

    if validated_args["config_path"]:
        table.add_row("Config", str(validated_args["config_path"]))

    # AI Engine
    ai_engine = validated_args["ai_engine"]
    if ai_engine == "local":
        table.add_row("AI Engine", "local (facebook/bart-large-cnn)")
    elif ai_engine == "claude":
        has_key = config.has_api_key("claude")
        status = "[green]configured[/green]" if has_key else "[red]missing API key[/red]"
        table.add_row("AI Engine", f"claude-3-5-haiku ({status})")
    elif ai_engine == "openai":
        has_key = config.has_api_key("openai")
        status = "[green]configured[/green]" if has_key else "[red]missing API key[/red]"
        table.add_row("AI Engine", f"gpt-4o-mini ({status})")

    # Processing options
    table.add_row("Batch Size", str(validated_args["batch_size"]))
    table.add_row("Max Retries", str(validated_args["max_retries"]))
    table.add_row("Resume", str(validated_args["resume"]))

    # Duplicate detection
    if validated_args["detect_duplicates"]:
        table.add_row(
            "Duplicates", f"detect ({validated_args['duplicate_strategy']})"
        )
    else:
        table.add_row("Duplicates", "disabled")

    console.print(table)


if RICH_AVAILABLE:

    @app.command()
    def process(
        input_file: Optional[Path] = typer.Option(
            None,
            "--input",
            "-i",
            help="Input file (raindrop.io CSV or Chrome HTML). Auto-detects if not specified.",
        ),
        output_file: Path = typer.Option(
            ...,
            "--output",
            "-o",
            help="Output CSV file (raindrop.io import format)",
        ),
        config_file: Optional[Path] = typer.Option(
            None,
            "--config",
            "-c",
            help="Custom configuration file path (TOML or JSON)",
        ),
        resume: bool = typer.Option(
            False,
            "--resume",
            "-r",
            help="Resume from existing checkpoint",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output with detailed information",
        ),
        batch_size: int = typer.Option(
            100,
            "--batch-size",
            "-b",
            min=10,
            max=1000,
            help="Processing batch size",
        ),
        max_retries: int = typer.Option(
            3,
            "--max-retries",
            "-m",
            min=0,
            max=10,
            help="Maximum retry attempts",
        ),
        clear_checkpoints: bool = typer.Option(
            False,
            "--clear-checkpoints",
            help="Clear existing checkpoints and start fresh",
        ),
        ai_engine: AIEngine = typer.Option(
            AIEngine.local,
            "--ai-engine",
            help="AI engine: local (free), claude (claude-3-5-haiku), openai (gpt-4o-mini)",
        ),
        no_duplicates: bool = typer.Option(
            False,
            "--no-duplicates",
            help="Disable duplicate URL detection",
        ),
        duplicate_strategy: DuplicateStrategy = typer.Option(
            DuplicateStrategy.highest_quality,
            "--duplicate-strategy",
            help="Strategy for resolving duplicates",
        ),
        generate_folders: bool = typer.Option(
            True,
            "--generate-folders/--no-folders",
            help="Generate AI-powered semantic folder structure",
        ),
        max_bookmarks_per_folder: int = typer.Option(
            20,
            "--max-bookmarks-per-folder",
            help="Maximum bookmarks per folder",
        ),
        chrome_html: bool = typer.Option(
            False,
            "--chrome-html",
            help="Generate Chrome HTML bookmark file in addition to CSV",
        ),
        html_output: Optional[Path] = typer.Option(
            None,
            "--html-output",
            help="Custom path for Chrome HTML output",
        ),
        html_title: str = typer.Option(
            "Enhanced Bookmarks",
            "--html-title",
            help="Title for Chrome HTML bookmark file",
        ),
        version: bool = typer.Option(
            False,
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ):
        """
        Process bookmark files with URL validation, AI descriptions, and smart tagging.

        [bold]Examples:[/bold]

            bookmark-processor -i bookmarks.csv -o enhanced.csv

            bookmark-processor -i chrome.html -o enhanced.csv --ai-engine claude

            bookmark-processor -i bookmarks.csv -o enhanced.csv --resume --verbose
        """
        try:
            # Validate arguments
            with console.status("[bold green]Validating arguments..."):
                input_path = validate_input_file(str(input_file) if input_file else None)
                if input_path is None:
                    validate_auto_detection_mode()
                output_path = validate_output_file(str(output_file))
                config_path = validate_config_file(
                    str(config_file) if config_file else None
                )
                validated_batch_size = validate_batch_size(batch_size)
                validated_max_retries = validate_max_retries(max_retries)
                validate_conflicting_arguments(resume, clear_checkpoints)

            validated_args = {
                "input_path": input_path,
                "output_path": output_path,
                "config_path": config_path,
                "resume": resume,
                "verbose": verbose,
                "batch_size": validated_batch_size,
                "max_retries": validated_max_retries,
                "clear_checkpoints": clear_checkpoints,
                "ai_engine": ai_engine.value,
                "detect_duplicates": not no_duplicates,
                "duplicate_strategy": duplicate_strategy.value,
                "generate_folders": generate_folders,
                "max_bookmarks_per_folder": max_bookmarks_per_folder,
                "generate_chrome_html": chrome_html,
                "chrome_html_output": str(html_output) if html_output else None,
                "html_title": html_title,
            }

            # Initialize configuration
            config = Configuration(validated_args["config_path"])
            config.update_from_args(validated_args)

            # Validate AI engine
            validated_args["ai_engine"] = validate_ai_engine(
                validated_args["ai_engine"], config
            )

            # Set up logging
            setup_logging(config)

            if verbose:
                console.print(
                    Panel.fit(
                        "[green]Arguments validated successfully![/green]",
                        title="Status",
                    )
                )
                print_config_details(validated_args, config)

            # Run processor
            logger = logging.getLogger(__name__)
            logger.info("Bookmark Processor CLI starting")
            logger.info(f"Input: {validated_args['input_path']}")
            logger.info(f"Output: {validated_args['output_path']}")
            logger.info(f"AI engine: {validated_args['ai_engine']}")

            processor = BookmarkProcessor(config)
            result = processor.run_cli(validated_args)

            if result == 0:
                console.print(
                    Panel.fit(
                        "[bold green]Processing completed successfully![/bold green]",
                        title="Done",
                    )
                )
            return result

        except ValidationError as e:
            console.print(f"[red]Validation Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            logger = logging.getLogger(__name__)
            logger.exception("Unexpected error in CLI")
            raise typer.Exit(1)

    @app.command()
    def create_config(
        template: ConfigTemplate = typer.Argument(
            ...,
            help="Configuration template type",
        ),
        output: Path = typer.Option(
            Path("user_config.toml"),
            "--output",
            "-o",
            help="Output configuration file path",
        ),
    ):
        """
        Create a configuration template file.

        [bold]Templates:[/bold]

            basic         - General purpose, balanced settings
            claude        - Optimized for Claude AI
            openai        - Optimized for OpenAI
            performance   - High-speed local processing
            large-dataset - Conservative for 3000+ bookmarks
        """
        import shutil

        template_files = {
            "basic": "user_config.toml.template",
            "claude": "claude_config.toml.template",
            "openai": "openai_config.toml.template",
            "performance": "local_performance.toml.template",
            "large-dataset": "large_dataset.toml.template",
        }

        template_name = template.value.replace("_", "-")
        template_file = template_files.get(template_name)

        if not template_file:
            console.print(f"[red]Unknown template: {template_name}[/red]")
            raise typer.Exit(1)

        try:
            config_dir = Path(__file__).parent / "config"
            template_path = config_dir / template_file

            if not template_path.exists():
                console.print(f"[red]Template not found: {template_path}[/red]")
                raise typer.Exit(1)

            if output.exists():
                if not typer.confirm(
                    f"Configuration file '{output}' exists. Overwrite?"
                ):
                    console.print("[yellow]Cancelled[/yellow]")
                    raise typer.Exit(0)

            shutil.copy2(template_path, output)

            console.print(
                Panel.fit(
                    f"[green]Created configuration file:[/green] {output}\n"
                    f"[dim]Template:[/dim] {template_name}",
                    title="Success",
                )
            )

            # Template-specific guidance
            guidance = {
                "basic": "Ready to use out of the box with local AI",
                "claude": "Add your Claude API key from console.anthropic.com",
                "openai": "Add your OpenAI API key from platform.openai.com",
                "performance": "Optimized for maximum speed with local AI",
                "large-dataset": "Conservative settings for 3000+ bookmarks",
            }

            console.print(f"\n[bold]Next steps:[/bold] {guidance.get(template_name, '')}")

        except Exception as e:
            console.print(f"[red]Error creating config: {e}[/red]")
            raise typer.Exit(1)


# Fallback CLI class for environments without Typer
class CLIInterface:
    """Legacy command line interface for backward compatibility."""

    def __init__(self):
        if RICH_AVAILABLE:
            # Use Typer app
            self.use_typer = True
        else:
            # Fallback to argparse
            self.use_typer = False
            self._setup_argparse()

    def _setup_argparse(self):
        """Set up argparse fallback."""
        import argparse

        self.parser = argparse.ArgumentParser(
            prog="bookmark-processor",
            description="Bookmark Validation and Enhancement Tool",
        )
        self.parser.add_argument("--version", "-V", action="version", version="2.0.0")
        self.parser.add_argument("--input", "-i", help="Input file")
        self.parser.add_argument("--output", "-o", required=True, help="Output file")
        self.parser.add_argument("--config", "-c", help="Config file")
        self.parser.add_argument("--resume", "-r", action="store_true")
        self.parser.add_argument("--verbose", "-v", action="store_true")
        self.parser.add_argument("--batch-size", "-b", type=int, default=100)
        self.parser.add_argument("--max-retries", "-m", type=int, default=3)
        self.parser.add_argument("--clear-checkpoints", action="store_true")
        self.parser.add_argument(
            "--ai-engine", choices=["local", "claude", "openai"], default="local"
        )
        self.parser.add_argument("--no-duplicates", action="store_true")
        self.parser.add_argument(
            "--duplicate-strategy",
            choices=["newest", "oldest", "most_complete", "highest_quality"],
            default="highest_quality",
        )
        self.parser.add_argument(
            "--generate-folders", action="store_true", default=True
        )
        self.parser.add_argument("--no-folders", action="store_true")
        self.parser.add_argument("--max-bookmarks-per-folder", type=int, default=20)
        self.parser.add_argument("--chrome-html", action="store_true")
        self.parser.add_argument("--html-output", help="Chrome HTML output path")
        self.parser.add_argument(
            "--html-title", default="Enhanced Bookmarks"
        )
        self.parser.add_argument(
            "--create-config",
            choices=["basic", "claude", "openai", "performance", "large-dataset"],
        )

    def run(self, args=None) -> int:
        """Run the CLI."""
        if self.use_typer:
            try:
                app()
                return 0
            except SystemExit as e:
                return e.code if e.code else 0
        else:
            return self._run_argparse(args)

    def _run_argparse(self, args=None) -> int:
        """Run with argparse fallback."""
        try:
            parsed = self.parser.parse_args(args)

            # Handle create-config
            if parsed.create_config:
                return self._handle_create_config(parsed.create_config)

            # Validate
            input_path = validate_input_file(parsed.input)
            if input_path is None:
                validate_auto_detection_mode()
            output_path = validate_output_file(parsed.output)
            config_path = validate_config_file(parsed.config)
            batch_size = validate_batch_size(parsed.batch_size)
            max_retries_val = validate_max_retries(parsed.max_retries)
            validate_conflicting_arguments(parsed.resume, parsed.clear_checkpoints)

            validated_args = {
                "input_path": input_path,
                "output_path": output_path,
                "config_path": config_path,
                "resume": parsed.resume,
                "verbose": parsed.verbose,
                "batch_size": batch_size,
                "max_retries": max_retries_val,
                "clear_checkpoints": parsed.clear_checkpoints,
                "ai_engine": parsed.ai_engine,
                "detect_duplicates": not parsed.no_duplicates,
                "duplicate_strategy": parsed.duplicate_strategy,
                "generate_folders": parsed.generate_folders and not parsed.no_folders,
                "max_bookmarks_per_folder": parsed.max_bookmarks_per_folder,
                "generate_chrome_html": parsed.chrome_html,
                "chrome_html_output": parsed.html_output,
                "html_title": parsed.html_title,
            }

            config = Configuration(validated_args["config_path"])
            config.update_from_args(validated_args)
            validated_args["ai_engine"] = validate_ai_engine(
                validated_args["ai_engine"], config
            )
            setup_logging(config)

            processor = BookmarkProcessor(config)
            return processor.run_cli(validated_args)

        except ValidationError as e:
            print(f"Validation Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _handle_create_config(self, config_type: str) -> int:
        """Handle config template creation."""
        import shutil

        template_files = {
            "basic": "user_config.toml.template",
            "claude": "claude_config.toml.template",
            "openai": "openai_config.toml.template",
            "performance": "local_performance.toml.template",
            "large-dataset": "large_dataset.toml.template",
        }

        template_file = template_files.get(config_type)
        if not template_file:
            print(f"Unknown template: {config_type}")
            return 1

        try:
            config_dir = Path(__file__).parent / "config"
            template_path = config_dir / template_file
            output_path = Path("user_config.toml")

            if not template_path.exists():
                print(f"Template not found: {template_path}")
                return 1

            if output_path.exists():
                response = input(f"'{output_path}' exists. Overwrite? (y/N): ")
                if response.lower() != "y":
                    print("Cancelled")
                    return 0

            shutil.copy2(template_path, output_path)
            print(f"Created: {output_path}")
            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1


def main(args=None):
    """Main entry point for the CLI."""
    cli = CLIInterface()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
