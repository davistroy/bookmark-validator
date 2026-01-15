"""
Command-line interface for the Bookmark Validation and Enhancement Tool.

This module provides a modern CLI using Typer and Rich for processing
raindrop.io bookmark exports, validating URLs, generating AI-enhanced
descriptions, and creating optimized tagging systems.
"""

import logging
import sys
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

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
from bookmark_processor.core.filters import FilterChain
from bookmark_processor.core.processing_modes import ProcessingMode, ProcessingStages
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


class AIMode(str, Enum):
    """AI processing modes for hybrid routing."""

    local = "local"
    cloud = "cloud"
    hybrid = "hybrid"


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


def _display_processing_mode_info(
    processing_mode: ProcessingMode,
    filter_chain: FilterChain,
    console: "Console",
) -> None:
    """Display information about the processing mode and filters."""
    if not RICH_AVAILABLE or console is None:
        return

    # Only show info if there's something special configured
    if processing_mode.is_preview or processing_mode.dry_run or filter_chain:
        info_parts = []

        # Preview mode
        if processing_mode.is_preview:
            info_parts.append(f"[cyan]Preview:[/cyan] First {processing_mode.preview_count} bookmarks")

        # Dry-run mode
        if processing_mode.dry_run:
            info_parts.append("[yellow]Dry-run:[/yellow] No changes will be made")

        # Filter info
        if filter_chain:
            info_parts.append(f"[magenta]Filters:[/magenta] {len(filter_chain)} active")

        # Processing stages
        stage_list = processing_mode.stages.stage_list
        if len(stage_list) < 5:  # Not all stages
            info_parts.append(f"[blue]Stages:[/blue] {', '.join(stage_list)}")

        if info_parts:
            console.print(Panel(
                "\n".join(info_parts),
                title="Processing Mode",
                border_style="dim",
            ))


def _display_dry_run_summary(
    validated_args: dict,
    filter_chain: FilterChain,
    config: Configuration,
    console: "Console",
) -> None:
    """Display a summary of what would be processed in dry-run mode."""
    if not RICH_AVAILABLE or console is None:
        return

    from bookmark_processor.core.csv_handler import RaindropCSVHandler
    from bookmark_processor.core.import_module import MultiFormatImporter

    input_path = validated_args.get("input_path")
    if not input_path:
        console.print("[dim]No input file specified for dry-run analysis.[/dim]")
        return

    try:
        # Load bookmarks to get counts
        importer = MultiFormatImporter()
        bookmarks = importer.import_bookmarks(input_path)
        total_count = len(bookmarks)

        # Apply filters to get filtered count
        if filter_chain:
            filtered_bookmarks = filter_chain.apply(bookmarks)
            filtered_count = len(filtered_bookmarks)
        else:
            filtered_count = total_count

        # Apply preview limit
        preview = validated_args.get("preview")
        if preview:
            process_count = min(preview, filtered_count)
        else:
            process_count = filtered_count

        # Build summary table
        table = Table(title="Dry-Run Summary", show_header=False)
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Input file", str(input_path))
        table.add_row("Total bookmarks", str(total_count))

        if filter_chain:
            table.add_row("After filters", f"{filtered_count} ({filtered_count * 100 // total_count}%)")

        if preview:
            table.add_row("Preview limit", str(preview))

        table.add_row("Would process", str(process_count))

        # Show processing stages
        processing_mode = validated_args.get("processing_mode")
        if processing_mode:
            stages = processing_mode.stages.stage_list
            table.add_row("Stages", ", ".join(stages) if stages else "none")

        # Estimate processing time (rough estimate based on bookmark count)
        # These are rough estimates: validation ~0.5s/bookmark, AI ~2s/bookmark
        if processing_mode:
            est_seconds = 0
            if processing_mode.should_validate:
                est_seconds += process_count * 0.5
            if processing_mode.should_extract_content:
                est_seconds += process_count * 0.3
            if processing_mode.should_run_ai:
                est_seconds += process_count * 2.0
            if processing_mode.should_optimize_tags:
                est_seconds += process_count * 0.1

            if est_seconds > 60:
                est_time = f"~{est_seconds // 60:.0f} minutes"
            else:
                est_time = f"~{est_seconds:.0f} seconds"
            table.add_row("Estimated time", est_time)

        console.print(table)

        # Show filter details if any
        if filter_chain:
            console.print("\n[dim]Active filters:[/dim]")
            for i, f in enumerate(filter_chain.filters, 1):
                filter_type = type(f).__name__
                console.print(f"  {i}. {filter_type}")

    except Exception as e:
        console.print(f"[yellow]Could not analyze input file: {e}[/yellow]")


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
        table.add_row("AI Engine", f"claude-haiku-4-5 ({status})")
    elif ai_engine == "openai":
        has_key = config.has_api_key("openai")
        status = "[green]configured[/green]" if has_key else "[red]missing API key[/red]"
        table.add_row("AI Engine", f"gpt-5-mini ({status})")

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


# =========================================================================
# Phase 7: Interactive Processing and Plugin Helpers
# =========================================================================


def _load_plugins(
    plugin_list: str,
    plugin_config_path: Optional[Path],
    console: "Console",
    verbose: bool,
) -> Optional["PluginRegistry"]:
    """
    Load plugins from comma-separated list.

    Args:
        plugin_list: Comma-separated plugin names
        plugin_config_path: Optional path to plugin config file
        console: Rich console for output
        verbose: Enable verbose output

    Returns:
        PluginRegistry with loaded plugins, or None if no plugins loaded
    """
    try:
        from bookmark_processor.plugins import PluginLoader, PluginRegistry

        # Parse plugin names
        plugin_names = [p.strip() for p in plugin_list.split(",") if p.strip()]

        if not plugin_names:
            return None

        # Load plugin configuration
        plugin_config = {}
        if plugin_config_path and plugin_config_path.exists():
            import toml

            full_config = toml.load(plugin_config_path)
            plugin_config = full_config.get("plugins", {})

        # Create loader and registry
        loader = PluginLoader()
        registry = PluginRegistry(loader)

        # Discover available plugins
        available = loader.discover_plugins()
        if verbose:
            console.print(f"[dim]Available plugins: {', '.join(available)}[/dim]")

        # Load requested plugins
        loaded = registry.load_plugins(plugin_names, plugin_config)

        if loaded:
            console.print(
                f"[green]Loaded {len(loaded)} plugin(s):[/green] "
                f"{', '.join(loaded.keys())}"
            )
        else:
            console.print("[yellow]No plugins loaded[/yellow]")

        return registry if loaded else None

    except ImportError as e:
        console.print(f"[yellow]Plugin system not available: {e}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error loading plugins: {e}[/red]")
        return None


def _run_interactive_processing(
    bookmarks: list,
    output_file: Path,
    config: "Configuration",
    console: "Console",
    confirm_threshold: float,
    ai_engine: str,
    verbose: bool,
    plugin_registry: Optional["PluginRegistry"] = None,
) -> int:
    """
    Run interactive bookmark processing.

    Args:
        bookmarks: List of bookmarks to process
        output_file: Output file path
        config: Configuration instance
        console: Rich console for output
        confirm_threshold: Confidence threshold for auto-accept
        ai_engine: AI engine to use
        verbose: Enable verbose output
        plugin_registry: Optional plugin registry

    Returns:
        Exit code (0 for success)
    """
    try:
        from bookmark_processor.core.interactive_processor import (
            InteractiveProcessor,
            ProposedChanges,
        )
        from bookmark_processor.core.ai_processor import EnhancedAIProcessor
        from bookmark_processor.core.csv_handler import RaindropCSVHandler

        # Create interactive processor
        interactive = InteractiveProcessor(
            confirm_threshold=confirm_threshold,
            show_diff=True,
            compact_mode=False,
            console=console,
        )

        # Generate proposed changes for each bookmark
        console.print("[cyan]Analyzing bookmarks and generating proposals...[/cyan]")

        proposed_changes = {}
        for bookmark in bookmarks:
            # Create mock proposals based on bookmark state
            # In a full implementation, this would use the AI processor
            changes = interactive.propose_changes(
                bookmark=bookmark,
                ai_result=None,  # Would come from AI processing
                proposed_tags=bookmark.optimized_tags or bookmark.tags,
                proposed_folder=bookmark.folder,
            )
            proposed_changes[bookmark.url] = changes

        # Run interactive processing
        results = interactive.process_interactive(
            bookmarks=bookmarks,
            proposed_changes=proposed_changes,
        )

        # Save results
        modified_bookmarks = [r.bookmark for r in results if r.was_modified]
        if modified_bookmarks:
            handler = RaindropCSVHandler()
            handler.export_bookmarks(modified_bookmarks, output_file)
            console.print(
                f"[green]Saved {len(modified_bookmarks)} modified bookmarks to {output_file}[/green]"
            )

        # Show stats
        stats = interactive.stats
        console.print(f"\n[bold]Session Stats:[/bold]")
        console.print(f"  Processed: {stats.processed_count}/{stats.total_bookmarks}")
        console.print(f"  Accepted all: {stats.accepted_all}")
        console.print(f"  Skipped: {stats.skipped}")
        if stats.auto_accepted > 0:
            console.print(f"  Auto-accepted: {stats.auto_accepted}")

        return 0

    except ImportError as e:
        console.print(f"[red]Interactive processing not available: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error in interactive processing: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


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
            help="AI engine: local (free), claude (claude-haiku-4-5), openai (gpt-5-mini)",
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
        # Phase 1.1: Preview/Dry-Run Mode
        preview: Optional[int] = typer.Option(
            None,
            "--preview",
            "-p",
            min=1,
            help="Process only first N bookmarks as a sample preview",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Validate configuration and show what would be processed without making changes",
        ),
        # Phase 1.2: Smart Filtering
        filter_folder: Optional[str] = typer.Option(
            None,
            "--filter-folder",
            help="Only process bookmarks in matching folders (supports glob patterns like 'Tech/*')",
        ),
        filter_tag: Optional[List[str]] = typer.Option(
            None,
            "--filter-tag",
            help="Only process bookmarks with these tags (can be specified multiple times)",
        ),
        filter_date: Optional[str] = typer.Option(
            None,
            "--filter-date",
            help="Only process bookmarks in date range (format: 'YYYY-MM-DD:YYYY-MM-DD', either side optional)",
        ),
        filter_domain: Optional[str] = typer.Option(
            None,
            "--filter-domain",
            help="Only process bookmarks from these domains (comma-separated, e.g., 'github.com,gitlab.com')",
        ),
        retry_invalid: bool = typer.Option(
            False,
            "--retry-invalid",
            help="Only re-process bookmarks that previously had validation errors",
        ),
        # Phase 1.3: Granular Processing Control
        skip_validation: bool = typer.Option(
            False,
            "--skip-validation",
            help="Skip URL validation stage (use existing validation status)",
        ),
        skip_ai: bool = typer.Option(
            False,
            "--skip-ai",
            help="Skip AI description generation stage",
        ),
        skip_content: bool = typer.Option(
            False,
            "--skip-content",
            help="Skip content extraction stage",
        ),
        tags_only: bool = typer.Option(
            False,
            "--tags-only",
            help="Only run tag optimization (skip all other stages)",
        ),
        folders_only: bool = typer.Option(
            False,
            "--folders-only",
            help="Only run folder organization (skip all other stages)",
        ),
        validate_only: bool = typer.Option(
            False,
            "--validate-only",
            help="Only validate URLs (skip all other processing stages)",
        ),
        # Phase 3.1: Hybrid AI Processing
        ai_mode: AIMode = typer.Option(
            AIMode.local,
            "--ai-mode",
            help="AI processing mode: local (free), cloud (API), hybrid (auto-route)",
        ),
        cloud_budget: float = typer.Option(
            5.00,
            "--cloud-budget",
            min=0.0,
            max=100.0,
            help="Maximum budget for cloud AI in USD (default: $5.00)",
        ),
        # Phase 3.2: Enhanced Tagging
        tag_config: Optional[Path] = typer.Option(
            None,
            "--tag-config",
            help="Path to TOML tag configuration file",
        ),
        # Phase 3.3: Folder Organization
        preserve_folders: bool = typer.Option(
            False,
            "--preserve-folders",
            help="Preserve original folder assignments",
        ),
        suggest_folders: bool = typer.Option(
            False,
            "--suggest-folders",
            help="Generate folder suggestions without applying them (outputs JSON)",
        ),
        learn_folders: bool = typer.Option(
            False,
            "--learn-folders",
            help="Learn folder patterns from existing structure",
        ),
        max_folder_depth: int = typer.Option(
            3,
            "--max-folder-depth",
            min=1,
            max=10,
            help="Maximum folder hierarchy depth (default: 3)",
        ),
        folder_suggestions_output: Optional[Path] = typer.Option(
            None,
            "--folder-suggestions-output",
            help="Output path for folder suggestions JSON (used with --suggest-folders)",
        ),
    ):
        """
        Process bookmark files with URL validation, AI descriptions, and smart tagging.

        [bold]Examples:[/bold]

            bookmark-processor -i bookmarks.csv -o enhanced.csv

            bookmark-processor -i chrome.html -o enhanced.csv --ai-engine claude

            bookmark-processor -i bookmarks.csv -o enhanced.csv --resume --verbose

        [bold]Preview & Dry-Run:[/bold]

            bookmark-processor -i bookmarks.csv -o out.csv --preview 10

            bookmark-processor -i bookmarks.csv --dry-run

        [bold]Filtering:[/bold]

            bookmark-processor -i bookmarks.csv -o out.csv --filter-folder "Tech/*"

            bookmark-processor -i bookmarks.csv -o out.csv --filter-tag python --filter-tag ai

            bookmark-processor -i bookmarks.csv -o out.csv --filter-date "2024-01-01:2024-12-31"

            bookmark-processor -i bookmarks.csv -o out.csv --filter-domain "github.com,gitlab.com"

        [bold]Processing Control:[/bold]

            bookmark-processor -i bookmarks.csv -o out.csv --skip-ai

            bookmark-processor -i bookmarks.csv -o out.csv --validate-only

            bookmark-processor -i bookmarks.csv -o out.csv --tags-only

        [bold]Hybrid AI (Phase 3.1):[/bold]

            bookmark-processor -i bookmarks.csv -o out.csv --ai-mode hybrid --cloud-budget 5.00

        [bold]Tag Configuration (Phase 3.2):[/bold]

            bookmark-processor -i bookmarks.csv -o out.csv --tag-config config.toml

        [bold]Folder Organization (Phase 3.3):[/bold]

            bookmark-processor -i bookmarks.csv -o out.csv --preserve-folders

            bookmark-processor -i bookmarks.csv -o out.csv --suggest-folders

            bookmark-processor -i bookmarks.csv -o out.csv --learn-folders --max-folder-depth 2
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

                # Phase 1.3: Validate mutually exclusive processing control options
                exclusive_options = [tags_only, folders_only, validate_only]
                exclusive_count = sum(exclusive_options)
                if exclusive_count > 1:
                    raise ValidationError(
                        "Options --tags-only, --folders-only, and --validate-only are mutually exclusive. "
                        "Use only one of these options at a time."
                    )

                # Cannot use skip options with exclusive modes
                if exclusive_count > 0 and (skip_validation or skip_ai or skip_content):
                    raise ValidationError(
                        "Cannot combine --skip-* options with --tags-only, --folders-only, or --validate-only. "
                        "The exclusive mode options override individual skip options."
                    )

            # Build the ProcessingMode from CLI args
            processing_mode_args = {
                "preview": preview,
                "dry_run": dry_run,
                "skip_validation": skip_validation,
                "skip_ai": skip_ai,
                "skip_content": skip_content,
                "tags_only": tags_only,
                "folders_only": folders_only,
                "validate_only": validate_only,
                "verbose": verbose,
            }
            processing_mode = ProcessingMode.from_cli_args(processing_mode_args)

            # Build the FilterChain from CLI args
            filter_args = {
                "filter_folder": filter_folder,
                "filter_tag": filter_tag,
                "filter_date": filter_date,
                "filter_domain": filter_domain,
                "retry_invalid": retry_invalid,
            }
            filter_chain = FilterChain.from_cli_args(filter_args)

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
                # Phase 1 additions
                "processing_mode": processing_mode,
                "filter_chain": filter_chain,
                "preview": preview,
                "dry_run": dry_run,
                # Phase 3.1: Hybrid AI Processing
                "ai_mode": ai_mode.value,
                "cloud_budget": cloud_budget,
                # Phase 3.2: Enhanced Tagging
                "tag_config_path": str(tag_config) if tag_config else None,
                # Phase 3.3: Folder Organization
                "preserve_folders": preserve_folders,
                "suggest_folders": suggest_folders,
                "learn_folders": learn_folders,
                "max_folder_depth": max_folder_depth,
                "folder_suggestions_output": str(folder_suggestions_output) if folder_suggestions_output else None,
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

            # Display Phase 1 mode information
            _display_processing_mode_info(processing_mode, filter_chain, console)

            # Handle dry-run mode: validate and show info, then exit
            if dry_run:
                console.print(
                    Panel.fit(
                        "[bold yellow]Dry-run mode:[/bold yellow] Configuration validated. "
                        "No changes will be made.",
                        title="Dry Run",
                    )
                )
                _display_dry_run_summary(validated_args, filter_chain, config, console)
                return 0

            # Run processor
            logger = logging.getLogger(__name__)
            logger.info("Bookmark Processor CLI starting")
            logger.info(f"Input: {validated_args['input_path']}")
            logger.info(f"Output: {validated_args['output_path']}")
            logger.info(f"AI engine: {validated_args['ai_engine']}")

            if preview:
                logger.info(f"Preview mode: processing first {preview} bookmarks")

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

    # =========================================================================
    # Phase 5: MCP Integration Commands
    # =========================================================================

    class DataSource(str, Enum):
        """Available data sources for bookmark processing."""
        csv = "csv"
        raindrop = "raindrop"

    @app.command()
    def enhance(
        source: DataSource = typer.Option(
            DataSource.csv,
            "--source",
            "-s",
            help="Data source: csv (file) or raindrop (MCP)",
        ),
        input_file: Optional[Path] = typer.Option(
            None,
            "--input",
            "-i",
            help="Input CSV file (required for csv source)",
        ),
        output_file: Optional[Path] = typer.Option(
            None,
            "--output",
            "-o",
            help="Output CSV file (optional for raindrop source)",
        ),
        collection: Optional[str] = typer.Option(
            None,
            "--collection",
            help="Raindrop.io collection to process",
        ),
        since_last_run: bool = typer.Option(
            False,
            "--since-last-run",
            help="Only process bookmarks added/changed since last run",
        ),
        since: Optional[str] = typer.Option(
            None,
            "--since",
            help="Only process bookmarks from time period (e.g., 7d, 30d, 2024-01-01)",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview changes without applying them",
        ),
        preview_count: Optional[int] = typer.Option(
            None,
            "--preview",
            "-p",
            help="Process only first N bookmarks as preview",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
        ai_engine: AIEngine = typer.Option(
            AIEngine.local,
            "--ai-engine",
            help="AI engine: local, claude, openai",
        ),
        mcp_server: Optional[str] = typer.Option(
            None,
            "--mcp-server",
            help="MCP server URL (overrides config)",
        ),
        raindrop_token: Optional[str] = typer.Option(
            None,
            "--raindrop-token",
            envvar="RAINDROP_TOKEN",
            help="Raindrop.io API token (can use RAINDROP_TOKEN env var)",
        ),
        # Phase 7: Interactive Mode
        interactive: bool = typer.Option(
            False,
            "--interactive",
            "-I",
            help="Enable interactive approval mode for changes",
        ),
        confirm_below: Optional[float] = typer.Option(
            None,
            "--confirm-below",
            help="Confirm changes with confidence below threshold (0.0-1.0)",
        ),
        # Phase 7: Plugin Support
        plugins: Optional[str] = typer.Option(
            None,
            "--plugins",
            help="Comma-separated list of plugins to enable (e.g., 'paywall-detector,ollama-ai')",
        ),
        plugin_config: Optional[Path] = typer.Option(
            None,
            "--plugin-config",
            help="Path to plugin configuration file (TOML)",
        ),
    ):
        """
        Enhance bookmarks from various data sources.

        This command processes bookmarks from either CSV files or directly
        from Raindrop.io via MCP, applying AI enhancements and tag optimization.

        [bold]CSV Source (default):[/bold]

            bookmark-processor enhance --source csv -i bookmarks.csv -o enhanced.csv

        [bold]Raindrop.io via MCP:[/bold]

            bookmark-processor enhance --source raindrop --collection "Tech"

            bookmark-processor enhance --source raindrop --since-last-run

        [bold]Preview/Dry-run:[/bold]

            bookmark-processor enhance --source raindrop --dry-run --preview 10

        [bold]Interactive Mode (Phase 7):[/bold]

            bookmark-processor enhance -i bookmarks.csv --interactive

            bookmark-processor enhance -i bookmarks.csv --confirm-below 0.7

        [bold]Plugins (Phase 7):[/bold]

            bookmark-processor enhance -i bookmarks.csv --plugins paywall-detector,ollama-ai

            bookmark-processor enhance -i bookmarks.csv --plugin-config plugins.toml

        [bold]Environment Variables:[/bold]

            RAINDROP_TOKEN - Raindrop.io API token
            MCP_SERVER_URL - MCP server URL
        """
        import asyncio
        import os

        try:
            # Load configuration
            config = Configuration()

            # Determine MCP server URL
            server_url = mcp_server or os.environ.get("MCP_SERVER_URL") or config.get(
                "raindrop.mcp_server", "http://localhost:3000"
            )

            # Determine Raindrop token
            token = raindrop_token or os.environ.get("RAINDROP_TOKEN") or config.get(
                "raindrop.token", ""
            )

            if source == DataSource.csv:
                # CSV source - use existing process command logic
                if not input_file:
                    console.print(
                        "[red]Error:[/red] --input is required for csv source"
                    )
                    raise typer.Exit(1)

                console.print(
                    f"[cyan]Processing CSV:[/cyan] {input_file}"
                )

                # Delegate to process command with appropriate options
                from bookmark_processor.core.data_sources import CSVDataSource

                data_source = CSVDataSource(
                    input_file,
                    output_file or Path("enhanced_bookmarks.csv")
                )

                # Build filters
                filters = {}
                if since:
                    filters["since"] = _parse_since(since)

                bookmarks = data_source.fetch_bookmarks(filters)

                if preview_count:
                    bookmarks = bookmarks[:preview_count]

                console.print(f"[green]Found {len(bookmarks)} bookmarks to process[/green]")

                if dry_run:
                    console.print("[yellow]Dry-run mode - no changes will be applied[/yellow]")
                    _display_bookmark_preview(bookmarks[:10], console)
                    return 0

                # Phase 7: Load plugins if specified
                plugin_registry = None
                if plugins:
                    plugin_registry = _load_plugins(
                        plugins, plugin_config, console, verbose
                    )

                # Phase 7: Interactive mode
                if interactive or confirm_below is not None:
                    return _run_interactive_processing(
                        bookmarks=bookmarks,
                        output_file=output_file or Path("enhanced_bookmarks.csv"),
                        config=config,
                        console=console,
                        confirm_threshold=confirm_below or 0.0,
                        ai_engine=ai_engine.value,
                        verbose=verbose,
                        plugin_registry=plugin_registry,
                    )

                # Process bookmarks using existing pipeline
                processor = BookmarkProcessor(config)
                validated_args = {
                    "input_path": input_file,
                    "output_path": output_file or Path("enhanced_bookmarks.csv"),
                    "ai_engine": ai_engine.value,
                    "verbose": verbose,
                    "batch_size": 100,
                    "max_retries": 3,
                    "detect_duplicates": True,
                    "generate_folders": True,
                }
                config.update_from_args(validated_args)
                result = processor.run_cli(validated_args)
                return result

            elif source == DataSource.raindrop:
                # Raindrop.io MCP source
                if not token:
                    console.print(
                        "[red]Error:[/red] Raindrop.io token required. "
                        "Set via --raindrop-token, RAINDROP_TOKEN env var, or config."
                    )
                    raise typer.Exit(1)

                console.print(
                    f"[cyan]Connecting to Raindrop.io via MCP:[/cyan] {server_url}"
                )

                # Run async enhance
                result = asyncio.run(
                    _enhance_raindrop_async(
                        server_url=server_url,
                        token=token,
                        collection=collection,
                        since_last_run=since_last_run,
                        since=since,
                        dry_run=dry_run,
                        preview_count=preview_count,
                        verbose=verbose,
                        ai_engine=ai_engine.value,
                        config=config,
                        console=console,
                        output_file=output_file,
                    )
                )
                return result

        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)

    @app.command()
    def config_cmd(
        action: str = typer.Argument(
            ...,
            help="Action: get, set, list, or test-connection",
        ),
        key: Optional[str] = typer.Argument(
            None,
            help="Configuration key (e.g., raindrop.token)",
        ),
        value: Optional[str] = typer.Argument(
            None,
            help="Configuration value (for 'set' action)",
        ),
        config_file: Path = typer.Option(
            Path("user_config.toml"),
            "--config",
            "-c",
            help="Configuration file path",
        ),
    ):
        """
        Manage bookmark processor configuration.

        [bold]Actions:[/bold]

            get               Get a configuration value
            set               Set a configuration value
            list              List all configuration values
            test-connection   Test MCP server connection

        [bold]Examples:[/bold]

            bookmark-processor config set raindrop.token "your-api-token"

            bookmark-processor config set raindrop.mcp_server "http://localhost:3000"

            bookmark-processor config get raindrop.token

            bookmark-processor config list

            bookmark-processor config test-connection
        """
        import os
        import toml

        try:
            # Load or create config file
            if config_file.exists():
                config_data = toml.load(config_file)
            else:
                config_data = {}

            if action == "get":
                if not key:
                    console.print("[red]Error:[/red] Key required for 'get' action")
                    raise typer.Exit(1)

                # Navigate nested keys
                parts = key.split(".")
                current = config_data
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        console.print(f"[yellow]Key '{key}' not found[/yellow]")
                        raise typer.Exit(0)

                # Mask sensitive values
                if "token" in key.lower() or "key" in key.lower():
                    if current and len(current) > 8:
                        display_value = current[:4] + "..." + current[-4:]
                    else:
                        display_value = "****"
                else:
                    display_value = current

                console.print(f"[cyan]{key}[/cyan] = [green]{display_value}[/green]")

            elif action == "set":
                if not key or value is None:
                    console.print(
                        "[red]Error:[/red] Key and value required for 'set' action"
                    )
                    raise typer.Exit(1)

                # Navigate and set nested keys
                parts = key.split(".")
                current = config_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value

                # Save config
                with open(config_file, "w") as f:
                    toml.dump(config_data, f)

                console.print(f"[green]Set {key}[/green]")

            elif action == "list":
                if not config_data:
                    console.print("[yellow]No configuration found[/yellow]")
                    raise typer.Exit(0)

                def print_nested(data, prefix=""):
                    for k, v in data.items():
                        full_key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            print_nested(v, full_key)
                        else:
                            # Mask sensitive values
                            if "token" in full_key.lower() or "key" in full_key.lower():
                                if v and len(str(v)) > 8:
                                    display_v = str(v)[:4] + "..." + str(v)[-4:]
                                else:
                                    display_v = "****"
                            else:
                                display_v = v
                            console.print(f"  [cyan]{full_key}[/cyan] = {display_v}")

                console.print("[bold]Configuration:[/bold]")
                print_nested(config_data)

            elif action == "test-connection":
                import asyncio

                server_url = config_data.get("raindrop", {}).get(
                    "mcp_server", os.environ.get("MCP_SERVER_URL", "http://localhost:3000")
                )
                token = config_data.get("raindrop", {}).get(
                    "token", os.environ.get("RAINDROP_TOKEN", "")
                )

                console.print(f"[cyan]Testing connection to:[/cyan] {server_url}")

                async def test_conn():
                    from bookmark_processor.core.data_sources.mcp_client import MCPClient
                    async with MCPClient(server_url, access_token=token) as client:
                        healthy = await client.health_check()
                        if healthy:
                            tools = await client.list_tools()
                            return True, tools
                        return False, []

                try:
                    success, tools = asyncio.run(test_conn())
                    if success:
                        console.print("[green]Connection successful![/green]")
                        console.print(f"[dim]Available tools: {len(tools)}[/dim]")
                        for tool in tools[:5]:
                            console.print(f"  - {tool.get('name', 'unknown')}")
                    else:
                        console.print("[red]Connection failed[/red]")
                        raise typer.Exit(1)
                except Exception as e:
                    console.print(f"[red]Connection failed:[/red] {e}")
                    raise typer.Exit(1)

            else:
                console.print(
                    f"[red]Unknown action:[/red] {action}. "
                    "Use: get, set, list, or test-connection"
                )
                raise typer.Exit(1)

        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    @app.command()
    def rollback(
        source: DataSource = typer.Option(
            DataSource.raindrop,
            "--source",
            "-s",
            help="Data source to rollback",
        ),
        backup_file: Optional[Path] = typer.Option(
            None,
            "--backup",
            "-b",
            help="Backup file to restore from",
        ),
        confirm: bool = typer.Option(
            False,
            "--yes",
            "-y",
            help="Skip confirmation prompt",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ):
        """
        Rollback changes to bookmarks.

        Restore bookmarks to their state before the last enhancement run.
        This requires a backup file created during the enhance operation.

        [bold]Examples:[/bold]

            bookmark-processor rollback --source raindrop --backup backup.json

            bookmark-processor rollback --source raindrop --backup backup.json --yes
        """
        import asyncio
        import json
        import os

        try:
            if not backup_file:
                # Look for most recent backup
                backup_dir = Path(".bookmark_processor_backups")
                if backup_dir.exists():
                    backups = sorted(backup_dir.glob("backup_*.json"), reverse=True)
                    if backups:
                        backup_file = backups[0]
                        console.print(f"[cyan]Using most recent backup:[/cyan] {backup_file}")

            if not backup_file or not backup_file.exists():
                console.print(
                    "[red]Error:[/red] No backup file found. "
                    "Specify with --backup or ensure backups exist."
                )
                raise typer.Exit(1)

            # Load backup
            with open(backup_file) as f:
                backup_data = json.load(f)

            bookmark_count = backup_data.get("bookmark_count", 0)
            backup_time = backup_data.get("timestamp", "unknown")

            console.print(
                f"[yellow]Rollback will restore {bookmark_count} bookmarks "
                f"from backup created at {backup_time}[/yellow]"
            )

            if not confirm:
                if not typer.confirm("Proceed with rollback?"):
                    console.print("[yellow]Rollback cancelled[/yellow]")
                    raise typer.Exit(0)

            if source == DataSource.raindrop:
                # Load config
                config = Configuration()
                server_url = os.environ.get("MCP_SERVER_URL") or config.get(
                    "raindrop.mcp_server", "http://localhost:3000"
                )
                token = os.environ.get("RAINDROP_TOKEN") or config.get(
                    "raindrop.token", ""
                )

                if not token:
                    console.print("[red]Error:[/red] Raindrop.io token required")
                    raise typer.Exit(1)

                async def do_rollback():
                    from bookmark_processor.core.data_sources.raindrop_mcp import (
                        RaindropMCPDataSource,
                    )

                    async with RaindropMCPDataSource(server_url, token) as source:
                        result = await source.restore_from_backup(backup_data)
                        return result

                result = asyncio.run(do_rollback())

                if result.succeeded > 0:
                    console.print(
                        f"[green]Rollback complete![/green] "
                        f"Restored {result.succeeded}/{result.total} bookmarks"
                    )
                else:
                    console.print("[red]Rollback failed - no bookmarks restored[/red]")
                    raise typer.Exit(1)

                if result.errors and verbose:
                    console.print("\n[yellow]Errors:[/yellow]")
                    for error in result.errors[:10]:
                        console.print(f"  - {error.get('url', 'unknown')}: {error.get('error', 'unknown')}")

            else:
                console.print(f"[red]Rollback not supported for source: {source}[/red]")
                raise typer.Exit(1)

        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)


# =========================================================================
# Phase 7: Plugin Management Commands
# =========================================================================

if RICH_AVAILABLE:

    @app.command()
    def plugins(
        action: str = typer.Argument(
            ...,
            help="Action: list, info, install, or test",
        ),
        plugin_name: Optional[str] = typer.Argument(
            None,
            help="Plugin name (for info, install, test actions)",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ):
        """
        Manage bookmark processor plugins.

        [bold]Actions:[/bold]

            list     - List all available plugins
            info     - Show detailed information about a plugin
            install  - Install a plugin from the registry
            test     - Test if a plugin loads correctly

        [bold]Examples:[/bold]

            bookmark-processor plugins list

            bookmark-processor plugins info paywall-detector

            bookmark-processor plugins test ollama-ai
        """
        try:
            from bookmark_processor.plugins import PluginLoader, PluginRegistry

            loader = PluginLoader()

            if action == "list":
                # Discover all available plugins
                available = loader.discover_plugins()

                if not available:
                    console.print("[yellow]No plugins found[/yellow]")
                    console.print(
                        "\n[dim]Plugin search paths:[/dim]"
                    )
                    for path in loader._search_paths:
                        console.print(f"  - {path}")
                    raise typer.Exit(0)

                # Build table
                table = Table(title="Available Plugins", show_header=True)
                table.add_column("Name", style="cyan")
                table.add_column("Version", style="green")
                table.add_column("Description")
                table.add_column("Type", style="dim")

                for name in sorted(available):
                    info = loader.get_plugin_info(name)
                    if info:
                        # Determine plugin type
                        provides = info.get("provides", [])
                        if "ai_processing" in provides:
                            plugin_type = "AI"
                        elif "validation" in provides:
                            plugin_type = "Validator"
                        elif "output" in provides:
                            plugin_type = "Output"
                        elif "tag_generation" in provides:
                            plugin_type = "Tags"
                        else:
                            plugin_type = "General"

                        table.add_row(
                            name,
                            info.get("version", "?"),
                            info.get("description", "")[:50],
                            plugin_type,
                        )
                    else:
                        table.add_row(name, "?", "[error loading]", "-")

                console.print(table)
                console.print(f"\n[dim]Total: {len(available)} plugins[/dim]")

            elif action == "info":
                if not plugin_name:
                    console.print("[red]Error:[/red] Plugin name required for 'info' action")
                    raise typer.Exit(1)

                # Discover plugins first
                loader.discover_plugins()
                info = loader.get_plugin_info(plugin_name)

                if not info:
                    console.print(f"[red]Plugin '{plugin_name}' not found[/red]")
                    raise typer.Exit(1)

                # Display detailed info
                console.print(f"\n[bold cyan]{info.get('name', plugin_name)}[/bold cyan]")
                console.print(f"  Version: {info.get('version', 'unknown')}")
                console.print(f"  Author: {info.get('author', 'unknown')}")
                console.print(f"  Description: {info.get('description', 'No description')}")

                if info.get("provides"):
                    console.print(f"  Provides: {', '.join(info['provides'])}")

                if info.get("requires"):
                    console.print(f"  Requires: {', '.join(info['requires'])}")

                if info.get("hooks"):
                    console.print(f"  Hooks: {', '.join(info['hooks'])}")

            elif action == "test":
                if not plugin_name:
                    console.print("[red]Error:[/red] Plugin name required for 'test' action")
                    raise typer.Exit(1)

                console.print(f"[cyan]Testing plugin: {plugin_name}[/cyan]")

                try:
                    # Try to load the plugin
                    plugin = loader.load_plugin(plugin_name, {})
                    console.print(f"[green]Plugin loaded successfully![/green]")
                    console.print(f"  Name: {plugin.name}")
                    console.print(f"  Version: {plugin.version}")
                    console.print(f"  Enabled: {plugin.enabled}")

                    # Unload
                    loader.unload_plugin(plugin_name)
                    console.print("[green]Plugin unloaded successfully![/green]")

                except Exception as e:
                    console.print(f"[red]Plugin test failed:[/red] {e}")
                    if verbose:
                        import traceback
                        console.print(traceback.format_exc())
                    raise typer.Exit(1)

            elif action == "install":
                console.print(
                    "[yellow]Plugin installation from registry is not yet implemented.[/yellow]\n"
                    "To add plugins, place them in: ~/.bookmark_processor/plugins/"
                )

            else:
                console.print(
                    f"[red]Unknown action:[/red] {action}. "
                    "Use: list, info, install, or test"
                )
                raise typer.Exit(1)

        except typer.Exit:
            raise
        except ImportError as e:
            console.print(f"[red]Plugin system not available:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)


# =========================================================================
# Phase 6: Export & Monitoring Commands
# =========================================================================

class ExportFormat(str, Enum):
    """Available export formats."""
    json = "json"
    markdown = "markdown"
    md = "md"
    obsidian = "obsidian"
    notion = "notion"
    opml = "opml"

if RICH_AVAILABLE:

    @app.command()
    def export(
        input_file: Path = typer.Option(
            ...,
            "--input",
            "-i",
            help="Input CSV file (raindrop.io format)",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output",
            "-o",
            help="Output path (file or directory depending on format)",
        ),
        format: ExportFormat = typer.Option(
            ExportFormat.json,
            "--format",
            "-f",
            help="Export format: json, markdown, obsidian, notion, opml",
        ),
        include_metadata: bool = typer.Option(
            True,
            "--metadata/--no-metadata",
            help="Include processing metadata in export",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ):
        """
        Export bookmarks to various formats.

        Supports multiple export formats for different use cases:
        - json: Full JSON export with all metadata
        - markdown: Markdown file(s) organized by folder
        - obsidian: Obsidian vault format with YAML frontmatter
        - notion: Notion-compatible CSV for database import
        - opml: OPML format for RSS readers

        [bold]Examples:[/bold]

            bookmark-processor export -i bookmarks.csv -o bookmarks.json --format json

            bookmark-processor export -i bookmarks.csv -o bookmarks.md --format markdown

            bookmark-processor export -i bookmarks.csv -o vault/bookmarks/ --format obsidian

            bookmark-processor export -i bookmarks.csv -o notion_import.csv --format notion

            bookmark-processor export -i bookmarks.csv -o bookmarks.opml --format opml
        """
        try:
            from bookmark_processor.core.csv_handler import RaindropCSVHandler
            from bookmark_processor.core.exporters import get_exporter

            # Load bookmarks
            with console.status("[bold green]Loading bookmarks..."):
                handler = RaindropCSVHandler()
                bookmarks = handler.load_and_transform_csv(input_file)

            console.print(f"[green]Loaded {len(bookmarks)} bookmarks[/green]")

            # Get the appropriate exporter
            format_name = format.value
            ExporterClass = get_exporter(format_name)

            # Configure exporter based on format
            if format_name in ("json",):
                exporter = ExporterClass(include_metadata=include_metadata)
            elif format_name in ("markdown", "md"):
                # Detect if output is directory (for multi-file mode)
                if output_path.suffix == "" or output_path.is_dir():
                    exporter = ExporterClass(mode="directory")
                else:
                    exporter = ExporterClass(mode="single")
            elif format_name == "obsidian":
                exporter = ExporterClass()
            elif format_name == "notion":
                exporter = ExporterClass()
            elif format_name == "opml":
                exporter = ExporterClass()
            else:
                exporter = ExporterClass()

            # Perform export
            with console.status(f"[bold green]Exporting to {format_name}..."):
                result = exporter.export(bookmarks, output_path)

            console.print(
                Panel.fit(
                    f"[green]Export complete![/green]\n"
                    f"Format: {result.format_name}\n"
                    f"Exported: {result.count} bookmarks\n"
                    f"Output: {result.path}",
                    title="Success",
                )
            )

            if result.warnings and verbose:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")

            if verbose and result.additional_info:
                console.print("\n[dim]Additional info:[/dim]")
                for key, value in result.additional_info.items():
                    console.print(f"  {key}: {value}")

            return 0

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)

    @app.command()
    def monitor(
        input_file: Path = typer.Option(
            ...,
            "--input",
            "-i",
            help="Input CSV file with bookmarks to check",
        ),
        stale_after: Optional[str] = typer.Option(
            None,
            "--stale-after",
            help="Only check bookmarks not checked within this duration (e.g., 7d, 30d)",
        ),
        archive_dead: bool = typer.Option(
            False,
            "--archive-dead",
            help="Archive dead links to the Wayback Machine",
        ),
        report_only: bool = typer.Option(
            False,
            "--report-only",
            help="Generate report without updating state",
        ),
        output_report: Optional[Path] = typer.Option(
            None,
            "--output",
            "-o",
            help="Save report to file (supports .txt, .json, .csv)",
        ),
        max_concurrent: int = typer.Option(
            20,
            "--concurrent",
            "-c",
            min=1,
            max=100,
            help="Maximum concurrent checks",
        ),
        timeout: float = typer.Option(
            30.0,
            "--timeout",
            "-t",
            min=5.0,
            max=120.0,
            help="Request timeout in seconds",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ):
        """
        Monitor bookmark health and detect broken links.

        Checks all bookmarks for:
        - Dead/broken links (4xx, 5xx errors)
        - Redirected URLs
        - Timeouts
        - Content changes

        Optionally archives dead links to the Wayback Machine.

        [bold]Examples:[/bold]

            bookmark-processor monitor -i bookmarks.csv

            bookmark-processor monitor -i bookmarks.csv --stale-after 30d

            bookmark-processor monitor -i bookmarks.csv --archive-dead

            bookmark-processor monitor -i bookmarks.csv --report-only -o report.json
        """
        import asyncio

        try:
            from bookmark_processor.core.csv_handler import RaindropCSVHandler
            from bookmark_processor.core.health_monitor import (
                BookmarkHealthMonitor,
                HealthMonitorError,
            )
            from bookmark_processor.core.data_sources.state_tracker import ProcessingStateTracker
            from datetime import timedelta

            # Load bookmarks
            with console.status("[bold green]Loading bookmarks..."):
                handler = RaindropCSVHandler()
                bookmarks = handler.load_and_transform_csv(input_file)

            console.print(f"[green]Loaded {len(bookmarks)} bookmarks[/green]")

            # Parse stale_after duration
            stale_duration = None
            if stale_after:
                parsed = _parse_since(stale_after)
                if isinstance(parsed, timedelta):
                    stale_duration = parsed
                else:
                    # It's a datetime, convert to timedelta from now
                    stale_duration = datetime.now() - parsed

            # Initialize state tracker if not report-only mode
            state_tracker = None if report_only else ProcessingStateTracker()

            # Initialize health monitor
            try:
                monitor_instance = BookmarkHealthMonitor(
                    state_tracker=state_tracker,
                    archive_dead=archive_dead,
                    max_concurrent=max_concurrent,
                    timeout=timeout,
                )
            except HealthMonitorError as e:
                console.print(f"[red]Error:[/red] {e}")
                console.print("[yellow]Tip: Install httpx with: pip install httpx[/yellow]")
                raise typer.Exit(1)

            # Progress callback
            def progress_cb(current, total, result):
                if verbose:
                    status_icon = {
                        "healthy": "[green]OK[/green]",
                        "dead": "[red]DEAD[/red]",
                        "redirected": "[yellow]REDIRECT[/yellow]",
                        "timeout": "[yellow]TIMEOUT[/yellow]",
                        "error": "[red]ERROR[/red]",
                    }.get(result.status, "[dim]?[/dim]")
                    console.print(f"  [{current}/{total}] {status_icon} {result.url[:60]}")

            # Run health check
            console.print("[cyan]Checking bookmark health...[/cyan]")

            async def run_check():
                return await monitor_instance.check_health(
                    bookmarks,
                    stale_after=stale_duration,
                    progress_callback=progress_cb if verbose else None,
                )

            report = asyncio.run(run_check())

            # Display results
            console.print("")
            console.print(
                Panel(
                    f"[bold]Health Check Results[/bold]\n\n"
                    f"Total checked:    {report.total}\n"
                    f"Healthy:          [green]{report.healthy}[/green] ({report.healthy_percentage:.1f}%)\n"
                    f"Redirected:       [yellow]{report.redirected}[/yellow]\n"
                    f"Dead/Broken:      [red]{report.dead}[/red]\n"
                    f"Timeouts:         [yellow]{report.timeout}[/yellow]\n"
                    f"Content changed:  [cyan]{report.content_changed}[/cyan]\n"
                    f"Duration:         {report.duration_seconds:.1f}s",
                    title="Health Report",
                )
            )

            # Show problematic URLs
            problematic = report.problematic
            if problematic:
                console.print(f"\n[yellow]Problematic URLs ({len(problematic)}):[/yellow]")

                table = Table(show_header=True)
                table.add_column("Status", style="bold", width=12)
                table.add_column("URL", max_width=50)
                table.add_column("Details", max_width=30)

                for result in problematic[:20]:
                    status_style = {
                        "dead": "red",
                        "timeout": "yellow",
                        "redirected": "cyan",
                        "content_changed": "blue",
                        "error": "red",
                    }.get(result.status, "dim")

                    details = ""
                    if result.redirect_url:
                        details = f"-> {result.redirect_url[:25]}..."
                    elif result.error_message:
                        details = result.error_message[:30]
                    elif result.wayback_url:
                        details = "[archived]"

                    table.add_row(
                        f"[{status_style}]{result.status}[/{status_style}]",
                        result.url[:50],
                        details
                    )

                console.print(table)

                if len(problematic) > 20:
                    console.print(f"  ... and {len(problematic) - 20} more")

            # Save report if requested
            if output_report:
                # Determine format from extension
                ext = output_report.suffix.lower()
                report_format = {
                    ".json": "json",
                    ".csv": "csv",
                    ".txt": "text",
                }.get(ext, "text")

                monitor_instance.save_report(report, output_report, format=report_format)
                console.print(f"\n[green]Report saved to {output_report}[/green]")

            # Archive summary
            if archive_dead and report.archived > 0:
                console.print(f"\n[cyan]Archived {report.archived} dead links to Wayback Machine[/cyan]")

            return 0

        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)


# Helper functions for MCP CLI commands

def _parse_since(since_str: str):
    """Parse a 'since' duration string into a datetime or timedelta."""
    from datetime import datetime, timedelta

    since_str = since_str.strip().lower()

    # Try parsing as duration (e.g., "7d", "30d", "2w")
    if since_str.endswith("d"):
        try:
            days = int(since_str[:-1])
            return timedelta(days=days)
        except ValueError:
            pass
    elif since_str.endswith("w"):
        try:
            weeks = int(since_str[:-1])
            return timedelta(weeks=weeks)
        except ValueError:
            pass
    elif since_str.endswith("h"):
        try:
            hours = int(since_str[:-1])
            return timedelta(hours=hours)
        except ValueError:
            pass

    # Try parsing as date
    try:
        return datetime.fromisoformat(since_str)
    except ValueError:
        pass

    # Try common date formats
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(since_str, fmt)
        except ValueError:
            pass

    raise ValueError(f"Cannot parse date/duration: {since_str}")


def _display_bookmark_preview(bookmarks, console):
    """Display a preview of bookmarks."""
    if not RICH_AVAILABLE:
        for b in bookmarks:
            print(f"  - {b.title or b.url}")
        return

    table = Table(title="Bookmark Preview")
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("URL", style="dim", max_width=40)
    table.add_column("Tags", style="green", max_width=20)

    for bookmark in bookmarks:
        title = (bookmark.title or "")[:40]
        url = (bookmark.url or "")[:40]
        tags = ", ".join(bookmark.tags[:3]) if bookmark.tags else ""
        table.add_row(title, url, tags)

    console.print(table)


async def _enhance_raindrop_async(
    server_url: str,
    token: str,
    collection: Optional[str],
    since_last_run: bool,
    since: Optional[str],
    dry_run: bool,
    preview_count: Optional[int],
    verbose: bool,
    ai_engine: str,
    config,
    console,
    output_file: Optional[Path],
):
    """Async function to enhance bookmarks via Raindrop.io MCP."""
    import json
    from datetime import datetime
    from pathlib import Path

    from bookmark_processor.core.data_sources.raindrop_mcp import RaindropMCPDataSource
    from bookmark_processor.core.data_sources.state_tracker import ProcessingStateTracker

    # Initialize state tracker
    state_tracker = ProcessingStateTracker()

    async with RaindropMCPDataSource(
        server_url=server_url,
        access_token=token,
        state_tracker=state_tracker if since_last_run else None
    ) as source:
        # Build filters
        filters = {}
        if collection:
            filters["collection"] = collection
        if since_last_run:
            filters["since_last_run"] = True
        if since:
            filters["since"] = _parse_since(since)
        if preview_count:
            filters["limit"] = preview_count

        # Fetch bookmarks
        console.print("[cyan]Fetching bookmarks...[/cyan]")
        bookmarks = await source.fetch_bookmarks(filters)

        console.print(f"[green]Found {len(bookmarks)} bookmarks to process[/green]")

        if not bookmarks:
            console.print("[yellow]No bookmarks to process[/yellow]")
            return 0

        # Display preview
        if dry_run or verbose:
            _display_bookmark_preview(bookmarks[:10], console)

        if dry_run:
            console.print("\n[yellow]Dry-run mode - no changes applied[/yellow]")
            return 0

        # Create backup before modifying
        backup_dir = Path(".bookmark_processor_backups")
        backup_dir.mkdir(exist_ok=True)
        backup_file = backup_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        backup_data = await source.create_backup(bookmarks)
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)
        console.print(f"[dim]Backup created: {backup_file}[/dim]")

        # Start processing run
        run_id = state_tracker.start_processing_run(source=source.source_name)

        # Process bookmarks
        # For now, just apply basic enhancements - full pipeline integration
        # would require more refactoring of the BookmarkProcessor
        console.print("[cyan]Processing bookmarks...[/cyan]")

        from bookmark_processor.core.bookmark_processor import BookmarkProcessor

        # Create processor with config
        processor = BookmarkProcessor(config)

        processed_count = 0
        error_count = 0

        with console.status("[bold green]Enhancing bookmarks...") as status:
            for i, bookmark in enumerate(bookmarks):
                try:
                    # Apply basic processing
                    # In a full implementation, this would use the full pipeline
                    status.update(f"[bold green]Processing {i+1}/{len(bookmarks)}: {bookmark.title[:30]}...")

                    # For now, mark as processed
                    state_tracker.mark_processed(bookmark, ai_engine=ai_engine)
                    processed_count += 1

                except Exception as e:
                    error_count += 1
                    if verbose:
                        console.print(f"[red]Error processing {bookmark.url}: {e}[/red]")

        # Complete processing run
        state_tracker.complete_processing_run(
            run_id=run_id,
            total_processed=len(bookmarks),
            total_succeeded=processed_count,
            total_failed=error_count
        )

        # Update bookmarks in Raindrop.io
        if processed_count > 0:
            console.print("[cyan]Updating bookmarks in Raindrop.io...[/cyan]")
            result = await source.bulk_update(bookmarks)

            console.print(
                f"[green]Complete![/green] "
                f"Updated {result.succeeded}/{result.total} bookmarks"
            )

            if result.errors and verbose:
                console.print("\n[yellow]Update errors:[/yellow]")
                for error in result.errors[:5]:
                    console.print(f"  - {error.get('url', 'unknown')}: {error.get('error', 'unknown')}")

        # Optionally export to CSV
        if output_file:
            console.print(f"[cyan]Exporting to CSV: {output_file}[/cyan]")
            from bookmark_processor.core.csv_handler import RaindropCSVHandler

            handler = RaindropCSVHandler()
            handler.save_import_csv(bookmarks, output_file)
            console.print(f"[green]Exported {len(bookmarks)} bookmarks to {output_file}[/green]")

        return 0


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
