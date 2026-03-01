"""
Comprehensive tests for CLI argument parser (cli_argparse.py).

Tests cover:
- Argument parser creation and configuration
- Argument validation for valid and invalid inputs
- Default values verification
- Conflicting arguments handling
- Help text generation
- Integration with validation utilities
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.cli_argparse import CLIInterface, main
from bookmark_processor.utils.validation import ValidationError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cli_interface():
    """Create a fresh CLIInterface instance."""
    return CLIInterface()


@pytest.fixture
def sample_input_csv(temp_dir):
    """Create a valid sample input CSV file."""
    csv_path = temp_dir / "input.csv"
    csv_content = (
        "id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n"
        "1,Test Title,Note,Excerpt,https://example.com,Test,tag1,2024-01-01T00:00:00Z,,,false\n"
    )
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def sample_output_csv(temp_dir):
    """Create a valid output path."""
    return temp_dir / "output.csv"


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a valid config file."""
    config_path = temp_dir / "config.ini"
    config_path.write_text("[processing]\nbatch_size = 50\n")
    return config_path


# ============================================================================
# Parser Creation Tests
# ============================================================================


class TestParserCreation:
    """Test argument parser creation and configuration."""

    def test_parser_is_created(self, cli_interface):
        """Test that parser is created on initialization."""
        assert cli_interface.parser is not None
        assert isinstance(cli_interface.parser, argparse.ArgumentParser)

    def test_parser_prog_name(self, cli_interface):
        """Test parser program name."""
        assert cli_interface.parser.prog == "bookmark-processor"

    def test_parser_description_set(self, cli_interface):
        """Test that parser has a description."""
        assert cli_interface.parser.description is not None
        assert "raindrop.io" in cli_interface.parser.description.lower()

    def test_parser_epilog_contains_examples(self, cli_interface):
        """Test that parser epilog contains usage examples."""
        assert cli_interface.parser.epilog is not None
        assert "Examples:" in cli_interface.parser.epilog
        assert "--input" in cli_interface.parser.epilog
        assert "--output" in cli_interface.parser.epilog

    def test_parser_formatter_class(self, cli_interface):
        """Test that parser uses RawDescriptionHelpFormatter."""
        assert (
            cli_interface.parser.formatter_class
            == argparse.RawDescriptionHelpFormatter
        )


class TestParserArguments:
    """Test that all arguments are defined correctly."""

    def test_input_argument_defined(self, cli_interface):
        """Test --input/-i argument exists."""
        # Parse with just input argument
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert hasattr(args, "input")
        assert args.input == "test.csv"

    def test_input_short_flag(self, cli_interface):
        """Test -i short flag for input."""
        args = cli_interface.parse_args(["-i", "test.csv", "--output", "out.csv"])
        assert args.input == "test.csv"

    def test_output_argument_defined(self, cli_interface):
        """Test --output/-o argument exists."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert hasattr(args, "output")
        assert args.output == "out.csv"

    def test_output_short_flag(self, cli_interface):
        """Test -o short flag for output."""
        args = cli_interface.parse_args(["--input", "test.csv", "-o", "out.csv"])
        assert args.output == "out.csv"

    def test_config_argument_defined(self, cli_interface):
        """Test --config/-c argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--config", "config.ini"]
        )
        assert hasattr(args, "config")
        assert args.config == "config.ini"

    def test_resume_argument_defined(self, cli_interface):
        """Test --resume/-r argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--resume"]
        )
        assert hasattr(args, "resume")
        assert args.resume is True

    def test_verbose_argument_defined(self, cli_interface):
        """Test --verbose/-v argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--verbose"]
        )
        assert hasattr(args, "verbose")
        assert args.verbose is True

    def test_batch_size_argument_defined(self, cli_interface):
        """Test --batch-size/-b argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--batch-size", "50"]
        )
        assert hasattr(args, "batch_size")
        assert args.batch_size == 50

    def test_max_retries_argument_defined(self, cli_interface):
        """Test --max-retries/-m argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--max-retries", "5"]
        )
        assert hasattr(args, "max_retries")
        assert args.max_retries == 5

    def test_clear_checkpoints_argument_defined(self, cli_interface):
        """Test --clear-checkpoints argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--clear-checkpoints"]
        )
        assert hasattr(args, "clear_checkpoints")
        assert args.clear_checkpoints is True

    def test_ai_engine_argument_defined(self, cli_interface):
        """Test --ai-engine argument exists with choices."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--ai-engine", "claude"]
        )
        assert hasattr(args, "ai_engine")
        assert args.ai_engine == "claude"

    def test_ai_engine_valid_choices(self, cli_interface):
        """Test --ai-engine accepts valid choices."""
        for engine in ["local", "claude", "openai"]:
            args = cli_interface.parse_args(
                ["--input", "test.csv", "--output", "out.csv", "--ai-engine", engine]
            )
            assert args.ai_engine == engine

    def test_ai_engine_invalid_choice(self, cli_interface):
        """Test --ai-engine rejects invalid choices."""
        with pytest.raises(SystemExit):
            cli_interface.parse_args(
                [
                    "--input",
                    "test.csv",
                    "--output",
                    "out.csv",
                    "--ai-engine",
                    "invalid",
                ]
            )

    def test_no_duplicates_argument_defined(self, cli_interface):
        """Test --no-duplicates argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--no-duplicates"]
        )
        assert hasattr(args, "no_duplicates")
        assert args.no_duplicates is True

    def test_duplicate_strategy_argument_defined(self, cli_interface):
        """Test --duplicate-strategy argument exists with choices."""
        args = cli_interface.parse_args(
            [
                "--input",
                "test.csv",
                "--output",
                "out.csv",
                "--duplicate-strategy",
                "newest",
            ]
        )
        assert hasattr(args, "duplicate_strategy")
        assert args.duplicate_strategy == "newest"

    def test_duplicate_strategy_valid_choices(self, cli_interface):
        """Test --duplicate-strategy accepts valid choices."""
        for strategy in ["newest", "oldest", "most_complete", "highest_quality"]:
            args = cli_interface.parse_args(
                [
                    "--input",
                    "test.csv",
                    "--output",
                    "out.csv",
                    "--duplicate-strategy",
                    strategy,
                ]
            )
            assert args.duplicate_strategy == strategy

    def test_since_last_run_argument_defined(self, cli_interface):
        """Test --since-last-run argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--since-last-run"]
        )
        assert hasattr(args, "since_last_run")
        assert args.since_last_run is True

    def test_clear_state_argument_defined(self, cli_interface):
        """Test --clear-state argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--clear-state"]
        )
        assert hasattr(args, "clear_state")
        assert args.clear_state is True

    def test_state_db_argument_defined(self, cli_interface):
        """Test --state-db argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--state-db", "custom.db"]
        )
        assert hasattr(args, "state_db")
        assert args.state_db == "custom.db"

    def test_generate_folders_argument_defined(self, cli_interface):
        """Test --generate-folders argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--generate-folders"]
        )
        assert hasattr(args, "generate_folders")
        assert args.generate_folders is True

    def test_no_folders_argument_defined(self, cli_interface):
        """Test --no-folders argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--no-folders"]
        )
        assert hasattr(args, "no_folders")
        assert args.no_folders is True

    def test_max_bookmarks_per_folder_argument_defined(self, cli_interface):
        """Test --max-bookmarks-per-folder argument exists."""
        args = cli_interface.parse_args(
            [
                "--input",
                "test.csv",
                "--output",
                "out.csv",
                "--max-bookmarks-per-folder",
                "15",
            ]
        )
        assert hasattr(args, "max_bookmarks_per_folder")
        assert args.max_bookmarks_per_folder == 15

    def test_chrome_html_argument_defined(self, cli_interface):
        """Test --chrome-html argument exists."""
        args = cli_interface.parse_args(
            ["--input", "test.csv", "--output", "out.csv", "--chrome-html"]
        )
        assert hasattr(args, "chrome_html")
        assert args.chrome_html is True

    def test_html_output_argument_defined(self, cli_interface):
        """Test --html-output argument exists."""
        args = cli_interface.parse_args(
            [
                "--input",
                "test.csv",
                "--output",
                "out.csv",
                "--html-output",
                "bookmarks.html",
            ]
        )
        assert hasattr(args, "html_output")
        assert args.html_output == "bookmarks.html"

    def test_html_title_argument_defined(self, cli_interface):
        """Test --html-title argument exists."""
        args = cli_interface.parse_args(
            [
                "--input",
                "test.csv",
                "--output",
                "out.csv",
                "--html-title",
                "My Bookmarks",
            ]
        )
        assert hasattr(args, "html_title")
        assert args.html_title == "My Bookmarks"

    def test_create_config_argument_defined(self, cli_interface):
        """Test --create-config argument exists with choices."""
        args = cli_interface.parse_args(["--create-config", "basic"])
        assert hasattr(args, "create_config")
        assert args.create_config == "basic"

    def test_create_config_valid_choices(self, cli_interface):
        """Test --create-config accepts valid choices."""
        for config_type in ["basic", "claude", "openai", "performance", "large-dataset"]:
            args = cli_interface.parse_args(["--create-config", config_type])
            assert args.create_config == config_type


# ============================================================================
# Default Values Tests
# ============================================================================


class TestDefaultValues:
    """Test default argument values."""

    def test_batch_size_default(self, cli_interface):
        """Test --batch-size default is 100."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.batch_size == 100

    def test_max_retries_default(self, cli_interface):
        """Test --max-retries default is 3."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.max_retries == 3

    def test_ai_engine_default(self, cli_interface):
        """Test --ai-engine default is 'local'."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.ai_engine == "local"

    def test_duplicate_strategy_default(self, cli_interface):
        """Test --duplicate-strategy default is 'highest_quality'."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.duplicate_strategy == "highest_quality"

    def test_max_bookmarks_per_folder_default(self, cli_interface):
        """Test --max-bookmarks-per-folder default is 20."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.max_bookmarks_per_folder == 20

    def test_html_title_default(self, cli_interface):
        """Test --html-title default is 'Enhanced Bookmarks'."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.html_title == "Enhanced Bookmarks"

    def test_generate_folders_default(self, cli_interface):
        """Test --generate-folders default is True."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.generate_folders is True

    def test_boolean_flags_default_false(self, cli_interface):
        """Test boolean flags default to False."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.resume is False
        assert args.verbose is False
        assert args.clear_checkpoints is False
        assert args.no_duplicates is False
        assert args.since_last_run is False
        assert args.clear_state is False
        assert args.no_folders is False
        assert args.chrome_html is False

    def test_optional_string_defaults_none(self, cli_interface):
        """Test optional string arguments default to None."""
        args = cli_interface.parse_args(["--input", "test.csv", "--output", "out.csv"])
        assert args.config is None
        assert args.state_db is None
        assert args.html_output is None


# ============================================================================
# Argument Validation Tests
# ============================================================================


class TestArgumentValidation:
    """Test argument validation logic."""

    def test_validate_args_requires_output(
        self, cli_interface, sample_input_csv, temp_dir
    ):
        """Test that output file is required."""
        args = cli_interface.parse_args(["--input", str(sample_input_csv)])
        with pytest.raises(ValidationError) as exc_info:
            cli_interface.validate_args(args)
        assert "Output file is required" in str(exc_info.value)

    def test_validate_args_valid_input_output(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test validation passes with valid input and output."""
        with patch(
            "bookmark_processor.cli_argparse.validate_input_file"
        ) as mock_validate_input:
            with patch(
                "bookmark_processor.cli_argparse.validate_output_file"
            ) as mock_validate_output:
                mock_validate_input.return_value = sample_input_csv
                mock_validate_output.return_value = sample_output_csv

                args = cli_interface.parse_args(
                    ["--input", str(sample_input_csv), "--output", str(sample_output_csv)]
                )
                validated = cli_interface.validate_args(args)

                assert validated["input_path"] == sample_input_csv
                assert validated["output_path"] == sample_output_csv

    def test_validate_args_returns_all_fields(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test that validate_args returns all expected fields."""
        with patch(
            "bookmark_processor.cli_argparse.validate_input_file"
        ) as mock_validate_input:
            with patch(
                "bookmark_processor.cli_argparse.validate_output_file"
            ) as mock_validate_output:
                with patch(
                    "bookmark_processor.cli_argparse.validate_batch_size"
                ) as mock_validate_batch:
                    with patch(
                        "bookmark_processor.cli_argparse.validate_max_retries"
                    ) as mock_validate_retries:
                        with patch(
                            "bookmark_processor.cli_argparse.validate_config_file"
                        ) as mock_validate_config:
                            with patch(
                                "bookmark_processor.cli_argparse.validate_conflicting_arguments"
                            ) as mock_validate_conflict:
                                mock_validate_input.return_value = sample_input_csv
                                mock_validate_output.return_value = sample_output_csv
                                mock_validate_batch.return_value = 100
                                mock_validate_retries.return_value = 3
                                mock_validate_config.return_value = None
                                mock_validate_conflict.return_value = None

                                args = cli_interface.parse_args(
                                    [
                                        "--input",
                                        str(sample_input_csv),
                                        "--output",
                                        str(sample_output_csv),
                                    ]
                                )
                                validated = cli_interface.validate_args(args)

                                expected_keys = [
                                    "input_path",
                                    "output_path",
                                    "config_path",
                                    "resume",
                                    "verbose",
                                    "batch_size",
                                    "max_retries",
                                    "clear_checkpoints",
                                    "ai_engine",
                                    "detect_duplicates",
                                    "duplicate_strategy",
                                    "generate_folders",
                                    "max_bookmarks_per_folder",
                                    "generate_chrome_html",
                                    "chrome_html_output",
                                    "html_title",
                                    "since_last_run",
                                    "clear_state",
                                    "state_db",
                                ]
                                for key in expected_keys:
                                    assert key in validated, f"Missing key: {key}"

    def test_validate_args_detect_duplicates_negation(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test that --no-duplicates sets detect_duplicates to False."""
        with patch(
            "bookmark_processor.cli_argparse.validate_input_file"
        ) as mock_validate_input:
            with patch(
                "bookmark_processor.cli_argparse.validate_output_file"
            ) as mock_validate_output:
                with patch(
                    "bookmark_processor.cli_argparse.validate_batch_size"
                ) as mock_batch:
                    with patch(
                        "bookmark_processor.cli_argparse.validate_max_retries"
                    ) as mock_retries:
                        with patch(
                            "bookmark_processor.cli_argparse.validate_config_file"
                        ) as mock_config:
                            with patch(
                                "bookmark_processor.cli_argparse.validate_conflicting_arguments"
                            ):
                                mock_validate_input.return_value = sample_input_csv
                                mock_validate_output.return_value = sample_output_csv
                                mock_batch.return_value = 100
                                mock_retries.return_value = 3
                                mock_config.return_value = None

                                args = cli_interface.parse_args(
                                    [
                                        "--input",
                                        str(sample_input_csv),
                                        "--output",
                                        str(sample_output_csv),
                                        "--no-duplicates",
                                    ]
                                )
                                validated = cli_interface.validate_args(args)

                                assert validated["detect_duplicates"] is False

    def test_validate_args_generate_folders_with_no_folders(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test that --no-folders disables folder generation."""
        with patch(
            "bookmark_processor.cli_argparse.validate_input_file"
        ) as mock_validate_input:
            with patch(
                "bookmark_processor.cli_argparse.validate_output_file"
            ) as mock_validate_output:
                with patch(
                    "bookmark_processor.cli_argparse.validate_batch_size"
                ) as mock_batch:
                    with patch(
                        "bookmark_processor.cli_argparse.validate_max_retries"
                    ) as mock_retries:
                        with patch(
                            "bookmark_processor.cli_argparse.validate_config_file"
                        ) as mock_config:
                            with patch(
                                "bookmark_processor.cli_argparse.validate_conflicting_arguments"
                            ):
                                mock_validate_input.return_value = sample_input_csv
                                mock_validate_output.return_value = sample_output_csv
                                mock_batch.return_value = 100
                                mock_retries.return_value = 3
                                mock_config.return_value = None

                                args = cli_interface.parse_args(
                                    [
                                        "--input",
                                        str(sample_input_csv),
                                        "--output",
                                        str(sample_output_csv),
                                        "--no-folders",
                                    ]
                                )
                                validated = cli_interface.validate_args(args)

                                assert validated["generate_folders"] is False


# ============================================================================
# Conflicting Arguments Tests
# ============================================================================


class TestConflictingArguments:
    """Test detection of conflicting arguments."""

    def test_resume_and_clear_checkpoints_conflict(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test that --resume and --clear-checkpoints cannot be used together."""
        with patch(
            "bookmark_processor.cli_argparse.validate_input_file"
        ) as mock_validate_input:
            with patch(
                "bookmark_processor.cli_argparse.validate_output_file"
            ) as mock_validate_output:
                with patch(
                    "bookmark_processor.cli_argparse.validate_batch_size"
                ) as mock_batch:
                    with patch(
                        "bookmark_processor.cli_argparse.validate_max_retries"
                    ) as mock_retries:
                        with patch(
                            "bookmark_processor.cli_argparse.validate_config_file"
                        ) as mock_config:
                            with patch(
                                "bookmark_processor.cli_argparse.validate_conflicting_arguments"
                            ) as mock_conflict:
                                mock_validate_input.return_value = sample_input_csv
                                mock_validate_output.return_value = sample_output_csv
                                mock_batch.return_value = 100
                                mock_retries.return_value = 3
                                mock_config.return_value = None
                                mock_conflict.side_effect = ValidationError(
                                    "Cannot use --resume and --clear-checkpoints together"
                                )

                                args = cli_interface.parse_args(
                                    [
                                        "--input",
                                        str(sample_input_csv),
                                        "--output",
                                        str(sample_output_csv),
                                        "--resume",
                                        "--clear-checkpoints",
                                    ]
                                )

                                with pytest.raises(ValidationError) as exc_info:
                                    cli_interface.validate_args(args)

                                assert (
                                    "Cannot use --resume and --clear-checkpoints together"
                                    in str(exc_info.value)
                                )


# ============================================================================
# Version Argument Tests
# ============================================================================


class TestVersionArgument:
    """Test version argument handling."""

    def test_version_argument_exists(self, cli_interface):
        """Test that --version/-V argument exists."""
        with pytest.raises(SystemExit) as exc_info:
            cli_interface.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_version_short_flag(self, cli_interface):
        """Test -V short flag for version."""
        with pytest.raises(SystemExit) as exc_info:
            cli_interface.parse_args(["-V"])
        assert exc_info.value.code == 0


# ============================================================================
# Help Text Tests
# ============================================================================


class TestHelpText:
    """Test help text generation."""

    def test_help_argument_works(self, cli_interface):
        """Test that --help exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            cli_interface.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_help_short_flag(self, cli_interface):
        """Test -h short flag for help."""
        with pytest.raises(SystemExit) as exc_info:
            cli_interface.parse_args(["-h"])
        assert exc_info.value.code == 0


# ============================================================================
# Create Config Handling Tests
# ============================================================================


class TestCreateConfigHandling:
    """Test --create-config option handling."""

    def test_handle_create_config_unknown_type(self, cli_interface, temp_dir, monkeypatch):
        """Test handling of unknown config type."""
        monkeypatch.chdir(temp_dir)
        result = cli_interface._handle_create_config("unknown")
        assert result == 1

    def test_handle_create_config_missing_template(
        self, cli_interface, temp_dir, monkeypatch, capsys
    ):
        """Test handling when template file is missing."""
        monkeypatch.chdir(temp_dir)
        # Create a temporary cli_argparse module file location
        # and point to a non-existent template directory
        with patch.object(
            Path, "exists", side_effect=lambda self: "user_config.toml" in str(self)
        ):
            # The template check happens inside the method, so we need to patch
            # differently - use the actual implementation but with a fake config dir
            import bookmark_processor.cli_argparse as cli_module

            original_file = cli_module.__file__
            # Create a fake path that doesn't have the template
            fake_config_dir = temp_dir / "config"
            fake_config_dir.mkdir(exist_ok=True)
            # Don't create the template file

            # Patch __file__ to point to our temp dir
            with patch.object(cli_module, "__file__", str(temp_dir / "cli_argparse.py")):
                result = cli_interface._handle_create_config("basic")
                # Should fail because template doesn't exist
                assert result == 1
                captured = capsys.readouterr()
                assert "Template file not found" in captured.out or result == 1

    def test_handle_create_config_existing_file_no_overwrite(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test handling when config file exists and user declines overwrite."""
        monkeypatch.chdir(temp_dir)
        # Create existing config file
        (temp_dir / "user_config.toml").write_text("existing")

        with patch("builtins.input", return_value="n"):
            # We need to provide a valid template path
            import bookmark_processor.cli_argparse
            config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
            template_file = config_dir / "user_config.toml.template"

            # Create the template if it doesn't exist for the test
            if config_dir.exists() and template_file.exists():
                result = cli_interface._handle_create_config("basic")
                assert result == 1

    def test_handle_create_config_existing_file_with_overwrite(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test handling when config file exists and user agrees to overwrite."""
        monkeypatch.chdir(temp_dir)
        # Create existing config file
        (temp_dir / "user_config.toml").write_text("existing")

        import bookmark_processor.cli_argparse
        config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
        template_file = config_dir / "user_config.toml.template"

        if config_dir.exists() and template_file.exists():
            with patch("builtins.input", return_value="y"):
                result = cli_interface._handle_create_config("basic")
                assert result == 0
                # Verify the file was created
                assert (temp_dir / "user_config.toml").exists()

    def test_handle_create_config_success_basic(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test successful creation of basic config."""
        monkeypatch.chdir(temp_dir)

        import bookmark_processor.cli_argparse
        config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
        template_file = config_dir / "user_config.toml.template"

        if config_dir.exists() and template_file.exists():
            result = cli_interface._handle_create_config("basic")
            assert result == 0
            assert (temp_dir / "user_config.toml").exists()

    def test_handle_create_config_success_claude(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test successful creation of claude config."""
        monkeypatch.chdir(temp_dir)

        import bookmark_processor.cli_argparse
        config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
        template_file = config_dir / "claude_config.toml.template"

        if config_dir.exists() and template_file.exists():
            result = cli_interface._handle_create_config("claude")
            assert result == 0

    def test_handle_create_config_success_openai(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test successful creation of openai config."""
        monkeypatch.chdir(temp_dir)

        import bookmark_processor.cli_argparse
        config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
        template_file = config_dir / "openai_config.toml.template"

        if config_dir.exists() and template_file.exists():
            result = cli_interface._handle_create_config("openai")
            assert result == 0

    def test_handle_create_config_success_performance(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test successful creation of performance config."""
        monkeypatch.chdir(temp_dir)

        import bookmark_processor.cli_argparse
        config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
        template_file = config_dir / "local_performance.toml.template"

        if config_dir.exists() and template_file.exists():
            result = cli_interface._handle_create_config("performance")
            assert result == 0

    def test_handle_create_config_success_large_dataset(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test successful creation of large-dataset config."""
        monkeypatch.chdir(temp_dir)

        import bookmark_processor.cli_argparse
        config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
        template_file = config_dir / "large_dataset.toml.template"

        if config_dir.exists() and template_file.exists():
            result = cli_interface._handle_create_config("large-dataset")
            assert result == 0

    def test_handle_create_config_exception(
        self, cli_interface, temp_dir, monkeypatch
    ):
        """Test exception handling during config creation."""
        monkeypatch.chdir(temp_dir)

        with patch("shutil.copy2", side_effect=Exception("Permission denied")):
            import bookmark_processor.cli_argparse
            config_dir = Path(bookmark_processor.cli_argparse.__file__).parent / "config"
            template_file = config_dir / "user_config.toml.template"

            if config_dir.exists() and template_file.exists():
                result = cli_interface._handle_create_config("basic")
                assert result == 1


# ============================================================================
# Run Method Tests
# ============================================================================


class TestRunMethod:
    """Test the run method of CLIInterface."""

    def test_run_returns_error_on_validation_error(self, cli_interface):
        """Test that run returns 1 on ValidationError."""
        result = cli_interface.run(["--input", "nonexistent.csv", "--output", "out.csv"])
        assert result == 1

    def test_run_with_create_config(self, cli_interface, temp_dir, monkeypatch):
        """Test run with --create-config option."""
        monkeypatch.chdir(temp_dir)
        with patch.object(cli_interface, "_handle_create_config", return_value=0):
            result = cli_interface.run(["--create-config", "basic"])
            assert result == 0

    def test_run_without_arguments_shows_error(self, cli_interface):
        """Test run without required arguments fails gracefully."""
        # Without --create-config, --output is required
        result = cli_interface.run([])
        # Should fail because output is required
        assert result == 1

    def test_run_with_verbose_and_local_ai(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test run with verbose mode and local AI engine."""
        with patch.object(cli_interface, "validate_args") as mock_validate:
            with patch.object(cli_interface, "process_arguments") as mock_process:
                with patch(
                    "bookmark_processor.cli_argparse.BookmarkProcessor"
                ) as mock_processor_class:
                    mock_validate.return_value = {
                        "input_path": sample_input_csv,
                        "output_path": sample_output_csv,
                        "config_path": None,
                        "resume": False,
                        "verbose": True,
                        "batch_size": 100,
                        "max_retries": 3,
                        "clear_checkpoints": False,
                        "ai_engine": "local",
                        "detect_duplicates": True,
                        "duplicate_strategy": "highest_quality",
                        "generate_folders": True,
                        "max_bookmarks_per_folder": 20,
                        "generate_chrome_html": False,
                        "chrome_html_output": None,
                        "html_title": "Enhanced Bookmarks",
                        "since_last_run": False,
                        "clear_state": False,
                        "state_db": None,
                    }

                    mock_config = MagicMock()
                    mock_process.return_value = mock_config

                    mock_processor = MagicMock()
                    mock_processor.run_cli.return_value = 0
                    mock_processor_class.return_value = mock_processor

                    result = cli_interface.run(
                        [
                            "--input",
                            str(sample_input_csv),
                            "--output",
                            str(sample_output_csv),
                            "--verbose",
                        ]
                    )

                    assert result == 0
                    mock_processor.run_cli.assert_called_once()

    def test_run_with_verbose_and_claude_ai(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test run with verbose mode and Claude AI engine."""
        with patch.object(cli_interface, "validate_args") as mock_validate:
            with patch.object(cli_interface, "process_arguments") as mock_process:
                with patch(
                    "bookmark_processor.cli_argparse.BookmarkProcessor"
                ) as mock_processor_class:
                    mock_validate.return_value = {
                        "input_path": sample_input_csv,
                        "output_path": sample_output_csv,
                        "config_path": None,
                        "resume": False,
                        "verbose": True,
                        "batch_size": 100,
                        "max_retries": 3,
                        "clear_checkpoints": False,
                        "ai_engine": "claude",
                        "detect_duplicates": True,
                        "duplicate_strategy": "highest_quality",
                        "generate_folders": True,
                        "max_bookmarks_per_folder": 20,
                        "generate_chrome_html": False,
                        "chrome_html_output": None,
                        "html_title": "Enhanced Bookmarks",
                        "since_last_run": False,
                        "clear_state": False,
                        "state_db": None,
                    }

                    mock_config = MagicMock()
                    mock_config.has_api_key.return_value = True
                    mock_config.get_rate_limit.return_value = 50
                    mock_config.get_batch_size.return_value = 10
                    mock_config.get_cost_tracking_settings.return_value = {
                        "cost_confirmation_interval": 10.0
                    }
                    mock_process.return_value = mock_config

                    mock_processor = MagicMock()
                    mock_processor.run_cli.return_value = 0
                    mock_processor_class.return_value = mock_processor

                    result = cli_interface.run(
                        [
                            "--input",
                            str(sample_input_csv),
                            "--output",
                            str(sample_output_csv),
                            "--verbose",
                            "--ai-engine",
                            "claude",
                        ]
                    )

                    assert result == 0

    def test_run_with_verbose_and_openai_ai(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test run with verbose mode and OpenAI AI engine."""
        with patch.object(cli_interface, "validate_args") as mock_validate:
            with patch.object(cli_interface, "process_arguments") as mock_process:
                with patch(
                    "bookmark_processor.cli_argparse.BookmarkProcessor"
                ) as mock_processor_class:
                    mock_validate.return_value = {
                        "input_path": sample_input_csv,
                        "output_path": sample_output_csv,
                        "config_path": None,
                        "resume": False,
                        "verbose": True,
                        "batch_size": 100,
                        "max_retries": 3,
                        "clear_checkpoints": False,
                        "ai_engine": "openai",
                        "detect_duplicates": False,
                        "duplicate_strategy": "highest_quality",
                        "generate_folders": True,
                        "max_bookmarks_per_folder": 20,
                        "generate_chrome_html": False,
                        "chrome_html_output": None,
                        "html_title": "Enhanced Bookmarks",
                        "since_last_run": False,
                        "clear_state": False,
                        "state_db": None,
                    }

                    mock_config = MagicMock()
                    mock_config.has_api_key.return_value = False
                    mock_config.get_rate_limit.return_value = 60
                    mock_config.get_batch_size.return_value = 20
                    mock_process.return_value = mock_config

                    mock_processor = MagicMock()
                    mock_processor.run_cli.return_value = 0
                    mock_processor_class.return_value = mock_processor

                    result = cli_interface.run(
                        [
                            "--input",
                            str(sample_input_csv),
                            "--output",
                            str(sample_output_csv),
                            "--verbose",
                            "--ai-engine",
                            "openai",
                        ]
                    )

                    assert result == 0

    def test_run_with_verbose_auto_detection_mode(
        self, cli_interface, sample_output_csv
    ):
        """Test run with verbose mode and auto-detection (no input file)."""
        with patch.object(cli_interface, "validate_args") as mock_validate:
            with patch.object(cli_interface, "process_arguments") as mock_process:
                with patch(
                    "bookmark_processor.cli_argparse.BookmarkProcessor"
                ) as mock_processor_class:
                    mock_validate.return_value = {
                        "input_path": None,  # Auto-detection mode
                        "output_path": sample_output_csv,
                        "config_path": None,
                        "resume": False,
                        "verbose": True,
                        "batch_size": 100,
                        "max_retries": 3,
                        "clear_checkpoints": False,
                        "ai_engine": "local",
                        "detect_duplicates": True,
                        "duplicate_strategy": "highest_quality",
                        "generate_folders": True,
                        "max_bookmarks_per_folder": 20,
                        "generate_chrome_html": False,
                        "chrome_html_output": None,
                        "html_title": "Enhanced Bookmarks",
                        "since_last_run": False,
                        "clear_state": False,
                        "state_db": None,
                    }

                    mock_config = MagicMock()
                    mock_process.return_value = mock_config

                    mock_processor = MagicMock()
                    mock_processor.run_cli.return_value = 0
                    mock_processor_class.return_value = mock_processor

                    result = cli_interface.run(
                        ["--output", str(sample_output_csv), "--verbose"]
                    )

                    assert result == 0

    def test_run_with_verbose_and_config_path(
        self, cli_interface, sample_input_csv, sample_output_csv, sample_config_file
    ):
        """Test run with verbose mode and custom config path."""
        with patch.object(cli_interface, "validate_args") as mock_validate:
            with patch.object(cli_interface, "process_arguments") as mock_process:
                with patch(
                    "bookmark_processor.cli_argparse.BookmarkProcessor"
                ) as mock_processor_class:
                    mock_validate.return_value = {
                        "input_path": sample_input_csv,
                        "output_path": sample_output_csv,
                        "config_path": sample_config_file,
                        "resume": False,
                        "verbose": True,
                        "batch_size": 100,
                        "max_retries": 3,
                        "clear_checkpoints": False,
                        "ai_engine": "local",
                        "detect_duplicates": True,
                        "duplicate_strategy": "highest_quality",
                        "generate_folders": True,
                        "max_bookmarks_per_folder": 20,
                        "generate_chrome_html": False,
                        "chrome_html_output": None,
                        "html_title": "Enhanced Bookmarks",
                        "since_last_run": False,
                        "clear_state": False,
                        "state_db": None,
                    }

                    mock_config = MagicMock()
                    mock_process.return_value = mock_config

                    mock_processor = MagicMock()
                    mock_processor.run_cli.return_value = 0
                    mock_processor_class.return_value = mock_processor

                    result = cli_interface.run(
                        [
                            "--input",
                            str(sample_input_csv),
                            "--output",
                            str(sample_output_csv),
                            "--verbose",
                            "--config",
                            str(sample_config_file),
                        ]
                    )

                    assert result == 0

    def test_run_catches_general_exception(self, cli_interface):
        """Test that run catches general exceptions and returns 1."""
        with patch.object(
            cli_interface, "parse_args", side_effect=Exception("Unexpected error")
        ):
            result = cli_interface.run(["--input", "test.csv", "--output", "out.csv"])
            assert result == 1

    def test_run_without_verbose_skips_detailed_output(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test run without verbose mode does not print detailed config."""
        with patch.object(cli_interface, "validate_args") as mock_validate:
            with patch.object(cli_interface, "process_arguments") as mock_process:
                with patch(
                    "bookmark_processor.cli_argparse.BookmarkProcessor"
                ) as mock_processor_class:
                    mock_validate.return_value = {
                        "input_path": sample_input_csv,
                        "output_path": sample_output_csv,
                        "config_path": None,
                        "resume": False,
                        "verbose": False,  # Not verbose
                        "batch_size": 100,
                        "max_retries": 3,
                        "clear_checkpoints": False,
                        "ai_engine": "local",
                        "detect_duplicates": True,
                        "duplicate_strategy": "highest_quality",
                        "generate_folders": True,
                        "max_bookmarks_per_folder": 20,
                        "generate_chrome_html": False,
                        "chrome_html_output": None,
                        "html_title": "Enhanced Bookmarks",
                        "since_last_run": False,
                        "clear_state": False,
                        "state_db": None,
                    }

                    mock_config = MagicMock()
                    mock_process.return_value = mock_config

                    mock_processor = MagicMock()
                    mock_processor.run_cli.return_value = 0
                    mock_processor_class.return_value = mock_processor

                    result = cli_interface.run(
                        [
                            "--input",
                            str(sample_input_csv),
                            "--output",
                            str(sample_output_csv),
                        ]
                    )

                    assert result == 0


# ============================================================================
# Main Function Tests
# ============================================================================


class TestMainFunction:
    """Test the main() entry point function."""

    def test_main_creates_cli_interface(self):
        """Test that main creates a CLIInterface."""
        with patch.object(CLIInterface, "run", return_value=0) as mock_run:
            result = main(["--create-config", "basic"])
            mock_run.assert_called_once()

    def test_main_passes_args_to_run(self):
        """Test that main passes arguments to run."""
        test_args = ["--input", "test.csv", "--output", "out.csv"]
        with patch.object(CLIInterface, "run", return_value=0) as mock_run:
            main(test_args)
            mock_run.assert_called_once_with(test_args)

    def test_main_returns_run_result(self):
        """Test that main returns the result of run."""
        with patch.object(CLIInterface, "run", return_value=42):
            result = main(["--create-config", "basic"])
            assert result == 42


# ============================================================================
# Process Arguments Tests
# ============================================================================


class TestProcessArguments:
    """Test process_arguments method."""

    def test_process_arguments_creates_configuration(
        self, cli_interface, sample_input_csv, sample_output_csv
    ):
        """Test that process_arguments creates a Configuration object."""
        validated_args = {
            "input_path": sample_input_csv,
            "output_path": sample_output_csv,
            "config_path": None,
            "resume": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "clear_checkpoints": False,
            "ai_engine": "local",
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_folders": True,
            "max_bookmarks_per_folder": 20,
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
            "since_last_run": False,
            "clear_state": False,
            "state_db": None,
        }

        with patch(
            "bookmark_processor.cli_argparse.Configuration"
        ) as mock_config_class:
            with patch(
                "bookmark_processor.cli_argparse.validate_ai_engine"
            ) as mock_validate_ai:
                with patch(
                    "bookmark_processor.cli_argparse.setup_logging"
                ) as mock_logging:
                    mock_config = MagicMock()
                    mock_config_class.return_value = mock_config
                    mock_validate_ai.return_value = "local"

                    result = cli_interface.process_arguments(validated_args)

                    mock_config_class.assert_called_once_with(None)
                    mock_config.update_from_args.assert_called_once()
                    assert result == mock_config


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input_argument(self, cli_interface):
        """Test parsing with empty input argument value."""
        args = cli_interface.parse_args(["--input", "", "--output", "out.csv"])
        assert args.input == ""

    def test_numeric_argument_type_enforcement(self, cli_interface):
        """Test that numeric arguments enforce type."""
        with pytest.raises(SystemExit):
            cli_interface.parse_args(
                ["--input", "test.csv", "--output", "out.csv", "--batch-size", "abc"]
            )

    def test_parse_args_with_none(self, cli_interface):
        """Test parse_args with None uses sys.argv."""
        # This is tricky to test properly without mocking sys.argv
        # Just verify the method accepts None
        with pytest.raises(SystemExit):
            # Should fail because sys.argv doesn't have required args
            cli_interface.parse_args(None)

    def test_multiple_short_flags_combined(self, cli_interface):
        """Test multiple arguments can be combined."""
        args = cli_interface.parse_args(
            [
                "-i",
                "test.csv",
                "-o",
                "out.csv",
                "-v",
                "-r",
                "-b",
                "50",
                "-m",
                "5",
            ]
        )
        assert args.input == "test.csv"
        assert args.output == "out.csv"
        assert args.verbose is True
        assert args.resume is True
        assert args.batch_size == 50
        assert args.max_retries == 5

    def test_auto_detection_mode_no_input(self, cli_interface):
        """Test validation with no input file (auto-detection mode)."""
        with patch(
            "bookmark_processor.cli_argparse.validate_input_file"
        ) as mock_validate_input:
            with patch(
                "bookmark_processor.cli_argparse.validate_auto_detection_mode"
            ) as mock_auto:
                with patch(
                    "bookmark_processor.cli_argparse.validate_output_file"
                ) as mock_output:
                    with patch(
                        "bookmark_processor.cli_argparse.validate_batch_size"
                    ) as mock_batch:
                        with patch(
                            "bookmark_processor.cli_argparse.validate_max_retries"
                        ) as mock_retries:
                            with patch(
                                "bookmark_processor.cli_argparse.validate_config_file"
                            ) as mock_config:
                                with patch(
                                    "bookmark_processor.cli_argparse.validate_conflicting_arguments"
                                ):
                                    mock_validate_input.return_value = None
                                    mock_output.return_value = Path("/tmp/out.csv")
                                    mock_batch.return_value = 100
                                    mock_retries.return_value = 3
                                    mock_config.return_value = None

                                    args = cli_interface.parse_args(
                                        ["--output", "out.csv"]
                                    )
                                    validated = cli_interface.validate_args(args)

                                    mock_auto.assert_called_once()
                                    assert validated["input_path"] is None


# ============================================================================
# Integration-Style Tests
# ============================================================================


class TestCLIIntegration:
    """Integration-style tests for the CLI."""

    def test_full_argument_parsing_scenario(self, cli_interface):
        """Test parsing a full set of arguments."""
        args = cli_interface.parse_args(
            [
                "--input",
                "input.csv",
                "--output",
                "output.csv",
                "--config",
                "config.ini",
                "--resume",
                "--verbose",
                "--batch-size",
                "75",
                "--max-retries",
                "5",
                "--ai-engine",
                "claude",
                "--duplicate-strategy",
                "newest",
                "--max-bookmarks-per-folder",
                "25",
                "--chrome-html",
                "--html-output",
                "bookmarks.html",
                "--html-title",
                "My Custom Bookmarks",
                "--since-last-run",
                "--state-db",
                "state.db",
            ]
        )

        assert args.input == "input.csv"
        assert args.output == "output.csv"
        assert args.config == "config.ini"
        assert args.resume is True
        assert args.verbose is True
        assert args.batch_size == 75
        assert args.max_retries == 5
        assert args.ai_engine == "claude"
        assert args.duplicate_strategy == "newest"
        assert args.max_bookmarks_per_folder == 25
        assert args.chrome_html is True
        assert args.html_output == "bookmarks.html"
        assert args.html_title == "My Custom Bookmarks"
        assert args.since_last_run is True
        assert args.state_db == "state.db"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
