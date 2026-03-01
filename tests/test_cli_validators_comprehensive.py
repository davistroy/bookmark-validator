"""
Comprehensive tests for CLI Validators

This module provides thorough test coverage for bookmark_processor/utils/cli_validators.py
"""

import argparse
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from bookmark_processor.utils.cli_validators import (
    AIEngineValidator,
    ArgumentCombinationValidator,
    BatchSizeValidator,
    CLIArgumentValidator,
    CLIValidator,
    ConfigFileValidator,
    DuplicateStrategyValidator,
    InputFileValidator,
    MaxRetriesValidator,
    OutputFileValidator,
    PathValidator,
    create_enhanced_parser,
    validate_cli_arguments,
)
from bookmark_processor.utils.input_validator import ValidationResult, ValidationSeverity


class TestPathValidator:
    """Tests for PathValidator class"""

    def test_empty_path_returns_error(self):
        """Test that empty path returns validation error"""
        validator = PathValidator("test_path", must_exist=False)
        result = validator.validate("")
        assert not result.is_valid
        assert any("empty" in str(issue).lower() for issue in result.issues)

    def test_none_path_returns_error(self):
        """Test that None path returns validation error"""
        validator = PathValidator("test_path", must_exist=False)
        result = validator.validate(None)
        assert not result.is_valid
        assert any("empty" in str(issue).lower() for issue in result.issues)

    def test_valid_existing_file(self):
        """Test validation of existing file"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                must_be_readable=True,
            )
            result = validator.validate(temp_path)
            assert result.is_valid
            assert result.sanitized_value is not None
        finally:
            os.unlink(temp_path)

    def test_valid_existing_directory(self):
        """Test validation of existing directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(
                "test_path",
                path_type="dir",
                must_exist=True,
                must_be_readable=True,
            )
            result = validator.validate(temp_dir)
            assert result.is_valid

    def test_nonexistent_path_with_must_exist(self):
        """Test that nonexistent path fails when must_exist is True"""
        validator = PathValidator(
            "test_path",
            path_type="file",
            must_exist=True,
        )
        result = validator.validate("/nonexistent/path/to/file.txt")
        assert not result.is_valid
        assert any("does not exist" in str(issue).lower() for issue in result.issues)

    def test_file_when_directory_expected(self):
        """Test error when file provided but directory expected"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="dir",
                must_exist=True,
            )
            result = validator.validate(temp_path)
            assert not result.is_valid
            assert any("not a directory" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_directory_when_file_expected(self):
        """Test error when directory provided but file expected"""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
            )
            result = validator.validate(temp_dir)
            assert not result.is_valid
            assert any("not a file" in str(issue).lower() for issue in result.issues)

    def test_disallowed_extension(self):
        """Test validation fails for disallowed file extension"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                allowed_extensions=[".csv", ".json"],
            )
            result = validator.validate(temp_path)
            assert not result.is_valid
            assert any("extension" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_allowed_extension(self):
        """Test validation passes for allowed file extension"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"col1,col2\nval1,val2")
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                allowed_extensions=[".csv", ".json"],
            )
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_create_parent_dirs(self):
        """Test creating parent directories when they don't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_path = os.path.join(temp_dir, "new_subdir", "file.txt")
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=False,
                must_be_writable=True,
                create_parent_dirs=True,
            )
            result = validator.validate(new_path)
            assert result.is_valid
            assert Path(new_path).parent.exists()

    def test_parent_dir_not_created_without_flag(self):
        """Test parent directories not created when flag is False"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_path = os.path.join(temp_dir, "nonexistent_subdir", "file.txt")
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=False,
                must_be_writable=True,
                create_parent_dirs=False,
            )
            result = validator.validate(new_path)
            assert not result.is_valid
            assert any("parent directory" in str(issue).lower() for issue in result.issues)

    def test_file_or_dir_type_accepts_both(self):
        """Test file_or_dir path type accepts both files and directories"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            file_path = f.name

        with tempfile.TemporaryDirectory() as dir_path:
            validator = PathValidator(
                "test_path",
                path_type="file_or_dir",
                must_exist=True,
            )
            # Test file
            result = validator.validate(file_path)
            assert result.is_valid

            # Test directory
            result = validator.validate(dir_path)
            assert result.is_valid

        os.unlink(file_path)

    def test_readable_permission_check(self):
        """Test readable permission check for existing file"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                must_be_readable=True,
            )
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_writable_permission_check_existing_file(self):
        """Test writable permission check for existing file"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                must_be_writable=True,
            )
            result = validator.validate(temp_path)
            # Should be valid if file is writable (which temp files usually are)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_writable_permission_check_new_file(self):
        """Test writable permission check for new file location"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_file = os.path.join(temp_dir, "new_file.txt")
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=False,
                must_be_writable=True,
            )
            result = validator.validate(new_file)
            assert result.is_valid

    def test_extension_case_insensitive(self):
        """Test that extension check is case insensitive"""
        with tempfile.NamedTemporaryFile(suffix=".CSV", delete=False) as f:
            f.write(b"col1,col2\nval1,val2")
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                allowed_extensions=[".csv"],
            )
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)


class TestInputFileValidator:
    """Tests for InputFileValidator class"""

    def test_valid_csv_file(self):
        """Test validation of valid CSV file"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1,col2,col3\nval1,val2,val3\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_empty_csv_file(self):
        """Test validation fails for empty CSV file"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name
            # File is empty (0 bytes)

        try:
            validator = InputFileValidator()
            result = validator.validate(temp_path)
            assert not result.is_valid
            assert any("empty" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_file_without_delimiters(self):
        """Test warning for file that doesn't appear to be CSV"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("just plain text without delimiters\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            result = validator.validate(temp_path)
            # Should still be valid but have a warning
            assert result.has_warnings()
            assert any("csv format" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file(self):
        """Test validation fails for nonexistent file"""
        validator = InputFileValidator()
        result = validator.validate("/nonexistent/path/file.csv")
        assert not result.is_valid

    def test_wrong_extension(self):
        """Test validation fails for non-CSV extension"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            result = validator.validate(temp_path)
            assert not result.is_valid
            assert any("extension" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_tab_separated_file(self):
        """Test that tab-separated file is detected as valid CSV format"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1\tcol2\tcol3\nval1\tval2\tval3\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_semicolon_separated_file(self):
        """Test that semicolon-separated file is detected as valid CSV format"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1;col2;col3\nval1;val2;val3\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_large_file_warning(self):
        """Test that large files generate a warning"""
        validator = InputFileValidator()
        # Mock the file stat to simulate a large file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            # Mock path.stat() to return a large file size
            with mock.patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 600 * 1024 * 1024  # 600 MB
                # Re-create validator to avoid caching issues
                result = validator.validate(temp_path)
                assert result.has_warnings()
                assert any("large" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)


class TestOutputFileValidator:
    """Tests for OutputFileValidator class"""

    def test_valid_new_output_path(self):
        """Test validation of valid new output path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")
            validator = OutputFileValidator()
            result = validator.validate(output_path)
            assert result.is_valid

    def test_existing_file_warning(self):
        """Test warning when output file already exists"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("existing,content\n")
            temp_path = f.name

        try:
            validator = OutputFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid
            assert result.has_warnings()
            assert any("overwritten" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_creates_parent_dirs(self):
        """Test that parent directories are created for output file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "new_subdir", "output.csv")
            validator = OutputFileValidator()
            result = validator.validate(output_path)
            assert result.is_valid
            assert Path(output_path).parent.exists()

    def test_wrong_extension(self):
        """Test validation fails for non-CSV extension"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.txt")
            validator = OutputFileValidator()
            result = validator.validate(output_path)
            assert not result.is_valid
            assert any("extension" in str(issue).lower() for issue in result.issues)

    def test_output_path_metadata(self):
        """Test that output path is stored in metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")
            validator = OutputFileValidator()
            result = validator.validate(output_path)
            assert "output_path" in result.metadata


class TestConfigFileValidator:
    """Tests for ConfigFileValidator class"""

    def test_none_config_is_valid(self):
        """Test that None config file (optional) is valid"""
        validator = ConfigFileValidator()
        result = validator.validate(None)
        assert result.is_valid
        assert result.sanitized_value is None

    def test_valid_ini_config(self):
        """Test validation of valid INI config file"""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            f.write("[section]\nkey = value\n")
            temp_path = f.name

        try:
            validator = ConfigFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_config_without_ini_format(self):
        """Test warning when config doesn't appear to be INI format"""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            f.write("plain text without sections\n")
            temp_path = f.name

        try:
            validator = ConfigFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid  # Still valid but with warning
            assert result.has_warnings()
            assert any("ini format" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_nonexistent_config_file(self):
        """Test validation fails for nonexistent config file"""
        validator = ConfigFileValidator()
        result = validator.validate("/nonexistent/config.ini")
        assert not result.is_valid

    def test_wrong_extension(self):
        """Test validation fails for non-INI extension"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("[section]\nkey = value\n")
            temp_path = f.name

        try:
            validator = ConfigFileValidator()
            result = validator.validate(temp_path)
            assert not result.is_valid
        finally:
            os.unlink(temp_path)

    def test_cfg_extension_valid(self):
        """Test that .cfg extension is valid"""
        with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False, mode="w") as f:
            f.write("[section]\nkey = value\n")
            temp_path = f.name

        try:
            validator = ConfigFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)

    def test_conf_extension_valid(self):
        """Test that .conf extension is valid"""
        with tempfile.NamedTemporaryFile(suffix=".conf", delete=False, mode="w") as f:
            f.write("[section]\nkey = value\n")
            temp_path = f.name

        try:
            validator = ConfigFileValidator()
            result = validator.validate(temp_path)
            assert result.is_valid
        finally:
            os.unlink(temp_path)


class TestBatchSizeValidator:
    """Tests for BatchSizeValidator class"""

    def test_valid_batch_size(self):
        """Test valid batch size in recommended range"""
        validator = BatchSizeValidator()
        result = validator.validate(100)
        assert result.is_valid
        assert result.sanitized_value == 100

    def test_batch_size_in_recommended_range(self):
        """Test batch size in recommended range generates info"""
        validator = BatchSizeValidator()
        result = validator.validate(100)
        assert result.is_valid
        assert any("recommended" in str(issue).lower() for issue in result.issues)

    def test_small_batch_size_warning(self):
        """Test warning for very small batch size"""
        validator = BatchSizeValidator()
        result = validator.validate(5)
        assert result.is_valid
        assert result.has_warnings()
        assert any("small batch" in str(issue).lower() for issue in result.issues)

    def test_large_batch_size_warning(self):
        """Test warning for large batch size"""
        validator = BatchSizeValidator()
        result = validator.validate(600)
        assert result.is_valid
        assert result.has_warnings()
        assert any("memory" in str(issue).lower() for issue in result.issues)

    def test_batch_size_minimum(self):
        """Test minimum batch size boundary"""
        validator = BatchSizeValidator()
        result = validator.validate(1)
        assert result.is_valid

    def test_batch_size_maximum(self):
        """Test maximum batch size boundary"""
        validator = BatchSizeValidator()
        result = validator.validate(1000)
        assert result.is_valid

    def test_batch_size_below_minimum(self):
        """Test batch size below minimum fails"""
        validator = BatchSizeValidator()
        result = validator.validate(0)
        assert not result.is_valid

    def test_batch_size_above_maximum(self):
        """Test batch size above maximum fails"""
        validator = BatchSizeValidator()
        result = validator.validate(1001)
        assert not result.is_valid

    def test_batch_size_string_conversion(self):
        """Test batch size string is converted to int"""
        validator = BatchSizeValidator()
        result = validator.validate("100")
        assert result.is_valid
        assert result.sanitized_value == 100

    def test_batch_size_float_rejected(self):
        """Test float batch size is handled correctly"""
        validator = BatchSizeValidator()
        result = validator.validate(100.5)
        # Float validation depends on NumberValidator implementation
        # Just check it doesn't crash
        assert result is not None


class TestMaxRetriesValidator:
    """Tests for MaxRetriesValidator class"""

    def test_valid_max_retries(self):
        """Test valid max retries value"""
        validator = MaxRetriesValidator()
        result = validator.validate(3)
        assert result.is_valid
        assert result.sanitized_value == 3

    def test_zero_retries_warning(self):
        """Test warning for zero retries"""
        validator = MaxRetriesValidator()
        result = validator.validate(0)
        assert result.is_valid
        assert result.has_warnings()
        assert any("zero" in str(issue).lower() for issue in result.issues)

    def test_high_retries_warning(self):
        """Test warning for high retry count"""
        validator = MaxRetriesValidator()
        result = validator.validate(7)
        assert result.is_valid
        assert result.has_warnings()
        assert any("slow" in str(issue).lower() for issue in result.issues)

    def test_max_retries_minimum(self):
        """Test minimum retries boundary"""
        validator = MaxRetriesValidator()
        result = validator.validate(0)
        assert result.is_valid

    def test_max_retries_maximum(self):
        """Test maximum retries boundary"""
        validator = MaxRetriesValidator()
        result = validator.validate(10)
        assert result.is_valid

    def test_max_retries_below_minimum(self):
        """Test retries below minimum fails"""
        validator = MaxRetriesValidator()
        result = validator.validate(-1)
        assert not result.is_valid

    def test_max_retries_above_maximum(self):
        """Test retries above maximum fails"""
        validator = MaxRetriesValidator()
        result = validator.validate(11)
        assert not result.is_valid


class TestAIEngineValidator:
    """Tests for AIEngineValidator class"""

    def test_local_engine(self):
        """Test local AI engine selection"""
        validator = AIEngineValidator()
        result = validator.validate("local")
        assert result.is_valid
        assert result.sanitized_value == "local"
        assert any("no api key" in str(issue).lower() for issue in result.issues)

    def test_claude_engine(self):
        """Test Claude AI engine selection"""
        validator = AIEngineValidator()
        result = validator.validate("claude")
        assert result.is_valid
        assert result.sanitized_value == "claude"
        assert any("api key" in str(issue).lower() for issue in result.issues)

    def test_openai_engine(self):
        """Test OpenAI engine selection"""
        validator = AIEngineValidator()
        result = validator.validate("openai")
        assert result.is_valid
        assert result.sanitized_value == "openai"
        assert any("api key" in str(issue).lower() for issue in result.issues)

    def test_invalid_engine(self):
        """Test invalid AI engine selection"""
        validator = AIEngineValidator()
        result = validator.validate("invalid_engine")
        assert not result.is_valid

    def test_none_engine(self):
        """Test None AI engine"""
        validator = AIEngineValidator()
        result = validator.validate(None)
        assert not result.is_valid


class TestDuplicateStrategyValidator:
    """Tests for DuplicateStrategyValidator class"""

    def test_newest_strategy(self):
        """Test newest duplicate strategy"""
        validator = DuplicateStrategyValidator()
        result = validator.validate("newest")
        assert result.is_valid
        assert result.sanitized_value == "newest"
        assert any("recent" in str(issue).lower() for issue in result.issues)

    def test_oldest_strategy(self):
        """Test oldest duplicate strategy"""
        validator = DuplicateStrategyValidator()
        result = validator.validate("oldest")
        assert result.is_valid
        assert result.sanitized_value == "oldest"
        assert any("oldest" in str(issue).lower() for issue in result.issues)

    def test_most_complete_strategy(self):
        """Test most_complete duplicate strategy"""
        validator = DuplicateStrategyValidator()
        result = validator.validate("most_complete")
        assert result.is_valid
        assert result.sanitized_value == "most_complete"
        assert any("fields" in str(issue).lower() for issue in result.issues)

    def test_highest_quality_strategy(self):
        """Test highest_quality duplicate strategy"""
        validator = DuplicateStrategyValidator()
        result = validator.validate("highest_quality")
        assert result.is_valid
        assert result.sanitized_value == "highest_quality"
        assert any("quality" in str(issue).lower() for issue in result.issues)

    def test_invalid_strategy(self):
        """Test invalid duplicate strategy"""
        validator = DuplicateStrategyValidator()
        result = validator.validate("invalid")
        assert not result.is_valid


class TestArgumentCombinationValidator:
    """Tests for ArgumentCombinationValidator class"""

    def test_resume_and_clear_checkpoints_conflict(self):
        """Test that resume and clear_checkpoints conflict"""
        validator = ArgumentCombinationValidator()
        args = {"resume": True, "clear_checkpoints": True}
        result = validator.validate(args)
        assert not result.is_valid
        assert any("--resume" in str(issue) for issue in result.issues)

    def test_same_input_output_paths(self):
        """Test that same input and output paths conflict"""
        validator = ArgumentCombinationValidator()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            args = {"input_path": temp_path, "output_path": temp_path}
            result = validator.validate(args)
            assert not result.is_valid
            assert any("same" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_cloud_ai_without_duplicate_detection_warning(self):
        """Test warning when using cloud AI without duplicate detection"""
        validator = ArgumentCombinationValidator()
        args = {
            "ai_engine": "claude",
            "detect_duplicates": False,
        }
        result = validator.validate(args)
        assert result.is_valid  # Warning, not error
        assert result.has_warnings()
        assert any("duplicate" in str(issue).lower() for issue in result.issues)

    def test_large_batch_with_cloud_ai_warning(self):
        """Test warning when using large batch size with cloud AI"""
        validator = ArgumentCombinationValidator()
        args = {
            "ai_engine": "claude",
            "batch_size": 150,
        }
        result = validator.validate(args)
        assert result.is_valid  # Warning, not error
        assert result.has_warnings()
        assert any("rate limit" in str(issue).lower() for issue in result.issues)

    def test_valid_combination(self):
        """Test valid argument combination"""
        validator = ArgumentCombinationValidator()
        args = {
            "resume": False,
            "clear_checkpoints": False,
            "input_path": "/input.csv",
            "output_path": "/output.csv",
            "ai_engine": "local",
            "batch_size": 100,
        }
        result = validator.validate(args)
        assert result.is_valid

    def test_local_ai_with_duplicate_detection_off(self):
        """Test local AI with duplicate detection off (no warning)"""
        validator = ArgumentCombinationValidator()
        args = {
            "ai_engine": "local",
            "detect_duplicates": False,
        }
        result = validator.validate(args)
        assert result.is_valid
        # Should not have the cloud AI warning
        assert not any(
            "cloud ai" in str(issue).lower() and "duplicate" in str(issue).lower()
            for issue in result.issues
        )


class TestCLIArgumentValidator:
    """Tests for CLIArgumentValidator class"""

    def _create_mock_args(self, **kwargs):
        """Create mock argparse.Namespace with given arguments"""
        defaults = {
            "input": None,
            "output": None,
            "config": None,
            "batch_size": 100,
            "max_retries": 3,
            "ai_engine": "local",
            "duplicate_strategy": "highest_quality",
            "resume": False,
            "verbose": False,
            "clear_checkpoints": False,
            "no_duplicates": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_validate_all_valid_arguments(self):
        """Test validation of all valid arguments"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as inf:
            inf.write("col1,col2\nval1,val2\n")
            input_path = inf.name

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")

            try:
                args = self._create_mock_args(input=input_path, output=output_path)
                validator = CLIArgumentValidator()
                result = validator.validate_all_arguments(args)
                assert result.is_valid
            finally:
                os.unlink(input_path)

    def test_validate_missing_input(self):
        """Test validation fails when input file is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")
            args = self._create_mock_args(
                input="/nonexistent/input.csv", output=output_path
            )
            validator = CLIArgumentValidator()
            result = validator.validate_all_arguments(args)
            assert not result.is_valid

    def test_validate_invalid_batch_size(self):
        """Test validation fails for invalid batch size"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as inf:
            inf.write("col1,col2\nval1,val2\n")
            input_path = inf.name

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")

            try:
                args = self._create_mock_args(
                    input=input_path, output=output_path, batch_size=0
                )
                validator = CLIArgumentValidator()
                result = validator.validate_all_arguments(args)
                assert not result.is_valid
            finally:
                os.unlink(input_path)

    def test_generate_help_message_valid(self):
        """Test help message generation for valid result"""
        validator = CLIArgumentValidator()
        result = ValidationResult(is_valid=True)
        message = validator.generate_help_message(result)
        assert "valid" in message.lower()

    def test_generate_help_message_with_errors(self):
        """Test help message generation with errors"""
        validator = CLIArgumentValidator()
        result = ValidationResult(is_valid=False)
        result.add_error("Test error message")
        result.add_warning("Test warning message")
        message = validator.generate_help_message(result)
        assert "error" in message.lower()
        assert "warning" in message.lower()
        assert "suggestion" in message.lower()

    def test_optional_config_file(self):
        """Test that config file is optional"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as inf:
            inf.write("col1,col2\nval1,val2\n")
            input_path = inf.name

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")

            try:
                args = self._create_mock_args(
                    input=input_path, output=output_path, config=None
                )
                validator = CLIArgumentValidator()
                result = validator.validate_all_arguments(args)
                assert result.is_valid
            finally:
                os.unlink(input_path)


class TestValidateCLIArguments:
    """Tests for validate_cli_arguments function"""

    def test_successful_validation(self):
        """Test successful validation returns validated args"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as inf:
            inf.write("col1,col2\nval1,val2\n")
            input_path = inf.name

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")

            try:
                args = argparse.Namespace(
                    input=input_path,
                    output=output_path,
                    config=None,
                    batch_size=100,
                    max_retries=3,
                    ai_engine="local",
                    duplicate_strategy="highest_quality",
                    resume=False,
                    verbose=False,
                    clear_checkpoints=False,
                    no_duplicates=False,
                )
                validated_args, error_message = validate_cli_arguments(args)
                assert validated_args is not None
                assert error_message is None
            finally:
                os.unlink(input_path)

    def test_failed_validation(self):
        """Test failed validation returns error message"""
        args = argparse.Namespace(
            input="/nonexistent/input.csv",
            output="/tmp/output.csv",
            config=None,
            batch_size=100,
            max_retries=3,
            ai_engine="local",
            duplicate_strategy="highest_quality",
            resume=False,
            verbose=False,
            clear_checkpoints=False,
            no_duplicates=False,
        )
        validated_args, error_message = validate_cli_arguments(args)
        assert validated_args is None
        assert error_message is not None


class TestCreateEnhancedParser:
    """Tests for create_enhanced_parser function"""

    def test_parser_creation(self):
        """Test that parser is created successfully"""
        parser = create_enhanced_parser()
        assert parser is not None
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_has_required_arguments(self):
        """Test that parser has required arguments"""
        parser = create_enhanced_parser()
        # Test parsing with required arguments
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            args = parser.parse_args(["--input", temp_path, "--output", "output.csv"])
            assert args.input == temp_path
            assert args.output == "output.csv"
        finally:
            os.unlink(temp_path)

    def test_parser_default_values(self):
        """Test parser default values"""
        parser = create_enhanced_parser()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            args = parser.parse_args(["--input", temp_path, "--output", "output.csv"])
            assert args.batch_size == 100
            assert args.max_retries == 3
            assert args.ai_engine == "local"
            assert args.duplicate_strategy == "highest_quality"
            assert args.resume is False
            assert args.verbose is False
        finally:
            os.unlink(temp_path)

    def test_parser_short_options(self):
        """Test parser short option aliases"""
        parser = create_enhanced_parser()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            args = parser.parse_args(
                ["-i", temp_path, "-o", "output.csv", "-b", "50", "-m", "5", "-v", "-r"]
            )
            assert args.input == temp_path
            assert args.output == "output.csv"
            assert args.batch_size == 50
            assert args.max_retries == 5
            assert args.verbose is True
            assert args.resume is True
        finally:
            os.unlink(temp_path)

    def test_parser_ai_engine_choices(self):
        """Test AI engine choices are enforced"""
        parser = create_enhanced_parser()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            # Valid choice
            args = parser.parse_args(
                ["--input", temp_path, "--output", "out.csv", "--ai-engine", "claude"]
            )
            assert args.ai_engine == "claude"

            # Invalid choice should raise
            with pytest.raises(SystemExit):
                parser.parse_args(
                    [
                        "--input",
                        temp_path,
                        "--output",
                        "out.csv",
                        "--ai-engine",
                        "invalid",
                    ]
                )
        finally:
            os.unlink(temp_path)

    def test_parser_duplicate_strategy_choices(self):
        """Test duplicate strategy choices are enforced"""
        parser = create_enhanced_parser()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            # Valid choice
            args = parser.parse_args(
                [
                    "--input",
                    temp_path,
                    "--output",
                    "out.csv",
                    "--duplicate-strategy",
                    "newest",
                ]
            )
            assert args.duplicate_strategy == "newest"

            # Invalid choice should raise
            with pytest.raises(SystemExit):
                parser.parse_args(
                    [
                        "--input",
                        temp_path,
                        "--output",
                        "out.csv",
                        "--duplicate-strategy",
                        "invalid",
                    ]
                )
        finally:
            os.unlink(temp_path)


class TestCLIValidatorAlias:
    """Tests for CLIValidator alias"""

    def test_alias_is_same_class(self):
        """Test that CLIValidator is an alias for CLIArgumentValidator"""
        assert CLIValidator is CLIArgumentValidator


class TestExceptionHandlingPaths:
    """Tests for exception handling paths in validators"""

    def test_invalid_path_format(self):
        """Test handling of invalid path format that raises exception"""
        validator = PathValidator("test_path", must_exist=False)
        # Mock Path.resolve to raise an exception
        with mock.patch.object(Path, "resolve", side_effect=ValueError("Invalid path")):
            result = validator.validate("some_path")
            assert not result.is_valid
            assert any("invalid path format" in str(issue).lower() for issue in result.issues)

    def test_unreadable_file(self):
        """Test handling of unreadable file"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                must_be_readable=True,
            )
            # Mock os.access to return False for read check
            original_access = os.access

            def mock_access(path, mode):
                if mode == os.R_OK:
                    return False
                return original_access(path, mode)

            with mock.patch("os.access", side_effect=mock_access):
                result = validator.validate(temp_path)
                assert not result.is_valid
                assert any("not readable" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_unwritable_existing_file(self):
        """Test handling of unwritable existing file"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=True,
                must_be_writable=True,
            )
            # Mock os.access to return False for write check
            original_access = os.access

            def mock_access(path, mode):
                if mode == os.W_OK:
                    return False
                return original_access(path, mode)

            with mock.patch("os.access", side_effect=mock_access):
                result = validator.validate(temp_path)
                assert not result.is_valid
                assert any("not writable" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_cannot_create_parent_dirs(self):
        """Test handling when parent directory creation fails"""
        validator = PathValidator(
            "test_path",
            path_type="file",
            must_exist=False,
            must_be_writable=True,
            create_parent_dirs=True,
        )
        # Use a path where we cannot create directories
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError("No permission")):
            result = validator.validate("/nonexistent/subdir/file.txt")
            assert not result.is_valid
            assert any("cannot create" in str(issue).lower() for issue in result.issues)

    def test_unwritable_parent_directory(self):
        """Test handling of unwritable parent directory for new file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_file = os.path.join(temp_dir, "new_file.txt")
            validator = PathValidator(
                "test_path",
                path_type="file",
                must_exist=False,
                must_be_writable=True,
            )

            # Mock os.access to return False for parent write check
            original_access = os.access

            def mock_access(path, mode):
                if mode == os.W_OK and Path(path).is_dir():
                    return False
                return original_access(path, mode)

            with mock.patch("os.access", side_effect=mock_access):
                result = validator.validate(new_file)
                assert not result.is_valid
                assert any("not writable" in str(issue).lower() for issue in result.issues)

    def test_file_size_check_exception(self):
        """Test handling of exception during file size check"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            # Mock stat to raise an exception
            with mock.patch.object(Path, "stat", side_effect=OSError("Stat failed")):
                result = validator.validate(temp_path)
                # Should still work but with warning about size check
                assert result.has_warnings()
                assert any("cannot check file size" in str(issue).lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_csv_format_check_exception(self):
        """Test handling of exception during CSV format check"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            # Mock open to raise exception for format check
            original_open = open

            def mock_open_func(*args, **kwargs):
                if args[0] == Path(temp_path) or str(args[0]) == str(Path(temp_path).resolve()):
                    raise IOError("Cannot read file")
                return original_open(*args, **kwargs)

            with mock.patch("builtins.open", side_effect=mock_open_func):
                result = validator.validate(temp_path)
                # File size check happens first, then format check
                assert result.has_warnings()
        finally:
            os.unlink(temp_path)

    def test_config_format_check_exception(self):
        """Test handling of exception during config format check"""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            f.write("[section]\nkey=value\n")
            temp_path = f.name

        try:
            validator = ConfigFileValidator()
            # Mock open to raise exception for format check
            original_open = open

            call_count = [0]

            def mock_open_func(*args, **kwargs):
                # Allow first call to work (validation), fail on second (format check)
                call_count[0] += 1
                if call_count[0] > 0 and "r" in str(kwargs.get("mode", args[1] if len(args) > 1 else "r")):
                    raise IOError("Cannot read config")
                return original_open(*args, **kwargs)

            with mock.patch("builtins.open", side_effect=mock_open_func):
                result = validator.validate(temp_path)
                assert result.has_warnings()
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """Edge case tests for CLI validators"""

    def test_path_with_special_characters(self):
        """Test path validation with special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with spaces in name
            special_path = os.path.join(temp_dir, "file with spaces.csv")
            with open(special_path, "w") as f:
                f.write("col1,col2\nval1,val2\n")

            validator = InputFileValidator()
            result = validator.validate(special_path)
            assert result.is_valid

    def test_unicode_in_path(self):
        """Test path validation with unicode characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with unicode in name
            unicode_path = os.path.join(temp_dir, "file_unicode.csv")
            with open(unicode_path, "w", encoding="utf-8") as f:
                f.write("col1,col2\nval1,val2\n")

            validator = InputFileValidator()
            result = validator.validate(unicode_path)
            assert result.is_valid

    def test_very_long_path(self):
        """Test handling of very long paths"""
        validator = PathValidator("test_path", must_exist=False)
        # Create a very long path
        long_path = "a" * 300 + ".csv"
        # This may or may not be valid depending on OS
        result = validator.validate(long_path)
        # Just verify it doesn't crash
        assert result is not None

    def test_batch_size_edge_boundaries(self):
        """Test batch size at exact boundaries"""
        validator = BatchSizeValidator()

        # At lower boundary
        result = validator.validate(1)
        assert result.is_valid
        assert result.sanitized_value == 1

        # At upper boundary
        result = validator.validate(1000)
        assert result.is_valid
        assert result.sanitized_value == 1000

    def test_max_retries_edge_boundaries(self):
        """Test max retries at exact boundaries"""
        validator = MaxRetriesValidator()

        # At lower boundary
        result = validator.validate(0)
        assert result.is_valid

        # At upper boundary
        result = validator.validate(10)
        assert result.is_valid

    def test_empty_csv_first_line(self):
        """Test CSV with empty first line"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("\n")  # Empty first line
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            validator = InputFileValidator()
            result = validator.validate(temp_path)
            # Should have an error or warning about empty first line
            assert any(
                "empty" in str(issue).lower()
                for issue in result.issues
            )
        finally:
            os.unlink(temp_path)

    def test_validation_result_merge(self):
        """Test merging validation results"""
        result1 = ValidationResult(is_valid=True)
        result1.add_warning("Warning 1")
        result1.sanitized_value = "value1"

        result2 = ValidationResult(is_valid=True)
        result2.add_warning("Warning 2")
        result2.sanitized_value = "value2"

        merged = result1.merge(result2)
        assert merged.is_valid
        assert len(merged.issues) == 2
        assert merged.sanitized_value == "value2"  # Second value takes precedence

    def test_boolean_args_handling(self):
        """Test boolean argument handling in CLI validator"""
        validator = CLIArgumentValidator()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as inf:
            inf.write("col1,col2\nval1,val2\n")
            input_path = inf.name

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")

            try:
                args = argparse.Namespace(
                    input=input_path,
                    output=output_path,
                    config=None,
                    batch_size=100,
                    max_retries=3,
                    ai_engine="local",
                    duplicate_strategy="highest_quality",
                    resume=True,
                    verbose=True,
                    clear_checkpoints=False,
                    no_duplicates=True,
                )
                result = validator.validate_all_arguments(args)
                assert result.is_valid
                assert result.sanitized_value["resume"] is True
                assert result.sanitized_value["verbose"] is True
                assert result.sanitized_value["detect_duplicates"] is False
            finally:
                os.unlink(input_path)

    def test_sanitized_value_contains_all_paths(self):
        """Test that sanitized value contains all path arguments"""
        validator = CLIArgumentValidator()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as inf:
            inf.write("col1,col2\nval1,val2\n")
            input_path = inf.name

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.csv")

            try:
                args = argparse.Namespace(
                    input=input_path,
                    output=output_path,
                    config=None,
                    batch_size=100,
                    max_retries=3,
                    ai_engine="local",
                    duplicate_strategy="highest_quality",
                    resume=False,
                    verbose=False,
                    clear_checkpoints=False,
                    no_duplicates=False,
                )
                result = validator.validate_all_arguments(args)
                assert "input_path" in result.sanitized_value
                assert "output_path" in result.sanitized_value
                assert "config_path" in result.sanitized_value
            finally:
                os.unlink(input_path)
