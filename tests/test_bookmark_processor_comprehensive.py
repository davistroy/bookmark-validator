"""
Comprehensive unit tests for bookmark_processor module.

Tests cover:
- BookmarkProcessor initialization
- process_bookmarks method
- resume_processing method
- _run_auto_detection_mode method
- _save_bookmarks_as_export_csv method
- run_cli method
- ProcessingResults class
- Error handling scenarios
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pandas as pd
import pytest

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.bookmark_processor import BookmarkProcessor, ProcessingResults
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.pipeline import PipelineConfig, PipelineResults
from tests.fixtures.mock_utilities import (
    MockAIProcessor,
    MockContentAnalyzer,
    MockEnhancedAIProcessor,
    MockRequestsSession,
    create_mock_pipeline_context,
)
from tests.fixtures.test_data import (
    create_expected_import_dataframe,
    create_sample_bookmark_objects,
    create_sample_export_dataframe,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="bookmark_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config():
    """Create a Configuration instance for testing."""
    return Configuration()


@pytest.fixture
def processor(config):
    """Create a BookmarkProcessor instance for testing."""
    return BookmarkProcessor(config)


@pytest.fixture
def sample_bookmarks() -> List[Bookmark]:
    """Create sample bookmark objects for testing."""
    return create_sample_bookmark_objects()


@pytest.fixture
def sample_input_file(temp_dir) -> Path:
    """Create a sample input CSV file for testing."""
    sample_df = create_sample_export_dataframe()
    input_file = temp_dir / "input.csv"
    sample_df.to_csv(input_file, index=False)
    return input_file


class TestProcessingResultsClass:
    """Test ProcessingResults class initialization and methods."""

    def test_init_with_pipeline_results(self):
        """Test ProcessingResults initialization with PipelineResults."""
        pipeline_results = PipelineResults(
            total_bookmarks=100,
            valid_bookmarks=95,
            invalid_bookmarks=5,
            ai_processed=90,
            tagged_bookmarks=85,
            unique_tags=50,
            processing_time=120.5,
            stages_completed=["validation", "content_extraction", "ai_processing"],
            error_summary={"timeout": 3, "not_found": 2},
            statistics={"avg_time": 1.2, "folder_generation": {"total_folders": 10}},
        )

        results = ProcessingResults(pipeline_results)

        assert results.total_bookmarks == 100
        assert results.processed_bookmarks == 95
        assert results.valid_bookmarks == 95
        assert results.invalid_bookmarks == 5
        assert results.ai_processed == 90
        assert results.tagged_bookmarks == 85
        assert results.unique_tags == 50
        assert results.processing_time == 120.5
        assert results.stages_completed == ["validation", "content_extraction", "ai_processing"]
        assert results.error_summary == {"timeout": 3, "not_found": 2}
        assert results.statistics == {"avg_time": 1.2, "folder_generation": {"total_folders": 10}}
        assert results.errors == []

    def test_init_without_pipeline_results(self):
        """Test ProcessingResults initialization without PipelineResults."""
        results = ProcessingResults()

        assert results.total_bookmarks == 0
        assert results.processed_bookmarks == 0
        assert results.valid_bookmarks == 0
        assert results.invalid_bookmarks == 0
        assert results.ai_processed == 0
        assert results.tagged_bookmarks == 0
        assert results.unique_tags == 0
        assert results.processing_time == 0.0
        assert results.stages_completed == []
        assert results.error_summary == {}
        assert results.statistics == {}
        assert results.errors == []

    def test_str_representation(self):
        """Test ProcessingResults __str__ method."""
        pipeline_results = PipelineResults(
            total_bookmarks=100,
            valid_bookmarks=95,
            invalid_bookmarks=5,
            ai_processed=90,
            tagged_bookmarks=85,
            unique_tags=50,
            processing_time=120.55,
            stages_completed=["validation"],
            error_summary={},
            statistics={},
        )

        results = ProcessingResults(pipeline_results)
        str_repr = str(results)

        assert "total=100" in str_repr
        assert "valid=95" in str_repr
        assert "invalid=5" in str_repr
        assert "ai_processed=90" in str_repr
        assert "tagged=85" in str_repr
        assert "unique_tags=50" in str_repr
        assert "errors=0" in str_repr
        assert "time=120.55s" in str_repr

    def test_str_representation_empty(self):
        """Test ProcessingResults __str__ method with empty results."""
        results = ProcessingResults()
        str_repr = str(results)

        assert "total=0" in str_repr
        assert "valid=0" in str_repr
        assert "time=0.00s" in str_repr


class TestBookmarkProcessorInitialization:
    """Test BookmarkProcessor initialization."""

    def test_init_with_config(self, config):
        """Test BookmarkProcessor initialization with Configuration."""
        processor = BookmarkProcessor(config)

        assert processor.config is config
        assert processor.checkpoint_manager is None
        assert processor.progress_tracker is None
        assert processor.logger is not None

    def test_init_preserves_config_reference(self, config):
        """Test that BookmarkProcessor preserves config reference."""
        processor = BookmarkProcessor(config)
        assert processor.config is config


class TestProcessBookmarksMethod:
    """Test process_bookmarks method."""

    def test_process_bookmarks_success(self, processor, temp_dir):
        """Test successful bookmark processing."""
        # Create test input file
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        # Mock the pipeline execution
        with patch("bookmark_processor.core.bookmark_processor.BookmarkProcessingPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = PipelineResults(
                total_bookmarks=5,
                valid_bookmarks=4,
                invalid_bookmarks=1,
                ai_processed=3,
                tagged_bookmarks=4,
                unique_tags=10,
                processing_time=1.5,
                stages_completed=["validation", "ai_processing"],
                error_summary={},
                statistics={},
            )
            mock_pipeline_class.return_value = mock_pipeline

            results = processor.process_bookmarks(
                input_file=input_file,
                output_file=output_file,
                resume=False,
            )

            assert isinstance(results, ProcessingResults)
            assert results.total_bookmarks == 5
            assert results.valid_bookmarks == 4
            assert results.invalid_bookmarks == 1
            assert results.ai_processed == 3
            assert results.processing_time == 1.5
            mock_pipeline.execute.assert_called_once()

    def test_process_bookmarks_with_all_options(self, processor, temp_dir):
        """Test process_bookmarks with all optional parameters."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        with patch("bookmark_processor.core.bookmark_processor.BookmarkProcessingPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = PipelineResults(
                total_bookmarks=5,
                valid_bookmarks=5,
                invalid_bookmarks=0,
                ai_processed=5,
                tagged_bookmarks=5,
                unique_tags=15,
                processing_time=2.0,
                stages_completed=["validation", "ai_processing", "tagging"],
                error_summary={},
                statistics={},
            )
            mock_pipeline_class.return_value = mock_pipeline

            results = processor.process_bookmarks(
                input_file=input_file,
                output_file=output_file,
                resume=True,
                verbose=True,
                batch_size=50,
                max_retries=5,
                clear_checkpoints=True,
                detect_duplicates=False,
                duplicate_strategy="keep_first",
                generate_folders=False,
                max_bookmarks_per_folder=30,
                ai_engine="claude",
                generate_chrome_html=True,
                chrome_html_output=str(temp_dir / "bookmarks.html"),
                html_title="My Enhanced Bookmarks",
            )

            assert results.valid_bookmarks == 5
            mock_pipeline_class.assert_called_once()

            # Verify PipelineConfig was created with correct parameters
            call_args = mock_pipeline_class.call_args
            config_arg = call_args[0][0]
            assert config_arg.verbose is True
            assert config_arg.batch_size == 50
            assert config_arg.max_retries == 5
            assert config_arg.clear_checkpoints is True
            assert config_arg.detect_duplicates is False
            assert config_arg.duplicate_strategy == "keep_first"
            assert config_arg.generate_folders is False
            assert config_arg.ai_engine == "claude"
            assert config_arg.generate_chrome_html is True

    def test_process_bookmarks_exception_handling(self, processor, temp_dir):
        """Test process_bookmarks exception handling."""
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"

        # Create an empty file that will cause an error
        input_file.touch()

        with patch("bookmark_processor.core.bookmark_processor.BookmarkProcessingPipeline") as mock_pipeline_class:
            mock_pipeline_class.side_effect = ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                processor.process_bookmarks(
                    input_file=input_file,
                    output_file=output_file,
                    resume=False,
                )

    def test_process_bookmarks_verbose_callback(self, processor, temp_dir):
        """Test that verbose mode triggers progress callback."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        with patch("bookmark_processor.core.bookmark_processor.BookmarkProcessingPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()

            # Capture the progress callback
            captured_callback = None
            def capture_execute(progress_callback=None):
                nonlocal captured_callback
                captured_callback = progress_callback
                return PipelineResults(
                    total_bookmarks=5,
                    valid_bookmarks=5,
                    invalid_bookmarks=0,
                    ai_processed=5,
                    tagged_bookmarks=5,
                    unique_tags=10,
                    processing_time=1.0,
                    stages_completed=["validation"],
                    error_summary={},
                    statistics={},
                )

            mock_pipeline.execute.side_effect = capture_execute
            mock_pipeline_class.return_value = mock_pipeline

            processor.process_bookmarks(
                input_file=input_file,
                output_file=output_file,
                resume=False,
                verbose=True,
            )

            # Verify callback was passed
            assert captured_callback is not None
            # Test that callback works without error
            captured_callback("Test message")


class TestResumeProcessingMethod:
    """Test resume_processing method."""

    def test_resume_processing_returns_empty_results(self, processor):
        """Test resume_processing returns empty ProcessingResults."""
        # Lines 158-160: Test the resume_processing stub method
        results = processor.resume_processing()

        assert isinstance(results, ProcessingResults)
        assert results.total_bookmarks == 0
        assert results.valid_bookmarks == 0
        assert results.processing_time == 0.0


class TestRunAutoDetectionMode:
    """Test _run_auto_detection_mode method."""

    def test_auto_detection_no_files_found(self, processor, temp_dir, capsys):
        """Test auto-detection when no bookmark files are found (lines 178-181)."""
        validated_args = {
            "input_path": None,
            "output_path": None,
            "resume": False,
            "clear_checkpoints": False,
        }

        with patch.object(processor, "_run_auto_detection_mode") as mock_method:
            # We need to test the actual implementation, so let's patch MultiFileProcessor
            pass

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.auto_detect_files.return_value = []
            mock_processor_class.return_value = mock_processor

            result = processor._run_auto_detection_mode(validated_args)

            assert result == 1
            captured = capsys.readouterr()
            assert "No bookmark files found" in captured.out

    def test_auto_detection_files_detected_but_no_bookmarks(self, processor, temp_dir, capsys):
        """Test auto-detection when files are found but no valid bookmarks (lines 192-194)."""
        validated_args = {
            "input_path": None,
            "output_path": None,
            "resume": False,
            "clear_checkpoints": False,
        }

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.auto_detect_files.return_value = [Path(temp_dir / "test.csv")]
            mock_processor.process_multiple_files.return_value = ([], {"file_errors": {}})
            mock_processor_class.return_value = mock_processor

            result = processor._run_auto_detection_mode(validated_args)

            assert result == 1
            captured = capsys.readouterr()
            assert "No valid bookmarks found" in captured.out

    def test_auto_detection_successful_processing(self, processor, temp_dir, capsys):
        """Test successful auto-detection processing (lines 196-280)."""
        validated_args = {
            "input_path": None,
            "output_path": temp_dir / "output.csv",
            "resume": True,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_folders": True,
            "max_bookmarks_per_folder": 20,
            "ai_engine": "local",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        sample_bookmarks = create_sample_bookmark_objects()

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.auto_detect_files.return_value = [
                Path(temp_dir / "file1.csv"),
                Path(temp_dir / "file2.csv"),
            ]
            mock_processor.process_multiple_files.return_value = (
                sample_bookmarks,
                {"file_errors": {}}
            )
            mock_processor_class.return_value = mock_processor

            with patch.object(processor, "_save_bookmarks_as_export_csv"):
                with patch.object(processor, "process_bookmarks") as mock_process:
                    mock_process.return_value = ProcessingResults(
                        PipelineResults(
                            total_bookmarks=5,
                            valid_bookmarks=4,
                            invalid_bookmarks=1,
                            ai_processed=3,
                            tagged_bookmarks=4,
                            unique_tags=10,
                            processing_time=1.5,
                            stages_completed=["validation", "ai_processing"],
                            error_summary={},
                            statistics={"folder_generation": {"total_folders": 5, "max_depth": 2}},
                        )
                    )

                    result = processor._run_auto_detection_mode(validated_args)

                    assert result == 0
                    captured = capsys.readouterr()
                    assert "Auto-detected" in captured.out
                    assert "Multi-file processing completed" in captured.out

    def test_auto_detection_with_file_errors(self, processor, temp_dir, capsys):
        """Test auto-detection with file processing errors (lines 272-278)."""
        validated_args = {
            "input_path": None,
            "output_path": temp_dir / "output.csv",
            "resume": True,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_folders": True,
            "max_bookmarks_per_folder": 20,
            "ai_engine": "local",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        sample_bookmarks = create_sample_bookmark_objects()

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.auto_detect_files.return_value = [Path(temp_dir / "file1.csv")]
            mock_processor.process_multiple_files.return_value = (
                sample_bookmarks,
                {"file_errors": {"file2.csv": "Invalid format"}}
            )
            mock_processor_class.return_value = mock_processor

            with patch.object(processor, "_save_bookmarks_as_export_csv"):
                with patch.object(processor, "process_bookmarks") as mock_process:
                    mock_process.return_value = ProcessingResults(
                        PipelineResults(
                            total_bookmarks=5,
                            valid_bookmarks=4,
                            invalid_bookmarks=1,
                            ai_processed=3,
                            tagged_bookmarks=4,
                            unique_tags=10,
                            processing_time=1.5,
                            stages_completed=["validation"],
                            error_summary={},
                            statistics={},
                        )
                    )

                    result = processor._run_auto_detection_mode(validated_args)

                    assert result == 0
                    captured = capsys.readouterr()
                    assert "File processing errors" in captured.out

    def test_auto_detection_without_folder_generation_stats(self, processor, temp_dir, capsys):
        """Test auto-detection output when folder generation stats are missing (line 267)."""
        validated_args = {
            "input_path": None,
            "output_path": temp_dir / "output.csv",
            "resume": False,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_folders": True,
            "max_bookmarks_per_folder": 20,
            "ai_engine": "local",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        sample_bookmarks = create_sample_bookmark_objects()

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.auto_detect_files.return_value = [Path(temp_dir / "file1.csv")]
            mock_processor.process_multiple_files.return_value = (
                sample_bookmarks,
                {"file_errors": {}}
            )
            mock_processor_class.return_value = mock_processor

            with patch.object(processor, "_save_bookmarks_as_export_csv"):
                with patch.object(processor, "process_bookmarks") as mock_process:
                    mock_process.return_value = ProcessingResults(
                        PipelineResults(
                            total_bookmarks=5,
                            valid_bookmarks=4,
                            invalid_bookmarks=1,
                            ai_processed=3,
                            tagged_bookmarks=4,
                            unique_tags=10,
                            processing_time=1.5,
                            stages_completed=["validation"],
                            error_summary={},
                            statistics={},  # No folder_generation key
                        )
                    )

                    result = processor._run_auto_detection_mode(validated_args)

                    assert result == 0
                    captured = capsys.readouterr()
                    assert "AI folder generation: Not completed" in captured.out

    def test_auto_detection_exception_handling(self, processor, temp_dir, capsys):
        """Test auto-detection exception handling (lines 286-289)."""
        validated_args = {
            "input_path": None,
            "output_path": None,
            "resume": False,
            "clear_checkpoints": False,
        }

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor_class.side_effect = Exception("Test exception")

            result = processor._run_auto_detection_mode(validated_args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Auto-detection processing failed" in captured.out

    def test_auto_detection_with_default_output_path(self, processor, temp_dir, capsys):
        """Test auto-detection generates timestamp output when no output specified (lines 202-207)."""
        validated_args = {
            "input_path": None,
            "output_path": None,  # No output path specified
            "resume": False,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_folders": True,
            "max_bookmarks_per_folder": 20,
            "ai_engine": "local",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        sample_bookmarks = create_sample_bookmark_objects()

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.auto_detect_files.return_value = [Path(temp_dir / "file1.csv")]
            mock_processor.process_multiple_files.return_value = (
                sample_bookmarks,
                {"file_errors": {}}
            )
            mock_processor_class.return_value = mock_processor

            with patch.object(processor, "_save_bookmarks_as_export_csv"):
                with patch.object(processor, "process_bookmarks") as mock_process:
                    mock_process.return_value = ProcessingResults(
                        PipelineResults(
                            total_bookmarks=5,
                            valid_bookmarks=5,
                            invalid_bookmarks=0,
                            ai_processed=5,
                            tagged_bookmarks=5,
                            unique_tags=10,
                            processing_time=1.0,
                            stages_completed=["validation"],
                            error_summary={},
                            statistics={},
                        )
                    )

                    result = processor._run_auto_detection_mode(validated_args)

                    assert result == 0
                    # Verify that process_bookmarks was called with a generated output file
                    call_args = mock_process.call_args
                    output_file = call_args[1]["output_file"]
                    assert "combined_bookmarks_" in str(output_file)


class TestSaveBookmarksAsExportCsv:
    """Test _save_bookmarks_as_export_csv method."""

    def test_save_bookmarks_as_export_csv(self, processor, temp_dir, sample_bookmarks):
        """Test saving bookmarks to export CSV format (lines 301-328)."""
        output_file = temp_dir / "export.csv"

        processor._save_bookmarks_as_export_csv(sample_bookmarks, output_file)

        assert output_file.exists()

        # Read the file and verify structure
        df = pd.read_csv(output_file)

        # Verify 11 columns (raindrop.io export format)
        expected_columns = [
            "id", "title", "note", "excerpt", "url", "folder",
            "tags", "created", "cover", "highlights", "favorite"
        ]
        assert list(df.columns) == expected_columns

        # Verify data was written correctly
        assert len(df) == len(sample_bookmarks)
        assert df.iloc[0]["url"] == sample_bookmarks[0].url
        assert df.iloc[0]["title"] == sample_bookmarks[0].title

    def test_save_bookmarks_with_empty_fields(self, processor, temp_dir):
        """Test saving bookmarks with empty/None fields."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title="",
                note=None,
                folder=None,
                tags=[],
                created=None,
            )
        ]

        output_file = temp_dir / "export.csv"
        processor._save_bookmarks_as_export_csv(bookmarks, output_file)

        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert df.iloc[0]["url"] == "https://example.com"

    def test_save_bookmarks_with_tags(self, processor, temp_dir):
        """Test saving bookmarks with tags."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title="Test",
                tags=["tag1", "tag2", "tag3"],
                created=datetime(2024, 1, 1, 12, 0, 0),
            )
        ]

        output_file = temp_dir / "export.csv"
        processor._save_bookmarks_as_export_csv(bookmarks, output_file)

        df = pd.read_csv(output_file)
        assert "tag1, tag2, tag3" in df.iloc[0]["tags"]

    def test_save_bookmarks_with_favorite(self, processor, temp_dir):
        """Test saving bookmarks with favorite field."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title="Test",
                favorite=True,
            )
        ]

        output_file = temp_dir / "export.csv"
        processor._save_bookmarks_as_export_csv(bookmarks, output_file)

        df = pd.read_csv(output_file)
        # The value may be read as boolean True or string "true" depending on pandas parsing
        assert str(df.iloc[0]["favorite"]).lower() == "true"


class TestRunCliMethod:
    """Test run_cli method."""

    def test_run_cli_auto_detection_mode(self, processor, capsys):
        """Test run_cli triggers auto-detection when no input (line 342-343)."""
        validated_args = {
            "input_path": None,
            "output_path": None,
            "resume": False,
            "clear_checkpoints": False,
        }

        with patch.object(processor, "_run_auto_detection_mode") as mock_auto:
            mock_auto.return_value = 0

            result = processor.run_cli(validated_args)

            assert result == 0
            mock_auto.assert_called_once_with(validated_args)

    def test_run_cli_single_file_processing(self, processor, temp_dir, capsys):
        """Test run_cli with single file processing (lines 346-373)."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        validated_args = {
            "input_path": input_file,
            "output_path": output_file,
            "resume": True,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        with patch.object(processor, "process_bookmarks") as mock_process:
            mock_process.return_value = ProcessingResults(
                PipelineResults(
                    total_bookmarks=5,
                    valid_bookmarks=4,
                    invalid_bookmarks=1,
                    ai_processed=3,
                    tagged_bookmarks=4,
                    unique_tags=10,
                    processing_time=1.5,
                    stages_completed=["validation", "ai_processing"],
                    error_summary={},
                    statistics={},
                )
            )

            result = processor.run_cli(validated_args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Processing completed successfully" in captured.out
            assert "Total bookmarks: 5" in captured.out
            assert "Valid bookmarks: 4" in captured.out
            assert "Processing time: 1.50s" in captured.out
            assert "validation, ai_processing" in captured.out

    def test_run_cli_with_error_summary(self, processor, temp_dir, capsys):
        """Test run_cli output with error summary (lines 375-378)."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        validated_args = {
            "input_path": input_file,
            "output_path": output_file,
            "resume": False,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        with patch.object(processor, "process_bookmarks") as mock_process:
            mock_process.return_value = ProcessingResults(
                PipelineResults(
                    total_bookmarks=5,
                    valid_bookmarks=3,
                    invalid_bookmarks=2,
                    ai_processed=3,
                    tagged_bookmarks=3,
                    unique_tags=10,
                    processing_time=1.5,
                    stages_completed=["validation"],
                    error_summary={"timeout": 1, "not_found": 1},
                    statistics={},
                )
            )

            result = processor.run_cli(validated_args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Error summary:" in captured.out
            assert "timeout: 1" in captured.out
            assert "not_found: 1" in captured.out

    def test_run_cli_with_critical_errors(self, processor, temp_dir, capsys):
        """Test run_cli output with critical errors (lines 380-384)."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        validated_args = {
            "input_path": input_file,
            "output_path": output_file,
            "resume": False,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        with patch.object(processor, "process_bookmarks") as mock_process:
            results = ProcessingResults(
                PipelineResults(
                    total_bookmarks=5,
                    valid_bookmarks=3,
                    invalid_bookmarks=2,
                    ai_processed=3,
                    tagged_bookmarks=3,
                    unique_tags=10,
                    processing_time=1.5,
                    stages_completed=["validation"],
                    error_summary={},
                    statistics={},
                )
            )
            results.errors = ["Critical error 1", "Critical error 2"]
            mock_process.return_value = results

            result = processor.run_cli(validated_args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Critical errors: 2" in captured.out

    def test_run_cli_exception_handling(self, processor, temp_dir, capsys):
        """Test run_cli exception handling (lines 388-391)."""
        validated_args = {
            "input_path": temp_dir / "nonexistent.csv",
            "output_path": temp_dir / "output.csv",
            "resume": False,
            "clear_checkpoints": False,
        }

        with patch.object(processor, "process_bookmarks") as mock_process:
            mock_process.side_effect = Exception("Test exception")

            result = processor.run_cli(validated_args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Processing failed:" in captured.out
            assert "Test exception" in captured.out


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_process_bookmarks_with_pathlib_path(self, processor, temp_dir):
        """Test process_bookmarks accepts pathlib.Path objects."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        with patch("bookmark_processor.core.bookmark_processor.BookmarkProcessingPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = PipelineResults(
                total_bookmarks=5,
                valid_bookmarks=5,
                invalid_bookmarks=0,
                ai_processed=5,
                tagged_bookmarks=5,
                unique_tags=10,
                processing_time=1.0,
                stages_completed=["validation"],
                error_summary={},
                statistics={},
            )
            mock_pipeline_class.return_value = mock_pipeline

            # Pass Path objects
            results = processor.process_bookmarks(
                input_file=Path(input_file),
                output_file=Path(output_file),
                resume=False,
            )

            assert results.total_bookmarks == 5

    def test_auto_detection_with_generate_folders_disabled(self, processor, temp_dir, capsys):
        """Test auto-detection output when generate_folders is disabled (line 258-268)."""
        validated_args = {
            "input_path": None,
            "output_path": temp_dir / "output.csv",
            "resume": False,
            "clear_checkpoints": False,
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_folders": False,  # Disabled
            "max_bookmarks_per_folder": 20,
            "ai_engine": "local",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Enhanced Bookmarks",
        }

        sample_bookmarks = create_sample_bookmark_objects()

        with patch("bookmark_processor.core.bookmark_processor.MultiFileProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.auto_detect_files.return_value = [Path(temp_dir / "file1.csv")]
            mock_processor.process_multiple_files.return_value = (
                sample_bookmarks,
                {"file_errors": {}}
            )
            mock_processor_class.return_value = mock_processor

            with patch.object(processor, "_save_bookmarks_as_export_csv"):
                with patch.object(processor, "process_bookmarks") as mock_process:
                    mock_process.return_value = ProcessingResults(
                        PipelineResults(
                            total_bookmarks=5,
                            valid_bookmarks=5,
                            invalid_bookmarks=0,
                            ai_processed=5,
                            tagged_bookmarks=5,
                            unique_tags=10,
                            processing_time=1.0,
                            stages_completed=["validation"],
                            error_summary={},
                            statistics={},
                        )
                    )

                    result = processor._run_auto_detection_mode(validated_args)

                    assert result == 0
                    captured = capsys.readouterr()
                    # Should not have folder generation output when disabled
                    assert "AI-generated folders" not in captured.out

    def test_pipeline_config_with_progress_level(self, processor, temp_dir):
        """Test that verbose flag properly sets progress_level."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        with patch("bookmark_processor.core.bookmark_processor.BookmarkProcessingPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = PipelineResults(
                total_bookmarks=5,
                valid_bookmarks=5,
                invalid_bookmarks=0,
                ai_processed=5,
                tagged_bookmarks=5,
                unique_tags=10,
                processing_time=1.0,
                stages_completed=["validation"],
                error_summary={},
                statistics={},
            )
            mock_pipeline_class.return_value = mock_pipeline

            processor.process_bookmarks(
                input_file=input_file,
                output_file=output_file,
                resume=False,
                verbose=True,
            )

            # Verify PipelineConfig was created with DETAILED progress level
            call_args = mock_pipeline_class.call_args
            config_arg = call_args[0][0]
            from bookmark_processor.utils.progress_tracker import ProgressLevel
            assert config_arg.progress_level == ProgressLevel.DETAILED


class TestIntegrationScenarios:
    """Integration-like tests for complete workflows."""

    def test_full_single_file_workflow(self, processor, temp_dir, capsys):
        """Test complete single file processing workflow."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        validated_args = {
            "input_path": input_file,
            "output_path": output_file,
            "resume": False,
            "clear_checkpoints": False,
            "verbose": True,
            "batch_size": 50,
            "max_retries": 2,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Test Bookmarks",
        }

        with patch.object(processor, "process_bookmarks") as mock_process:
            mock_process.return_value = ProcessingResults(
                PipelineResults(
                    total_bookmarks=5,
                    valid_bookmarks=4,
                    invalid_bookmarks=1,
                    ai_processed=4,
                    tagged_bookmarks=4,
                    unique_tags=15,
                    processing_time=2.5,
                    stages_completed=["validation", "content_extraction", "ai_processing", "tagging"],
                    error_summary={"timeout": 1},
                    statistics={},
                )
            )

            result = processor.run_cli(validated_args)

            assert result == 0
            mock_process.assert_called_once()

            # Verify correct parameters were passed
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["input_file"] == input_file
            assert call_kwargs["output_file"] == output_file
            assert call_kwargs["verbose"] is True
            assert call_kwargs["batch_size"] == 50

    def test_resume_and_clear_checkpoints_interaction(self, processor, temp_dir):
        """Test that clear_checkpoints overrides resume flag."""
        sample_df = create_sample_export_dataframe()
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"
        sample_df.to_csv(input_file, index=False)

        validated_args = {
            "input_path": input_file,
            "output_path": output_file,
            "resume": True,
            "clear_checkpoints": True,  # Should override resume
            "verbose": False,
            "batch_size": 100,
            "max_retries": 3,
            "detect_duplicates": True,
            "duplicate_strategy": "highest_quality",
            "generate_chrome_html": False,
            "chrome_html_output": None,
            "html_title": "Test",
        }

        with patch.object(processor, "process_bookmarks") as mock_process:
            mock_process.return_value = ProcessingResults(
                PipelineResults(
                    total_bookmarks=5,
                    valid_bookmarks=5,
                    invalid_bookmarks=0,
                    ai_processed=5,
                    tagged_bookmarks=5,
                    unique_tags=10,
                    processing_time=1.0,
                    stages_completed=["validation"],
                    error_summary={},
                    statistics={},
                )
            )

            processor.run_cli(validated_args)

            call_kwargs = mock_process.call_args[1]
            # When clear_checkpoints is True, resume should be False
            assert call_kwargs["resume"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
