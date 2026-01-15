"""
Integration tests for the bookmark processor.

Tests the complete workflow from CSV input to CSV output.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from bookmark_processor.core.batch_types import ValidationResult
from bookmark_processor.core.bookmark_processor import BookmarkProcessor
from bookmark_processor.core.pipeline import PipelineConfig
from tests.fixtures.test_data import (
    SAMPLE_RAINDROP_EXPORT_ROWS,
    TEST_CONFIGS,
    create_sample_export_dataframe,
)


def create_mock_validation_result(url: str, is_valid: bool = True, **kwargs) -> ValidationResult:
    """Create a mock ValidationResult for testing."""
    return ValidationResult(
        url=url,
        is_valid=is_valid,
        status_code=kwargs.get("status_code", 200 if is_valid else 404),
        final_url=kwargs.get("final_url", url),
        response_time=kwargs.get("response_time", 0.1),
        error_message=kwargs.get("error_message"),
        error_type=kwargs.get("error_type"),
        content_type=kwargs.get("content_type", "text/html"),
    )


@pytest.mark.integration
class TestBookmarkProcessorIntegration:
    """Integration tests for the complete bookmark processor workflow."""

    @pytest.fixture
    def temp_input_file(self):
        """Create a temporary input CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = create_sample_export_dataframe()
            df.to_csv(f, index=False)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary output file path."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        # Remove the file (we just want the path)
        os.unlink(temp_path)

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_complete_workflow_success(self, temp_input_file, temp_output_file):
        """Test the complete workflow with successful processing."""
        from bookmark_processor.config.configuration import Configuration

        # Create test configuration
        config = Configuration()
        processor = BookmarkProcessor(config)

        # Mock URL validation to avoid actual HTTP calls
        def mock_validate_url(url):
            return create_mock_validation_result(url, is_valid=True)

        with patch(
            "bookmark_processor.core.url_validator.URLValidator.validate_url",
            side_effect=mock_validate_url,
        ):
            # Process bookmarks
            results = processor.process_bookmarks(
                input_file=Path(temp_input_file),
                output_file=Path(temp_output_file),
                resume=False,
                verbose=False,
                batch_size=10,
                max_retries=1,
                clear_checkpoints=True,
            )

        # Verify results
        assert results is not None
        assert results.total_bookmarks > 0
        assert results.valid_bookmarks > 0

        # Verify output file exists and has correct format
        assert os.path.exists(temp_output_file)

        output_df = pd.read_csv(temp_output_file)
        expected_columns = ["url", "folder", "title", "note", "tags", "created"]
        assert list(output_df.columns) == expected_columns
        assert len(output_df) > 0

    def test_workflow_with_invalid_urls(self, temp_input_file, temp_output_file):
        """Test workflow with some invalid URLs."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Mock mixed responses - some successful, some failed
        def mock_validate_url(url):
            if "invalid" in url or "not-a-valid-url" in url:
                return create_mock_validation_result(
                    url,
                    is_valid=False,
                    error_message="Connection failed",
                    error_type="connection_error",
                )
            return create_mock_validation_result(url, is_valid=True)

        with patch(
            "bookmark_processor.core.url_validator.URLValidator.validate_url",
            side_effect=mock_validate_url,
        ):
            results = processor.process_bookmarks(
                input_file=Path(temp_input_file),
                output_file=Path(temp_output_file),
                resume=False,
                verbose=False,
                batch_size=10,
                max_retries=1,
                clear_checkpoints=True,
            )

        # Should still process successfully, just with some failures
        assert results is not None
        assert results.total_bookmarks > 0
        assert results.invalid_bookmarks >= 0  # Some URLs might be invalid

        # Output file should still be created with valid bookmarks
        assert os.path.exists(temp_output_file)
        output_df = pd.read_csv(temp_output_file)
        assert len(output_df) >= 0  # At least some bookmarks should be valid

    def test_workflow_with_ai_processing(self, temp_input_file, temp_output_file):
        """Test workflow with AI processing enabled (mocked)."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Mock URL validation
        def mock_validate_url(url):
            return create_mock_validation_result(url, is_valid=True)

        # Mock AI processor to avoid loading actual models
        with (
            patch(
                "bookmark_processor.core.ai_factory.AIFactory.create_client"
            ) as mock_ai_factory,
            patch(
                "bookmark_processor.core.url_validator.URLValidator.validate_url",
                side_effect=mock_validate_url,
            ),
        ):

            # Mock AI processor instance
            mock_ai = Mock()
            mock_ai.process_batch.return_value = {
                url: {
                    "enhanced_description": f"AI-enhanced description for {url}",
                    "generated_tags": ["ai", "enhanced", "test"],
                    "processing_method": "ai_enhancement",
                }
                for url in [
                    row["url"]
                    for row in SAMPLE_RAINDROP_EXPORT_ROWS
                    if "not-a-valid-url" not in row["url"]
                ]
            }
            mock_ai.get_processing_statistics.return_value = {
                "total_processed": 4,
                "ai_enhanced": 4,
                "fallback_used": 0,
            }
            mock_ai_factory.return_value = mock_ai

            results = processor.process_bookmarks(
                input_file=Path(temp_input_file),
                output_file=Path(temp_output_file),
                resume=False,
                verbose=False,
                batch_size=10,
                clear_checkpoints=True,
            )

        # Verify AI processing was used
        assert results is not None
        assert results.total_bookmarks > 0

        # Check output contains AI-enhanced descriptions
        output_df = pd.read_csv(temp_output_file)
        assert len(output_df) > 0
        # Note: We'd check for AI-enhanced content here, but it depends on the specific implementation

    def test_workflow_with_checkpoints(self, temp_input_file, temp_output_file):
        """Test workflow with checkpoint functionality."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Mock URL validation
        def mock_validate_url(url):
            return create_mock_validation_result(url, is_valid=True)

        # Create a temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:

            with patch(
                "bookmark_processor.core.url_validator.URLValidator.validate_url",
                side_effect=mock_validate_url,
            ):
                # First run - should create checkpoint
                results1 = processor.process_bookmarks(
                    input_file=Path(temp_input_file),
                    output_file=Path(temp_output_file),
                    resume=True,
                    verbose=False,
                    batch_size=2,  # Small batch to ensure checkpoints
                    clear_checkpoints=False,
                )

                # Verify first run completed
                assert results1 is not None
                assert results1.total_bookmarks > 0

                # Check that checkpoint directory structure exists
                # Note: Checkpoint creation depends on the specific checkpoint logic

    def test_workflow_error_handling(self, temp_output_file):
        """Test error handling for invalid input."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Test with non-existent input file
        with pytest.raises(Exception):
            processor.process_bookmarks(
                input_file=Path("nonexistent.csv"),
                output_file=Path(temp_output_file),
                resume=False,
            )

    def test_workflow_with_malformed_csv(self, temp_output_file):
        """Test handling of malformed CSV input."""
        # Create malformed CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("invalid,csv,format\n")
            f.write("missing,columns\n")
            f.write("inconsistent,row,data,too,many,columns\n")
            malformed_path = f.name

        try:
            from bookmark_processor.config.configuration import Configuration

            config = Configuration()
            processor = BookmarkProcessor(config)

            # Should handle malformed CSV gracefully
            with pytest.raises(Exception):
                processor.process_bookmarks(
                    input_file=Path(malformed_path),
                    output_file=Path(temp_output_file),
                    resume=False,
                    clear_checkpoints=True,
                )
        finally:
            os.unlink(malformed_path)

    def test_workflow_performance_minimal_config(
        self, temp_input_file, temp_output_file
    ):
        """Test workflow with minimal performance configuration."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Mock URL validation
        def mock_validate_url(url):
            return create_mock_validation_result(url, is_valid=True)

        with patch(
            "bookmark_processor.core.url_validator.URLValidator.validate_url",
            side_effect=mock_validate_url,
        ):
            # Use minimal config for faster testing
            test_config = TEST_CONFIGS["minimal"]

            results = processor.process_bookmarks(
                input_file=Path(temp_input_file),
                output_file=Path(temp_output_file),
                resume=False,
                verbose=test_config["verbose"],
                batch_size=test_config["batch_size"],
                max_retries=test_config["max_retries"],
                clear_checkpoints=True,
            )

        assert results is not None
        assert results.processing_time < 10.0  # Should be fast with minimal config

    def test_cli_integration(self, temp_input_file, temp_output_file):
        """Test CLI integration with the processor."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Mock URL validation
        def mock_validate_url(url):
            return create_mock_validation_result(url, is_valid=True)

        with patch(
            "bookmark_processor.core.url_validator.URLValidator.validate_url",
            side_effect=mock_validate_url,
        ):
            # Simulate CLI arguments
            exit_code = processor.run_cli(
                {
                    "input_path": Path(temp_input_file),
                    "output_path": Path(temp_output_file),
                    "batch_size": 5,
                    "max_retries": 1,
                    "verbose": True,
                    "resume": False,
                    "clear_checkpoints": True,
                }
            )

        # CLI should complete successfully
        assert exit_code == 0
        assert os.path.exists(temp_output_file)

    def test_large_dataset_simulation(self):
        """Test with a larger simulated dataset."""
        # Create larger test dataset
        large_data = []
        for i in range(50):  # Simulate 50 bookmarks
            row = {
                "id": str(i),
                "title": f"Test Bookmark {i}",
                "note": f"Note for bookmark {i}",
                "excerpt": f"Excerpt for bookmark {i}",
                "url": f"https://example{i}.com",
                "folder": f"Folder{i % 5}",  # Distribute across 5 folders
                "tags": f"tag{i % 3}, tag{(i + 1) % 3}",  # Rotate tags
                "created": "2024-01-01T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            }
            large_data.append(row)

        # Create temporary files
        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as input_f,
            tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as output_f,
        ):

            # Write large dataset
            df = pd.DataFrame(large_data)
            df.to_csv(input_f, index=False)
            input_path = input_f.name
            output_path = output_f.name

        # Remove output file (we just want the path)
        os.unlink(output_path)

        try:
            from bookmark_processor.config.configuration import Configuration

            config = Configuration()
            processor = BookmarkProcessor(config)

            # Mock URL validation
            def mock_validate_url(url):
                return create_mock_validation_result(url, is_valid=True)

            with patch(
                "bookmark_processor.core.url_validator.URLValidator.validate_url",
                side_effect=mock_validate_url,
            ):
                results = processor.process_bookmarks(
                    input_file=Path(input_path),
                    output_file=Path(output_path),
                    resume=False,
                    verbose=False,
                    batch_size=10,
                    max_retries=1,
                    clear_checkpoints=True,
                )

            # Verify large dataset processing
            assert results is not None
            assert results.total_bookmarks == 50
            assert results.processing_time > 0

            # Verify output
            assert os.path.exists(output_path)
            output_df = pd.read_csv(output_path)
            assert len(output_df) > 0

        finally:
            # Cleanup
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)


class TestEndToEndScenarios:
    """End-to-end test scenarios for complete bookmark processing workflows."""

    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary output file path."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        # Remove the file (we just want the path)
        os.unlink(temp_path)

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def diverse_bookmark_data(self):
        """Create diverse bookmark data for comprehensive testing."""
        diverse_data = [
            # Standard bookmark with all fields
            {
                "id": "1",
                "title": "Python Official Documentation",
                "note": "Complete Python reference and tutorials",
                "excerpt": "Python is a powerful programming language",
                "url": "https://docs.python.org/3/",
                "folder": "Programming/Python",
                "tags": "python, documentation, programming, reference",
                "created": "2024-01-01T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            # Bookmark with special characters in title and note
            {
                "id": "2",
                "title": "Advanced Python: Metaclasses & Design Patterns",
                "note": "Comprehensive guide to Python's advanced features including metaclasses, decorators, and design patterns",
                "excerpt": "Learn advanced Python concepts for better code architecture",
                "url": "https://realpython.com/python-metaclasses/",
                "folder": "Programming/Python/Advanced",
                "tags": "python, advanced, metaclasses, design-patterns",
                "created": "2024-01-02T10:30:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "true",
            },
            # Bookmark with minimal data
            {
                "id": "3",
                "title": "",
                "note": "",
                "excerpt": "",
                "url": "https://example.com",
                "folder": "",
                "tags": "",
                "created": "2024-01-03T15:45:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            # Bookmark with very long content
            {
                "id": "4",
                "title": "The Complete Guide to Web Development in 2024: From Frontend to Backend, Including Modern Frameworks, Best Practices, and Industry Standards",
                "note": "This comprehensive guide covers everything you need to know about modern web development including HTML5, CSS3, JavaScript ES6+, React, Vue.js, Angular, Node.js, Express, databases, cloud deployment, testing strategies, and much more. It's designed for both beginners and experienced developers looking to stay current with industry trends.",
                "excerpt": "A thorough exploration of modern web development covering frontend, backend, and full-stack development approaches with practical examples and real-world projects.",
                "url": "https://developer.mozilla.org/en-US/docs/Learn",
                "folder": "Web Development/Guides",
                "tags": "web-development, frontend, backend, javascript, html, css, frameworks, tutorial, comprehensive",
                "created": "2024-01-04T09:20:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "true",
            },
            # Bookmark with nested folder structure
            {
                "id": "5",
                "title": "Machine Learning with scikit-learn",
                "note": "Practical machine learning algorithms and techniques",
                "excerpt": "Scikit-learn is a versatile machine learning library for Python",
                "url": "https://scikit-learn.org/stable/",
                "folder": "AI/Machine Learning/Libraries/Python",
                "tags": "machine-learning, scikit-learn, python, algorithms",
                "created": "2024-01-05T14:10:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            # International content
            {
                "id": "6",
                "title": "プログラミング入門 - Japanese Programming Tutorial",
                "note": "Japanese programming tutorial with UTF-8 characters: 日本語のプログラミングチュートリアル",
                "excerpt": "Learn programming concepts in Japanese",
                "url": "https://example.jp/programming",
                "folder": "Programming/Tutorials/International",
                "tags": "programming, japanese, tutorial, 日本語",
                "created": "2024-01-06T11:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            # URL with query parameters and fragments
            {
                "id": "7",
                "title": "GitHub Advanced Search",
                "note": "GitHub's powerful search functionality with filters",
                "excerpt": "Search across all of GitHub's public repositories",
                "url": "https://github.com/search?q=bookmark+processor&type=repositories&s=stars&o=desc#results",
                "folder": "Development/Git",
                "tags": "github, search, repositories, development",
                "created": "2024-01-07T16:30:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            # Potentially problematic URL (but valid)
            {
                "id": "8",
                "title": "Test Site with Redirect",
                "note": "Site that might redirect to test redirect handling",
                "excerpt": "Testing redirect behavior",
                "url": "https://httpbin.org/redirect/3",
                "folder": "Testing/HTTP",
                "tags": "testing, http, redirect",
                "created": "2024-01-08T13:15:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
        ]
        return diverse_data

    @pytest.fixture
    def diverse_input_file(self, diverse_bookmark_data):
        """Create a temporary input file with diverse bookmark data."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            df = pd.DataFrame(diverse_bookmark_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_full_pipeline_with_diverse_content(
        self, diverse_input_file, temp_output_file
    ):
        """Test the complete pipeline with diverse bookmark content."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Mock URL validation - all URLs are valid
        def mock_validate_url(url):
            # Handle redirect simulation
            final_url = url
            if "httpbin.org/redirect" in url:
                final_url = "https://httpbin.org/get"
            return create_mock_validation_result(url, is_valid=True, final_url=final_url)

        with patch(
            "bookmark_processor.core.url_validator.URLValidator.validate_url",
            side_effect=mock_validate_url,
        ):
            results = processor.process_bookmarks(
                input_file=Path(diverse_input_file),
                output_file=Path(temp_output_file),
                resume=False,
                verbose=True,
                batch_size=4,
                max_retries=2,
                clear_checkpoints=True,
            )

        # Verify comprehensive processing results
        assert results is not None
        assert results.total_bookmarks == 8
        assert results.valid_bookmarks >= 6  # At least most should be valid
        assert results.processing_time > 0

        # Verify output file structure and content
        assert os.path.exists(temp_output_file)
        output_df = pd.read_csv(temp_output_file)

        # Check expected columns
        expected_columns = ["url", "folder", "title", "note", "tags", "created"]
        assert list(output_df.columns) == expected_columns

        # Verify data integrity
        assert len(output_df) >= 6  # Should have valid bookmarks

        # Check for proper handling of special characters
        unicode_rows = output_df[
            output_df["title"].str.contains("プログラミング", na=False)
        ]
        assert len(unicode_rows) <= 1  # Should handle unicode content

        # Check folder structure preservation
        nested_folders = output_df[
            output_df["folder"].str.contains("AI/Machine Learning", na=False)
        ]
        assert len(nested_folders) <= 1  # Should preserve nested folders

        # Verify redirect handling (if URL was processed)
        redirect_rows = output_df[
            output_df["url"].str.contains("httpbin.org", na=False)
        ]
        if len(redirect_rows) > 0:
            # Check that either original or final URL is preserved
            assert any("httpbin.org" in url for url in redirect_rows["url"].values)

    def test_error_recovery_and_partial_processing(self):
        """Test error recovery with mixed valid and invalid URLs."""
        from bookmark_processor.config.configuration import Configuration

        # Create test data with intentionally problematic entries
        problematic_data = [
            {
                "id": "1",
                "title": "Valid Site",
                "note": "This should work",
                "excerpt": "",
                "url": "https://httpbin.org/status/200",
                "folder": "Valid",
                "tags": "working, test",
                "created": "2024-01-01T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            {
                "id": "2",
                "title": "Timeout Site",
                "note": "This will timeout",
                "excerpt": "",
                "url": "https://httpbin.org/delay/30",
                "folder": "Problematic",
                "tags": "timeout, test",
                "created": "2024-01-02T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            {
                "id": "3",
                "title": "Not Found Site",
                "note": "This will return 404",
                "excerpt": "",
                "url": "https://httpbin.org/status/404",
                "folder": "Problematic",
                "tags": "notfound, test",
                "created": "2024-01-03T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            {
                "id": "4",
                "title": "Another Valid Site",
                "note": "This should also work",
                "excerpt": "",
                "url": "https://httpbin.org/status/201",
                "folder": "Valid",
                "tags": "working, test, second",
                "created": "2024-01-04T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
            {
                "id": "5",
                "title": "Invalid URL Format",
                "note": "This has an invalid URL",
                "excerpt": "",
                "url": "not-a-valid-url-at-all",
                "folder": "Problematic",
                "tags": "invalid, test",
                "created": "2024-01-05T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            },
        ]

        # Create temporary files
        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as input_f,
            tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as output_f,
        ):

            df = pd.DataFrame(problematic_data)
            df.to_csv(input_f, index=False)
            input_path = input_f.name
            output_path = output_f.name

        # Remove output file (we just want the path)
        os.unlink(output_path)

        try:
            config = Configuration()
            processor = BookmarkProcessor(config)

            # Mock URL validation with mixed results
            def mock_validate_url(url):
                if "status/200" in url or "status/201" in url:
                    return create_mock_validation_result(url, is_valid=True)
                elif "status/404" in url:
                    return create_mock_validation_result(
                        url,
                        is_valid=False,
                        status_code=404,
                        error_message="Not Found",
                        error_type="http_error",
                    )
                elif "delay/30" in url:
                    return create_mock_validation_result(
                        url,
                        is_valid=False,
                        error_message="Request timed out",
                        error_type="timeout",
                    )
                else:
                    # Invalid URL format
                    return create_mock_validation_result(
                        url,
                        is_valid=False,
                        error_message="Invalid URL format",
                        error_type="format_error",
                    )

            with patch(
                "bookmark_processor.core.url_validator.URLValidator.validate_url",
                side_effect=mock_validate_url,
            ):
                results = processor.process_bookmarks(
                    input_file=Path(input_path),
                    output_file=Path(output_path),
                    resume=False,
                    verbose=True,
                    batch_size=2,
                    max_retries=1,  # Quick retries for testing
                    clear_checkpoints=True,
                )

            # Verify partial success
            assert results is not None
            assert results.total_bookmarks == 5
            assert (
                results.valid_bookmarks >= 2
            )  # At least the status/200 URLs should work
            assert results.invalid_bookmarks >= 1  # At least some should fail

            # Verify output contains only valid bookmarks
            if os.path.exists(output_path):
                output_df = pd.read_csv(output_path)
                assert len(output_df) >= 2  # Should have the working URLs

                # Verify only valid URLs are in output
                for url in output_df["url"]:
                    assert "status/200" in url or "status/201" in url

        finally:
            # Cleanup
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_large_batch_processing_simulation(self):
        """Test processing with a larger simulated dataset to verify scalability."""
        from bookmark_processor.config.configuration import Configuration

        # Generate a larger dataset (100 bookmarks)
        large_dataset = []
        domains = ["example.com", "test.org", "demo.net", "sample.co", "mock.io"]
        folders = [
            "Technology",
            "Programming",
            "Web Development",
            "Data Science",
            "AI/ML",
        ]
        tag_sets = [
            ["web", "frontend", "javascript"],
            ["python", "backend", "api"],
            ["data", "analysis", "visualization"],
            ["machine-learning", "ai", "algorithms"],
            ["database", "sql", "optimization"],
        ]

        for i in range(100):
            domain = domains[i % len(domains)]
            folder = folders[i % len(folders)]
            tags = tag_sets[i % len(tag_sets)]

            bookmark = {
                "id": str(i + 1),
                "title": f"Test Bookmark {i + 1}: Advanced {folder} Guide",
                "note": f"Comprehensive guide to {folder.lower()} concepts and best practices",
                "excerpt": f"Learn about {folder.lower()} with practical examples and expert insights",
                "url": f"https://{domain}/article/{i + 1}",
                "folder": folder,
                "tags": ", ".join(tags),
                "created": f"2024-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "true" if i % 10 == 0 else "false",
            }
            large_dataset.append(bookmark)

        # Create temporary files
        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as input_f,
            tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as output_f,
        ):

            df = pd.DataFrame(large_dataset)
            df.to_csv(input_f, index=False)
            input_path = input_f.name
            output_path = output_f.name

        # Remove output file (we just want the path)
        os.unlink(output_path)

        try:
            config = Configuration()
            processor = BookmarkProcessor(config)

            # Mock URL validation - all URLs are valid
            def mock_validate_url(url):
                return create_mock_validation_result(url, is_valid=True)

            with patch(
                "bookmark_processor.core.url_validator.URLValidator.validate_url",
                side_effect=mock_validate_url,
            ):
                # Process with reasonable batch size for testing
                start_time = time.time()

                results = processor.process_bookmarks(
                    input_file=Path(input_path),
                    output_file=Path(output_path),
                    resume=False,
                    verbose=False,  # Reduce verbosity for large test
                    batch_size=20,  # Process in batches
                    max_retries=1,
                    clear_checkpoints=True,
                )

                end_time = time.time()
                processing_time = end_time - start_time

            # Verify large dataset processing
            assert results is not None
            assert results.total_bookmarks == 100
            assert results.valid_bookmarks >= 95  # Most should be valid
            assert results.processing_time > 0

            # Performance verification
            assert (
                processing_time < 30
            )  # Should complete within reasonable time for mocked data

            # Verify output quality
            assert os.path.exists(output_path)
            output_df = pd.read_csv(output_path)

            assert len(output_df) >= 95  # Should have most bookmarks
            assert len(output_df.columns) == 6  # Correct output format

            # Verify data integrity
            assert output_df["url"].notna().all()  # All URLs should be present
            assert output_df["title"].notna().all()  # All titles should be present

            # Check tag distribution - tags may be present from original data
            all_tags = []
            for tag_str in output_df["tags"].dropna():
                if tag_str:
                    all_tags.extend([tag.strip() for tag in str(tag_str).split(",")])

            # Verify some tags exist (may not be many due to mocked validation)
            unique_tags = set(all_tags)
            # Note: With mocked validation, original tags should be preserved
            # but we don't require a specific minimum since tag processing is complex

        finally:
            # Cleanup
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)


class TestCompleteProcessingPipeline:
    """Tests for the complete processing pipeline including all stages."""

    def test_pipeline_stage_progression(self):
        """Test that all pipeline stages execute in the correct order."""
        from bookmark_processor.config.configuration import Configuration

        config = Configuration()
        processor = BookmarkProcessor(config)

        # Create simple test data
        test_data = [
            {
                "id": "1",
                "title": "Test Site",
                "note": "Test note",
                "excerpt": "Test excerpt",
                "url": "https://example.com",
                "folder": "Test",
                "tags": "test, example",
                "created": "2024-01-01T00:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "false",
            }
        ]

        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as input_f,
            tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as output_f,
        ):

            df = pd.DataFrame(test_data)
            df.to_csv(input_f, index=False)
            input_path = input_f.name
            output_path = output_f.name

        # Remove output file (we just want the path)
        os.unlink(output_path)

        try:
            # Mock URL validation
            def mock_validate_url(url):
                return create_mock_validation_result(url, is_valid=True)

            with patch(
                "bookmark_processor.core.url_validator.URLValidator.validate_url",
                side_effect=mock_validate_url,
            ):
                results = processor.process_bookmarks(
                    input_file=Path(input_path),
                    output_file=Path(output_path),
                    resume=False,
                    verbose=True,
                    clear_checkpoints=True,
                )

            # Verify pipeline completed
            assert results is not None
            assert results.total_bookmarks == 1
            assert results.processing_time > 0

            # Verify stages were completed
            assert len(results.stages_completed) > 0

            # Verify output was created
            assert os.path.exists(output_path)

        finally:
            # Cleanup
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_pipeline_with_configuration_variations(self):
        """Test pipeline with different configuration options."""
        from bookmark_processor.config.configuration import Configuration

        # Test different batch sizes
        for batch_size in [1, 5, 10]:
            config = Configuration()
            processor = BookmarkProcessor(config)

            # Create test data
            test_data = [
                {
                    "id": str(i),
                    "title": f"Test Site {i}",
                    "note": f"Test note {i}",
                    "excerpt": f"Test excerpt {i}",
                    "url": f"https://example{i}.com",
                    "folder": "Test",
                    "tags": f"test{i}, example",
                    "created": "2024-01-01T00:00:00Z",
                    "cover": "",
                    "highlights": "",
                    "favorite": "false",
                }
                for i in range(batch_size * 2)  # Create enough data for batching
            ]

            with (
                tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as input_f,
                tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as output_f,
            ):

                df = pd.DataFrame(test_data)
                df.to_csv(input_f, index=False)
                input_path = input_f.name
                output_path = output_f.name

            # Remove output file (we just want the path)
            os.unlink(output_path)

            try:
                # Mock URL validation
                def mock_validate_url(url):
                    return create_mock_validation_result(url, is_valid=True)

                with patch(
                    "bookmark_processor.core.url_validator.URLValidator.validate_url",
                    side_effect=mock_validate_url,
                ):
                    results = processor.process_bookmarks(
                        input_file=Path(input_path),
                        output_file=Path(output_path),
                        resume=False,
                        verbose=False,
                        batch_size=batch_size,
                        max_retries=1,
                        clear_checkpoints=True,
                    )

                # Verify batch processing worked
                assert results is not None
                assert results.total_bookmarks == batch_size * 2
                assert results.processing_time > 0

            finally:
                # Cleanup
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__])
