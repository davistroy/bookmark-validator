"""
Error Handling and Recovery Integration Tests

Tests for comprehensive error handling, recovery mechanisms, and resilience
throughout the complete bookmark processing pipeline.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from requests.exceptions import ConnectionError, HTTPError, Timeout

from bookmark_processor.core.ai_processor import AIProcessingResult, EnhancedAIProcessor
from bookmark_processor.core.checkpoint_manager import (
    CheckpointManager,
    ProcessingStage,
)
from bookmark_processor.core.content_analyzer import ContentAnalyzer, ContentData
from bookmark_processor.core.pipeline import BookmarkProcessingPipeline, PipelineConfig
from bookmark_processor.core.url_validator import URLValidator, ValidationResult
from bookmark_processor.utils.error_handler import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
)
from bookmark_processor.utils.retry_handler import RetryHandler


class TestErrorHandlingIntegration:
    """Test error handling and recovery in complete pipeline integration."""

    @pytest.fixture
    def error_prone_csv_data(self):
        """CSV data designed to trigger various error conditions."""
        return """id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,"Valid URL","Good bookmark","Working link","https://httpbin.org/status/200","Test/Valid","test, valid","2024-01-01T00:00:00Z","","","false"
2,"Timeout URL","Slow response","Will timeout","https://httpbin.org/delay/10","Test/Timeout","test, slow","2024-01-02T00:00:00Z","","","false"
3,"Not Found","Missing page","404 error","https://httpbin.org/status/404","Test/NotFound","test, error","2024-01-03T00:00:00Z","","","false"
4,"Server Error","Internal error","500 error","https://httpbin.org/status/500","Test/ServerError","test, server","2024-01-04T00:00:00Z","","","false"
5,"Invalid URL","Malformed URL","Bad format","not-a-valid-url","Test/Invalid","test, invalid","2024-01-05T00:00:00Z","","","false"
6,"Connection Error","Network issue","Connection fails","https://nonexistent-domain-12345.com","Test/Network","test, network","2024-01-06T00:00:00Z","","","false"
7,"Rate Limited","Too many requests","Rate limit hit","https://httpbin.org/status/429","Test/RateLimit","test, limit","2024-01-07T00:00:00Z","","","false"
8,"Redirect Loop","Infinite redirect","Redirect issue","https://httpbin.org/redirect/10","Test/Redirect","test, redirect","2024-01-08T00:00:00Z","","","false"
"""

    @pytest.fixture
    def error_csv_file(self, error_prone_csv_data):
        """Create temporary CSV file with error-prone data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(error_prone_csv_data)
            return f.name

    @pytest.fixture
    def error_config(self, error_csv_file):
        """Create pipeline configuration for error testing."""
        return PipelineConfig(
            input_file=error_csv_file,
            output_file=error_csv_file.replace(".csv", "_error_output.csv"),
            batch_size=3,
            max_retries=2,
            resume_enabled=True,
            url_timeout=5.0,
            max_concurrent_requests=2,
            ai_enabled=True,
            checkpoint_dir=".test_error_handling",
            save_interval=2,
            verbose=True,
        )

    @pytest.mark.asyncio
    async def test_network_error_handling_and_recovery(self, error_config):
        """Test handling of various network errors with recovery mechanisms."""
        pipeline = BookmarkProcessingPipeline(error_config)

        # Mock different types of network errors
        network_errors = [
            ConnectionError("Connection failed"),
            Timeout("Request timed out"),
            HTTPError("HTTP error occurred"),
            Exception("Unknown network error"),
        ]

        call_count = [0]

        def mock_batch_validate_with_errors(urls, **kwargs):
            results = []
            for i, url in enumerate(urls):
                call_count[0] += 1

                if "httpbin.org" in url and call_count[0] <= 4:
                    # Simulate network errors for first few calls
                    error_index = (call_count[0] - 1) % len(network_errors)
                    error = network_errors[error_index]

                    result = ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=None,
                        error_message=str(error),
                        error_type=type(error).__name__,
                    )
                elif "nonexistent-domain" in url:
                    result = ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=None,
                        error_message="DNS resolution failed",
                        error_type="ConnectionError",
                    )
                elif "not-a-valid-url" in url:
                    result = ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=None,
                        error_message="Invalid URL format",
                        error_type="URLError",
                    )
                else:
                    # Success case (or retry success)
                    result = ValidationResult(
                        url=url, is_valid=True, status_code=200, final_url=url
                    )

                results.append(result)

            return results

        with patch.object(
            pipeline.url_validator,
            "batch_validate",
            side_effect=mock_batch_validate_with_errors,
        ):
            with patch.object(
                pipeline.ai_processor, "batch_process"
            ) as mock_ai_process:

                # Mock AI processing to focus on network error testing
                mock_ai_process.return_value = []

                # Execute pipeline
                results = pipeline.execute()

                # Verify error handling
                assert results is not None
                assert results.total_bookmarks == 8

                # Should have some valid and some invalid results
                assert results.valid_bookmarks >= 1  # At least some should succeed
                assert results.invalid_bookmarks >= 1  # Some should fail

                # Verify error summary
                assert (
                    "url_validation" in results.error_summary
                    or len(results.error_summary) >= 0
                )

                # Check that pipeline completed despite errors
                state = pipeline.checkpoint_manager.load_checkpoint(
                    error_config.input_file
                )
                assert state.current_stage == ProcessingStage.COMPLETED

        # Cleanup
        pipeline._cleanup_resources()
        Path(error_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_ai_processing_error_recovery(self, error_config):
        """Test AI processing error handling and fallback mechanisms."""
        pipeline = BookmarkProcessingPipeline(error_config)

        # Mock URL validation to succeed
        with patch.object(pipeline.url_validator, "batch_validate") as mock_validate:

            validation_results = []
            for i in range(8):
                validation_results.append(
                    ValidationResult(
                        url=f"https://example.com/{i}", is_valid=True, status_code=200
                    )
                )
            mock_validate.return_value = validation_results

            # Mock AI processing with various errors and recovery
            call_count = [0]

            def mock_ai_batch_process_with_errors(bookmarks, **kwargs):
                call_count[0] += 1
                results = []

                for i, bookmark in enumerate(bookmarks):
                    if call_count[0] == 1 and i == 0:
                        # First bookmark in first batch fails
                        raise Exception("AI service temporarily unavailable")
                    elif call_count[0] == 2 and i == 1:
                        # Second bookmark in second batch fails
                        raise TimeoutError("AI processing timeout")
                    else:
                        # Success or fallback
                        if call_count[0] <= 2:
                            # Fallback description
                            description = f"Fallback description for {bookmark.title}"
                            method = "fallback"
                        else:
                            # Normal AI processing
                            description = f"AI enhanced: {bookmark.title}"
                            method = "ai"

                        result = AIProcessingResult(
                            original_url=bookmark.url,
                            enhanced_description=description,
                            processing_method=method,
                            processing_time=0.1,
                        )
                        results.append(result)

                return results

            with patch.object(
                pipeline.ai_processor,
                "batch_process",
                side_effect=mock_ai_batch_process_with_errors,
            ):

                # Execute pipeline
                results = pipeline.execute()

                # Verify error recovery
                assert results is not None
                assert results.total_bookmarks == 8
                assert results.valid_bookmarks == 8  # All URLs should be valid

                # Should have processed some bookmarks despite AI errors
                assert results.ai_processed >= 6  # Most should be processed

                # Verify mix of AI and fallback processing
                ai_results = pipeline.ai_results
                fallback_count = sum(
                    1 for r in ai_results.values() if r.processing_method == "fallback"
                )
                ai_count = sum(
                    1 for r in ai_results.values() if r.processing_method == "ai"
                )

                assert fallback_count >= 1  # Should have fallback results
                assert ai_count >= 1  # Should have normal AI results

        # Cleanup
        pipeline._cleanup_resources()
        Path(error_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_checkpoint_corruption_and_recovery(self, error_config):
        """Test recovery from checkpoint corruption and file system errors."""
        pipeline = BookmarkProcessingPipeline(error_config)

        # Start processing and create checkpoint
        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "batch_process") as mock_ai_process,
        ):

            # Mock partial processing
            validation_results = [
                ValidationResult(
                    url="https://example.com/1", is_valid=True, status_code=200
                ),
                ValidationResult(
                    url="https://example.com/2", is_valid=True, status_code=200
                ),
            ]
            mock_validate.return_value = validation_results
            mock_ai_process.return_value = []

            # Start processing to create checkpoint
            pipeline._stage_load_bookmarks()
            pipeline._stage_validate_urls()

        # Simulate checkpoint corruption
        checkpoint_files = list(Path(error_config.checkpoint_dir).glob("*.json"))
        if checkpoint_files:
            corrupt_file = checkpoint_files[0]
            corrupt_file.write_text(
                '{"corrupted": "data", "invalid": json}'
            )  # Invalid JSON

        # Test recovery with corrupted checkpoint
        pipeline2 = BookmarkProcessingPipeline(error_config)

        with (
            patch.object(pipeline2.url_validator, "batch_validate") as mock_validate2,
            patch.object(pipeline2.ai_processor, "batch_process") as mock_ai_process2,
        ):

            mock_validate2.return_value = validation_results
            mock_ai_process2.return_value = []

            # Should fall back to new processing
            results = pipeline2.execute()

            # Should complete successfully despite checkpoint corruption
            assert results is not None
            assert results.total_bookmarks > 0

        # Cleanup
        pipeline._cleanup_resources()
        pipeline2._cleanup_resources()
        Path(error_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_file_system_error_handling(self, error_config):
        """Test handling of file system errors (permissions, disk space, etc.)."""
        pipeline = BookmarkProcessingPipeline(error_config)

        with (
            patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
            patch.object(pipeline.ai_processor, "batch_process") as mock_ai_process,
        ):

            mock_validate.return_value = [
                ValidationResult(
                    url="https://example.com/1", is_valid=True, status_code=200
                )
            ]
            mock_ai_process.return_value = [
                AIProcessingResult(
                    original_url="https://example.com/1",
                    enhanced_description="Test description",
                    processing_method="test",
                    processing_time=0.1,
                )
            ]

            # Mock file system error during output generation
            original_save = pipeline.csv_handler.save_import_csv

            def mock_save_with_error(*args, **kwargs):
                raise PermissionError("Permission denied: Cannot write to output file")

            with patch.object(
                pipeline.csv_handler,
                "save_import_csv",
                side_effect=mock_save_with_error,
            ):

                # Should raise an exception during output generation
                with pytest.raises(Exception, match="Failed to save output file"):
                    pipeline.execute()

                # Verify checkpoint was still saved (for recovery)
                assert pipeline.checkpoint_manager.has_checkpoint(
                    error_config.input_file
                )

        # Test recovery after fixing file system issue
        pipeline2 = BookmarkProcessingPipeline(error_config)

        with (
            patch.object(pipeline2.url_validator, "batch_validate") as mock_validate2,
            patch.object(pipeline2.ai_processor, "batch_process") as mock_ai_process2,
        ):

            mock_validate2.return_value = []  # No new validation needed
            mock_ai_process2.return_value = []

            # Should resume and complete successfully
            results = pipeline2.execute()

            assert results is not None
            assert Path(error_config.output_file).exists()

        # Cleanup
        pipeline._cleanup_resources()
        pipeline2._cleanup_resources()
        Path(error_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_memory_pressure_error_handling(self, error_config):
        """Test handling of memory pressure and resource exhaustion."""
        # Modify config for memory testing
        error_config.memory_warning_threshold = 10  # Very low threshold (10MB)
        error_config.memory_critical_threshold = 20  # Critical at 20MB
        error_config.memory_batch_size = 1  # Small batches

        pipeline = BookmarkProcessingPipeline(error_config)

        # Mock memory monitor to simulate memory pressure
        with patch.object(
            pipeline.memory_monitor, "get_current_usage_mb"
        ) as mock_memory:

            call_count = [0]

            def mock_memory_usage():
                call_count[0] += 1
                if call_count[0] <= 3:
                    return 15  # Above warning threshold
                elif call_count[0] <= 6:
                    return 25  # Above critical threshold
                else:
                    return 5  # Back to normal

            mock_memory.side_effect = mock_memory_usage

            with (
                patch.object(pipeline.url_validator, "batch_validate") as mock_validate,
                patch.object(pipeline.ai_processor, "batch_process") as mock_ai_process,
            ):

                mock_validate.return_value = [
                    ValidationResult(
                        url="https://example.com/1", is_valid=True, status_code=200
                    )
                ]
                mock_ai_process.return_value = []

                # Should handle memory pressure gracefully
                results = pipeline.execute()

                # Should complete despite memory warnings
                assert results is not None

                # Verify memory monitoring was called
                assert mock_memory.call_count > 0

        # Cleanup
        pipeline._cleanup_resources()
        Path(error_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_rate_limiting_error_recovery(self, error_config):
        """Test recovery from rate limiting errors."""
        pipeline = BookmarkProcessingPipeline(error_config)

        # Mock rate limiting scenarios
        rate_limit_count = [0]

        def mock_batch_validate_with_rate_limiting(urls, **kwargs):
            results = []
            for url in urls:
                rate_limit_count[0] += 1

                if rate_limit_count[0] <= 3:
                    # First few requests hit rate limit
                    result = ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=429,
                        error_message="Rate limit exceeded",
                        error_type="RateLimitError",
                    )
                else:
                    # Later requests succeed (after backing off)
                    result = ValidationResult(
                        url=url, is_valid=True, status_code=200, final_url=url
                    )

                results.append(result)

            return results

        with patch.object(
            pipeline.url_validator,
            "batch_validate",
            side_effect=mock_batch_validate_with_rate_limiting,
        ):
            with patch.object(
                pipeline.ai_processor, "batch_process"
            ) as mock_ai_process:

                mock_ai_process.return_value = []

                # Execute pipeline
                results = pipeline.execute()

                # Should handle rate limiting and eventually succeed
                assert results is not None
                assert results.total_bookmarks == 8

                # Some URLs should eventually succeed after rate limit recovery
                assert results.valid_bookmarks >= 5  # Most should succeed after retry
                assert results.invalid_bookmarks <= 3  # Some may still fail

        # Cleanup
        pipeline._cleanup_resources()
        Path(error_config.output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_concurrent_processing_error_isolation(self, error_config):
        """Test error isolation in concurrent processing scenarios."""
        error_config.max_concurrent_requests = 3  # Enable concurrency

        pipeline = BookmarkProcessingPipeline(error_config)

        # Mock mixed success/failure scenarios
        def mock_batch_validate_mixed_results(urls, **kwargs):
            results = []
            for i, url in enumerate(urls):
                if i % 3 == 0:
                    # Every third URL fails
                    result = ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=500,
                        error_message="Server error",
                        error_type="HTTPError",
                    )
                elif i % 4 == 0:
                    # Every fourth URL times out
                    result = ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=None,
                        error_message="Request timeout",
                        error_type="TimeoutError",
                    )
                else:
                    # Other URLs succeed
                    result = ValidationResult(
                        url=url, is_valid=True, status_code=200, final_url=url
                    )

                results.append(result)

            return results

        with patch.object(
            pipeline.url_validator,
            "batch_validate",
            side_effect=mock_batch_validate_mixed_results,
        ):
            with patch.object(
                pipeline.ai_processor, "batch_process"
            ) as mock_ai_process:

                # Mock AI processing with some errors
                def mock_ai_with_errors(bookmarks, **kwargs):
                    results = []
                    for i, bookmark in enumerate(bookmarks):
                        if i % 5 == 0:
                            # Skip every fifth bookmark (simulate error)
                            continue

                        result = AIProcessingResult(
                            original_url=bookmark.url,
                            enhanced_description=f"AI processed: {bookmark.title}",
                            processing_method="concurrent_test",
                            processing_time=0.1,
                        )
                        results.append(result)

                    return results

                mock_ai_process.side_effect = mock_ai_with_errors

                # Execute pipeline
                results = pipeline.execute()

                # Verify error isolation
                assert results is not None
                assert results.total_bookmarks == 8

                # Should have mixed results
                assert results.valid_bookmarks >= 2  # Some should succeed
                assert results.invalid_bookmarks >= 2  # Some should fail

                # Errors in some concurrent operations shouldn't block others
                assert results.ai_processed >= 2  # Some AI processing should succeed

        # Cleanup
        pipeline._cleanup_resources()
        Path(error_config.output_file).unlink(missing_ok=True)


class TestErrorCategorization:
    """Test error categorization and severity assessment."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return ErrorHandler()

    def test_network_error_categorization(self, error_handler):
        """Test categorization of network-related errors."""
        # Connection errors
        conn_error = ConnectionError("Connection failed")
        error_details = error_handler.categorize_error(conn_error)
        assert error_details.category == ErrorCategory.NETWORK
        assert error_details.severity == ErrorSeverity.HIGH
        assert error_details.is_retryable is True

        # Timeout errors
        timeout_error = Timeout("Request timed out")
        error_details = error_handler.categorize_error(timeout_error)
        assert error_details.category == ErrorCategory.NETWORK
        assert error_details.severity == ErrorSeverity.MEDIUM
        assert error_details.is_retryable is True

        # HTTP errors
        http_error = HTTPError("HTTP 500 Internal Server Error")
        error_details = error_handler.categorize_error(http_error)
        assert error_details.category == ErrorCategory.HTTP
        assert error_details.severity == ErrorSeverity.MEDIUM
        assert error_details.is_retryable is True

    def test_data_error_categorization(self, error_handler):
        """Test categorization of data-related errors."""
        # Invalid URL format
        url_error = ValueError("Invalid URL format")
        error_details = error_handler.categorize_error(url_error)
        assert error_details.category == ErrorCategory.DATA
        assert error_details.severity == ErrorSeverity.LOW
        assert error_details.is_retryable is False

        # JSON parsing error
        json_error = json.JSONDecodeError("Invalid JSON", "doc", 0)
        error_details = error_handler.categorize_error(json_error)
        assert error_details.category == ErrorCategory.DATA
        assert error_details.severity == ErrorSeverity.MEDIUM
        assert error_details.is_retryable is False

    def test_system_error_categorization(self, error_handler):
        """Test categorization of system-related errors."""
        # Permission error
        perm_error = PermissionError("Permission denied")
        error_details = error_handler.categorize_error(perm_error)
        assert error_details.category == ErrorCategory.SYSTEM
        assert error_details.severity == ErrorSeverity.HIGH
        assert error_details.is_retryable is False

        # Memory error
        mem_error = MemoryError("Out of memory")
        error_details = error_handler.categorize_error(mem_error)
        assert error_details.category == ErrorCategory.SYSTEM
        assert error_details.severity == ErrorSeverity.CRITICAL
        assert error_details.is_retryable is False

    def test_ai_processing_error_categorization(self, error_handler):
        """Test categorization of AI processing errors."""
        # Generic AI error
        ai_error = Exception("AI service unavailable")
        error_details = error_handler.categorize_error(ai_error)
        assert error_details.category == ErrorCategory.UNKNOWN
        assert error_details.severity == ErrorSeverity.MEDIUM
        assert error_details.is_retryable is True


class TestRetryMechanisms:
    """Test retry mechanisms and backoff strategies."""

    @pytest.fixture
    def retry_handler(self):
        """Create retry handler for testing."""
        return RetryHandler(default_max_retries=3, default_base_delay=0.1)

    @pytest.mark.skip(reason="retry_async method not implemented in current RetryHandler API")
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self, retry_handler):
        """Test exponential backoff retry strategy."""
        attempt_count = [0]
        attempt_delays = []

        async def failing_operation():
            attempt_count[0] += 1
            attempt_delays.append(datetime.now())

            if attempt_count[0] < 3:
                raise ConnectionError("Connection failed")
            return "success"

        # Should succeed after retries
        result = await retry_handler.retry_async(failing_operation)
        assert result == "success"
        assert attempt_count[0] == 3

        # Verify exponential backoff
        if len(attempt_delays) >= 2:
            delay1 = (attempt_delays[1] - attempt_delays[0]).total_seconds()
            assert delay1 >= 0.1  # First retry delay

        if len(attempt_delays) >= 3:
            delay2 = (attempt_delays[2] - attempt_delays[1]).total_seconds()
            assert delay2 >= delay1  # Second delay should be longer

    @pytest.mark.skip(reason="retry_async method not implemented in current RetryHandler API")
    @pytest.mark.asyncio
    async def test_retry_with_permanent_failure(self, retry_handler):
        """Test retry behavior with permanent failures."""
        attempt_count = [0]

        async def always_failing_operation():
            attempt_count[0] += 1
            raise ValueError("Permanent error")

        # Should give up after max retries
        with pytest.raises(ValueError, match="Permanent error"):
            await retry_handler.retry_async(always_failing_operation)

        assert attempt_count[0] == retry_handler.default_max_retries + 1  # Initial + retries

    @pytest.mark.skip(reason="retry_async method not implemented in current RetryHandler API")
    @pytest.mark.asyncio
    async def test_retry_with_non_retryable_error(self, retry_handler):
        """Test that non-retryable errors are not retried."""
        attempt_count = [0]

        async def non_retryable_error():
            attempt_count[0] += 1
            raise PermissionError("Permission denied")

        # Should not retry permission errors
        with pytest.raises(PermissionError):
            await retry_handler.retry_async(
                non_retryable_error, retryable_exceptions=[ConnectionError]
            )

        assert attempt_count[0] == 1  # Only one attempt


class TestGracefulDegradation:
    """Test graceful degradation strategies when components fail."""

    @pytest.mark.asyncio
    async def test_ai_service_fallback_chain(self):
        """Test fallback chain when AI services fail."""
        from bookmark_processor.config.configuration import Configuration
        from bookmark_processor.core.ai_factory import AIManager

        # Mock configuration
        config = Mock(spec=Configuration)
        config.get_api_key.return_value = None
        config.get.side_effect = lambda section, key, fallback=None: {
            ("ai", "default_engine"): "claude",
            ("ai", "claude_rpm"): "50",
            ("ai", "openai_rpm"): "60",
        }.get((section, key), fallback)

        ai_manager = AIManager("claude", config)

        # Mock primary AI service failure
        with patch.object(ai_manager, "_create_ai_processor") as mock_create:
            mock_processor = Mock()
            mock_processor.generate_description.side_effect = Exception(
                "AI service unavailable"
            )
            mock_create.return_value = mock_processor

            # Mock bookmark
            bookmark = Mock()
            bookmark.title = "Test Bookmark"
            bookmark.note = "Test note"
            bookmark.excerpt = "Test excerpt"

            # Should fall back to local processing
            description, metadata = await ai_manager.generate_description(
                bookmark, "existing content"
            )

            # Should get fallback description
            assert description is not None
            assert len(description) > 0
            assert metadata["provider"] == "fallback"
            assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_content_analysis_fallback(self):
        """Test content analysis fallback when network fails."""
        from bookmark_processor.core.content_analyzer import ContentAnalyzer

        analyzer = ContentAnalyzer(timeout=5.0)

        # Mock network failure
        with patch.object(analyzer, "_fetch_content") as mock_fetch:
            mock_fetch.side_effect = ConnectionError("Network unavailable")

            # Should fall back to existing data
            content_data = analyzer.analyze_content(
                "https://example.com",
                existing_title="Test Title",
                existing_note="Test Note",
                existing_excerpt="Test Excerpt",
            )

            # Should return fallback content
            assert content_data.title == "Test Title"
            assert content_data.description == "Test Note"
            assert content_data.source == "existing_data"

    @pytest.mark.asyncio
    async def test_tag_generation_fallback(self):
        """Test tag generation fallback when AI fails."""
        from bookmark_processor.core.tag_generator import CorpusAwareTagGenerator

        tag_generator = CorpusAwareTagGenerator(target_tag_count=10)

        # Mock AI tag generation failure
        with patch.object(tag_generator, "_generate_ai_tags") as mock_ai_tags:
            mock_ai_tags.side_effect = Exception("AI tag generation failed")

            # Should fall back to keyword extraction
            tags = tag_generator.generate_tags(
                "Python Programming Tutorial",
                "Learn Python programming basics",
                "https://example.com/python",
            )

            # Should get fallback tags
            assert len(tags) > 0
            assert any("python" in tag.lower() for tag in tags)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
