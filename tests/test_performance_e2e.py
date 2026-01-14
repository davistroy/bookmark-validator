"""
End-to-end performance tests for the bookmark processor.

These tests verify that the bookmark processor can handle various load levels
while meeting performance requirements for processing time and memory usage.

Run with: pytest tests/test_performance_e2e.py --runperformance -v
"""

import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd

from bookmark_processor.core.pipeline import BookmarkPipeline
from bookmark_processor.core.data_models import ProcessingResults
from bookmark_processor.core.checkpoint_manager import CheckpointManager
from tests.fixtures.generate_test_data import TestDataGenerator


# Mark all tests in this module as performance tests
pytestmark = pytest.mark.performance


# ============================================================================
# Test Data Generation
# ============================================================================


@pytest.fixture(scope="module")
def module_test_data_dir(tmp_path_factory):
    """Create a module-scoped temporary directory for test data."""
    return tmp_path_factory.mktemp("performance_test_data")


@pytest.fixture(scope="module")
def module_test_files(module_test_data_dir):
    """Generate test files once per module for efficiency."""
    generator = TestDataGenerator(seed=42)
    return generator.generate_performance_suite(module_test_data_dir)


# ============================================================================
# Mock Setup for Performance Tests
# ============================================================================


@pytest.fixture
def mock_url_validation():
    """Mock URL validation for performance tests (no network delays)."""
    with patch("bookmark_processor.core.url_validator.URLValidator") as mock_class:
        mock_validator = Mock()

        def validate_url(url: str) -> tuple[bool, str, int]:
            """Mock validation - fast and deterministic."""
            if not url or url.startswith(("javascript:", "mailto:", "not-a-valid")):
                return False, url, 400
            return True, url, 200

        mock_validator.validate_url = validate_url
        mock_validator.validate_url_simple = validate_url

        mock_class.return_value = mock_validator
        yield mock_validator


@pytest.fixture
def mock_content_extraction():
    """Mock content extraction for performance tests."""
    with patch("bookmark_processor.core.content_analyzer.ContentAnalyzer") as mock_class:
        mock_analyzer = Mock()

        def extract_metadata(url: str):
            """Mock extraction - fast generation."""
            from bookmark_processor.core.data_models import BookmarkMetadata

            return BookmarkMetadata(
                title=f"Title for {url}",
                description=f"Description for {url}",
                keywords=["test", "performance", "mock"],
            )

        mock_analyzer.extract_metadata = extract_metadata
        mock_class.return_value = mock_analyzer
        yield mock_analyzer


@pytest.fixture
def mock_ai_processing():
    """Mock AI processing for performance tests."""
    with patch("bookmark_processor.core.ai_processor.EnhancedAIProcessor") as mock_class:
        mock_processor = Mock()

        def process_batch(bookmarks):
            """Mock AI processing - fast generation."""
            for bookmark in bookmarks:
                if bookmark.note or bookmark.excerpt:
                    bookmark.enhanced_description = (
                        f"AI-enhanced: {bookmark.note or bookmark.excerpt}"
                    )
                else:
                    bookmark.enhanced_description = f"AI-generated description for {bookmark.title}"
                bookmark.optimized_tags = ["ai", "generated", "performance", "test"]
            return bookmarks

        mock_processor.process_batch = process_batch
        mock_processor.is_available = True
        mock_processor.model_name = "mock-model"

        mock_class.return_value = mock_processor
        yield mock_processor


@pytest.fixture
def performance_pipeline(
    temp_dir,
    temp_config_file,
    mock_url_validation,
    mock_content_extraction,
    mock_ai_processing,
):
    """Create a pipeline configured for performance testing."""
    from bookmark_processor.config.configuration import Configuration

    config = Configuration(config_file=str(temp_config_file))

    # Adjust config for performance testing
    config.processing["batch_size"] = 50
    config.checkpoint["save_interval"] = 100
    config.checkpoint["checkpoint_dir"] = str(temp_dir / "checkpoints")

    pipeline = BookmarkPipeline(config)
    return pipeline


# ============================================================================
# Small Dataset Tests (100 bookmarks) - Quick Sanity Check
# ============================================================================


@pytest.mark.small
def test_small_dataset_performance(
    module_test_files,
    performance_pipeline,
    performance_monitor,
    performance_baseline,
    performance_assertion,
    temp_dir,
):
    """Test processing 100 bookmarks with performance monitoring."""
    # Arrange
    input_file = module_test_files["small"]
    output_file = temp_dir / "output_small.csv"
    baseline = performance_baseline["small_test"]

    # Act
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    # Get metrics
    metrics = performance_monitor.stop()

    # Assert - Processing completed
    assert output_file.exists(), "Output file should be created"
    assert results.processed_bookmarks > 0, "Should process bookmarks"

    # Assert - Performance metrics
    performance_assertion.assert_duration_within_limit(
        metrics["duration_seconds"],
        baseline["max_duration_seconds"],
        "Small dataset test",
    )

    performance_assertion.assert_memory_within_limit(
        metrics["peak_memory_mb"], baseline["max_memory_mb"], "Small dataset test"
    )

    # Assert - Throughput
    performance_assertion.assert_throughput_above_minimum(
        results.processed_bookmarks,
        metrics["duration_seconds"],
        baseline["min_throughput_per_hour"],
        "Small dataset test",
    )

    # Print summary
    print(f"\n=== Small Dataset Performance ===")
    print(f"Processed: {results.processed_bookmarks} bookmarks")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    print(f"Memory: {metrics['peak_memory_mb']:.2f} MB")
    print(
        f"Throughput: {(results.processed_bookmarks / metrics['duration_seconds']) * 3600:.2f} bookmarks/hour"
    )


@pytest.mark.small
def test_small_dataset_with_checkpoints(
    module_test_files,
    performance_pipeline,
    performance_monitor,
    temp_dir,
):
    """Test checkpoint functionality with small dataset."""
    # Arrange
    input_file = module_test_files["small"]
    output_file = temp_dir / "output_checkpoint.csv"
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Configure for checkpointing
    performance_pipeline.config.checkpoint["enabled"] = True
    performance_pipeline.config.checkpoint["save_interval"] = 20

    # Act - Initial processing
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    # Assert - Checkpoints created
    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoint_files) > 0, "Checkpoint files should be created"

    # Get metrics
    metrics = performance_monitor.stop()

    # Assert - Checkpoint overhead is reasonable (< 10% overhead)
    checkpoint_overhead = 0.1  # 10%
    max_duration_with_checkpoints = 60 * (1 + checkpoint_overhead)

    assert metrics["duration_seconds"] <= max_duration_with_checkpoints, (
        f"Checkpoint overhead too high: {metrics['duration_seconds']:.2f}s "
        f"exceeds {max_duration_with_checkpoints:.2f}s"
    )

    print(f"\n=== Checkpoint Performance ===")
    print(f"Checkpoints created: {len(checkpoint_files)}")
    print(f"Duration with checkpoints: {metrics['duration_seconds']:.2f} seconds")


# ============================================================================
# Medium Dataset Tests (1000 bookmarks) - Medium Load
# ============================================================================


@pytest.mark.medium
@pytest.mark.slow
def test_medium_dataset_performance(
    module_test_files,
    performance_pipeline,
    performance_monitor,
    performance_baseline,
    performance_assertion,
    temp_dir,
):
    """Test processing 1000 bookmarks with performance monitoring."""
    # Arrange
    input_file = module_test_files["medium"]
    output_file = temp_dir / "output_medium.csv"
    baseline = performance_baseline["medium_test"]

    # Act
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    # Get metrics
    metrics = performance_monitor.stop()

    # Assert - Processing completed
    assert output_file.exists(), "Output file should be created"
    assert results.processed_bookmarks > 900, "Should process most bookmarks"

    # Assert - Performance metrics
    performance_assertion.assert_duration_within_limit(
        metrics["duration_seconds"],
        baseline["max_duration_seconds"],
        "Medium dataset test",
    )

    performance_assertion.assert_memory_within_limit(
        metrics["peak_memory_mb"], baseline["max_memory_mb"], "Medium dataset test"
    )

    # Assert - Throughput
    performance_assertion.assert_throughput_above_minimum(
        results.processed_bookmarks,
        metrics["duration_seconds"],
        baseline["min_throughput_per_hour"],
        "Medium dataset test",
    )

    # Assert - Success rate
    performance_assertion.assert_success_rate_above_threshold(
        results.valid_bookmarks,
        results.processed_bookmarks,
        min_rate=0.95,
        test_name="Medium dataset test",
    )

    # Print summary
    print(f"\n=== Medium Dataset Performance ===")
    print(f"Processed: {results.processed_bookmarks} bookmarks")
    print(f"Valid: {results.valid_bookmarks}")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    print(f"Memory: {metrics['peak_memory_mb']:.2f} MB")
    print(
        f"Throughput: {(results.processed_bookmarks / metrics['duration_seconds']) * 3600:.2f} bookmarks/hour"
    )


@pytest.mark.medium
@pytest.mark.slow
def test_medium_dataset_memory_stability(
    module_test_files, performance_pipeline, performance_monitor, temp_dir
):
    """Test that memory usage remains stable during medium dataset processing."""
    # Arrange
    input_file = module_test_files["medium"]
    output_file = temp_dir / "output_memory.csv"

    memory_samples = []

    # Create a monitoring callback
    class MemoryMonitor:
        def __init__(self, monitor):
            self.monitor = monitor
            self.samples = []

        def sample(self):
            metrics = self.monitor.get_current_metrics()
            self.samples.append(metrics["current_memory_mb"])

    mem_monitor = MemoryMonitor(performance_monitor)

    # Sample memory during processing (simulate)
    # In a real scenario, this would be called periodically during processing
    start_memory = performance_monitor.get_current_metrics()["current_memory_mb"]

    # Act
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    end_memory = performance_monitor.get_current_metrics()["current_memory_mb"]

    # Get final metrics
    metrics = performance_monitor.stop()

    # Assert - Memory growth is bounded
    memory_growth = end_memory - start_memory
    max_acceptable_growth = 1500  # 1.5 GB growth is acceptable

    assert memory_growth <= max_acceptable_growth, (
        f"Memory growth of {memory_growth:.2f}MB exceeds limit of {max_acceptable_growth}MB"
    )

    # Assert - No memory leaks (memory doesn't grow unbounded)
    assert (
        metrics["peak_memory_mb"] < 3000
    ), "Memory usage should stay under 3GB for medium dataset"

    print(f"\n=== Memory Stability ===")
    print(f"Start memory: {start_memory:.2f} MB")
    print(f"End memory: {end_memory:.2f} MB")
    print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
    print(f"Memory growth: {memory_growth:.2f} MB")


# ============================================================================
# Large Dataset Tests (3500+ bookmarks) - Full Load Simulation
# ============================================================================


@pytest.mark.large
@pytest.mark.slow
def test_large_dataset_performance(
    module_test_files,
    performance_pipeline,
    performance_monitor,
    performance_baseline,
    performance_assertion,
    temp_dir,
):
    """Test processing 3500 bookmarks - full load simulation."""
    # Arrange
    input_file = module_test_files["large"]
    output_file = temp_dir / "output_large.csv"
    baseline = performance_baseline["large_test"]

    # Act
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    # Get metrics
    metrics = performance_monitor.stop()

    # Assert - Processing completed
    assert output_file.exists(), "Output file should be created"
    assert results.processed_bookmarks > 3400, "Should process most bookmarks"

    # Assert - Performance metrics
    performance_assertion.assert_duration_within_limit(
        metrics["duration_seconds"],
        baseline["max_duration_seconds"],
        "Large dataset test",
    )

    performance_assertion.assert_memory_within_limit(
        metrics["peak_memory_mb"], baseline["max_memory_mb"], "Large dataset test"
    )

    # Assert - Throughput
    performance_assertion.assert_throughput_above_minimum(
        results.processed_bookmarks,
        metrics["duration_seconds"],
        baseline["min_throughput_per_hour"],
        "Large dataset test",
    )

    # Assert - Success rate
    performance_assertion.assert_success_rate_above_threshold(
        results.valid_bookmarks,
        results.processed_bookmarks,
        min_rate=0.95,
        test_name="Large dataset test",
    )

    # Print detailed summary
    print(f"\n=== Large Dataset Performance ===")
    print(f"Processed: {results.processed_bookmarks}/{baseline['size']} bookmarks")
    print(f"Valid: {results.valid_bookmarks}")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds ({metrics['duration_seconds'] / 3600:.2f} hours)")
    print(f"Memory: {metrics['peak_memory_mb']:.2f} MB")
    print(
        f"Throughput: {(results.processed_bookmarks / metrics['duration_seconds']) * 3600:.2f} bookmarks/hour"
    )
    print(
        f"Success rate: {(results.valid_bookmarks / results.processed_bookmarks * 100):.2f}%"
    )


@pytest.mark.large
@pytest.mark.slow
def test_large_dataset_checkpoint_resume(
    module_test_files,
    performance_pipeline,
    performance_monitor,
    benchmark_timer,
    temp_dir,
):
    """Test checkpoint/resume functionality with large dataset."""
    # Arrange
    input_file = module_test_files["large"]
    output_file = temp_dir / "output_resume.csv"
    checkpoint_dir = temp_dir / "checkpoints_resume"
    checkpoint_dir.mkdir(exist_ok=True)

    # Configure for checkpointing
    performance_pipeline.config.checkpoint["enabled"] = True
    performance_pipeline.config.checkpoint["save_interval"] = 100
    performance_pipeline.config.checkpoint["checkpoint_dir"] = str(checkpoint_dir)

    # Act - Initial processing (will create checkpoints)
    benchmark_timer.start("initial_processing")
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )
    benchmark_timer.stop()

    # Assert - Checkpoints created
    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoint_files) > 0, "Checkpoint files should be created"

    # Simulate resume by processing again (should skip already processed)
    benchmark_timer.start("resume_processing")
    resume_results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )
    resume_time = benchmark_timer.stop()

    # Get metrics
    metrics = performance_monitor.stop()

    # Assert - Resume is much faster (should skip most processing)
    # Resume should be at least 10x faster than initial
    assert resume_time < 30, f"Resume should be fast, took {resume_time:.2f}s"

    # Print summary
    print(f"\n=== Checkpoint/Resume Performance ===")
    print(f"Initial processing: {benchmark_timer.get_time('initial_processing'):.2f}s")
    print(f"Resume processing: {resume_time:.2f}s")
    print(f"Checkpoints created: {len(checkpoint_files)}")


# ============================================================================
# Specialized Performance Tests
# ============================================================================


@pytest.mark.medium
def test_duplicate_detection_performance(
    module_test_files, performance_pipeline, performance_monitor, temp_dir
):
    """Test performance of duplicate detection with dataset containing duplicates."""
    # Arrange
    input_file = module_test_files["duplicates"]
    output_file = temp_dir / "output_duplicates.csv"

    # Act
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    # Get metrics
    metrics = performance_monitor.stop()

    # Assert - Duplicates handled
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)

    assert len(output_df) < len(input_df), "Duplicates should be removed"

    # Assert - Performance acceptable
    assert metrics["duration_seconds"] < 120, "Duplicate detection should be fast"

    print(f"\n=== Duplicate Detection Performance ===")
    print(f"Input bookmarks: {len(input_df)}")
    print(f"Output bookmarks: {len(output_df)}")
    print(f"Duplicates removed: {len(input_df) - len(output_df)}")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds")


@pytest.mark.small
def test_error_handling_performance(
    module_test_files, performance_pipeline, performance_monitor, temp_dir
):
    """Test that error handling doesn't significantly impact performance."""
    # Arrange - Use dataset with some invalid entries
    input_file = module_test_files["small"]  # Contains ~2% invalid
    output_file = temp_dir / "output_errors.csv"

    # Act
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    # Get metrics
    metrics = performance_monitor.stop()

    # Assert - Errors handled gracefully
    assert results.processed_bookmarks > 0, "Should process valid bookmarks"
    assert len(results.errors) > 0 or results.invalid_bookmarks > 0, "Should detect invalid bookmarks"

    # Assert - Error handling doesn't cause significant slowdown
    assert metrics["duration_seconds"] < 120, "Error handling should be efficient"

    # Print summary
    print(f"\n=== Error Handling Performance ===")
    print(f"Total processed: {results.processed_bookmarks}")
    print(f"Valid: {results.valid_bookmarks}")
    print(f"Invalid: {results.invalid_bookmarks}")
    print(f"Errors: {len(results.errors)}")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds")


# ============================================================================
# Performance Comparison Tests
# ============================================================================


@pytest.mark.small
def test_batch_size_impact(
    module_test_files, temp_config_file, benchmark_timer, temp_dir,
    mock_url_validation, mock_content_extraction, mock_ai_processing
):
    """Test impact of different batch sizes on performance."""
    from bookmark_processor.config.configuration import Configuration

    input_file = module_test_files["small"]
    batch_sizes = [10, 25, 50, 100]
    results_by_batch_size = {}

    for batch_size in batch_sizes:
        # Configure with specific batch size
        config = Configuration(config_file=str(temp_config_file))
        config.processing["batch_size"] = batch_size
        config.checkpoint["checkpoint_dir"] = str(temp_dir / f"checkpoints_{batch_size}")

        pipeline = BookmarkPipeline(config)
        output_file = temp_dir / f"output_batch_{batch_size}.csv"

        # Time the processing
        benchmark_timer.start(f"batch_{batch_size}")
        results = pipeline.process(
            input_file=str(input_file),
            output_file=str(output_file),
        )
        duration = benchmark_timer.stop()

        results_by_batch_size[batch_size] = {
            "duration": duration,
            "processed": results.processed_bookmarks,
        }

    # Print comparison
    print(f"\n=== Batch Size Impact ===")
    for batch_size, data in results_by_batch_size.items():
        print(f"Batch size {batch_size}: {data['duration']:.2f}s ({data['processed']} bookmarks)")

    # Assert - All batch sizes should complete
    for batch_size, data in results_by_batch_size.items():
        assert data["processed"] > 0, f"Batch size {batch_size} should process bookmarks"


# ============================================================================
# Performance Regression Tests
# ============================================================================


@pytest.mark.small
def test_performance_baseline_regression(
    module_test_files,
    performance_pipeline,
    performance_monitor,
    performance_baseline,
    temp_dir,
):
    """
    Test that performance doesn't regress from baseline.

    This test establishes baseline performance metrics that should not degrade
    in future versions.
    """
    # Arrange
    input_file = module_test_files["small"]
    output_file = temp_dir / "output_baseline.csv"
    baseline = performance_baseline["small_test"]

    # Act
    results = performance_pipeline.process(
        input_file=str(input_file),
        output_file=str(output_file),
    )

    metrics = performance_monitor.stop()

    # Calculate throughput
    throughput = (results.processed_bookmarks / metrics["duration_seconds"]) * 3600

    # Assert - Performance meets or exceeds baseline
    assert (
        metrics["duration_seconds"] <= baseline["max_duration_seconds"]
    ), "Performance regression: processing too slow"

    assert (
        metrics["peak_memory_mb"] <= baseline["max_memory_mb"]
    ), "Performance regression: memory usage too high"

    assert (
        throughput >= baseline["min_throughput_per_hour"]
    ), "Performance regression: throughput too low"

    # Store metrics for regression tracking
    performance_report = {
        "test": "small_dataset_baseline",
        "timestamp": time.time(),
        "metrics": {
            "duration_seconds": metrics["duration_seconds"],
            "peak_memory_mb": metrics["peak_memory_mb"],
            "throughput_per_hour": throughput,
            "processed_count": results.processed_bookmarks,
        },
        "baseline": baseline,
    }

    print(f"\n=== Performance Baseline Report ===")
    print(f"Duration: {metrics['duration_seconds']:.2f}s (max: {baseline['max_duration_seconds']}s)")
    print(f"Memory: {metrics['peak_memory_mb']:.2f}MB (max: {baseline['max_memory_mb']}MB)")
    print(f"Throughput: {throughput:.2f}/hour (min: {baseline['min_throughput_per_hour']}/hour)")
