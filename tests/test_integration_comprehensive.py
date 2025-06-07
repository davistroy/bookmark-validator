"""
Comprehensive Integration Tests for Bookmark Processor

This module contains comprehensive integration tests that use the enhanced
testing framework to validate complete end-to-end functionality.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from bookmark_processor.core.bookmark_processor import BookmarkProcessor
from tests.framework import (
    CompositeValidator,
    IntegrationTestFixtures,
    ScenarioRunner,
    StandardScenarios,
    TestEnvironmentManager,
)


@pytest.mark.integration
class TestComprehensiveIntegration:
    """Comprehensive integration tests using the enhanced framework."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment for each test."""
        self.env_manager = TestEnvironmentManager()
        self.logger = logging.getLogger("test_comprehensive_integration")

    def teardown_method(self, method):
        """Clean up after each test."""
        if hasattr(self, "env_manager"):
            self.env_manager.cleanup_all()

    def test_basic_processing_scenario(self):
        """Test basic bookmark processing workflow."""

        with self.env_manager.temporary_environment("basic_processing") as env:
            # Set up fixtures
            with IntegrationTestFixtures(env) as fixtures:
                # Create test data
                input_file = fixtures.create_test_dataset(
                    name="basic_test", size=10, include_invalid=False
                )

                # Set up mocks
                mocks = fixtures.setup_standard_mocks(
                    network_success_rate=0.95, ai_quality="good"
                )

                # Run scenario
                scenario_runner = ScenarioRunner(env)
                scenario = StandardScenarios.get_basic_scenarios()[
                    0
                ]  # BasicProcessingScenario

                result = scenario_runner.run_scenario(scenario)

                # Validate results
                assert result.status.value == "completed"
                assert result.processing_results is not None
                assert result.processing_results.total_bookmarks == 10
                assert (
                    result.processing_results.valid_bookmarks >= 8
                )  # 80% success rate
                assert len(result.errors) == 0

    def test_checkpoint_resume_functionality(self):
        """Test checkpoint creation and resume functionality."""

        with self.env_manager.temporary_environment("checkpoint_resume") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create larger test dataset
                input_file = fixtures.create_test_dataset(
                    name="checkpoint_test", size=20, include_invalid=False
                )

                # Set up mocks
                mocks = fixtures.setup_standard_mocks()

                # Create initial checkpoint scenario
                checkpoint_file = fixtures.create_checkpoint_scenario(
                    checkpoint_id="test_checkpoint", processed_count=5, total_count=20
                )

                # Run checkpoint scenario
                scenario_runner = ScenarioRunner(env)
                checkpoint_scenario = StandardScenarios.get_comprehensive_scenarios()[
                    1
                ]  # CheckpointResumeScenario

                result = scenario_runner.run_scenario(checkpoint_scenario)

                # Validate checkpoint functionality
                checkpoint_dir = env.get_directory("checkpoints")
                checkpoint_files = list(checkpoint_dir.glob("*.json"))

                assert len(checkpoint_files) > 0, "Checkpoint files should be created"
                assert result.status.value == "completed"
                assert result.processing_results.total_bookmarks == 20

    def test_error_handling_resilience(self):
        """Test error handling and recovery mechanisms."""

        with self.env_manager.temporary_environment("error_handling") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create test data with intentional errors
                input_file = fixtures.create_error_test_dataset("error_test")

                # Set up mocks with high error rate
                mocks = fixtures.setup_standard_mocks(
                    network_success_rate=0.6, ai_quality="poor"  # 40% error rate
                )

                # Run error handling scenario
                scenario_runner = ScenarioRunner(env)
                error_scenario = StandardScenarios.get_comprehensive_scenarios()[
                    2
                ]  # ErrorHandlingScenario

                result = scenario_runner.run_scenario(error_scenario)

                # Validate error handling
                assert result.status.value == "completed"
                assert result.processing_results is not None
                assert (
                    len(result.processing_results.errors) > 0
                ), "Should have recorded errors"
                assert (
                    result.processing_results.valid_bookmarks > 0
                ), "Should have some valid results despite errors"
                assert (
                    result.processing_results.invalid_bookmarks > 0
                ), "Should have some invalid results"

    def test_performance_under_load(self):
        """Test performance with larger datasets."""

        with self.env_manager.temporary_environment("performance_test") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create performance test dataset
                input_file = fixtures.create_performance_dataset(
                    name="performance_test", size=50
                )

                # Set up fast mocks for performance testing
                mocks = fixtures.setup_standard_mocks(
                    network_success_rate=0.95, ai_quality="good"
                )

                # Run performance scenario
                scenario_runner = ScenarioRunner(env)
                performance_scenario = StandardScenarios.get_performance_scenarios()[
                    2
                ]  # 50 items

                start_time = time.time()
                result = scenario_runner.run_scenario(performance_scenario)
                end_time = time.time()

                # Validate performance
                assert result.status.value == "completed"
                assert result.processing_results.total_bookmarks == 50

                # Performance criteria
                processing_time = end_time - start_time
                assert (
                    processing_time < 30.0
                ), f"Processing took too long: {processing_time:.2f}s"

                # Throughput validation
                throughput = 50 / processing_time
                assert (
                    throughput > 2.0
                ), f"Throughput too low: {throughput:.2f} items/second"

    def test_malformed_input_handling(self):
        """Test handling of malformed input data."""

        with self.env_manager.temporary_environment("malformed_input") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create malformed CSV
                malformed_file = fixtures.create_malformed_csv("malformed_test")

                # Set up mocks
                mocks = fixtures.setup_standard_mocks()

                # Create output file path
                output_file = env.get_directory("output") / "malformed_output.csv"

                # Try to process malformed input
                processor = BookmarkProcessor()

                with pytest.raises(Exception):
                    # Should raise an exception for malformed input
                    processor.process_bookmarks(
                        input_file=str(malformed_file),
                        output_file=str(output_file),
                        batch_size=5,
                        enable_checkpoints=False,
                    )

    def test_network_condition_simulation(self):
        """Test behavior under different network conditions."""

        network_conditions = ["fast", "slow", "unstable"]

        for condition in network_conditions:
            with self.env_manager.temporary_environment(f"network_{condition}") as env:
                with IntegrationTestFixtures(env) as fixtures:
                    # Create test data
                    input_file = fixtures.create_test_dataset(
                        name=f"network_{condition}", size=15, include_invalid=False
                    )

                    # Set up network condition specific mocks
                    if condition == "fast":
                        mocks = fixtures.setup_standard_mocks(network_success_rate=0.95)
                    elif condition == "slow":
                        mocks = fixtures.setup_standard_mocks(network_success_rate=0.85)
                    else:  # unstable
                        mocks = fixtures.setup_standard_mocks(network_success_rate=0.7)

                    # Run basic scenario
                    scenario_runner = ScenarioRunner(env)
                    scenario = StandardScenarios.get_basic_scenarios()[0]

                    result = scenario_runner.run_scenario(scenario)

                    # Validate that processing completes despite network conditions
                    assert result.status.value == "completed"
                    assert result.processing_results.total_bookmarks == 15

                    # Adjust expectations based on network condition
                    if condition == "fast":
                        assert result.processing_results.valid_bookmarks >= 14
                    elif condition == "slow":
                        assert result.processing_results.valid_bookmarks >= 12
                    else:  # unstable
                        assert result.processing_results.valid_bookmarks >= 10

    def test_comprehensive_validation_suite(self):
        """Test comprehensive validation of all aspects."""

        with self.env_manager.temporary_environment("comprehensive_validation") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create comprehensive test dataset
                input_file = fixtures.create_test_dataset(
                    name="validation_test", size=25, include_invalid=True
                )

                # Set up mocks
                mocks = fixtures.setup_standard_mocks()

                # Run processing
                output_file = env.get_directory("output") / "validation_output.csv"
                processor = BookmarkProcessor()

                results = processor.process_bookmarks(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    batch_size=10,
                    enable_checkpoints=True,
                    checkpoint_dir=str(env.get_directory("checkpoints")),
                )

                # Comprehensive validation
                validator = CompositeValidator()

                validation_results = validator.validate_integration_test(
                    results=results,
                    output_file=output_file,
                    checkpoint_dir=env.get_directory("checkpoints"),
                    expected_results={
                        "min_success_rate": 0.7,
                        "should_complete": True,
                        "max_duration": 30.0,
                    },
                )

                # Check validation results
                assert validator.get_overall_result(
                    validation_results
                ), f"Validation failed: {[r.message for r in validation_results.values() if not r.passed]}"

                # Generate and verify validation report
                report = validator.generate_validation_report(validation_results)
                assert report["overall_passed"]
                assert report["failed_count"] == 0

    def test_stress_scenario_execution(self):
        """Test stress scenarios with multiple challenging conditions."""

        with self.env_manager.temporary_environment("stress_test") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create large dataset with errors
                input_file = fixtures.create_test_dataset(
                    name="stress_test", size=100, include_invalid=True
                )

                # Set up challenging conditions
                mocks = fixtures.setup_standard_mocks(
                    network_success_rate=0.75, ai_quality="random"  # 25% error rate
                )

                # Run all stress scenarios
                scenario_runner = ScenarioRunner(env)
                stress_scenarios = StandardScenarios.get_stress_scenarios()

                results = scenario_runner.run_scenarios(stress_scenarios)

                # Validate stress test results
                assert len(results) == len(stress_scenarios)

                # At least some scenarios should complete successfully
                completed_count = sum(
                    1 for r in results.values() if r.status.value == "completed"
                )
                assert (
                    completed_count >= len(stress_scenarios) // 2
                ), "At least half of stress scenarios should complete"

                # Generate summary report
                summary = scenario_runner.get_summary_report()
                assert summary["total_scenarios"] == len(stress_scenarios)
                assert summary["success_rate"] >= 0.5  # At least 50% success rate

    def test_end_to_end_workflow_validation(self):
        """Test complete end-to-end workflow with full validation."""

        with self.env_manager.temporary_environment("end_to_end") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create realistic test dataset
                input_file = fixtures.create_performance_dataset(
                    name="end_to_end", size=30
                )

                # Set up realistic mocks
                mocks = fixtures.setup_standard_mocks(
                    network_success_rate=0.9, ai_quality="good"
                )

                # Configure output
                output_file = env.get_directory("output") / "end_to_end_output.csv"

                # Run complete processing workflow
                processor = BookmarkProcessor()

                processing_results = processor.process_bookmarks(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    batch_size=10,
                    max_retries=2,
                    timeout=10,
                    enable_ai_processing=True,
                    enable_checkpoints=True,
                    checkpoint_dir=str(env.get_directory("checkpoints")),
                )

                # Verify output file integrity
                output_validation = fixtures.verify_test_output(
                    output_file=output_file,
                    expected_structure=True,
                    min_rows=25,  # Expect at least 25 valid results from 30 input
                )

                assert output_validation["file_exists"]
                assert output_validation["structure_valid"]
                assert len(output_validation["errors"]) == 0

                # Verify processing results
                assert processing_results is not None
                assert processing_results.total_bookmarks == 30
                assert processing_results.valid_bookmarks >= 25
                assert processing_results.processing_time > 0

                # Verify checkpoint creation
                checkpoint_files = list(env.get_directory("checkpoints").glob("*.json"))
                assert (
                    len(checkpoint_files) > 0
                ), "Checkpoints should be created for long processing"

    @pytest.mark.slow
    def test_large_dataset_processing(self):
        """Test processing of larger datasets (marked as slow test)."""

        with self.env_manager.temporary_environment("large_dataset") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create large test dataset
                input_file = fixtures.create_performance_dataset(
                    name="large_dataset", size=200
                )

                # Set up optimized mocks for large dataset
                mocks = fixtures.setup_standard_mocks(
                    network_success_rate=0.9, ai_quality="good"
                )

                # Configure for large dataset processing
                output_file = env.get_directory("output") / "large_dataset_output.csv"

                # Process with optimized settings
                processor = BookmarkProcessor()

                start_time = time.time()
                processing_results = processor.process_bookmarks(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    batch_size=20,  # Larger batches for efficiency
                    max_retries=1,  # Reduced retries for speed
                    timeout=15,
                    enable_ai_processing=False,  # Disable AI for speed
                    enable_checkpoints=True,
                    checkpoint_dir=str(env.get_directory("checkpoints")),
                )
                end_time = time.time()

                # Validate large dataset processing
                assert processing_results is not None
                assert processing_results.total_bookmarks == 200
                assert processing_results.valid_bookmarks >= 160  # 80% success rate

                # Performance validation
                processing_time = end_time - start_time
                assert (
                    processing_time < 120.0
                ), f"Large dataset processing took too long: {processing_time:.2f}s"

                # Throughput validation
                throughput = 200 / processing_time
                assert (
                    throughput > 2.0
                ), f"Large dataset throughput too low: {throughput:.2f} items/second"

                # Memory efficiency validation (indirect through successful completion)
                assert processing_results.processing_time > 0

                self.logger.info(
                    f"Large dataset test completed: {throughput:.2f} items/second"
                )


@pytest.mark.integration
@pytest.mark.network
class TestNetworkIntegration:
    """Network-specific integration tests."""

    def test_offline_mode_handling(self):
        """Test behavior when network is completely unavailable."""

        env_manager = TestEnvironmentManager()

        with env_manager.temporary_environment("offline_test") as env:
            with IntegrationTestFixtures(env) as fixtures:
                # Create test data
                input_file = fixtures.create_test_dataset(
                    name="offline_test", size=5, include_invalid=False
                )

                # Set up complete network failure
                mocks = fixtures.setup_standard_mocks(
                    network_success_rate=0.0  # Complete network failure
                )

                # Configure output
                output_file = env.get_directory("output") / "offline_output.csv"

                # Process in offline conditions
                processor = BookmarkProcessor()

                processing_results = processor.process_bookmarks(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    batch_size=5,
                    max_retries=1,
                    timeout=5,
                    enable_checkpoints=False,
                )

                # Validate offline behavior
                assert processing_results is not None
                assert processing_results.total_bookmarks == 5
                assert (
                    processing_results.invalid_bookmarks == 5
                )  # All should fail due to network
                assert (
                    len(processing_results.errors) >= 5
                )  # Should have error for each bookmark


if __name__ == "__main__":
    pytest.main([__file__])
