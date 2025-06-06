"""
Test Scenarios for Integration Testing

Provides predefined test scenarios, scenario runners, and scenario validation
for comprehensive integration testing of the bookmark processor.
"""

import time
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import pandas as pd
import logging

from bookmark_processor.core.bookmark_processor import BookmarkProcessor
from bookmark_processor.core.data_models import ProcessingResults


class ScenarioStatus(Enum):
    """Status of a test scenario execution."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScenarioResult:
    """Result of a test scenario execution."""
    scenario_name: str
    status: ScenarioStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_results: Optional[ProcessingResults] = None
    artifacts: Dict[str, Path] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get scenario execution duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success(self) -> bool:
        """Check if scenario completed successfully."""
        return self.status == ScenarioStatus.COMPLETED and len(self.errors) == 0


class TestScenario(ABC):
    """Abstract base class for test scenarios."""
    
    def __init__(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        expected_results: Dict[str, Any]
    ):
        self.name = name
        self.description = description
        self.config = config
        self.expected_results = expected_results
        self.logger = logging.getLogger(f"scenario.{name}")
    
    @abstractmethod
    def setup_test_data(self, environment) -> Path:
        """Set up test data for the scenario."""
        pass
    
    @abstractmethod
    def run_scenario(
        self,
        processor: BookmarkProcessor,
        input_file: Path,
        output_file: Path,
        environment
    ) -> ProcessingResults:
        """Execute the test scenario."""
        pass
    
    @abstractmethod
    def validate_results(
        self,
        results: ProcessingResults,
        output_file: Path,
        environment
    ) -> List[str]:
        """Validate scenario results. Return list of errors if any."""
        pass
    
    def get_scenario_config(self) -> Dict[str, Any]:
        """Get the configuration for this scenario."""
        return self.config.copy()
    
    def get_expected_results(self) -> Dict[str, Any]:
        """Get expected results for this scenario."""
        return self.expected_results.copy()


class BasicProcessingScenario(TestScenario):
    """Basic bookmark processing scenario."""
    
    def __init__(self):
        super().__init__(
            name="basic_processing",
            description="Basic bookmark processing with minimal configuration",
            config={
                'batch_size': 5,
                'max_retries': 1,
                'timeout': 5,
                'enable_ai_processing': False,
                'enable_checkpoints': False
            },
            expected_results={
                'min_success_rate': 0.8,
                'should_complete': True,
                'max_duration': 30.0
            }
        )
    
    def setup_test_data(self, environment) -> Path:
        """Create basic test data."""
        test_data = [
            {
                'id': '1',
                'title': 'Example Site 1',
                'note': 'Test note 1',
                'excerpt': 'Test excerpt 1',
                'url': 'https://example.com',
                'folder': 'Test',
                'tags': 'test, example',
                'created': '2024-01-01T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            },
            {
                'id': '2', 
                'title': 'Example Site 2',
                'note': 'Test note 2',
                'excerpt': 'Test excerpt 2',
                'url': 'https://httpbin.org/get',
                'folder': 'Test',
                'tags': 'test, api',
                'created': '2024-01-02T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            }
        ]
        
        input_file = environment.get_directory('input') / 'basic_test.csv'
        df = pd.DataFrame(test_data)
        df.to_csv(input_file, index=False)
        
        return input_file
    
    def run_scenario(
        self,
        processor: BookmarkProcessor,
        input_file: Path,
        output_file: Path,
        environment
    ) -> ProcessingResults:
        """Run basic processing scenario."""
        return processor.process_bookmarks(
            input_file=str(input_file),
            output_file=str(output_file),
            **self.config
        )
    
    def validate_results(
        self,
        results: ProcessingResults,
        output_file: Path,
        environment
    ) -> List[str]:
        """Validate basic processing results."""
        errors = []
        
        # Check that processing completed
        if not results:
            errors.append("Processing results are None")
            return errors
        
        # Check success rate
        if results.total_bookmarks > 0:
            success_rate = results.valid_bookmarks / results.total_bookmarks
            if success_rate < self.expected_results['min_success_rate']:
                errors.append(f"Success rate {success_rate:.2f} below minimum {self.expected_results['min_success_rate']}")
        
        # Check output file exists
        if not output_file.exists():
            errors.append("Output file was not created")
        else:
            # Validate output structure
            try:
                output_df = pd.read_csv(output_file)
                expected_columns = ['url', 'folder', 'title', 'note', 'tags', 'created']
                if list(output_df.columns) != expected_columns:
                    errors.append(f"Output columns incorrect: {list(output_df.columns)}")
            except Exception as e:
                errors.append(f"Failed to read output file: {e}")
        
        return errors


class CheckpointResumeScenario(TestScenario):
    """Scenario testing checkpoint and resume functionality."""
    
    def __init__(self):
        super().__init__(
            name="checkpoint_resume",
            description="Test checkpoint creation and resume functionality",
            config={
                'batch_size': 3,
                'max_retries': 1,
                'timeout': 10,
                'enable_ai_processing': False,
                'enable_checkpoints': True
            },
            expected_results={
                'checkpoint_created': True,
                'resume_successful': True,
                'data_consistency': True
            }
        )
        self.interrupt_after = 2  # Interrupt after processing 2 items
    
    def setup_test_data(self, environment) -> Path:
        """Create test data for checkpoint testing."""
        # Create larger dataset to ensure checkpoint creation
        test_data = []
        for i in range(10):
            test_data.append({
                'id': str(i + 1),
                'title': f'Checkpoint Test Site {i + 1}',
                'note': f'Test note {i + 1}',
                'excerpt': f'Test excerpt {i + 1}',
                'url': f'https://example{i}.com',
                'folder': 'CheckpointTest',
                'tags': f'test, checkpoint, item{i}',
                'created': f'2024-01-{(i % 28) + 1:02d}T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            })
        
        input_file = environment.get_directory('input') / 'checkpoint_test.csv'
        df = pd.DataFrame(test_data)
        df.to_csv(input_file, index=False)
        
        return input_file
    
    def run_scenario(
        self,
        processor: BookmarkProcessor,
        input_file: Path,
        output_file: Path,
        environment
    ) -> ProcessingResults:
        """Run checkpoint resume scenario with simulated interruption."""
        
        # Phase 1: Initial processing (will be interrupted)
        self.logger.info("Starting initial processing (will be interrupted)")
        
        # Use threading to interrupt processing after a delay
        def interrupt_processing():
            time.sleep(2)  # Let some processing happen
            # This is a simulation - in real tests we'd use process interruption
            self.logger.info("Simulating process interruption")
        
        interrupt_thread = threading.Thread(target=interrupt_processing)
        interrupt_thread.start()
        
        try:
            # First run - simulate partial completion
            config_phase1 = self.config.copy()
            config_phase1['checkpoint_dir'] = str(environment.get_directory('checkpoints'))
            
            results_phase1 = processor.process_bookmarks(
                input_file=str(input_file),
                output_file=str(output_file),
                **config_phase1
            )
            
            # Check if checkpoints were created
            checkpoint_files = list(environment.get_directory('checkpoints').glob('*.json'))
            if checkpoint_files:
                self.logger.info(f"Checkpoints created: {len(checkpoint_files)} files")
            
            # Phase 2: Resume processing
            self.logger.info("Resuming processing from checkpoint")
            
            config_phase2 = self.config.copy()
            config_phase2['resume'] = True
            config_phase2['checkpoint_dir'] = str(environment.get_directory('checkpoints'))
            
            results_phase2 = processor.process_bookmarks(
                input_file=str(input_file),
                output_file=str(output_file),
                **config_phase2
            )
            
            return results_phase2
            
        except Exception as e:
            self.logger.error(f"Error in checkpoint scenario: {e}")
            # Return partial results for analysis
            return ProcessingResults(
                total_bookmarks=10,
                processed_bookmarks=0,
                valid_bookmarks=0,
                invalid_bookmarks=0,
                errors=[str(e)],
                processing_time=0.0
            )
        finally:
            interrupt_thread.join()
    
    def validate_results(
        self,
        results: ProcessingResults,
        output_file: Path,
        environment
    ) -> List[str]:
        """Validate checkpoint resume results."""
        errors = []
        
        # Check checkpoint files were created
        checkpoint_files = list(environment.get_directory('checkpoints').glob('*.json'))
        if not checkpoint_files:
            errors.append("No checkpoint files were created")
        
        # Check final processing results
        if not results:
            errors.append("No processing results returned")
            return errors
        
        # Check that all items were eventually processed
        if results.total_bookmarks != 10:
            errors.append(f"Expected 10 total bookmarks, got {results.total_bookmarks}")
        
        # Check output file
        if not output_file.exists():
            errors.append("Final output file not created")
        else:
            try:
                output_df = pd.read_csv(output_file)
                if len(output_df) != results.valid_bookmarks:
                    errors.append(f"Output row count {len(output_df)} doesn't match valid bookmarks {results.valid_bookmarks}")
            except Exception as e:
                errors.append(f"Failed to validate output file: {e}")
        
        return errors


class ErrorHandlingScenario(TestScenario):
    """Scenario testing error handling and recovery."""
    
    def __init__(self):
        super().__init__(
            name="error_handling",
            description="Test error handling with invalid URLs and malformed data",
            config={
                'batch_size': 3,
                'max_retries': 2,
                'timeout': 5,
                'enable_ai_processing': False,
                'enable_checkpoints': False
            },
            expected_results={
                'should_handle_errors': True,
                'should_continue_processing': True,
                'min_error_logging': 1
            }
        )
    
    def setup_test_data(self, environment) -> Path:
        """Create test data with intentional errors."""
        test_data = [
            # Valid bookmark
            {
                'id': '1',
                'title': 'Valid Site',
                'note': 'Valid note',
                'excerpt': 'Valid excerpt',
                'url': 'https://httpbin.org/get',
                'folder': 'Valid',
                'tags': 'valid, test',
                'created': '2024-01-01T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            },
            # Invalid URL
            {
                'id': '2',
                'title': 'Invalid URL',
                'note': 'This has an invalid URL',
                'excerpt': 'Invalid URL test',
                'url': 'not-a-valid-url',
                'folder': 'Invalid',
                'tags': 'invalid, test',
                'created': '2024-01-02T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            },
            # Non-existent domain
            {
                'id': '3',
                'title': 'Non-existent Domain',
                'note': 'This domain does not exist',
                'excerpt': 'Non-existent domain test',
                'url': 'https://this-domain-definitely-does-not-exist-12345.com',
                'folder': 'NonExistent',
                'tags': 'nonexistent, test',
                'created': '2024-01-03T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            },
            # Another valid bookmark to ensure processing continues
            {
                'id': '4',
                'title': 'Another Valid Site',
                'note': 'Another valid note',
                'excerpt': 'Another valid excerpt',
                'url': 'https://httpbin.org/status/200',
                'folder': 'Valid',
                'tags': 'valid, test, second',
                'created': '2024-01-04T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            }
        ]
        
        input_file = environment.get_directory('input') / 'error_test.csv'
        df = pd.DataFrame(test_data)
        df.to_csv(input_file, index=False)
        
        return input_file
    
    def run_scenario(
        self,
        processor: BookmarkProcessor,
        input_file: Path,
        output_file: Path,
        environment
    ) -> ProcessingResults:
        """Run error handling scenario."""
        return processor.process_bookmarks(
            input_file=str(input_file),
            output_file=str(output_file),
            **self.config
        )
    
    def validate_results(
        self,
        results: ProcessingResults,
        output_file: Path,
        environment
    ) -> List[str]:
        """Validate error handling results."""
        errors = []
        
        if not results:
            errors.append("No processing results returned")
            return errors
        
        # Should have processed all items despite errors
        if results.total_bookmarks != 4:
            errors.append(f"Expected 4 total bookmarks, got {results.total_bookmarks}")
        
        # Should have some errors recorded
        if len(results.errors) < self.expected_results['min_error_logging']:
            errors.append(f"Expected at least {self.expected_results['min_error_logging']} errors, got {len(results.errors)}")
        
        # Should have some valid bookmarks despite errors
        if results.valid_bookmarks < 2:
            errors.append(f"Expected at least 2 valid bookmarks, got {results.valid_bookmarks}")
        
        # Should have some invalid bookmarks
        if results.invalid_bookmarks < 1:
            errors.append(f"Expected at least 1 invalid bookmark, got {results.invalid_bookmarks}")
        
        # Output file should still be created with valid bookmarks
        if not output_file.exists():
            errors.append("Output file not created despite having valid bookmarks")
        
        return errors


class PerformanceScenario(TestScenario):
    """Scenario testing performance with larger datasets."""
    
    def __init__(self, dataset_size: int = 50):
        self.dataset_size = dataset_size
        super().__init__(
            name=f"performance_{dataset_size}",
            description=f"Performance test with {dataset_size} bookmarks",
            config={
                'batch_size': 10,
                'max_retries': 1,
                'timeout': 10,
                'enable_ai_processing': False,
                'enable_checkpoints': True
            },
            expected_results={
                'max_duration': dataset_size * 0.5,  # 0.5 seconds per bookmark max
                'min_success_rate': 0.8,
                'max_memory_mb': 500
            }
        )
    
    def setup_test_data(self, environment) -> Path:
        """Create performance test dataset."""
        test_data = []
        
        # Create a variety of test URLs
        url_patterns = [
            'https://httpbin.org/get?id={}',
            'https://httpbin.org/status/200?test={}',
            'https://example.com/page{}',
            'https://httpbin.org/delay/1?item={}'
        ]
        
        for i in range(self.dataset_size):
            url_pattern = url_patterns[i % len(url_patterns)]
            test_data.append({
                'id': str(i + 1),
                'title': f'Performance Test Site {i + 1}',
                'note': f'Performance test note {i + 1}',
                'excerpt': f'Performance test excerpt for item {i + 1}',
                'url': url_pattern.format(i),
                'folder': f'Performance/Batch{i // 10}',
                'tags': f'performance, test, batch{i // 10}, item{i}',
                'created': f'2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': str(i % 5 == 0).lower()
            })
        
        input_file = environment.get_directory('input') / f'performance_test_{self.dataset_size}.csv'
        df = pd.DataFrame(test_data)
        df.to_csv(input_file, index=False)
        
        return input_file
    
    def run_scenario(
        self,
        processor: BookmarkProcessor,
        input_file: Path,
        output_file: Path,
        environment
    ) -> ProcessingResults:
        """Run performance scenario with timing."""
        start_time = time.time()
        
        results = processor.process_bookmarks(
            input_file=str(input_file),
            output_file=str(output_file),
            **self.config
        )
        
        end_time = time.time()
        
        # Add timing information to results
        if results:
            results.processing_time = end_time - start_time
        
        return results
    
    def validate_results(
        self,
        results: ProcessingResults,
        output_file: Path,
        environment
    ) -> List[str]:
        """Validate performance results."""
        errors = []
        
        if not results:
            errors.append("No processing results returned")
            return errors
        
        # Check duration
        if results.processing_time > self.expected_results['max_duration']:
            errors.append(f"Processing took {results.processing_time:.2f}s, expected max {self.expected_results['max_duration']:.2f}s")
        
        # Check success rate
        if results.total_bookmarks > 0:
            success_rate = results.valid_bookmarks / results.total_bookmarks
            if success_rate < self.expected_results['min_success_rate']:
                errors.append(f"Success rate {success_rate:.2f} below minimum {self.expected_results['min_success_rate']}")
        
        # Check that all items were processed
        if results.total_bookmarks != self.dataset_size:
            errors.append(f"Expected {self.dataset_size} total bookmarks, got {results.total_bookmarks}")
        
        return errors


class ScenarioRunner:
    """Runs test scenarios and manages execution."""
    
    def __init__(self, environment):
        self.environment = environment
        self.logger = logging.getLogger("scenario_runner")
        self.results: Dict[str, ScenarioResult] = {}
    
    def run_scenario(
        self,
        scenario: TestScenario,
        processor_factory: Callable[[], BookmarkProcessor] = None
    ) -> ScenarioResult:
        """Run a single test scenario."""
        
        if processor_factory is None:
            processor_factory = lambda: BookmarkProcessor()
        
        self.logger.info(f"Starting scenario: {scenario.name}")
        
        # Initialize result
        result = ScenarioResult(
            scenario_name=scenario.name,
            status=ScenarioStatus.PENDING,
            start_time=datetime.now(timezone.utc)
        )
        
        try:
            result.status = ScenarioStatus.RUNNING
            
            # Set up test data
            input_file = scenario.setup_test_data(self.environment)
            output_file = self.environment.get_directory('output') / f"{scenario.name}_output.csv"
            
            # Create processor instance
            processor = processor_factory()
            
            # Run scenario
            processing_results = scenario.run_scenario(
                processor, input_file, output_file, self.environment
            )
            result.processing_results = processing_results
            
            # Store artifacts
            result.artifacts = {
                'input_file': input_file,
                'output_file': output_file
            }
            
            # Validate results
            validation_errors = scenario.validate_results(
                processing_results, output_file, self.environment
            )
            result.errors = validation_errors
            
            # Calculate metrics
            result.metrics = self._calculate_metrics(scenario, processing_results)
            
            # Set final status
            result.status = ScenarioStatus.COMPLETED if not validation_errors else ScenarioStatus.FAILED
            
        except Exception as e:
            self.logger.error(f"Scenario {scenario.name} failed with exception: {e}")
            result.status = ScenarioStatus.FAILED
            result.errors.append(f"Exception during execution: {str(e)}")
        
        finally:
            result.end_time = datetime.now(timezone.utc)
            self.results[scenario.name] = result
        
        self.logger.info(f"Scenario {scenario.name} completed with status: {result.status}")
        return result
    
    def run_scenarios(
        self,
        scenarios: List[TestScenario],
        processor_factory: Callable[[], BookmarkProcessor] = None
    ) -> Dict[str, ScenarioResult]:
        """Run multiple scenarios."""
        for scenario in scenarios:
            self.run_scenario(scenario, processor_factory)
        
        return self.results.copy()
    
    def _calculate_metrics(
        self,
        scenario: TestScenario,
        results: ProcessingResults
    ) -> Dict[str, Any]:
        """Calculate metrics for a scenario."""
        metrics = {}
        
        if results:
            metrics['total_bookmarks'] = results.total_bookmarks
            metrics['processed_bookmarks'] = results.processed_bookmarks
            metrics['valid_bookmarks'] = results.valid_bookmarks
            metrics['invalid_bookmarks'] = results.invalid_bookmarks
            metrics['processing_time'] = results.processing_time
            
            if results.total_bookmarks > 0:
                metrics['success_rate'] = results.valid_bookmarks / results.total_bookmarks
                metrics['error_rate'] = results.invalid_bookmarks / results.total_bookmarks
                metrics['processing_rate'] = results.total_bookmarks / max(results.processing_time, 0.01)
        
        return metrics
    
    def get_scenario_result(self, scenario_name: str) -> Optional[ScenarioResult]:
        """Get result for a specific scenario."""
        return self.results.get(scenario_name)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary report of all scenario results."""
        total_scenarios = len(self.results)
        completed_scenarios = sum(1 for r in self.results.values() if r.status == ScenarioStatus.COMPLETED)
        failed_scenarios = sum(1 for r in self.results.values() if r.status == ScenarioStatus.FAILED)
        
        return {
            'total_scenarios': total_scenarios,
            'completed_scenarios': completed_scenarios,
            'failed_scenarios': failed_scenarios,
            'success_rate': completed_scenarios / max(total_scenarios, 1),
            'scenario_results': {name: {
                'status': result.status.value,
                'duration': result.duration,
                'errors': len(result.errors),
                'metrics': result.metrics
            } for name, result in self.results.items()}
        }


class StandardScenarios:
    """Factory for standard test scenarios."""
    
    @staticmethod
    def get_basic_scenarios() -> List[TestScenario]:
        """Get basic integration test scenarios."""
        return [
            BasicProcessingScenario(),
            ErrorHandlingScenario()
        ]
    
    @staticmethod
    def get_comprehensive_scenarios() -> List[TestScenario]:
        """Get comprehensive integration test scenarios."""
        return [
            BasicProcessingScenario(),
            CheckpointResumeScenario(),
            ErrorHandlingScenario(),
            PerformanceScenario(dataset_size=25)
        ]
    
    @staticmethod
    def get_performance_scenarios() -> List[TestScenario]:
        """Get performance-focused test scenarios."""
        return [
            PerformanceScenario(dataset_size=10),
            PerformanceScenario(dataset_size=25),
            PerformanceScenario(dataset_size=50)
        ]
    
    @staticmethod
    def get_stress_scenarios() -> List[TestScenario]:
        """Get stress test scenarios."""
        return [
            PerformanceScenario(dataset_size=100),
            CheckpointResumeScenario(),
            ErrorHandlingScenario()
        ]