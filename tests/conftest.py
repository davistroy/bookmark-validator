"""
Pytest configuration and shared fixtures for bookmark processor tests.

This module provides common fixtures, mocks, and test utilities that are
shared across multiple test modules.
"""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import requests

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.data_models import (
    Bookmark,
    BookmarkMetadata,
    ProcessingResults,
    ProcessingStatus,
)
from tests.fixtures.test_data import (
    EXPECTED_RAINDROP_IMPORT_ROWS,
    MOCK_AI_RESULTS,
    MOCK_CONTENT_DATA,
    SAMPLE_RAINDROP_EXPORT_ROWS,
    TEST_CONFIGS,
    create_expected_import_dataframe,
    create_invalid_csv_content,
    create_malformed_bookmark_data,
    create_sample_bookmark_objects,
    create_sample_export_dataframe,
    create_sample_processed_bookmark,
)

# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set environment variables for testing
    os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = "true"
    os.environ["BOOKMARK_PROCESSOR_LOG_LEVEL"] = "DEBUG"

    # Disable network access by default
    os.environ["BOOKMARK_PROCESSOR_OFFLINE_MODE"] = "true"


def pytest_unconfigure(config):
    """Clean up after all tests."""
    # Clean up environment variables
    test_env_vars = [
        "BOOKMARK_PROCESSOR_TEST_MODE",
        "BOOKMARK_PROCESSOR_LOG_LEVEL",
        "BOOKMARK_PROCESSOR_OFFLINE_MODE",
    ]

    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


def pytest_runtest_setup(item):
    """Set up before each test."""
    # Skip slow tests unless specifically requested
    if "slow" in item.keywords and not item.config.getoption(
        "--runslow", default=False
    ):
        pytest.skip("need --runslow option to run")

    # Skip network tests in offline mode
    if (
        "network" in item.keywords
        and os.environ.get("BOOKMARK_PROCESSOR_OFFLINE_MODE") == "true"
    ):
        pytest.skip("network tests skipped in offline mode")

    # Skip performance tests unless specifically requested
    if "performance" in item.keywords and not item.config.getoption(
        "--runperformance", default=False
    ):
        pytest.skip("need --runperformance option to run")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runnetwork",
        action="store_true",
        default=False,
        help="run network tests (disable offline mode)",
    )
    parser.addoption(
        "--runperformance",
        action="store_true",
        default=False,
        help="run performance tests (can be time-consuming)",
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="enable benchmarking for performance tests",
    )


# ============================================================================
# Temporary Directory and File Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="bookmark_test_"))
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def temp_csv_file(temp_dir: Path) -> Path:
    """Create a temporary CSV file path."""
    return temp_dir / "test_bookmarks.csv"


@pytest.fixture
def temp_config_file(temp_dir: Path) -> Path:
    """Create a temporary configuration file."""
    config_path = temp_dir / "test_config.ini"

    # Create a basic test configuration
    config_content = """
[network]
timeout = 10
max_retries = 2
default_delay = 0.1
max_concurrent_requests = 5

[processing]
batch_size = 10
max_tags_per_bookmark = 5
target_unique_tags = 50

[ai]
default_engine = local
claude_rpm = 10
openai_rpm = 10

[checkpoint]
enabled = true
save_interval = 5
checkpoint_dir = checkpoints
auto_cleanup = true

[logging]
log_level = DEBUG
console_output = true
"""

    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def temp_checkpoint_dir(temp_dir: Path) -> Path:
    """Create a temporary checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def test_config(temp_config_file: Path) -> Configuration:
    """Create a test configuration instance."""
    config = Configuration(config_file=str(temp_config_file))
    return config


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Minimal configuration for fast tests."""
    return TEST_CONFIGS["minimal"]


@pytest.fixture
def standard_config() -> Dict[str, Any]:
    """Standard configuration for normal tests."""
    return TEST_CONFIGS["standard"]


@pytest.fixture
def performance_config() -> Dict[str, Any]:
    """Performance configuration for load tests."""
    return TEST_CONFIGS["performance"]


# ============================================================================
# Data Fixtures - Raw Data
# ============================================================================


@pytest.fixture
def sample_raindrop_export_data() -> List[Dict[str, str]]:
    """Sample raindrop.io export data."""
    return SAMPLE_RAINDROP_EXPORT_ROWS.copy()


@pytest.fixture
def expected_raindrop_import_data() -> List[Dict[str, str]]:
    """Expected raindrop.io import data."""
    return EXPECTED_RAINDROP_IMPORT_ROWS.copy()


@pytest.fixture
def mock_content_data() -> Dict[str, Dict[str, Any]]:
    """Mock content extraction data."""
    return MOCK_CONTENT_DATA.copy()


@pytest.fixture
def mock_ai_results() -> Dict[str, Dict[str, Any]]:
    """Mock AI processing results."""
    return MOCK_AI_RESULTS.copy()


@pytest.fixture
def malformed_bookmark_data() -> List[Dict[str, Any]]:
    """Malformed bookmark data for testing error handling."""
    return create_malformed_bookmark_data()


@pytest.fixture
def invalid_csv_content() -> str:
    """Invalid CSV content for testing error handling."""
    return create_invalid_csv_content()


# ============================================================================
# Data Fixtures - DataFrames
# ============================================================================


@pytest.fixture
def sample_export_dataframe() -> pd.DataFrame:
    """Sample raindrop.io export DataFrame."""
    return create_sample_export_dataframe()


@pytest.fixture
def expected_import_dataframe() -> pd.DataFrame:
    """Expected raindrop.io import DataFrame."""
    return create_expected_import_dataframe()


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(
        columns=[
            "id",
            "title",
            "note",
            "excerpt",
            "url",
            "folder",
            "tags",
            "created",
            "cover",
            "highlights",
            "favorite",
        ]
    )


@pytest.fixture
def large_sample_dataframe() -> pd.DataFrame:
    """Large sample DataFrame for performance testing."""
    # Create a larger dataset based on the sample data
    base_data = SAMPLE_RAINDROP_EXPORT_ROWS[:3]  # Use first 3 rows

    large_data = []
    for i in range(100):  # Create 100 bookmarks
        for j, row in enumerate(base_data):
            new_row = row.copy()
            new_row["id"] = str(i * len(base_data) + j + 1)
            new_row["url"] = f"{row['url']}?test={i}_{j}"
            new_row["title"] = f"{row['title']} - Test {i}_{j}"
            large_data.append(new_row)

    return pd.DataFrame(large_data)


# ============================================================================
# Data Fixtures - Objects
# ============================================================================


@pytest.fixture
def sample_bookmark_objects() -> List[Bookmark]:
    """Sample Bookmark objects."""
    return create_sample_bookmark_objects()


@pytest.fixture
def sample_processed_bookmark() -> Bookmark:
    """Fully processed bookmark for testing."""
    return create_sample_processed_bookmark()


@pytest.fixture
def valid_bookmark() -> Bookmark:
    """Simple valid bookmark for testing."""
    return Bookmark(
        id="test1",
        title="Test Bookmark",
        url="https://example.com",
        folder="Test",
        tags=["test", "example"],
        note="Test note",
    )


@pytest.fixture
def invalid_bookmark() -> Bookmark:
    """Invalid bookmark for testing error handling."""
    return Bookmark(
        id="invalid", title="", url="", folder="", tags=[], note=""  # Invalid empty URL
    )


@pytest.fixture
def processing_results() -> ProcessingResults:
    """Sample processing results for testing."""
    return ProcessingResults(
        total_bookmarks=100,
        processed_bookmarks=95,
        valid_bookmarks=90,
        invalid_bookmarks=10,
        url_validation_success=92,
        url_validation_failed=8,
        ai_processing_success=85,
        ai_processing_failed=15,
        errors=["Error 1", "Error 2"],
        processing_time=120.5,
    )


# ============================================================================
# Mock Fixtures - Network and External Services
# ============================================================================


@pytest.fixture
def mock_requests_session():
    """Mock requests.Session for URL validation tests."""
    with patch("requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Default successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = (
            "<html><head><title>Test</title></head><body>Test content</body></html>"
        )
        mock_response.history = []

        mock_session.get.return_value = mock_response
        mock_session.head.return_value = mock_response

        yield mock_session


@pytest.fixture
def mock_ai_processor():
    """Mock AI processor for testing without actual AI models."""
    with patch(
        "bookmark_processor.core.ai_processor.EnhancedAIProcessor"
    ) as mock_class:
        mock_processor = Mock()

        # Mock successful AI processing
        def mock_process_batch(bookmarks):
            for bookmark in bookmarks:
                url = bookmark.url
                if url in MOCK_AI_RESULTS:
                    result = MOCK_AI_RESULTS[url]
                    bookmark.enhanced_description = result["enhanced_description"]
                    bookmark.optimized_tags = result["generated_tags"]
                else:
                    bookmark.enhanced_description = (
                        f"AI description for {bookmark.title}"
                    )
                    bookmark.optimized_tags = ["ai", "generated", "tag"]
            return bookmarks

        mock_processor.process_batch = mock_process_batch
        mock_processor.is_available = True
        mock_processor.model_name = "test-model"

        mock_class.return_value = mock_processor
        yield mock_processor


@pytest.fixture
def mock_content_extractor():
    """Mock content extractor for testing without actual web requests."""
    with patch(
        "bookmark_processor.core.content_analyzer.ContentAnalyzer"
    ) as mock_class:
        mock_extractor = Mock()

        def mock_extract_content(url):
            if url in MOCK_CONTENT_DATA:
                data = MOCK_CONTENT_DATA[url]
                return BookmarkMetadata(
                    title=data["title"],
                    description=data["description"],
                    keywords=data["meta_keywords"],
                )
            else:
                return BookmarkMetadata(
                    title=f"Mock title for {url}",
                    description=f"Mock description for {url}",
                    keywords=["mock", "test"],
                )

        mock_extractor.extract_metadata = mock_extract_content
        mock_class.return_value = mock_extractor
        yield mock_extractor


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing without delays."""
    with patch(
        "bookmark_processor.utils.intelligent_rate_limiter.IntelligentRateLimiter"
    ) as mock_class:
        mock_limiter = Mock()
        mock_limiter.get_delay.return_value = 0.0  # No delays in tests
        mock_limiter.should_retry.return_value = True
        mock_limiter.record_request.return_value = None
        mock_limiter.record_error.return_value = None

        mock_class.return_value = mock_limiter
        yield mock_limiter


# ============================================================================
# File System Fixtures
# ============================================================================


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_export_dataframe: pd.DataFrame) -> Path:
    """Create a sample CSV file with test data."""
    csv_path = temp_dir / "sample_export.csv"
    sample_export_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def invalid_csv_file(temp_dir: Path, invalid_csv_content: str) -> Path:
    """Create an invalid CSV file for error testing."""
    csv_path = temp_dir / "invalid.csv"
    csv_path.write_text(invalid_csv_content)
    return csv_path


@pytest.fixture
def empty_csv_file(temp_dir: Path) -> Path:
    """Create an empty CSV file."""
    csv_path = temp_dir / "empty.csv"
    csv_path.write_text("")
    return csv_path


@pytest.fixture
def large_csv_file(temp_dir: Path, large_sample_dataframe: pd.DataFrame) -> Path:
    """Create a large CSV file for performance testing."""
    csv_path = temp_dir / "large_sample.csv"
    large_sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# Integration Test Environment Fixtures
# ============================================================================


@pytest.fixture
def integration_test_environment(
    temp_dir: Path, temp_config_file: Path, temp_checkpoint_dir: Path
):
    """Set up a complete integration test environment."""
    # Create necessary directories
    data_dir = temp_dir / "data"
    logs_dir = temp_dir / "logs"
    output_dir = temp_dir / "output"

    for directory in [data_dir, logs_dir, output_dir]:
        directory.mkdir(exist_ok=True)

    # Set up environment variables for integration testing
    test_env = {
        "BOOKMARK_PROCESSOR_CONFIG": str(temp_config_file),
        "BOOKMARK_PROCESSOR_CHECKPOINT_DIR": str(temp_checkpoint_dir),
        "BOOKMARK_PROCESSOR_TEST_MODE": "true",
        "BOOKMARK_PROCESSOR_LOG_LEVEL": "DEBUG",
        "BOOKMARK_PROCESSOR_DATA_DIR": str(data_dir),
        "BOOKMARK_PROCESSOR_LOGS_DIR": str(logs_dir),
        "BOOKMARK_PROCESSOR_OUTPUT_DIR": str(output_dir),
        "BOOKMARK_PROCESSOR_OFFLINE_MODE": "true",  # Default to offline
    }

    # Store original environment
    original_env = {}
    for key, value in test_env.items():
        if key in os.environ:
            original_env[key] = os.environ[key]
        os.environ[key] = value

    # Create test environment object
    environment = {
        "temp_dir": temp_dir,
        "data_dir": data_dir,
        "logs_dir": logs_dir,
        "output_dir": output_dir,
        "config_file": temp_config_file,
        "checkpoint_dir": temp_checkpoint_dir,
        "env_vars": test_env,
    }

    yield environment

    # Restore original environment
    for key in test_env.keys():
        if key in original_env:
            os.environ[key] = original_env[key]
        elif key in os.environ:
            del os.environ[key]


@pytest.fixture
def isolated_test_environment(integration_test_environment):
    """Create an isolated test environment for each test."""
    # Create unique subdirectories for this test instance
    test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    base_dir = integration_test_environment["temp_dir"]

    test_env = {
        "test_id": test_id,
        "test_dir": base_dir / test_id,
        "input_dir": base_dir / test_id / "input",
        "output_dir": base_dir / test_id / "output",
        "checkpoint_dir": base_dir / test_id / "checkpoints",
        "logs_dir": base_dir / test_id / "logs",
        "config_dir": base_dir / test_id / "config",
    }

    # Create all directories
    for directory in test_env.values():
        if isinstance(directory, Path):
            directory.mkdir(parents=True, exist_ok=True)

    # Copy base environment
    test_env.update(integration_test_environment)

    yield test_env

    # Cleanup is handled by the parent fixture


@pytest.fixture
def database_test_environment(integration_test_environment):
    """Set up test environment with database-like storage simulation."""
    # Create SQLite-like storage for checkpoint testing
    db_dir = integration_test_environment["temp_dir"] / "db"
    db_dir.mkdir(exist_ok=True)

    # Create mock database files
    checkpoint_db = db_dir / "checkpoints.db"
    progress_db = db_dir / "progress.db"

    # Initialize with empty JSON structure (simulating database)
    checkpoint_db.write_text('{"checkpoints": {}, "metadata": {"version": "1.0"}}')
    progress_db.write_text('{"sessions": {}, "statistics": {}}')

    db_env = {
        "db_dir": db_dir,
        "checkpoint_db": checkpoint_db,
        "progress_db": progress_db,
        "BOOKMARK_PROCESSOR_CHECKPOINT_DB": str(checkpoint_db),
        "BOOKMARK_PROCESSOR_PROGRESS_DB": str(progress_db),
    }

    # Update environment
    integration_test_environment.update(db_env)

    yield integration_test_environment


# ============================================================================
# Environment and State Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment before and after each test."""
    # Store original values
    original_env = {}
    
    # Environment variables that should be cleaned between tests
    # but NOT the core test mode variables that should persist
    env_vars_to_clean = [
        "BOOKMARK_PROCESSOR_CONFIG",
        "BOOKMARK_PROCESSOR_CHECKPOINT_DIR",
        "TRANSFORMERS_CACHE",
        "CLAUDE_API_KEY",
        "OPENAI_API_KEY",
        "BOOKMARK_PROCESSOR_DATA_DIR",
        "BOOKMARK_PROCESSOR_LOGS_DIR",
        "BOOKMARK_PROCESSOR_OUTPUT_DIR",
        "BOOKMARK_PROCESSOR_CHECKPOINT_DB",
        "BOOKMARK_PROCESSOR_PROGRESS_DB",
    ]
    
    # Preserve essential test environment variables set by pytest_configure
    essential_test_vars = [
        "BOOKMARK_PROCESSOR_TEST_MODE",
        "BOOKMARK_PROCESSOR_LOG_LEVEL", 
        "BOOKMARK_PROCESSOR_OFFLINE_MODE",
    ]

    for var in env_vars_to_clean:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore original values but don't override essential test variables
    for var, value in original_env.items():
        if var not in essential_test_vars:
            os.environ[var] = value


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    with patch("bookmark_processor.core.data_models.datetime") as mock_dt:
        mock_dt.now.return_value = test_datetime
        mock_dt.utcnow.return_value = test_datetime
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield test_datetime


# ============================================================================
# Integration Test Helpers and Utilities
# ============================================================================


class IntegrationTestHelper:
    """Helper class for integration testing."""

    def __init__(self, test_environment: Dict[str, Any]):
        self.env = test_environment
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for integration tests."""
        import logging

        log_file = self.env.get("logs_dir", Path()) / "integration_test.log"

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("integration_test")

    def create_test_data_file(self, filename: str, data: List[Dict[str, str]]) -> Path:
        """Create a test data file in the test environment."""
        file_path = self.env["input_dir"] / filename
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Created test data file: {file_path}")
        return file_path

    def create_checkpoint_scenario(
        self, checkpoint_id: str, progress: Dict[str, Any]
    ) -> Path:
        """Create a checkpoint scenario for testing resume functionality."""
        checkpoint_file = self.env["checkpoint_dir"] / f"{checkpoint_id}.json"
        import json

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "progress": progress,
            "metadata": {"test_mode": True, "created_by": "integration_test"},
        }

        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
        self.logger.info(f"Created checkpoint scenario: {checkpoint_file}")
        return checkpoint_file

    def simulate_network_conditions(self, condition: str) -> Dict[str, Any]:
        """Simulate different network conditions for testing."""
        conditions = {
            "fast": {"delay": 0.1, "timeout": 5, "error_rate": 0.01},
            "slow": {"delay": 2.0, "timeout": 30, "error_rate": 0.05},
            "unstable": {"delay": 1.0, "timeout": 10, "error_rate": 0.15},
            "offline": {"delay": 0, "timeout": 1, "error_rate": 1.0},
        }

        return conditions.get(condition, conditions["fast"])

    def verify_output_structure(self, output_file: Path) -> bool:
        """Verify that output file has correct structure."""
        if not output_file.exists():
            return False

        try:
            df = pd.read_csv(output_file)
            expected_columns = ["url", "folder", "title", "note", "tags", "created"]
            return list(df.columns) == expected_columns
        except Exception as e:
            self.logger.error(f"Error verifying output structure: {e}")
            return False

    def collect_test_artifacts(self) -> Dict[str, Path]:
        """Collect all test artifacts for analysis."""
        artifacts = {}

        # Log files
        log_files = list(self.env["logs_dir"].glob("*.log"))
        if log_files:
            artifacts["logs"] = log_files

        # Checkpoint files
        checkpoint_files = list(self.env["checkpoint_dir"].glob("*.json"))
        if checkpoint_files:
            artifacts["checkpoints"] = checkpoint_files

        # Output files
        output_files = list(self.env["output_dir"].glob("*.csv"))
        if output_files:
            artifacts["outputs"] = output_files

        return artifacts


@pytest.fixture
def integration_test_helper(integration_test_environment):
    """Create an integration test helper instance."""
    return IntegrationTestHelper(integration_test_environment)


class ProcessingScenario:
    """Represents a processing scenario for integration testing."""

    def __init__(
        self, name: str, config: Dict[str, Any], expected_results: Dict[str, Any]
    ):
        self.name = name
        self.config = config
        self.expected_results = expected_results
        self.actual_results = None
        self.artifacts = {}

    def run(self, processor, input_file: Path, output_file: Path):
        """Run the processing scenario."""
        # This will be implemented by specific test cases
        raise NotImplementedError

    def verify_results(self) -> bool:
        """Verify that actual results match expected results."""
        if not self.actual_results:
            return False

        for key, expected_value in self.expected_results.items():
            actual_value = getattr(self.actual_results, key, None)
            if actual_value != expected_value:
                return False

        return True


# Standard processing scenarios for reuse across tests
STANDARD_SCENARIOS = {
    "minimal_processing": ProcessingScenario(
        name="Minimal Processing",
        config={
            "batch_size": 5,
            "max_retries": 1,
            "timeout": 5,
            "enable_ai_processing": False,
            "enable_checkpoints": False,
        },
        expected_results={"should_complete": True, "min_success_rate": 0.8},
    ),
    "full_processing": ProcessingScenario(
        name="Full Processing with AI",
        config={
            "batch_size": 10,
            "max_retries": 3,
            "timeout": 30,
            "enable_ai_processing": True,
            "enable_checkpoints": True,
        },
        expected_results={"should_complete": True, "min_success_rate": 0.7},
    ),
    "error_resilience": ProcessingScenario(
        name="Error Resilience",
        config={
            "batch_size": 3,
            "max_retries": 2,
            "timeout": 5,
            "enable_ai_processing": False,
            "enable_checkpoints": True,
            "simulate_errors": True,
        },
        expected_results={"should_complete": True, "min_success_rate": 0.5},
    ),
}


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_bookmark_valid(bookmark: Bookmark) -> None:
    """Assert that a bookmark is valid."""
    assert bookmark.is_valid(), f"Bookmark should be valid: {bookmark}"
    assert bookmark.url, "Bookmark must have a URL"
    assert bookmark.get_effective_title(), "Bookmark must have a title"


def assert_dataframe_structure(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """Assert that a DataFrame has the expected structure."""
    assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"
    assert (
        list(df.columns) == expected_columns
    ), f"Expected columns {expected_columns}, got {list(df.columns)}"


def assert_processing_results_valid(results: ProcessingResults) -> None:
    """Assert that processing results are valid."""
    assert results.total_bookmarks >= 0, "Total bookmarks should be non-negative"
    assert (
        results.processed_bookmarks <= results.total_bookmarks
    ), "Processed should not exceed total"
    assert (
        results.valid_bookmarks <= results.processed_bookmarks
    ), "Valid should not exceed processed"
    assert isinstance(results.errors, list), "Errors should be a list"
    assert results.processing_time >= 0, "Processing time should be non-negative"


def assert_integration_test_complete(
    test_helper: IntegrationTestHelper,
    input_file: Path,
    output_file: Path,
    results: ProcessingResults,
) -> None:
    """Assert that an integration test completed successfully."""
    # Input file should exist and be readable
    assert input_file.exists(), f"Input file should exist: {input_file}"
    input_df = pd.read_csv(input_file)
    assert len(input_df) > 0, "Input file should contain data"

    # Output file should be created with correct structure
    assert output_file.exists(), f"Output file should be created: {output_file}"
    assert test_helper.verify_output_structure(
        output_file
    ), "Output should have correct structure"

    # Results should be valid
    assert_processing_results_valid(results)

    # Processing should have made progress
    assert results.processed_bookmarks > 0, "Should have processed some bookmarks"


def assert_checkpoint_functionality(
    checkpoint_dir: Path, initial_count: int, resumed_count: int
) -> None:
    """Assert that checkpoint functionality works correctly."""
    # Checkpoint files should exist
    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoint_files) > 0, "Checkpoint files should be created"

    # Resumed processing should continue from where it left off
    assert (
        resumed_count >= initial_count
    ), "Resumed processing should continue from checkpoint"


def assert_error_handling(
    results: ProcessingResults, expected_error_types: List[str]
) -> None:
    """Assert that error handling works correctly."""
    # Should have some errors if we're testing error handling
    assert len(results.errors) > 0, "Should have recorded errors"

    # Should continue processing despite errors
    assert results.processed_bookmarks > 0, "Should continue processing despite errors"

    # Error types should be as expected
    for error_type in expected_error_types:
        assert any(
            error_type in str(error) for error in results.errors
        ), f"Should have {error_type} errors"


# ============================================================================
# Parametrized Test Data
# ============================================================================

# URL test cases for parametrized tests
URL_TEST_CASES = [
    # (input_url, expected_normalized, is_valid)
    ("https://example.com", "https://example.com", True),
    ("http://example.com", "http://example.com", True),
    ("example.com", "https://example.com", True),
    ("www.example.com", "https://www.example.com", True),
    ("", "", False),
    ("not-a-url", "not-a-url", False),
    ("javascript:void(0)", "javascript:void(0)", False),
    ("mailto:test@example.com", "mailto:test@example.com", False),
    ("https://EXAMPLE.COM/PATH", "https://example.com/PATH", True),
    ("https://example.com:443/", "https://example.com", True),
    ("https://example.com///path//", "https://example.com/path", True),
]

# Tag formatting test cases
TAG_FORMAT_TEST_CASES = [
    # (tags_list, expected_formatted)
    ([], ""),
    (["single"], "single"),
    (["tag1", "tag2"], "tag1, tag2"),
    (["tag1", "tag2", "tag3"], "tag1, tag2, tag3"),
    (["  spaced  ", "tags"], "spaced, tags"),
    (["", "valid", ""], "valid"),
]

# Date parsing test cases
DATE_PARSING_TEST_CASES = [
    # (date_string, expected_datetime_or_none)
    ("", None),
    ("2024-01-01T00:00:00Z", datetime(2024, 1, 1, tzinfo=timezone.utc)),
    (
        "2024-01-01T12:30:45+00:00",
        datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc),
    ),
    ("invalid-date", None),
    ("2024-13-01T00:00:00Z", None),  # Invalid month
]


# ============================================================================
# Performance Testing Fixtures
# ============================================================================


import time
import psutil
from typing import Generator, Dict


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process()

    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        self.end_time = time.time()
        self.update_peak_memory()

        return {
            "duration_seconds": self.end_time - self.start_time,
            "start_memory_mb": self.start_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": self.peak_memory - self.start_memory,
        }

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics without stopping."""
        self.update_peak_memory()
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        return {
            "elapsed_seconds": current_time - self.start_time if self.start_time else 0,
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory,
        }


@pytest.fixture
def performance_monitor() -> Generator[PerformanceMonitor, None, None]:
    """Fixture for monitoring performance during tests."""
    monitor = PerformanceMonitor()
    monitor.start()
    yield monitor
    metrics = monitor.stop()

    # Log metrics
    print(f"\n=== Performance Metrics ===")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    print(f"Start Memory: {metrics['start_memory_mb']:.2f} MB")
    print(f"Peak Memory: {metrics['peak_memory_mb']:.2f} MB")
    print(f"Memory Increase: {metrics['memory_increase_mb']:.2f} MB")


@pytest.fixture
def benchmark_timer():
    """Simple timing fixture for performance benchmarks."""

    class Timer:
        def __init__(self):
            self.times = {}
            self.current_label = None
            self.start_time = None

        def start(self, label: str = "default"):
            """Start timing for a labeled operation."""
            self.current_label = label
            self.start_time = time.time()

        def stop(self) -> float:
            """Stop timing and return elapsed time."""
            if self.start_time is None:
                return 0.0
            elapsed = time.time() - self.start_time
            if self.current_label:
                self.times[self.current_label] = elapsed
            self.start_time = None
            return elapsed

        def get_time(self, label: str) -> float:
            """Get time for a specific label."""
            return self.times.get(label, 0.0)

        def get_all_times(self) -> Dict[str, float]:
            """Get all recorded times."""
            return self.times.copy()

        def print_summary(self):
            """Print a summary of all timings."""
            print("\n=== Timing Summary ===")
            for label, duration in sorted(self.times.items()):
                print(f"{label}: {duration:.4f} seconds")

    return Timer()


@pytest.fixture
def performance_test_data_dir(temp_dir: Path) -> Path:
    """Create directory for performance test data."""
    perf_dir = temp_dir / "performance_data"
    perf_dir.mkdir(exist_ok=True)
    return perf_dir


@pytest.fixture
def generated_test_files(
    performance_test_data_dir: Path,
) -> Generator[Dict[str, Path], None, None]:
    """Generate test files for performance testing on demand."""
    from tests.fixtures.generate_test_data import TestDataGenerator

    generator = TestDataGenerator(seed=42)
    test_files = generator.generate_performance_suite(performance_test_data_dir)

    yield test_files

    # Cleanup is handled by temp_dir fixture


@pytest.fixture
def performance_baseline() -> Dict[str, Any]:
    """Define baseline performance expectations."""
    return {
        "small_test": {
            "size": 100,
            "max_duration_seconds": 60,  # 1 minute
            "max_memory_mb": 1000,  # 1 GB
            "min_throughput_per_hour": 100,
        },
        "medium_test": {
            "size": 1000,
            "max_duration_seconds": 600,  # 10 minutes
            "max_memory_mb": 2000,  # 2 GB
            "min_throughput_per_hour": 300,
        },
        "large_test": {
            "size": 3500,
            "max_duration_seconds": 28800,  # 8 hours
            "max_memory_mb": 4000,  # 4 GB
            "min_throughput_per_hour": 400,
        },
    }


class PerformanceAssertion:
    """Helper for asserting performance metrics."""

    @staticmethod
    def assert_duration_within_limit(
        actual_seconds: float, max_seconds: float, test_name: str = "test"
    ):
        """Assert that duration is within acceptable limits."""
        assert (
            actual_seconds <= max_seconds
        ), f"{test_name} took {actual_seconds:.2f}s, exceeds limit of {max_seconds}s"

    @staticmethod
    def assert_memory_within_limit(
        peak_memory_mb: float, max_memory_mb: float, test_name: str = "test"
    ):
        """Assert that memory usage is within acceptable limits."""
        assert (
            peak_memory_mb <= max_memory_mb
        ), f"{test_name} used {peak_memory_mb:.2f}MB, exceeds limit of {max_memory_mb}MB"

    @staticmethod
    def assert_throughput_above_minimum(
        processed_count: int,
        duration_seconds: float,
        min_per_hour: float,
        test_name: str = "test",
    ):
        """Assert that processing throughput meets minimum requirements."""
        if duration_seconds == 0:
            return  # Skip if test was too fast

        actual_per_hour = (processed_count / duration_seconds) * 3600
        assert actual_per_hour >= min_per_hour, (
            f"{test_name} throughput was {actual_per_hour:.2f} items/hour, "
            f"below minimum of {min_per_hour} items/hour"
        )

    @staticmethod
    def assert_success_rate_above_threshold(
        successful: int, total: int, min_rate: float = 0.8, test_name: str = "test"
    ):
        """Assert that success rate meets minimum threshold."""
        if total == 0:
            return  # Skip if no items processed

        actual_rate = successful / total
        assert actual_rate >= min_rate, (
            f"{test_name} success rate was {actual_rate:.2%}, "
            f"below minimum of {min_rate:.2%}"
        )


@pytest.fixture
def performance_assertion() -> PerformanceAssertion:
    """Fixture for performance assertions."""
    return PerformanceAssertion()


# Export these for use in tests
__all__ = [
    "temp_dir",
    "temp_csv_file",
    "temp_config_file",
    "temp_checkpoint_dir",
    "test_config",
    "minimal_config",
    "standard_config",
    "performance_config",
    "sample_raindrop_export_data",
    "expected_raindrop_import_data",
    "mock_content_data",
    "mock_ai_results",
    "malformed_bookmark_data",
    "sample_export_dataframe",
    "expected_import_dataframe",
    "empty_dataframe",
    "large_sample_dataframe",
    "sample_bookmark_objects",
    "sample_processed_bookmark",
    "valid_bookmark",
    "invalid_bookmark",
    "processing_results",
    "mock_requests_session",
    "mock_ai_processor",
    "mock_content_extractor",
    "mock_rate_limiter",
    "sample_csv_file",
    "invalid_csv_file",
    "empty_csv_file",
    "large_csv_file",
    "clean_environment",
    "mock_datetime",
    "assert_bookmark_valid",
    "assert_dataframe_structure",
    "assert_processing_results_valid",
    "URL_TEST_CASES",
    "TAG_FORMAT_TEST_CASES",
    "DATE_PARSING_TEST_CASES",
    "performance_monitor",
    "benchmark_timer",
    "performance_test_data_dir",
    "generated_test_files",
    "performance_baseline",
    "performance_assertion",
    "PerformanceMonitor",
    "PerformanceAssertion",
]
