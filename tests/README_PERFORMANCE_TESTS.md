# Performance Testing Suite

## Overview

This directory contains comprehensive end-to-end performance tests for the bookmark processor. These tests validate that the system can handle various load levels while meeting performance requirements for processing time, memory usage, and throughput.

## Test Files

### Core Files

- **`test_performance_e2e.py`**: Main performance test suite with tests for small (100), medium (1000), and large (3500+) datasets
- **`fixtures/generate_test_data.py`**: Test data generator for creating realistic raindrop.io CSV exports
- **`conftest.py`**: Enhanced with performance monitoring fixtures and utilities

## Test Categories

### Small Tests (100 bookmarks)
- Quick sanity checks
- Checkpoint functionality validation
- Error handling performance
- Batch size impact analysis

### Medium Tests (1000 bookmarks)
- Medium load validation
- Memory stability testing
- Performance baseline regression tests

### Large Tests (3500+ bookmarks)
- Full load simulation matching production requirements
- Extended checkpoint/resume testing
- 8-hour processing validation

## Setup

### Prerequisites

Install all dependencies including performance monitoring tools:

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/WSL
# or
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Verify psutil is installed (required for memory monitoring)
python -c "import psutil; print('psutil version:', psutil.__version__)"
```

### Generate Test Data

Generate the complete performance test suite:

```bash
# Generate all test files (100, 1000, 3500 bookmarks + specialized tests)
python -m tests.fixtures.generate_test_data --suite

# Or generate a custom size file
python -m tests.fixtures.generate_test_data --size 500 --output tests/fixtures/performance_data/custom_500.csv
```

This creates test files in `tests/fixtures/performance_data/`:
- `performance_test_100.csv` - Small dataset
- `performance_test_1000.csv` - Medium dataset
- `performance_test_3500.csv` - Large dataset
- `performance_test_duplicates.csv` - Dataset with duplicates
- `performance_test_clean.csv` - All valid bookmarks

## Running Performance Tests

### Quick Start

```bash
# Run all performance tests (requires --runperformance flag)
pytest tests/test_performance_e2e.py --runperformance -v

# Run only small tests (fast, < 2 minutes)
pytest tests/test_performance_e2e.py --runperformance -v -m "small"

# Run medium tests (moderate, 5-10 minutes)
pytest tests/test_performance_e2e.py --runperformance -v -m "medium"

# Run large tests (slow, may take hours)
pytest tests/test_performance_e2e.py --runperformance -v -m "large"
```

### Test Markers

Performance tests use pytest markers for fine-grained control:

- `@pytest.mark.performance` - All performance tests
- `@pytest.mark.small` - Small dataset tests (100 bookmarks)
- `@pytest.mark.medium` - Medium dataset tests (1000 bookmarks)
- `@pytest.mark.large` - Large dataset tests (3500+ bookmarks)
- `@pytest.mark.slow` - Tests that take > 1 minute

### Running Specific Test Scenarios

```bash
# Run only checkpoint/resume tests
pytest tests/test_performance_e2e.py::test_small_dataset_with_checkpoints --runperformance -v
pytest tests/test_performance_e2e.py::test_large_dataset_checkpoint_resume --runperformance -v

# Run memory stability tests
pytest tests/test_performance_e2e.py::test_medium_dataset_memory_stability --runperformance -v

# Run duplicate detection performance
pytest tests/test_performance_e2e.py::test_duplicate_detection_performance --runperformance -v

# Run batch size comparison
pytest tests/test_performance_e2e.py::test_batch_size_impact --runperformance -v
```

### Advanced Options

```bash
# Run with verbose output and show print statements
pytest tests/test_performance_e2e.py --runperformance -v -s

# Run and capture performance output
pytest tests/test_performance_e2e.py --runperformance -v --tb=short > performance_results.txt 2>&1

# Run with pytest-benchmark integration (if available)
pytest tests/test_performance_e2e.py --runperformance -v --benchmark

# Skip slow tests
pytest tests/test_performance_e2e.py --runperformance -v -m "not slow"
```

## Performance Baselines

The tests enforce the following performance baselines:

### Small Dataset (100 bookmarks)
- **Max Duration**: 60 seconds
- **Max Memory**: 1 GB
- **Min Throughput**: 100 bookmarks/hour

### Medium Dataset (1000 bookmarks)
- **Max Duration**: 600 seconds (10 minutes)
- **Max Memory**: 2 GB
- **Min Throughput**: 300 bookmarks/hour

### Large Dataset (3500 bookmarks)
- **Max Duration**: 28,800 seconds (8 hours)
- **Max Memory**: 4 GB
- **Min Throughput**: 400 bookmarks/hour

## Performance Metrics Tracked

Each test monitors and reports:

1. **Processing Time**
   - Total duration in seconds
   - Throughput (bookmarks per hour)

2. **Memory Usage**
   - Start memory (MB)
   - Peak memory (MB)
   - Memory increase during processing

3. **Success Rates**
   - Bookmarks processed
   - Valid vs invalid bookmarks
   - Error rates

4. **Checkpoint Performance**
   - Checkpoint save frequency
   - Resume time
   - Checkpoint overhead

## Understanding Test Output

### Sample Output

```
=== Performance Metrics ===
Duration: 45.23 seconds
Start Memory: 234.56 MB
Peak Memory: 567.89 MB
Memory Increase: 333.33 MB

=== Small Dataset Performance ===
Processed: 98 bookmarks
Duration: 45.23 seconds
Memory: 567.89 MB
Throughput: 7798.23 bookmarks/hour
```

### Interpreting Results

- **Green (PASSED)**: Test completed within performance limits
- **Red (FAILED)**: Performance regression detected or limits exceeded
- **Metrics**: Compare actual values against baselines

## Troubleshooting

### Tests are Skipped

If you see "need --runperformance option to run":

```bash
# Make sure to include the --runperformance flag
pytest tests/test_performance_e2e.py --runperformance -v
```

### Memory Errors

If tests fail due to memory issues:

1. Check system resources: `free -h` (Linux) or Task Manager (Windows)
2. Close other applications
3. Reduce batch sizes in test configuration
4. Run smaller test subsets

### Test Data Not Found

If tests fail because CSV files are missing:

```bash
# Regenerate test data
python -m tests.fixtures.generate_test_data --suite
```

### Slow Performance

If tests are running slower than expected:

1. Check that mocks are properly configured (network requests should be mocked)
2. Verify you're not running other resource-intensive processes
3. Check disk I/O performance
4. Review test output for bottlenecks

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  workflow_dispatch:  # Manual trigger

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Generate test data
        run: |
          python -m tests.fixtures.generate_test_data --suite
      - name: Run performance tests
        run: |
          pytest tests/test_performance_e2e.py --runperformance -v -m "small or medium"
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: performance-results
          path: performance_results.txt
```

## Customization

### Adjusting Baselines

Edit `conftest.py` to modify performance baselines:

```python
@pytest.fixture
def performance_baseline() -> Dict[str, Any]:
    return {
        "small_test": {
            "size": 100,
            "max_duration_seconds": 60,
            "max_memory_mb": 1000,
            "min_throughput_per_hour": 100,
        },
        # ... adjust as needed
    }
```

### Custom Test Data

Generate custom test datasets:

```python
from tests.fixtures.generate_test_data import TestDataGenerator

generator = TestDataGenerator(seed=42)
bookmarks = generator.generate_dataset(
    size=500,
    include_invalid=True,
    duplicate_rate=0.05  # 5% duplicates
)
generator.save_to_csv(bookmarks, Path("custom_test.csv"))
```

## Best Practices

1. **Run Regularly**: Schedule weekly or monthly performance test runs
2. **Track Trends**: Save and compare performance metrics over time
3. **Isolate Environment**: Run on dedicated test machines when possible
4. **Monitor Resources**: Use system monitoring tools during long tests
5. **Document Changes**: Note performance impacts in commit messages

## Contributing

When adding new performance tests:

1. Use appropriate markers (`@pytest.mark.small`, `@pytest.mark.medium`, etc.)
2. Include performance assertions using `performance_assertion` fixture
3. Document expected performance characteristics
4. Add to this README if introducing new test categories

## Related Documentation

- [Main README](../README.md) - Project overview
- [Testing Guide](../docs/testing.md) - General testing guidelines
- [CLAUDE.md](../CLAUDE.md) - AI development guide with performance requirements

## Support

For issues or questions about performance testing:

1. Check test output and logs
2. Review this documentation
3. Check existing GitHub issues
4. Create a new issue with performance metrics and system details
