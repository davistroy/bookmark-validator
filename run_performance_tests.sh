#!/bin/bash
# Performance Test Runner for Bookmark Processor
# This script generates test data and runs performance tests with various configurations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Bookmark Processor - Performance Tests${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Virtual environment not detected"
    if [ -d "venv" ]; then
        echo "Attempting to activate virtual environment..."
        source venv/bin/activate
        print_status "Virtual environment activated"
    elif [ -d "test_env" ]; then
        echo "Attempting to activate test_env virtual environment..."
        source test_env/bin/activate
        print_status "Test environment activated"
    else
        print_error "No virtual environment found. Please create one first:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import pytest" 2>/dev/null || {
    print_error "pytest not installed. Installing test dependencies..."
    pip install pytest pytest-cov psutil
}

python -c "import psutil" 2>/dev/null || {
    print_error "psutil not installed. Installing..."
    pip install psutil
}

python -c "import pandas" 2>/dev/null || {
    print_error "pandas not installed. Installing core dependencies..."
    pip install -r requirements.txt
}

print_status "All dependencies installed"
echo

# Generate test data
echo -e "${BLUE}Generating test data...${NC}"
if [ ! -d "tests/fixtures/performance_data" ]; then
    print_warning "Performance test data not found. Generating..."
    python -m tests.fixtures.generate_test_data --suite
    print_status "Test data generated"
else
    print_status "Test data already exists"
fi
echo

# Parse command line arguments
TEST_LEVEL="${1:-small}"
VERBOSE="${2:-}"

# Run tests based on level
case "$TEST_LEVEL" in
    small)
        echo -e "${BLUE}Running SMALL performance tests (100 bookmarks)...${NC}"
        echo "Expected duration: < 2 minutes"
        echo
        if [ "$VERBOSE" == "-v" ] || [ "$VERBOSE" == "--verbose" ]; then
            pytest tests/test_performance_e2e.py --runperformance -v -s -m "small"
        else
            pytest tests/test_performance_e2e.py --runperformance -v -m "small"
        fi
        ;;

    medium)
        echo -e "${BLUE}Running MEDIUM performance tests (1000 bookmarks)...${NC}"
        echo "Expected duration: 5-10 minutes"
        echo
        if [ "$VERBOSE" == "-v" ] || [ "$VERBOSE" == "--verbose" ]; then
            pytest tests/test_performance_e2e.py --runperformance -v -s -m "medium"
        else
            pytest tests/test_performance_e2e.py --runperformance -v -m "medium"
        fi
        ;;

    large)
        echo -e "${YELLOW}WARNING: Large performance tests can take several hours${NC}"
        echo "Expected duration: up to 8 hours for full dataset"
        echo
        read -p "Do you want to continue? (yes/no): " -n 3 -r
        echo
        if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            echo -e "${BLUE}Running LARGE performance tests (3500+ bookmarks)...${NC}"
            if [ "$VERBOSE" == "-v" ] || [ "$VERBOSE" == "--verbose" ]; then
                pytest tests/test_performance_e2e.py --runperformance -v -s -m "large"
            else
                pytest tests/test_performance_e2e.py --runperformance -v -m "large"
            fi
        else
            print_warning "Large tests cancelled"
            exit 0
        fi
        ;;

    all)
        echo -e "${BLUE}Running ALL performance tests...${NC}"
        echo "Expected duration: varies (small + medium tests)"
        echo
        if [ "$VERBOSE" == "-v" ] || [ "$VERBOSE" == "--verbose" ]; then
            pytest tests/test_performance_e2e.py --runperformance -v -s
        else
            pytest tests/test_performance_e2e.py --runperformance -v
        fi
        ;;

    quick)
        echo -e "${BLUE}Running QUICK performance validation...${NC}"
        echo "Running a single small test for quick validation"
        echo
        pytest tests/test_performance_e2e.py::test_small_dataset_performance --runperformance -v
        ;;

    help|--help|-h)
        echo "Usage: $0 [TEST_LEVEL] [OPTIONS]"
        echo
        echo "TEST_LEVEL:"
        echo "  small     Run small dataset tests (100 bookmarks) - default"
        echo "  medium    Run medium dataset tests (1000 bookmarks)"
        echo "  large     Run large dataset tests (3500+ bookmarks)"
        echo "  all       Run all performance tests"
        echo "  quick     Run a single quick validation test"
        echo "  help      Show this help message"
        echo
        echo "OPTIONS:"
        echo "  -v, --verbose    Show detailed output including print statements"
        echo
        echo "Examples:"
        echo "  $0                    # Run small tests (default)"
        echo "  $0 small -v           # Run small tests with verbose output"
        echo "  $0 medium             # Run medium tests"
        echo "  $0 large              # Run large tests (with confirmation)"
        echo "  $0 all                # Run all tests"
        echo "  $0 quick              # Quick validation"
        echo
        exit 0
        ;;

    *)
        print_error "Unknown test level: $TEST_LEVEL"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo
    print_status "Performance tests completed successfully!"
    echo
    echo -e "${GREEN}Next steps:${NC}"
    echo "  - Review test output above for performance metrics"
    echo "  - Check tests/README_PERFORMANCE_TESTS.md for detailed documentation"
    echo "  - Run with -v flag for verbose output: $0 $TEST_LEVEL -v"
else
    echo
    print_error "Performance tests failed!"
    echo
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  - Check test output above for error messages"
    echo "  - Verify all dependencies are installed: pip install -r requirements.txt"
    echo "  - Ensure test data exists: python -m tests.fixtures.generate_test_data --suite"
    echo "  - See tests/README_PERFORMANCE_TESTS.md for more help"
    exit 1
fi
