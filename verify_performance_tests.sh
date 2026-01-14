#!/bin/bash
# Quick verification script for performance test suite
# This checks syntax and structure without running full tests

set -e

echo "================================================"
echo "Performance Test Suite - Verification"
echo "================================================"
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() {
    echo -e "${GREEN}✓${NC} $1"
}

fail() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

# Check Python syntax
echo "Checking Python syntax..."

python -m py_compile tests/test_performance_e2e.py 2>/dev/null && \
    pass "test_performance_e2e.py syntax OK" || \
    fail "test_performance_e2e.py syntax error"

python -m py_compile tests/fixtures/generate_test_data.py 2>/dev/null && \
    pass "generate_test_data.py syntax OK" || \
    fail "generate_test_data.py syntax error"

python -m py_compile tests/conftest.py 2>/dev/null && \
    pass "conftest.py syntax OK" || \
    fail "conftest.py syntax error"

echo

# Check file structure
echo "Checking file structure..."

[ -f "tests/test_performance_e2e.py" ] && \
    pass "test_performance_e2e.py exists" || \
    fail "test_performance_e2e.py missing"

[ -f "tests/fixtures/generate_test_data.py" ] && \
    pass "generate_test_data.py exists" || \
    fail "generate_test_data.py missing"

[ -f "tests/README_PERFORMANCE_TESTS.md" ] && \
    pass "README_PERFORMANCE_TESTS.md exists" || \
    fail "README_PERFORMANCE_TESTS.md missing"

[ -f "run_performance_tests.sh" ] && \
    pass "run_performance_tests.sh exists" || \
    fail "run_performance_tests.sh missing"

[ -f "PERFORMANCE_TEST_SUMMARY.md" ] && \
    pass "PERFORMANCE_TEST_SUMMARY.md exists" || \
    fail "PERFORMANCE_TEST_SUMMARY.md missing"

[ -x "run_performance_tests.sh" ] && \
    pass "run_performance_tests.sh is executable" || \
    fail "run_performance_tests.sh not executable"

echo

# Check pytest markers in pyproject.toml
echo "Checking pytest configuration..."

if grep -q "performance: Performance and load tests" pyproject.toml; then
    pass "Performance marker configured"
else
    fail "Performance marker missing in pyproject.toml"
fi

if grep -q "small: Small dataset performance tests" pyproject.toml; then
    pass "Small marker configured"
else
    fail "Small marker missing in pyproject.toml"
fi

if grep -q "medium: Medium dataset performance tests" pyproject.toml; then
    pass "Medium marker configured"
else
    fail "Medium marker missing in pyproject.toml"
fi

if grep -q "large: Large dataset performance tests" pyproject.toml; then
    pass "Large marker configured"
else
    fail "Large marker missing in pyproject.toml"
fi

echo

# Check test structure
echo "Checking test structure..."

if grep -q "pytest.mark.performance" tests/test_performance_e2e.py; then
    pass "Performance markers present in tests"
else
    fail "Performance markers missing in tests"
fi

if grep -q "def test_small_dataset_performance" tests/test_performance_e2e.py; then
    pass "Small dataset tests found"
else
    fail "Small dataset tests missing"
fi

if grep -q "def test_medium_dataset_performance" tests/test_performance_e2e.py; then
    pass "Medium dataset tests found"
else
    fail "Medium dataset tests missing"
fi

if grep -q "def test_large_dataset_performance" tests/test_performance_e2e.py; then
    pass "Large dataset tests found"
else
    fail "Large dataset tests missing"
fi

echo

# Check fixtures
echo "Checking performance fixtures..."

if grep -q "def performance_monitor" tests/conftest.py; then
    pass "performance_monitor fixture found"
else
    fail "performance_monitor fixture missing"
fi

if grep -q "def performance_baseline" tests/conftest.py; then
    pass "performance_baseline fixture found"
else
    fail "performance_baseline fixture missing"
fi

if grep -q "class PerformanceMonitor" tests/conftest.py; then
    pass "PerformanceMonitor class found"
else
    fail "PerformanceMonitor class missing"
fi

if grep -q "class PerformanceAssertion" tests/conftest.py; then
    pass "PerformanceAssertion class found"
else
    fail "PerformanceAssertion class missing"
fi

echo

# Check test data generator
echo "Checking test data generator..."

if grep -q "class TestDataGenerator" tests/fixtures/generate_test_data.py; then
    pass "TestDataGenerator class found"
else
    fail "TestDataGenerator class missing"
fi

if grep -q "def generate_dataset" tests/fixtures/generate_test_data.py; then
    pass "generate_dataset method found"
else
    fail "generate_dataset method missing"
fi

if grep -q "def generate_performance_suite" tests/fixtures/generate_test_data.py; then
    pass "generate_performance_suite method found"
else
    fail "generate_performance_suite method missing"
fi

echo
echo "================================================"
echo -e "${GREEN}All verifications passed!${NC}"
echo "================================================"
echo
echo "Performance test suite is properly configured."
echo
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Generate test data: python -m tests.fixtures.generate_test_data --suite"
echo "  3. Run tests: ./run_performance_tests.sh quick"
echo
echo "See PERFORMANCE_TEST_SUMMARY.md for complete documentation."
