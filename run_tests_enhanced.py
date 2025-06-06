#!/usr/bin/env python3
"""
Enhanced test runner for bookmark processor.

Provides comprehensive test execution with coverage reporting and CI integration.
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


def setup_environment():
    """Set up test environment variables."""
    os.environ['BOOKMARK_PROCESSOR_TEST_MODE'] = 'true'
    os.environ['BOOKMARK_PROCESSOR_LOG_LEVEL'] = 'DEBUG'
    os.environ['BOOKMARK_PROCESSOR_OFFLINE_MODE'] = 'true'
    
    # Ensure test directories exist
    test_dirs = [
        'htmlcov',
        'junit',
        '.pytest_cache'
    ]
    
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)


def install_dependencies():
    """Install test dependencies if needed."""
    print("📦 Installing test dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            '-r', 'requirements-dev.txt'
        ], check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Output: {e.stdout.decode()}")
        print(f"Error: {e.stderr.decode()}")
        return False


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        print(f"✅ {description} completed in {duration:.2f}s")
        
        # Print output for important commands
        if result.stdout and ('pytest' in cmd[0] or 'coverage' in cmd):
            print(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ {description} failed after {duration:.2f}s")
        print(f"Exit code: {e.returncode}")
        
        if e.stdout:
            print(f"Output:\n{e.stdout}")
        if e.stderr:
            print(f"Error:\n{e.stderr}")
        
        return False


def run_linting() -> bool:
    """Run code linting checks."""
    print("\n🔍 Running code quality checks...")
    
    checks = [
        ([sys.executable, '-m', 'black', '--check', 'bookmark_processor', 'tests'], 
         "Code formatting check (black)"),
        ([sys.executable, '-m', 'isort', '--check-only', 'bookmark_processor', 'tests'], 
         "Import sorting check (isort)"),
        ([sys.executable, '-m', 'flake8', 'bookmark_processor', 'tests'], 
         "Linting check (flake8)"),
        ([sys.executable, '-m', 'mypy', 'bookmark_processor', '--ignore-missing-imports'], 
         "Type checking (mypy)"),
        ([sys.executable, '-m', 'bandit', '-r', 'bookmark_processor', '-ll'], 
         "Security check (bandit)")
    ]
    
    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def run_tests(test_type: str, coverage: bool, verbose: bool, fast: bool, file: Optional[str]) -> bool:
    """Run tests with specified parameters."""
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test path
    if file:
        cmd.append(file)
    else:
        cmd.append('tests/')
    
    # Add test type markers
    if test_type == 'unit':
        cmd.extend(['-m', 'unit or not integration'])
    elif test_type == 'integration':
        cmd.extend(['-m', 'integration'])
    elif test_type == 'all':
        pass  # Run all tests
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            '--cov=bookmark_processor',
            '--cov-report=html:htmlcov',
            '--cov-report=term-missing',
            '--cov-report=xml',
            '--cov-fail-under=80'
        ])
    
    # Add verbosity
    if verbose:
        cmd.extend(['-v', '--tb=short'])
    else:
        cmd.extend(['--tb=short'])
    
    # Add fast mode (skip slow tests)
    if fast:
        cmd.extend(['-m', 'not slow'])
    
    # Add JUnit XML output for CI
    if os.environ.get('CI'):
        cmd.extend([f'--junitxml=junit/test-results-{test_type}.xml'])
    
    # Add color output unless in CI
    if not os.environ.get('NO_COLOR'):
        cmd.append('--color=yes')
    
    # Disable warnings unless verbose
    if not verbose:
        cmd.append('--disable-warnings')
    
    return run_command(cmd, f"Running {test_type} tests")


def generate_coverage_report() -> bool:
    """Generate comprehensive coverage reports."""
    print("\n📊 Generating coverage reports...")
    
    reports = [
        ([sys.executable, '-m', 'coverage', 'html'], 
         "HTML coverage report"),
        ([sys.executable, '-m', 'coverage', 'xml'], 
         "XML coverage report"),
        ([sys.executable, '-m', 'coverage', 'json'], 
         "JSON coverage report"),
        ([sys.executable, '-m', 'coverage', 'report'], 
         "Terminal coverage report")
    ]
    
    all_passed = True
    for cmd, description in reports:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def run_performance_tests() -> bool:
    """Run performance tests."""
    print("\n⚡ Running performance tests...")
    
    cmd = [
        sys.executable, '-m', 'pytest', 
        'tests/', '-m', 'performance',
        '--timeout=300', '-v'
    ]
    
    return run_command(cmd, "Performance tests")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run bookmark processor tests')
    parser.add_argument(
        '--test-type', 
        choices=['unit', 'integration', 'all'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Run tests with coverage reporting'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Verbose test output'
    )
    parser.add_argument(
        '--fast', 
        action='store_true',
        help='Run only fast tests (skip slow tests)'
    )
    parser.add_argument(
        '--file', 
        help='Run tests from specific file'
    )
    parser.add_argument(
        '--install-deps', 
        action='store_true',
        help='Install test dependencies first'
    )
    parser.add_argument(
        '--no-lint', 
        action='store_true',
        help='Skip linting checks'
    )
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance tests'
    )
    parser.add_argument(
        '--ci-mode', 
        action='store_true',
        help='Run in CI mode with full checks'
    )
    
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    print("🧪 Bookmark Processor Test Runner")
    print("=" * 50)
    
    success = True
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            return 1
    
    # Run linting checks (unless disabled)
    if not args.no_lint and (args.ci_mode or args.test_type == 'all'):
        if not run_linting():
            success = False
            if args.ci_mode:
                print("❌ Linting failed in CI mode, stopping")
                return 1
    
    # Run main tests
    if not run_tests(args.test_type, args.coverage, args.verbose, args.fast, args.file):
        success = False
    
    # Generate coverage reports if requested
    if args.coverage:
        if not generate_coverage_report():
            success = False
    
    # Run performance tests if requested
    if args.performance:
        if not run_performance_tests():
            success = False
    
    # Print summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        
        # Print coverage summary if available
        if args.coverage and Path('htmlcov/index.html').exists():
            print(f"📊 Coverage report: file://{Path('htmlcov/index.html').absolute()}")
        
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())