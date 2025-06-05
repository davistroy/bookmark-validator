#!/usr/bin/env python3
"""
Test runner script for bookmark processor.

Provides various test execution options and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"❌ {description} failed")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run bookmark processor tests")
    parser.add_argument(
        "--test-type", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Run only fast tests (skip slow tests)"
    )
    parser.add_argument(
        "--file", 
        type=str,
        help="Run tests from specific file"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install test dependencies first"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    
    # Install dependencies if requested
    if args.install_deps:
        deps_cmd = "pip install pytest pytest-cov pytest-mock pytest-timeout"
        if not run_command(deps_cmd, "Installing test dependencies"):
            return 1
    
    # Build test command
    cmd_parts = ["python", "-m", "pytest"]
    
    # Add coverage if requested
    if args.coverage:
        cmd_parts.extend([
            "--cov=bookmark_processor",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=75"
        ])
    
    # Add verbosity
    if args.verbose:
        cmd_parts.append("-vv")
    else:
        cmd_parts.append("-v")
    
    # Add test type filters
    if args.test_type == "unit":
        cmd_parts.extend(["-m", "unit"])
    elif args.test_type == "integration":
        cmd_parts.extend(["-m", "integration"])
    
    # Skip slow tests if requested
    if args.fast:
        cmd_parts.extend(["-m", "not slow"])
    
    # Specific file
    if args.file:
        cmd_parts.append(f"tests/{args.file}")
    else:
        cmd_parts.append("tests/")
    
    # Additional options
    cmd_parts.extend([
        "--tb=short",
        "--color=yes",
        "--disable-warnings"
    ])
    
    # Run the tests
    cmd = " ".join(cmd_parts)
    print(f"🚀 Running tests: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(cmd, shell=True, cwd=project_root)
    
    if result.returncode == 0:
        print("-" * 60)
        print("🎉 All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/")
            print("   Open htmlcov/index.html in your browser to view detailed coverage")
    else:
        print("-" * 60)
        print("❌ Some tests failed")
        return result.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())