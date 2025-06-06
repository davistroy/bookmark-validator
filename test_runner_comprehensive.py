#!/usr/bin/env python3
"""
Comprehensive test runner and reporting system for bookmark processor.

This script runs the complete test suite, generates coverage reports,
and provides detailed analysis of test results.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class TestResult:
    """Container for test result information."""
    name: str
    passed: bool
    duration: float
    output: str
    error: Optional[str] = None
    coverage: Optional[float] = None


@dataclass
class TestSuite:
    """Container for test suite results."""
    name: str
    results: List[TestResult]
    total_duration: float
    passed_count: int
    failed_count: int
    coverage: Optional[float] = None


class TestRunner:
    """Comprehensive test runner with reporting."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results: List[TestSuite] = []
        self.start_time = datetime.now(timezone.utc)
        self.setup_environment()
    
    def setup_environment(self):
        """Set up test environment."""
        os.environ.update({
            'BOOKMARK_PROCESSOR_TEST_MODE': 'true',
            'BOOKMARK_PROCESSOR_LOG_LEVEL': 'ERROR',
            'BOOKMARK_PROCESSOR_OFFLINE_MODE': 'true',
            'PYTHONPATH': str(self.project_root)
        })
        
        # Create necessary directories
        for dir_name in ['htmlcov', 'junit', 'reports', '.pytest_cache']:
            (self.project_root / dir_name).mkdir(exist_ok=True)
    
    def run_command(self, cmd: List[str], description: str, timeout: int = 300) -> TestResult:
        """Run a command and capture results."""
        print(f"\n🔄 Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                print(f"✅ {description} completed in {duration:.2f}s")
            else:
                print(f"❌ {description} failed in {duration:.2f}s")
                print(f"Exit code: {result.returncode}")
            
            return TestResult(
                name=description,
                passed=success,
                duration=duration,
                output=result.stdout,
                error=result.stderr if not success else None
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {description} timed out after {duration:.2f}s")
            
            return TestResult(
                name=description,
                passed=False,
                duration=duration,
                output="",
                error=f"Test timed out after {timeout} seconds"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {description} crashed after {duration:.2f}s: {e}")
            
            return TestResult(
                name=description,
                passed=False,
                duration=duration,
                output="",
                error=str(e)
            )
    
    def run_lint_checks(self) -> TestSuite:
        """Run all linting and code quality checks."""
        print("\n🔍 Running Code Quality Checks")
        print("=" * 50)
        
        lint_tests = [
            ([sys.executable, '-m', 'black', '--check', 'bookmark_processor', 'tests'], 
             "Code formatting (black)"),
            ([sys.executable, '-m', 'isort', '--check-only', 'bookmark_processor', 'tests'], 
             "Import sorting (isort)"),
            ([sys.executable, '-m', 'flake8', 'bookmark_processor'], 
             "Linting (flake8)"),
            ([sys.executable, '-m', 'mypy', 'bookmark_processor', '--ignore-missing-imports'], 
             "Type checking (mypy)"),
            ([sys.executable, '-m', 'bandit', '-r', 'bookmark_processor', '-ll'], 
             "Security check (bandit)")
        ]
        
        results = []
        total_duration = 0
        
        for cmd, description in lint_tests:
            result = self.run_command(cmd, description)
            results.append(result)
            total_duration += result.duration
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        return TestSuite(
            name="Code Quality",
            results=results,
            total_duration=total_duration,
            passed_count=passed_count,
            failed_count=failed_count
        )
    
    def run_unit_tests(self) -> TestSuite:
        """Run unit tests with coverage."""
        print("\n🧪 Running Unit Tests")
        print("=" * 50)
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '-m', 'unit or not integration',
            '--cov=bookmark_processor',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml',
            '--cov-report=json:coverage.json',
            '--cov-report=term-missing',
            '--junitxml=junit/unit-tests.xml',
            '-v',
            '--tb=short'
        ]
        
        result = self.run_command(cmd, "Unit tests with coverage", timeout=600)
        
        # Extract coverage percentage
        coverage = self.extract_coverage_percentage(result.output)
        result.coverage = coverage
        
        return TestSuite(
            name="Unit Tests",
            results=[result],
            total_duration=result.duration,
            passed_count=1 if result.passed else 0,
            failed_count=0 if result.passed else 1,
            coverage=coverage
        )
    
    def run_integration_tests(self) -> TestSuite:
        """Run integration tests."""
        print("\n🔗 Running Integration Tests")
        print("=" * 50)
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '-m', 'integration',
            '--junitxml=junit/integration-tests.xml',
            '-v',
            '--tb=short'
        ]
        
        result = self.run_command(cmd, "Integration tests", timeout=900)
        
        return TestSuite(
            name="Integration Tests",
            results=[result],
            total_duration=result.duration,
            passed_count=1 if result.passed else 0,
            failed_count=0 if result.passed else 1
        )
    
    def run_performance_tests(self) -> TestSuite:
        """Run performance tests."""
        print("\n⚡ Running Performance Tests")
        print("=" * 50)
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '-m', 'performance',
            '--junitxml=junit/performance-tests.xml',
            '-v',
            '--tb=short',
            '--timeout=300'
        ]
        
        result = self.run_command(cmd, "Performance tests", timeout=600)
        
        return TestSuite(
            name="Performance Tests",
            results=[result],
            total_duration=result.duration,
            passed_count=1 if result.passed else 0,
            failed_count=0 if result.passed else 1
        )
    
    def run_security_tests(self) -> TestSuite:
        """Run security-focused tests."""
        print("\n🔒 Running Security Tests")
        print("=" * 50)
        
        security_tests = [
            ([sys.executable, '-m', 'pytest', 'tests/', '-m', 'security', '-v'], 
             "Security tests"),
            ([sys.executable, '-m', 'bandit', '-r', 'bookmark_processor', '-f', 'json', '-o', 'reports/bandit.json'], 
             "Security scan (bandit)"),
            (['safety', 'check', '--json', '--output', 'reports/safety.json'], 
             "Dependency security check")
        ]
        
        results = []
        total_duration = 0
        
        for cmd, description in security_tests:
            result = self.run_command(cmd, description)
            results.append(result)
            total_duration += result.duration
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        return TestSuite(
            name="Security Tests",
            results=results,
            total_duration=total_duration,
            passed_count=passed_count,
            failed_count=failed_count
        )
    
    def extract_coverage_percentage(self, output: str) -> Optional[float]:
        """Extract coverage percentage from pytest output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                try:
                    # Look for percentage in the line
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            return float(part.rstrip('%'))
                except (ValueError, IndexError):
                    continue
        return None
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - self.start_time).total_seconds()
        
        total_passed = sum(suite.passed_count for suite in self.test_results)
        total_failed = sum(suite.failed_count for suite in self.test_results)
        total_tests = total_passed + total_failed
        
        # Get overall coverage
        coverage_suite = next((s for s in self.test_results if s.coverage is not None), None)
        overall_coverage = coverage_suite.coverage if coverage_suite else None
        
        summary = {
            "timestamp": end_time.isoformat(),
            "total_duration": total_duration,
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "coverage": overall_coverage,
            "suites": [asdict(suite) for suite in self.test_results],
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "test_mode": os.environ.get('BOOKMARK_PROCESSOR_TEST_MODE'),
                "offline_mode": os.environ.get('BOOKMARK_PROCESSOR_OFFLINE_MODE')
            }
        }
        
        return summary
    
    def save_report(self, summary: Dict[str, Any]):
        """Save test report to files."""
        reports_dir = self.project_root / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_report = reports_dir / 'test-summary.json'
        with open(json_report, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save human-readable report
        text_report = reports_dir / 'test-summary.txt'
        with open(text_report, 'w') as f:
            self.write_text_report(f, summary)
        
        print(f"\n📊 Reports saved:")
        print(f"   JSON: {json_report}")
        print(f"   Text: {text_report}")
    
    def write_text_report(self, file, summary: Dict[str, Any]):
        """Write human-readable test report."""
        file.write("BOOKMARK PROCESSOR - TEST SUMMARY REPORT\n")
        file.write("=" * 50 + "\n\n")
        
        file.write(f"Test Date: {summary['timestamp']}\n")
        file.write(f"Total Duration: {summary['total_duration']:.2f} seconds\n")
        file.write(f"Python Version: {summary['environment']['python_version']}\n")
        file.write(f"Platform: {summary['environment']['platform']}\n\n")
        
        file.write("OVERALL RESULTS\n")
        file.write("-" * 20 + "\n")
        file.write(f"Total Tests: {summary['total_tests']}\n")
        file.write(f"Passed: {summary['passed']}\n")
        file.write(f"Failed: {summary['failed']}\n")
        file.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
        
        if summary['coverage']:
            file.write(f"Code Coverage: {summary['coverage']:.1f}%\n")
        
        file.write("\nTEST SUITE BREAKDOWN\n")
        file.write("-" * 25 + "\n")
        
        for suite_data in summary['suites']:
            file.write(f"\n{suite_data['name']}:\n")
            file.write(f"  Duration: {suite_data['total_duration']:.2f}s\n")
            file.write(f"  Passed: {suite_data['passed_count']}\n")
            file.write(f"  Failed: {suite_data['failed_count']}\n")
            
            if suite_data['coverage']:
                file.write(f"  Coverage: {suite_data['coverage']:.1f}%\n")
            
            # Show failed tests
            failed_tests = [r for r in suite_data['results'] if not r['passed']]
            if failed_tests:
                file.write("  Failed Tests:\n")
                for test in failed_tests:
                    file.write(f"    - {test['name']}\n")
                    if test['error']:
                        file.write(f"      Error: {test['error'][:100]}...\n")
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary to console."""
        print("\n" + "=" * 60)
        print("🧪 BOOKMARK PROCESSOR TEST SUMMARY")
        print("=" * 60)
        
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['coverage']:
            coverage_emoji = "🎯" if summary['coverage'] >= 80 else "⚠️"
            print(f"{coverage_emoji} Code Coverage: {summary['coverage']:.1f}%")
        
        print("\nSUITE BREAKDOWN:")
        for suite_data in summary['suites']:
            status_emoji = "✅" if suite_data['failed_count'] == 0 else "❌"
            print(f"{status_emoji} {suite_data['name']}: "
                  f"{suite_data['passed_count']}/{suite_data['passed_count'] + suite_data['failed_count']} "
                  f"({suite_data['total_duration']:.1f}s)")
        
        # Show coverage report location
        if summary['coverage'] and (self.project_root / 'htmlcov' / 'index.html').exists():
            print(f"\n📊 HTML Coverage Report: file://{self.project_root / 'htmlcov' / 'index.html'}")
        
        overall_success = summary['failed'] == 0
        if overall_success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n💥 {summary['failed']} TEST(S) FAILED")
        
        print("=" * 60)
        
        return overall_success
    
    def run_full_suite(self) -> bool:
        """Run complete test suite."""
        print("🚀 Starting Comprehensive Test Suite")
        print(f"Project: {self.project_root}")
        print(f"Start Time: {self.start_time.isoformat()}")
        
        # Run all test suites
        test_suites = [
            self.run_lint_checks,
            self.run_unit_tests,
            self.run_integration_tests,
            self.run_performance_tests,
            self.run_security_tests
        ]
        
        for suite_runner in test_suites:
            try:
                suite_result = suite_runner()
                self.test_results.append(suite_result)
            except Exception as e:
                print(f"❌ Suite runner failed: {e}")
                # Continue with other suites
        
        # Generate and save reports
        summary = self.generate_summary_report()
        self.save_report(summary)
        
        # Print final summary
        return self.print_summary(summary)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent
    runner = TestRunner(project_root)
    
    success = runner.run_full_suite()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())