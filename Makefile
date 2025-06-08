# Makefile for Bookmark Processor

.PHONY: help install install-dev clean test test-unit test-integration test-performance test-ci lint format type-check security-check coverage build docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install production dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  clean           - Clean build artifacts and cache files"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-performance - Run performance tests"
	@echo "  test-ci         - Run tests in CI mode (full checks)"
	@echo "  lint            - Run all linting checks"
	@echo "  format          - Format code with black and isort"
	@echo "  type-check      - Run type checking with mypy"
	@echo "  security-check  - Run security checks with bandit"
	@echo "  coverage        - Run tests with coverage reporting"
	@echo "  build           - Build package and executable"
	@echo "  docs            - Generate documentation"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

# Cleaning targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Testing targets
test:
	python run_tests_enhanced.py --test-type all --coverage

test-unit:
	python run_tests_enhanced.py --test-type unit --coverage

test-integration:
	python run_tests_enhanced.py --test-type integration

test-performance:
	python run_tests_enhanced.py --performance

test-ci:
	python run_tests_enhanced.py --ci-mode --coverage

test-fast:
	python run_tests_enhanced.py --fast

# Code quality targets
lint:
	python run_tests_enhanced.py --test-type unit --no-test --lint-only

format:
	black bookmark_processor tests
	isort bookmark_processor tests

type-check:
	mypy bookmark_processor --ignore-missing-imports

security-check:
	bandit -r bookmark_processor -ll
	safety check

# Coverage targets
coverage:
	python run_tests_enhanced.py --test-type all --coverage
	@echo "Coverage report generated in htmlcov/"
	@echo "Open htmlcov/index.html in your browser"

coverage-report:
	coverage report --show-missing
	coverage html
	coverage xml

# Build targets
build:
	python setup.py sdist bdist_wheel

build-executable:
	@echo "Validating platform compatibility..."
	python3 scripts/validate_platform.py
	@echo "Building Linux/WSL executable..."
	chmod +x build_linux.sh
	bash build_linux.sh

# Documentation targets
docs:
	@echo "Checking documentation..."
	python -c "import markdown; markdown.markdown(open('README.md').read())"
	@echo "Documentation check passed"

# Development workflow targets
dev-setup: install-dev
	@echo "Development environment setup complete"

pre-commit: format lint test-unit
	@echo "Pre-commit checks completed"

ci-test: clean install-dev test-ci
	@echo "CI testing completed"

# Quick development targets
quick-test:
	pytest tests/ -x --tb=short

watch-test:
	pytest-watch tests/

# Database and migration targets (if applicable)
reset-test-db:
	rm -f test_database.db

# Performance profiling
profile:
	python -m cProfile -o profile.out -m bookmark_processor --help
	python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	python -m memory_profiler run_tests.py

# Dependency management
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

check-deps:
	pip-audit
	safety check

# Release targets
version-bump-patch:
	bump2version patch

version-bump-minor:
	bump2version minor

version-bump-major:
	bump2version major

# Container targets (if using Docker)
docker-build:
	docker build -t bookmark-processor .

docker-test:
	docker run --rm bookmark-processor make test

# Benchmark targets
benchmark:
	python -m pytest tests/ -m benchmark --benchmark-only

# All quality checks
quality: format lint type-check security-check test-unit
	@echo "All quality checks passed"

# Full CI pipeline
ci: clean install-dev quality test coverage build
	@echo "Full CI pipeline completed successfully"