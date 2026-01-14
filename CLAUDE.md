# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bookmark Validation and Enhancement Tool - a Python CLI that processes raindrop.io bookmark exports (11-column CSV) to validate URLs, generate AI-enhanced descriptions, and output optimized 6-column CSV imports. Designed for large datasets (3,500+ bookmarks) with checkpoint/resume capability for long-running processes.

**Platform:** Linux/WSL only (Windows native not supported)

## Common Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Run the application
python -m bookmark_processor --input raindrop_export.csv --output enhanced.csv
python -m bookmark_processor --input file.csv --output out.csv --resume  # Resume from checkpoint
python -m bookmark_processor --input file.csv --output out.csv --ai-engine claude  # Use Claude API

# Testing
python -m pytest tests/ -v                          # All tests
python -m pytest tests/test_csv_handler.py -v       # Single test file
python -m pytest tests/ -k "test_url_validation"    # Run tests matching pattern
python -m pytest tests/ --cov=bookmark_processor    # With coverage

# Code quality
black bookmark_processor tests && isort bookmark_processor tests  # Format
flake8 bookmark_processor tests                                   # Lint
mypy bookmark_processor                                           # Type check

# Makefile shortcuts
make test           # Run all tests with coverage
make test-unit      # Unit tests only
make format         # Format code
make quality        # All quality checks (format, lint, type-check, security, test)
```

## Architecture

### Data Flow
```
Input CSV (11 columns) → Bookmark objects → Processing Pipeline → Output CSV (6 columns)
                                                ↓
                    URL Validation → Content Analysis → AI Enhancement → Tag Optimization
```

### Key Components

**Entry Points:**
- `bookmark_processor/main.py` - CLI entry point
- `bookmark_processor/cli.py` / `cli_argparse.py` - Command line interface

**Core Pipeline (`bookmark_processor/core/`):**
- `pipeline.py` - `BookmarkProcessingPipeline` orchestrates all processing stages
- `data_models.py` - `Bookmark`, `BookmarkMetadata`, `ProcessingStatus` dataclasses
- `csv_handler.py` - `RaindropCSVHandler` for raindrop.io format I/O (11→6 columns)
- `url_validator.py` - URL validation with retry logic and rate limiting
- `content_analyzer.py` - Web content extraction and metadata parsing
- `ai_processor.py` - `EnhancedAIProcessor` for description generation
- `tag_generator.py` - `CorpusAwareTagGenerator` for optimized tagging (100-200 unique tags)
- `checkpoint_manager.py` - Checkpoint/resume functionality for long processes
- `duplicate_detector.py` - URL deduplication with multiple resolution strategies

**AI Backends (`bookmark_processor/core/`):**
- `ai_factory.py` - Factory for creating AI processors
- `claude_api_client.py` - Claude API integration
- `openai_api_client.py` - OpenAI API integration
- Default: local facebook/bart-large-cnn model

**Utilities (`bookmark_processor/utils/`):**
- `intelligent_rate_limiter.py` - Site-specific rate limiting
- `progress_tracker.py` - Progress bars and ETA estimation
- `memory_optimizer.py` - Batch processing for memory efficiency
- `browser_simulator.py` - User agent rotation

### Configuration
- `bookmark_processor/config/pydantic_config.py` - Pydantic-based config with validation
- `bookmark_processor/config/configuration.py` - Legacy configuration management

### Data Formats

**Input (raindrop.io export):**
```csv
id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
```

**Output (raindrop.io import):**
```csv
url,folder,title,note,tags,created
```

Tags format: single tag unquoted, multiple tags quoted with commas (`"ai, research, tech"`)

## Key Design Decisions

1. **Dependency Injection:** Pipeline components are injectable for testing and flexibility
2. **Checkpoint System:** Saves every 50 items; auto-resumes on restart
3. **Rate Limiting:** Per-domain delays (Google 2s, GitHub 1.5s, YouTube 2s, LinkedIn 2s)
4. **AI Fallback Hierarchy:** AI with existing content → existing excerpt → meta description → title-based
5. **Tag Strategy:** Replaces original tags entirely; uses existing tags as context for AI generation
6. **Memory Efficiency:** Batch processing with configurable batch sizes (default 100)

## Testing

Test markers defined in `pyproject.toml`:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests >5 seconds
- `@pytest.mark.network` - Network-dependent tests (mocked)
- `@pytest.mark.ai` - AI processing tests

Run specific markers: `python -m pytest -m unit`

## Docker

```bash
docker-compose build
docker-compose run --rm bookmark-processor --input /app/data/input.csv --output /app/data/output.csv
```

See `DOCKER.md` for complete Docker setup guide.
