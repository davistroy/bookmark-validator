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
- `bookmark_processor/cli.py` / `cli_argparse.py` - Command line interface with advanced options

**Core Pipeline (`bookmark_processor/core/`):**
- `pipeline.py` - `BookmarkProcessingPipeline` orchestrates all processing stages
- `async_pipeline.py` - `AsyncPipelineExecutor` for concurrent processing (10x throughput)
- `data_models.py` - `Bookmark`, `BookmarkMetadata`, `ProcessingStatus` dataclasses
- `csv_handler.py` - `RaindropCSVHandler` for raindrop.io format I/O (11→6 columns)
- `url_validator.py` - URL validation with retry logic and rate limiting
- `content_analyzer.py` - Web content extraction and metadata parsing
- `ai_processor.py` - `EnhancedAIProcessor` for description generation
- `ai_router.py` - `AIRouter` for hybrid local/cloud AI selection
- `tag_generator.py` - `CorpusAwareTagGenerator` and `EnhancedTagGenerator` for optimized tagging
- `tag_config.py` - `TagConfig` for user-defined vocabulary via TOML
- `folder_generator.py` - `EnhancedFolderGenerator` with semantic folder suggestions
- `checkpoint_manager.py` - Checkpoint/resume functionality for long processes
- `duplicate_detector.py` - URL deduplication with multiple resolution strategies
- `filters.py` - Composable `BookmarkFilter` system (folder, tag, date, domain, status)
- `processing_modes.py` - `ProcessingStages` flags and `ProcessingMode` for granular control
- `quality_reporter.py` - `QualityReporter` with metrics calculation
- `interactive_processor.py` - `InteractiveProcessor` for approval workflows
- `health_monitor.py` - `BookmarkHealthMonitor` with Wayback Machine integration
- `database.py` - `BookmarkDatabase` with FTS5 search and run comparison

**Data Sources (`bookmark_processor/core/data_sources/`):**
- `protocol.py` - `BookmarkDataSource` protocol for abstraction
- `csv_source.py` - `CSVDataSource` implementation
- `state_tracker.py` - `ProcessingStateTracker` with SQLite persistence
- `mcp_client.py` - `MCPClient` for MCP integration
- `raindrop_mcp.py` - `RaindropMCPDataSource` for direct Raindrop.io sync

**Streaming (`bookmark_processor/core/streaming/`):**
- `reader.py` - `StreamingBookmarkReader` for generator-based reading
- `writer.py` - `StreamingBookmarkWriter` for incremental writing
- `pipeline.py` - `StreamingPipeline` for memory-efficient processing

**Exporters (`bookmark_processor/core/exporters/`):**
- `base.py` - `BookmarkExporter` ABC and `ExportResult`
- `json_exporter.py` - JSON format export
- `markdown_exporter.py` - Markdown format export
- `obsidian_exporter.py` - Obsidian vault-compatible export
- `notion_exporter.py` - Notion-compatible export
- `opml_exporter.py` - OPML format for feed readers

**Plugins (`bookmark_processor/plugins/`):**
- `base.py` - `BookmarkPlugin`, `ValidatorPlugin`, `AIProcessorPlugin`, `OutputPlugin`
- `loader.py` - `PluginLoader` for discovery
- `registry.py` - `PluginRegistry` for management
- `examples/paywall_detector.py` - Sample validator plugin
- `examples/ollama_ai.py` - Sample AI plugin for Ollama

**AI Backends (`bookmark_processor/core/`):**
- `ai_factory.py` - Factory for creating AI processors
- `claude_api_client.py` - Claude API integration
- `openai_api_client.py` - OpenAI API integration
- Default: local facebook/bart-large-cnn model

**Utilities (`bookmark_processor/utils/`):**
- `intelligent_rate_limiter.py` - Site-specific rate limiting
- `progress_tracker.py` - Progress bars and ETA estimation
- `enhanced_progress.py` - `EnhancedProgressTracker` with multi-stage weighted ETA
- `memory_optimizer.py` - Batch processing for memory efficiency
- `browser_simulator.py` - User agent rotation
- `report_generator.py` - `ReportGenerator` for Rich/JSON/Markdown output
- `report_styles.py` - `ReportStyle` enum and style configuration

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
7. **Protocol-Based Abstraction:** Data sources implement `BookmarkDataSource` protocol for flexibility
8. **Composable Filters:** Filter chain with AND/OR operators for complex queries
9. **Plugin Architecture:** Loader/registry pattern for extensibility
10. **Streaming Pipeline:** Generator-based processing for constant memory usage
11. **Async Concurrency:** Semaphore-controlled async for network-bound operations
12. **SQLite State:** Database-backed state with FTS5 for search and history

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
