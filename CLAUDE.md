## Learning Capture — Every Session

After any non-trivial finding (URL validation failure, rate limit, checkpoint/resume bug, async concurrency surprise, AI backend behavior difference, raindrop.io CSV quirk, multi-attempt fix):
1. Update `CLAUDE.md` — add/update bullet in relevant section
2. Update memory file — `C:\Users\Troy Davis\.claude\projects\C--Users-Troy-Davis-dev-personal-bookmark-validator\memory\`
3. Update `MEMORY.md` — concise bullet + link to topic file

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Operational rules, always enforced |
| `memory/MEMORY.md` | Concise index, survives compaction |
| `memory/pipeline-learnings.md` | Async pipeline, concurrency, batch sizing |
| `memory/ai-backend-learnings.md` | Claude/OpenAI/local model quirks |
| `memory/data-format-learnings.md` | CSV format, raindrop.io import/export quirks |

### Verified Operational Rules

*(None yet — add as discovered)*

---

# CLAUDE.md

## Project Overview

Python CLI that processes raindrop.io bookmark exports (11-column CSV) to validate URLs, generate AI-enhanced descriptions, and output optimized 6-column CSV imports. Designed for large datasets (3,500+ bookmarks) with checkpoint/resume.

**Platform:** Linux/WSL only (Windows native not supported)

## Common Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Run
python -m bookmark_processor --input raindrop_export.csv --output enhanced.csv
python -m bookmark_processor --input file.csv --output out.csv --resume
python -m bookmark_processor --input file.csv --output out.csv --ai-engine claude

# Testing
python -m pytest tests/ -v
python -m pytest tests/test_csv_handler.py -v
python -m pytest tests/ -k "test_url_validation"
python -m pytest tests/ --cov=bookmark_processor

# Quality
black bookmark_processor tests && isort bookmark_processor tests
flake8 bookmark_processor tests
mypy bookmark_processor

# Makefile
make test / make test-unit / make format / make quality
```

## Architecture

```
Input CSV (11 cols) → Bookmark objects → Pipeline → Output CSV (6 cols)
                                              ↓
                  URL Validation → Content Analysis → AI Enhancement → Tag Optimization
```

**Core Pipeline (`bookmark_processor/core/`):**

| Module | Purpose |
|--------|---------|
| `pipeline.py` | `BookmarkProcessingPipeline` orchestrator |
| `async_pipeline.py` | `AsyncPipelineExecutor` (10x throughput) |
| `data_models.py` | `Bookmark`, `BookmarkMetadata`, `ProcessingStatus` |
| `csv_handler.py` | `RaindropCSVHandler` (11→6 columns) |
| `url_validator.py` | URL validation with retry + rate limiting |
| `content_analyzer.py` | Web content extraction |
| `ai_processor.py` | `EnhancedAIProcessor` |
| `ai_router.py` | `AIRouter` (hybrid local/cloud) |
| `tag_generator.py` | `CorpusAwareTagGenerator`, `EnhancedTagGenerator` |
| `tag_config.py` | User-defined vocabulary via TOML |
| `folder_generator.py` | Semantic folder suggestions |
| `checkpoint_manager.py` | Checkpoint/resume for long processes |
| `duplicate_detector.py` | URL dedup with multiple resolution strategies |
| `filters.py` | Composable `BookmarkFilter` (AND/OR operators) |
| `database.py` | `BookmarkDatabase` with FTS5 search + run comparison |
| `streaming/` | Generator-based reader/writer/pipeline |
| `data_sources/` | Protocol + CSV/MCP/Raindrop sources |
| `exporters/` | JSON, Markdown, Obsidian, Notion, OPML |

**AI Backends:** `claude_api_client.py`, `openai_api_client.py`, default: local `facebook/bart-large-cnn`

**Utils:** `intelligent_rate_limiter.py`, `memory_optimizer.py`, `browser_simulator.py`, `report_generator.py`

## Data Formats

**Input:** `id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite`
**Output:** `url,folder,title,note,tags,created`
**Tags:** single unquoted, multiple quoted with commas (`"ai, research, tech"`)

## Key Design Decisions

- Checkpoint saves every 50 items; auto-resumes on restart
- Rate limiting: Google 2s, GitHub 1.5s, YouTube 2s, LinkedIn 2s per domain
- AI fallback: AI with content → existing excerpt → meta description → title-based
- Tag strategy: replaces original tags entirely; uses existing as AI context
- Streaming pipeline for constant memory usage
- Semaphore-controlled async for network-bound ops
- SQLite state with FTS5 for search and history

## Testing

```bash
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m slow
python -m pytest -m network
python -m pytest -m ai
```

## Docker

```bash
docker-compose build
docker-compose run --rm bookmark-processor --input /app/data/input.csv --output /app/data/output.csv
```
