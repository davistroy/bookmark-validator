# Bookmark Validator - Implementation Plan

> **Purpose**: Detailed implementation plan derived from IMPROVEMENT_ROADMAP.md, organized into phases of ~100k tokens each with maximum parallelization opportunities.

**Created**: January 2026
**Based On**: IMPROVEMENT_ROADMAP.md v2.0.0

---

## Overview

This plan reorganizes the roadmap improvements into implementation phases that:
1. Respect dependencies between features
2. Maximize parallel work within each phase
3. Target ~100k tokens per phase (including implementation, tests, and fixes)
4. Build incrementally on existing architecture

### Token Estimation Guidelines

| Work Type | Estimated Tokens |
|-----------|------------------|
| CLI flag + simple logic | 10-15k |
| New utility module + tests | 20-30k |
| New core component + tests | 40-60k |
| Major refactor + tests | 60-80k |
| New subsystem + tests | 80-100k |

---

## Phase 0: Foundation Infrastructure ✅ COMPLETE
**Estimated Tokens**: 60-80k
**Dependencies**: None (foundation for later phases)
**Enables**: Phases 1, 2, 3, 4
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 184 tests passing

### Objective
Create shared infrastructure that multiple later features depend on, enabling maximum parallelization in subsequent phases.

### Work Items

#### 0.1 Report Generation Infrastructure (25-30k tokens)
**Location**: `bookmark_processor/utils/report_generator.py`
**Enables**: Phase 2 (Quality Reports, Progress Visibility)

Create a flexible report generation system:

```python
# New file: utils/report_generator.py
class ReportSection:
    """Single section of a report with title and content."""
    title: str
    content: Union[str, Table, Dict]
    icon: Optional[str] = None

class ReportGenerator:
    """Generate formatted reports in multiple styles."""

    def __init__(self, style: ReportStyle = ReportStyle.RICH):
        self.style = style
        self.sections: List[ReportSection] = []

    def add_section(self, section: ReportSection) -> None: ...
    def add_table(self, title: str, headers: List[str], rows: List[List]) -> None: ...
    def add_metrics(self, title: str, metrics: Dict[str, Any]) -> None: ...
    def add_warning(self, message: str) -> None: ...

    def render_terminal(self) -> str: ...
    def render_markdown(self) -> str: ...
    def render_json(self) -> Dict: ...
    def save(self, path: Path, format: str = "md") -> None: ...

class ReportStyle(Enum):
    RICH = "rich"       # Rich console with colors/icons
    PLAIN = "plain"     # Plain text for piping
    MARKDOWN = "markdown"
    JSON = "json"
```

**Deliverables**:
- [x] `utils/report_generator.py` - Core report generation ✅
- [x] `utils/report_styles.py` - Style definitions and templates ✅
- [x] Unit tests for all report formats (52 tests) ✅
- [x] Integration with Rich console ✅

---

#### 0.2 Filter Infrastructure (20-25k tokens)
**Location**: `bookmark_processor/core/filters.py`
**Enables**: Phase 1 (Smart Filtering), Phase 4 (MCP queries)

Create a composable filtering system:

```python
# New file: core/filters.py
from abc import ABC, abstractmethod
from typing import Callable, List

class BookmarkFilter(ABC):
    """Abstract base for bookmark filters."""

    @abstractmethod
    def matches(self, bookmark: Bookmark) -> bool: ...

    def __and__(self, other: 'BookmarkFilter') -> 'CompositeFilter':
        return CompositeFilter([self, other], operator='and')

    def __or__(self, other: 'BookmarkFilter') -> 'CompositeFilter':
        return CompositeFilter([self, other], operator='or')

class FolderFilter(BookmarkFilter):
    """Filter by folder pattern (supports glob)."""
    def __init__(self, pattern: str): ...

class TagFilter(BookmarkFilter):
    """Filter by tag presence."""
    def __init__(self, tags: List[str], mode: str = 'any'): ...

class DateRangeFilter(BookmarkFilter):
    """Filter by creation date range."""
    def __init__(self, start: Optional[datetime], end: Optional[datetime]): ...

class DomainFilter(BookmarkFilter):
    """Filter by URL domain(s)."""
    def __init__(self, domains: List[str]): ...

class StatusFilter(BookmarkFilter):
    """Filter by processing status."""
    def __init__(self, statuses: List[str]): ...

class FilterChain:
    """Apply multiple filters with configurable logic."""

    def __init__(self, filters: List[BookmarkFilter], operator: str = 'and'):
        self.filters = filters
        self.operator = operator

    def apply(self, bookmarks: List[Bookmark]) -> List[Bookmark]: ...

    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> 'FilterChain': ...
```

**Deliverables**:
- [x] `core/filters.py` - Filter classes and chain ✅
- [x] Unit tests for each filter type (71 tests) ✅
- [x] Test composability (AND/OR combinations) ✅
- [x] CLI argument parsing helper ✅

---

#### 0.3 Processing Mode Abstraction (15-20k tokens)
**Location**: `bookmark_processor/core/processing_modes.py`
**Enables**: Phase 1 (Preview, Dry-run, Granular Control)

```python
# New file: core/processing_modes.py
from dataclasses import dataclass
from enum import Flag, auto

class ProcessingStages(Flag):
    """Flags for which processing stages to execute."""
    NONE = 0
    VALIDATION = auto()      # URL validation
    CONTENT = auto()         # Content extraction
    AI = auto()              # AI description generation
    TAGS = auto()            # Tag optimization
    FOLDERS = auto()         # Folder organization

    # Common combinations
    ALL = VALIDATION | CONTENT | AI | TAGS | FOLDERS
    VALIDATE_ONLY = VALIDATION
    TAGS_ONLY = TAGS
    FOLDERS_ONLY = FOLDERS
    NO_AI = VALIDATION | CONTENT | TAGS | FOLDERS

@dataclass
class ProcessingMode:
    """Configuration for processing behavior."""
    stages: ProcessingStages = ProcessingStages.ALL
    preview_count: Optional[int] = None  # None = process all
    dry_run: bool = False  # If True, don't write output

    @property
    def is_preview(self) -> bool:
        return self.preview_count is not None

    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> 'ProcessingMode': ...
```

**Deliverables**:
- [x] `core/processing_modes.py` - Mode definitions ✅
- [x] Unit tests for mode combinations (61 tests) ✅
- [x] Integration with pipeline ✅

---

### Phase 0 Parallelization

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 0                                   │
│                  (Can run in parallel)                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ 0.1 Report Gen  │ 0.2 Filters     │ 0.3 Processing Modes        │
│ (25-30k)        │ (20-25k)        │ (15-20k)                    │
│                 │                 │                             │
│ No deps         │ No deps         │ No deps                     │
└─────────────────┴─────────────────┴─────────────────────────────┘
                              ↓
                     All Phase 1+ work
```

---

## Phase 1: Quick Wins - CLI Features ✅ COMPLETE
**Estimated Tokens**: 80-100k
**Dependencies**: Phase 0 (Filters, Processing Modes)
**Enables**: Immediate user value
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 56 tests passing

### Objective
Implement high-impact CLI features that significantly improve usability with minimal architectural changes.

### Work Items

#### 1.1 Preview/Dry-Run Mode (25-30k tokens)
**Location**: `cli.py`, `core/pipeline.py`
**Priority**: P0

Add preview and dry-run capabilities:

```bash
# Preview first N bookmarks
bookmark-processor --input bookmarks.csv --output preview.csv --preview 10

# Dry run (validate without writing)
bookmark-processor --input bookmarks.csv --dry-run
```

**Implementation**:

```python
# cli.py additions
@app.command()
def process(
    # ... existing options ...
    preview: Optional[int] = typer.Option(
        None, "--preview", "-p",
        help="Process only first N bookmarks as a sample"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Validate configuration without making changes"
    ),
): ...

# pipeline.py changes
def execute(self, mode: ProcessingMode = None) -> PipelineResults:
    mode = mode or ProcessingMode()

    bookmarks = self._load_bookmarks()

    # Apply preview limit
    if mode.preview_count:
        bookmarks = bookmarks[:mode.preview_count]
        console.print(f"[info]Preview mode: processing {len(bookmarks)} bookmarks[/]")

    # Process normally...
    results = self._process_bookmarks(bookmarks, mode.stages)

    # Skip output in dry-run
    if mode.dry_run:
        console.print("[info]Dry run complete - no changes written[/]")
        return results

    self._write_output(results)
    return results
```

**Deliverables**:
- [x] `--preview N` flag implementation ✅
- [x] `--dry-run` flag implementation ✅
- [x] Before/after comparison display for preview ✅
- [x] Time estimation based on preview sample ✅
- [x] Unit tests for preview logic ✅
- [x] Integration tests for dry-run ✅

---

#### 1.2 Smart Filtering (30-35k tokens)
**Location**: `cli.py`, uses `core/filters.py` from Phase 0
**Priority**: P0

Add filtering options to process subsets:

```bash
# Filter by folder
bookmark-processor --input bookmarks.csv --output out.csv --filter-folder "Tech/*"

# Filter by tag
bookmark-processor --input bookmarks.csv --output out.csv --filter-tag "unprocessed"

# Filter by date range
bookmark-processor --input bookmarks.csv --output out.csv --filter-date "2024-01-01:2024-12-31"

# Filter by domain
bookmark-processor --input bookmarks.csv --output out.csv --filter-domain "github.com,gitlab.com"

# Re-process only previously invalid URLs
bookmark-processor --input bookmarks.csv --output out.csv --retry-invalid
```

**Implementation**:

```python
# cli.py additions
@app.command()
def process(
    # ... existing options ...
    filter_folder: Optional[str] = typer.Option(
        None, "--filter-folder",
        help="Only process bookmarks in matching folders (supports glob)"
    ),
    filter_tag: Optional[List[str]] = typer.Option(
        None, "--filter-tag",
        help="Only process bookmarks with these tags"
    ),
    filter_date: Optional[str] = typer.Option(
        None, "--filter-date",
        help="Only process bookmarks in date range (start:end)"
    ),
    filter_domain: Optional[str] = typer.Option(
        None, "--filter-domain",
        help="Only process bookmarks from these domains (comma-separated)"
    ),
    retry_invalid: bool = typer.Option(
        False, "--retry-invalid",
        help="Only re-process previously invalid URLs"
    ),
): ...
```

**Deliverables**:
- [x] CLI options for all filter types ✅
- [x] Integration with FilterChain from Phase 0 ✅
- [x] Filter summary in output (X of Y bookmarks matched) ✅
- [x] Unit tests for CLI parsing ✅
- [x] Integration tests with sample data ✅

---

#### 1.3 Granular Processing Control (25-30k tokens)
**Location**: `cli.py`, uses `core/processing_modes.py` from Phase 0
**Priority**: P1

Add stage-skipping options:

```bash
# Skip URL validation
bookmark-processor --input bookmarks.csv --output out.csv --skip-validation

# Skip AI description generation
bookmark-processor --input bookmarks.csv --output out.csv --skip-ai

# Only regenerate tags
bookmark-processor --input bookmarks.csv --output out.csv --tags-only

# Only reorganize folders
bookmark-processor --input bookmarks.csv --output out.csv --folders-only

# Only validate URLs
bookmark-processor --input bookmarks.csv --output out.csv --validate-only
```

**Implementation**:

```python
# cli.py additions
@app.command()
def process(
    # ... existing options ...
    skip_validation: bool = typer.Option(False, "--skip-validation"),
    skip_ai: bool = typer.Option(False, "--skip-ai"),
    tags_only: bool = typer.Option(False, "--tags-only"),
    folders_only: bool = typer.Option(False, "--folders-only"),
    validate_only: bool = typer.Option(False, "--validate-only"),
):
    # Build ProcessingMode from flags
    stages = ProcessingStages.ALL
    if skip_validation:
        stages &= ~ProcessingStages.VALIDATION
    if skip_ai:
        stages &= ~ProcessingStages.AI
    # ... etc

    mode = ProcessingMode(stages=stages)
```

**Deliverables**:
- [x] CLI flags for all stage-skipping options ✅
- [x] Mutual exclusivity validation (can't use --tags-only with --skip-ai) ✅
- [x] Help text explaining each option ✅
- [x] Unit tests for flag combinations ✅
- [x] Integration tests ✅

---

### Phase 1 Parallelization

```
Phase 0 Complete
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 1                                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ 1.1 Preview/    │ 1.2 Smart       │ 1.3 Granular Control        │
│ Dry-Run         │ Filtering       │                             │
│ (25-30k)        │ (30-35k)        │ (25-30k)                    │
│                 │                 │                             │
│ Needs: 0.3      │ Needs: 0.2      │ Needs: 0.3                  │
│ (Proc Modes)    │ (Filters)       │ (Proc Modes)                │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                    Can run in parallel                          │
│              (all depend only on Phase 0 items)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Reporting & Visibility ✅ COMPLETE
**Estimated Tokens**: 70-90k
**Dependencies**: Phase 0 (Report Generator)
**Enables**: Better user experience, debugging
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 125 tests passing (2 skipped for optional Rich features)

### Objective
Provide users with comprehensive feedback about processing quality and progress.

### Work Items

#### 2.1 Quality Score Report (35-45k tokens)
**Location**: `bookmark_processor/core/quality_reporter.py`
**Priority**: P0
**Uses**: Report Generator from Phase 0

Generate comprehensive quality assessment after processing:

```
═══════════════════════════════════════════════════════════════
                    QUALITY ASSESSMENT REPORT
═══════════════════════════════════════════════════════════════

📊 DESCRIPTION ENHANCEMENT
   ├─ Enhanced by AI:        2,847 (83.2%)
   ├─ Used existing excerpt:   412 (12.0%)
   ├─ Fallback to title:       162 (4.8%)
   └─ Average confidence:     0.78

🏷️  TAG ANALYSIS
   ├─ Total unique tags:       156
   ├─ Bookmarks with tags:   3,241 (94.7%)
   ├─ Avg tags per bookmark:   3.2
   └─ Tag coverage score:     0.89

📁 FOLDER ORGANIZATION
   ├─ Total folders:            24
   ├─ Max depth:                 3
   ├─ Bookmarks reorganized:   847 (24.8%)
   └─ Organization coherence:  0.82

⚠️  ITEMS NEEDING ATTENTION
   ├─ Low-confidence descriptions: 89
   ├─ Untagged bookmarks:          23
   └─ Suggested for manual review: 112
```

**Implementation**:

```python
# New file: core/quality_reporter.py
from dataclasses import dataclass
from typing import List, Dict
from ..utils.report_generator import ReportGenerator, ReportSection

@dataclass
class QualityMetrics:
    """Collected quality metrics from processing."""
    # Description metrics
    ai_enhanced_count: int
    excerpt_used_count: int
    title_fallback_count: int
    avg_confidence: float

    # Tag metrics
    unique_tags: int
    tagged_bookmarks: int
    avg_tags_per_bookmark: float
    tag_coverage: float

    # Folder metrics
    total_folders: int
    max_depth: int
    reorganized_count: int
    coherence_score: float

    # Attention items
    low_confidence_urls: List[str]
    untagged_urls: List[str]
    review_suggested_urls: List[str]

class QualityReporter:
    """Generate quality assessment reports."""

    def __init__(self, results: PipelineResults):
        self.results = results
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> QualityMetrics: ...

    def generate_report(self, style: str = "rich") -> str:
        generator = ReportGenerator(style=style)

        # Description section
        generator.add_metrics("DESCRIPTION ENHANCEMENT", {
            "Enhanced by AI": f"{self.metrics.ai_enhanced_count} ({self._pct('ai')}%)",
            "Used existing excerpt": f"{self.metrics.excerpt_used_count} ({self._pct('excerpt')}%)",
            "Fallback to title": f"{self.metrics.title_fallback_count} ({self._pct('title')}%)",
            "Average confidence": f"{self.metrics.avg_confidence:.2f}",
        }, icon="📊")

        # ... more sections ...

        return generator.render()

    def get_items_for_review(self) -> List[Bookmark]:
        """Return bookmarks that need manual attention."""
        ...

    def export_review_csv(self, path: Path) -> None:
        """Export items needing review to separate CSV."""
        ...
```

**Deliverables**:
- [x] `core/quality_reporter.py` - Metrics calculation and report generation ✅
- [x] Terminal output with Rich formatting ✅
- [x] Markdown export option ✅
- [x] JSON export for programmatic access ✅
- [x] `--export-review` flag to output items needing attention ✅
- [x] Unit tests for metric calculations (63 tests) ✅
- [x] Integration tests with real pipeline output ✅

---

#### 2.2 Enhanced Progress Visibility (35-45k tokens)
**Location**: `bookmark_processor/utils/progress_tracker.py` (enhance existing)
**Priority**: P1

Improve progress display with stage-based ETA:

```
═══════════════════════════════════════════════════════════════
📊 PROCESSING STATUS - 2h 15m elapsed
═══════════════════════════════════════════════════════════════

Stage 1: URL Validation      ████████████████████ 100% ✓ (32m)
Stage 2: Content Analysis    ████████████░░░░░░░░  62% ⏳ (45m / ~28m left)
Stage 3: AI Processing       ░░░░░░░░░░░░░░░░░░░░   0% ⏸ (~2h 30m)
Stage 4: Tag Generation      ░░░░░░░░░░░░░░░░░░░░   0% ⏸ (~15m)
Stage 5: Output Generation   ░░░░░░░░░░░░░░░░░░░░   0% ⏸ (~2m)

Overall: ████████░░░░░░░░░░░░  38% | ETA: 3h 15m | Memory: 1.8GB
───────────────────────────────────────────────────────────────
Current: Analyzing github.com/example/repo (1,247/2,012)
Speed: 18.3 URLs/min | Errors: 23 (1.8%)
```

**Implementation**:

```python
# Enhance utils/progress_tracker.py
class StageProgress:
    """Track progress for a single processing stage."""
    name: str
    total: int
    completed: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_duration: Optional[timedelta]

    @property
    def status(self) -> str:
        if self.completed_at:
            return "complete"
        elif self.started_at:
            return "in_progress"
        return "pending"

    @property
    def eta(self) -> Optional[timedelta]:
        if not self.started_at or self.completed == 0:
            return self.estimated_duration
        elapsed = datetime.now() - self.started_at
        rate = self.completed / elapsed.total_seconds()
        remaining = self.total - self.completed
        return timedelta(seconds=remaining / rate)

class EnhancedProgressTracker:
    """Multi-stage progress tracking with ETA estimation."""

    STAGE_WEIGHTS = {
        ProcessingStage.URL_VALIDATION: 0.15,
        ProcessingStage.CONTENT_ANALYSIS: 0.25,
        ProcessingStage.AI_PROCESSING: 0.45,
        ProcessingStage.TAG_OPTIMIZATION: 0.10,
        ProcessingStage.OUTPUT_GENERATION: 0.05,
    }

    def __init__(self, total_bookmarks: int):
        self.total = total_bookmarks
        self.stages: Dict[ProcessingStage, StageProgress] = {}
        self._init_stages()

    def start_stage(self, stage: ProcessingStage) -> None: ...
    def update_stage(self, stage: ProcessingStage, completed: int) -> None: ...
    def complete_stage(self, stage: ProcessingStage) -> None: ...

    def render_progress(self) -> str:
        """Render Rich-formatted progress display."""
        ...

    def get_overall_eta(self) -> timedelta:
        """Calculate overall ETA based on stage weights and progress."""
        ...
```

**Deliverables**:
- [x] Enhanced `StageProgress` class ✅
- [x] `EnhancedProgressTracker` with multi-stage support ✅
- [x] Rich console rendering with live update ✅
- [x] Per-stage ETA calculation ✅
- [x] Overall weighted ETA ✅
- [x] Memory usage display ✅
- [x] Error rate tracking ✅
- [x] Unit tests for ETA calculations (62 tests) ✅
- [x] Visual tests (manual verification) ✅

---

### Phase 2 Parallelization

```
Phase 0 Complete
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 2                                   │
├───────────────────────────────┬─────────────────────────────────┤
│ 2.1 Quality Score Report      │ 2.2 Enhanced Progress           │
│ (35-45k)                      │ (35-45k)                        │
│                               │                                 │
│ Needs: 0.1 (Report Gen)       │ Needs: None (enhances existing) │
├───────────────────────────────┴─────────────────────────────────┤
│                    Can run in parallel                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: AI & Tagging Improvements ✅ COMPLETE
**Estimated Tokens**: 90-100k
**Dependencies**: None (works with existing components)
**Enables**: Better output quality
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 86 tests passing

### Objective
Improve AI processing quality and tag generation accuracy.

### Work Items

#### 3.1 Hybrid AI Processing (40-50k tokens)
**Location**: `bookmark_processor/core/ai_processor.py`, `core/ai_router.py`
**Priority**: P1

Route bookmarks to appropriate AI based on complexity:

```python
# New file: core/ai_router.py
class AIRouter:
    """Route bookmarks to optimal AI engine based on content."""

    def __init__(
        self,
        local_processor: EnhancedAIProcessor,
        cloud_processor: Optional[BaseAPIClient] = None,
        config: Optional[HybridAIConfig] = None
    ):
        self.local = local_processor
        self.cloud = cloud_processor
        self.config = config or HybridAIConfig()
        self.cost_tracker = CostTracker()

    def route(self, bookmark: Bookmark, content: ContentData) -> str:
        """Determine which AI engine to use."""

        # Budget exhausted → local only
        if self.cost_tracker.total >= self.config.budget_cap:
            return "local"

        # Simple content → local
        if content.word_count < self.config.simple_threshold:
            return "local"

        # Cloud-required content types
        if content.content_type in self.config.cloud_required_types:
            return "cloud"

        # Try local first, escalate if low confidence
        local_result = self.local.process(bookmark, content)
        if local_result.confidence < self.config.escalation_threshold:
            return "cloud"

        return "local"

    async def process(
        self,
        bookmark: Bookmark,
        content: ContentData
    ) -> AIProcessingResult:
        engine = self.route(bookmark, content)
        if engine == "cloud" and self.cloud:
            return await self.cloud.process_bookmark(bookmark)
        return self.local.process(bookmark, content)

@dataclass
class HybridAIConfig:
    """Configuration for hybrid AI routing."""
    mode: str = "hybrid"  # local, cloud, hybrid
    escalation_threshold: float = 0.7
    budget_cap: float = 5.00  # USD
    simple_threshold: int = 200  # words
    cloud_required_types: List[str] = field(default_factory=lambda: [
        "documentation", "research", "technical"
    ])
```

**CLI Usage**:
```bash
bookmark-processor --input bookmarks.csv --output out.csv \
    --ai-mode hybrid --cloud-budget 5.00
```

**Deliverables**:
- [x] `core/ai_router.py` - Routing logic ✅
- [x] `HybridAIConfig` dataclass ✅
- [x] CLI options `--ai-mode` and `--cloud-budget` ✅
- [x] Cost tracking integration ✅
- [x] Budget exhaustion handling ✅
- [x] Unit tests for routing decisions (24 tests) ✅
- [x] Integration tests with mocked APIs ✅

---

#### 3.2 Improved Tag Generation (30-35k tokens)
**Location**: `bookmark_processor/core/tag_generator.py` (enhance existing)
**Priority**: P2

Add tag hierarchy support and user-defined vocabulary:

```python
# Enhance core/tag_generator.py

@dataclass
class TagConfig:
    """User-configurable tag settings."""
    # Protected tags (never consolidated)
    protected_tags: Set[str] = field(default_factory=lambda: {
        "important", "to-read", "reference", "archived"
    })

    # Synonym mappings
    synonyms: Dict[str, str] = field(default_factory=dict)

    # Hierarchy definitions
    hierarchy: Dict[str, str] = field(default_factory=dict)

    # Target counts
    target_unique_tags: int = 150
    max_tags_per_bookmark: int = 5

class EnhancedTagGenerator(CorpusAwareTagGenerator):
    """Tag generation with hierarchy and user vocabulary support."""

    def __init__(self, config: Optional[TagConfig] = None):
        super().__init__()
        self.config = config or TagConfig()

    def normalize_tag(self, tag: str) -> str:
        """Apply synonyms and hierarchy."""
        # Apply synonym mapping
        normalized = self.config.synonyms.get(tag.lower(), tag.lower())

        # Apply hierarchy if defined
        if normalized in self.config.hierarchy:
            return self.config.hierarchy[normalized]

        return normalized

    def is_protected(self, tag: str) -> bool:
        """Check if tag should be preserved."""
        return tag.lower() in self.config.protected_tags

    def generate_with_confidence(
        self,
        bookmarks: List[Bookmark]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Generate tags with confidence scores."""
        results = {}
        for bookmark in bookmarks:
            tags_with_scores = self._score_tags(bookmark)
            results[bookmark.url] = tags_with_scores
        return results
```

**Configuration file support** (`config.toml`):
```toml
[tags]
protected_tags = ["important", "to-read", "reference"]

[tags.synonyms]
"artificial-intelligence" = "ai"
"ml" = "machine-learning"
"js" = "javascript"

[tags.hierarchy]
"ai" = "technology/ai"
"python" = "technology/programming/python"
```

**Deliverables**:
- [x] `TagConfig` dataclass with TOML loading ✅
- [x] Protected tag handling ✅
- [x] Synonym resolution ✅
- [x] Hierarchy support ✅
- [x] Confidence scores in output ✅
- [x] CLI option `--tag-config` ✅
- [x] Unit tests for all tag transformations (36 tests) ✅
- [x] Sample config file ✅

---

#### 3.3 Folder Organization Improvements (20-25k tokens)
**Location**: `bookmark_processor/core/folder_generator.py` (enhance existing)
**Priority**: P2

Add folder preservation and suggestion modes:

```bash
# Preserve existing folders
bookmark-processor --input bookmarks.csv --output out.csv --preserve-folders

# Suggest folders without auto-assigning
bookmark-processor --input bookmarks.csv --output out.csv --suggest-folders

# Learn from existing structure
bookmark-processor --input bookmarks.csv --output out.csv --learn-folders

# Limit nesting depth
bookmark-processor --input bookmarks.csv --output out.csv --max-folder-depth 2
```

**Implementation**:

```python
# Enhance core/folder_generator.py

class EnhancedFolderGenerator(AIFolderGenerator):
    """Folder generation with preservation and learning modes."""

    def __init__(
        self,
        preserve_existing: bool = False,
        suggest_only: bool = False,
        learn_from_existing: bool = False,
        max_depth: int = 3
    ):
        super().__init__()
        self.preserve_existing = preserve_existing
        self.suggest_only = suggest_only
        self.learn_from_existing = learn_from_existing
        self.max_depth = max_depth

    def generate(self, bookmarks: List[Bookmark]) -> FolderGenerationResult:
        if self.learn_from_existing:
            self._learn_patterns(bookmarks)

        assignments = {}
        suggestions = []

        for bookmark in bookmarks:
            if self.preserve_existing and bookmark.folder:
                assignments[bookmark.url] = bookmark.folder
                continue

            suggestion = self._suggest_folder(bookmark)

            if self.suggest_only:
                suggestions.append(FolderSuggestion(
                    url=bookmark.url,
                    current_folder=bookmark.folder,
                    suggested_folder=suggestion.folder,
                    confidence=suggestion.confidence,
                    reason=suggestion.reason
                ))
            else:
                assignments[bookmark.url] = suggestion.folder

        return FolderGenerationResult(
            assignments=assignments,
            suggestions=suggestions if self.suggest_only else None
        )
```

**Deliverables**:
- [x] `--preserve-folders` flag ✅
- [x] `--suggest-folders` flag with JSON output ✅
- [x] `--learn-folders` pattern learning ✅
- [x] `--max-folder-depth` limit ✅
- [x] Folder suggestions file format ✅
- [x] Unit tests for each mode (26 tests) ✅
- [x] Integration tests ✅

---

### Phase 3 Parallelization

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 3                                   │
│               (No dependencies on Phases 1-2)                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ 3.1 Hybrid AI   │ 3.2 Improved    │ 3.3 Folder Improvements     │
│ (40-50k)        │ Tags (30-35k)   │ (20-25k)                    │
│                 │                 │                             │
│ Independent     │ Independent     │ Independent                 │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                    Can run in parallel                          │
│                  (all enhance existing components)              │
└─────────────────────────────────────────────────────────────────┘
```

**Note**: Phase 3 can run in parallel with Phases 1 and 2 since they all build on different parts of the existing codebase.

---

## Phase 4: Data Source Abstraction ✅ COMPLETE
**Estimated Tokens**: 80-100k
**Dependencies**: None
**Enables**: Phase 5 (MCP Integration)
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 84 tests passing

### Objective
Create an abstraction layer that enables multiple data sources (CSV, MCP, future sources).

### Work Items

#### 4.1 Data Source Protocol (30-35k tokens)
**Location**: `bookmark_processor/core/data_sources/`
**Priority**: P1

```python
# New file: core/data_sources/protocol.py
from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Dict, Any

class BookmarkDataSource(Protocol):
    """Protocol for bookmark data sources."""

    @abstractmethod
    def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        """Fetch bookmarks with optional filtering."""
        ...

    @abstractmethod
    def update_bookmark(self, bookmark: Bookmark) -> bool:
        """Update a single bookmark. Returns success status."""
        ...

    @abstractmethod
    def bulk_update(
        self,
        bookmarks: List[Bookmark]
    ) -> 'BulkUpdateResult':
        """Bulk update multiple bookmarks."""
        ...

    @property
    @abstractmethod
    def supports_incremental(self) -> bool:
        """Whether this source supports incremental updates."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name for this source."""
        ...

@dataclass
class BulkUpdateResult:
    """Result of a bulk update operation."""
    total: int
    succeeded: int
    failed: int
    errors: List[Dict[str, Any]]
```

**Deliverables**:
- [x] `core/data_sources/protocol.py` - Abstract protocol ✅
- [x] `BulkUpdateResult` dataclass ✅
- [x] Type hints and documentation ✅

---

#### 4.2 CSV Data Source (20-25k tokens)
**Location**: `bookmark_processor/core/data_sources/csv_source.py`

Wrap existing CSV handler to implement the new protocol:

```python
# New file: core/data_sources/csv_source.py
class CSVDataSource(BookmarkDataSource):
    """CSV-based data source (wraps existing RaindropCSVHandler)."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        csv_handler: Optional[RaindropCSVHandler] = None
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.handler = csv_handler or RaindropCSVHandler()
        self._bookmarks: Optional[List[Bookmark]] = None

    def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        if self._bookmarks is None:
            df = self.handler.read_csv_file(self.input_path)
            self._bookmarks = self._df_to_bookmarks(df)

        if filters:
            chain = FilterChain.from_dict(filters)
            return chain.apply(self._bookmarks)

        return self._bookmarks

    def update_bookmark(self, bookmark: Bookmark) -> bool:
        # Update in-memory, write happens at end
        for i, b in enumerate(self._bookmarks):
            if b.url == bookmark.url:
                self._bookmarks[i] = bookmark
                return True
        return False

    def bulk_update(self, bookmarks: List[Bookmark]) -> BulkUpdateResult:
        succeeded = 0
        failed = 0
        errors = []

        for bookmark in bookmarks:
            if self.update_bookmark(bookmark):
                succeeded += 1
            else:
                failed += 1
                errors.append({"url": bookmark.url, "error": "Not found"})

        return BulkUpdateResult(
            total=len(bookmarks),
            succeeded=succeeded,
            failed=failed,
            errors=errors
        )

    def save(self) -> None:
        """Write all bookmarks to output file."""
        self.handler.write_csv_file(self._bookmarks, self.output_path)

    @property
    def supports_incremental(self) -> bool:
        return False

    @property
    def source_name(self) -> str:
        return "CSV File"
```

**Deliverables**:
- [x] `core/data_sources/csv_source.py` ✅
- [x] Unit tests (29 tests) ✅
- [x] Integration with existing pipeline ✅

---

#### 4.3 Processing State Tracker (30-35k tokens)
**Location**: `bookmark_processor/core/data_sources/state_tracker.py`

Track which bookmarks have been processed for incremental updates:

```python
# New file: core/data_sources/state_tracker.py
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Set, Optional

class ProcessingStateTracker:
    """Track processing state for incremental updates."""

    DB_SCHEMA = """
    CREATE TABLE IF NOT EXISTS processed_bookmarks (
        url TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        processed_at TIMESTAMP NOT NULL,
        ai_engine TEXT,
        description TEXT,
        tags TEXT
    );

    CREATE TABLE IF NOT EXISTS processing_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        source TEXT NOT NULL,
        total_processed INTEGER DEFAULT 0,
        config_hash TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_processed_at
        ON processed_bookmarks(processed_at);
    """

    def __init__(self, db_path: Path = Path(".bookmark_processor.db")):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(self.DB_SCHEMA)
        self.conn.commit()

    def mark_processed(
        self,
        bookmark: Bookmark,
        content_hash: str,
        ai_engine: str
    ) -> None:
        """Mark a bookmark as processed."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO processed_bookmarks
            (url, content_hash, processed_at, ai_engine, description, tags)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                bookmark.url,
                content_hash,
                datetime.now().isoformat(),
                ai_engine,
                bookmark.enhanced_description,
                ",".join(bookmark.optimized_tags)
            )
        )
        self.conn.commit()

    def needs_processing(self, bookmark: Bookmark) -> bool:
        """Check if bookmark needs (re)processing."""
        current_hash = self._compute_hash(bookmark)
        cursor = self.conn.execute(
            "SELECT content_hash FROM processed_bookmarks WHERE url = ?",
            (bookmark.url,)
        )
        row = cursor.fetchone()

        if row is None:
            return True  # Never processed

        return row[0] != current_hash  # Content changed

    def get_unprocessed(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """Filter to only unprocessed bookmarks."""
        return [b for b in bookmarks if self.needs_processing(b)]

    def _compute_hash(self, bookmark: Bookmark) -> str:
        """Compute content hash for change detection."""
        import hashlib
        content = f"{bookmark.title}|{bookmark.note}|{bookmark.excerpt}"
        return hashlib.md5(content.encode()).hexdigest()
```

**Deliverables**:
- [x] `core/data_sources/state_tracker.py` ✅
- [x] SQLite schema ✅
- [x] Change detection via hashing ✅
- [x] Run history tracking ✅
- [x] Unit tests (35 tests) ✅
- [x] CLI option `--since-last-run` ✅

---

### Phase 4 Parallelization

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 4                                   │
│               (No dependencies on Phases 1-3)                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ 4.1 Data Source │ 4.2 CSV Source  │ 4.3 State Tracker           │
│ Protocol        │ Implementation  │                             │
│ (30-35k)        │ (20-25k)        │ (30-35k)                    │
│                 │                 │                             │
│ Needs: Nothing  │ Needs: 4.1      │ Needs: Nothing              │
├─────────────────┴─────────────────┴─────────────────────────────┤
│     4.1 and 4.3 can run in parallel                             │
│     4.2 must wait for 4.1                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 5: MCP Integration ✅ COMPLETE
**Estimated Tokens**: 90-100k
**Dependencies**: Phase 4 (Data Source Abstraction)
**Enables**: One-command workflow
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 64 tests passing (3 skipped for platform-specific tests)

### Objective
Enable direct Raindrop.io integration via MCP, eliminating manual CSV export/import.

### Work Items

#### 5.1 MCP Client Foundation (35-40k tokens)
**Location**: `bookmark_processor/core/data_sources/mcp_client.py`

```python
# New file: core/data_sources/mcp_client.py
from typing import Any, Dict, List, Optional
import httpx

class MCPClient:
    """Client for communicating with MCP servers."""

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> 'MCPClient':
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool with arguments."""
        response = await self._client.post(
            f"{self.server_url}/tools/{tool_name}",
            json={"arguments": arguments}
        )
        response.raise_for_status()
        return response.json()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        response = await self._client.get(f"{self.server_url}/tools")
        response.raise_for_status()
        return response.json()["tools"]
```

**Deliverables**:
- [x] `core/data_sources/mcp_client.py` ✅
- [x] Async context manager ✅
- [x] Tool calling interface ✅
- [x] Error handling ✅
- [x] Unit tests with mocked server (25 tests) ✅

---

#### 5.2 Raindrop MCP Data Source (40-45k tokens)
**Location**: `bookmark_processor/core/data_sources/raindrop_mcp.py`

```python
# New file: core/data_sources/raindrop_mcp.py
class RaindropMCPDataSource(BookmarkDataSource):
    """Raindrop.io data source via MCP server."""

    def __init__(
        self,
        server_url: str,
        access_token: str,
        state_tracker: Optional[ProcessingStateTracker] = None
    ):
        self.client = MCPClient(server_url)
        self.token = access_token
        self.tracker = state_tracker or ProcessingStateTracker()

    async def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        """Fetch bookmarks from Raindrop.io via MCP."""
        async with self.client:
            # Use bookmark_search MCP tool
            params = {"access_token": self.token}

            if filters:
                if "collection" in filters:
                    params["collection_id"] = filters["collection"]
                if "tags" in filters:
                    params["tags"] = filters["tags"]
                if "query" in filters:
                    params["query"] = filters["query"]

            result = await self.client.call_tool("bookmark_search", params)

            return [
                self._api_to_bookmark(item)
                for item in result.get("raindrops", [])
            ]

    async def update_bookmark(self, bookmark: Bookmark) -> bool:
        """Update a single bookmark in Raindrop.io."""
        async with self.client:
            try:
                await self.client.call_tool("bookmark_manage", {
                    "access_token": self.token,
                    "action": "update",
                    "id": bookmark.id,
                    "updates": self._bookmark_to_api_update(bookmark)
                })
                return True
            except Exception as e:
                logger.error(f"Failed to update {bookmark.url}: {e}")
                return False

    async def bulk_update(
        self,
        bookmarks: List[Bookmark]
    ) -> BulkUpdateResult:
        """Bulk update bookmarks via MCP."""
        async with self.client:
            result = await self.client.call_tool("bulk_edit_raindrops", {
                "access_token": self.token,
                "ids": [b.id for b in bookmarks],
                "updates": [self._bookmark_to_api_update(b) for b in bookmarks]
            })

            return BulkUpdateResult(
                total=len(bookmarks),
                succeeded=result.get("modified", 0),
                failed=len(bookmarks) - result.get("modified", 0),
                errors=result.get("errors", [])
            )

    def _api_to_bookmark(self, data: Dict[str, Any]) -> Bookmark:
        """Convert Raindrop API format to Bookmark."""
        return Bookmark(
            id=str(data["_id"]),
            title=data.get("title", ""),
            note=data.get("note", ""),
            excerpt=data.get("excerpt", ""),
            url=data["link"],
            folder=data.get("collection", {}).get("title", ""),
            tags=data.get("tags", []),
            created=datetime.fromisoformat(data["created"]),
            # ... map remaining fields
        )

    def _bookmark_to_api_update(self, bookmark: Bookmark) -> Dict[str, Any]:
        """Convert Bookmark to Raindrop API update format."""
        return {
            "title": bookmark.get_effective_title(),
            "note": bookmark.get_effective_description(),
            "tags": bookmark.optimized_tags or bookmark.tags,
            "collection": {"$id": self._folder_to_collection_id(bookmark.folder)}
        }

    @property
    def supports_incremental(self) -> bool:
        return True

    @property
    def source_name(self) -> str:
        return "Raindrop.io (MCP)"
```

**Deliverables**:
- [x] `core/data_sources/raindrop_mcp.py` ✅
- [x] API format conversion ✅
- [x] Collection/folder mapping ✅
- [x] Incremental update support ✅
- [x] Unit tests with mocked MCP (21 tests) ✅
- [x] Integration tests (requires real MCP server) ✅

---

#### 5.3 MCP CLI Commands (15-20k tokens)
**Location**: `bookmark_processor/cli.py`

Add MCP-specific commands:

```bash
# Configure Raindrop.io connection
bookmark-processor config set raindrop.token "your-api-token"
bookmark-processor config set raindrop.mcp_server "http://localhost:3000"

# Process via MCP
bookmark-processor enhance --source raindrop

# Process specific collection
bookmark-processor enhance --source raindrop --collection "Tech"

# Process only new bookmarks
bookmark-processor enhance --source raindrop --since-last-run

# Dry run
bookmark-processor enhance --source raindrop --dry-run --preview 10

# Rollback
bookmark-processor rollback --source raindrop
```

**Implementation**:

```python
# cli.py additions
@app.command()
def enhance(
    source: str = typer.Option(
        "csv", "--source", "-s",
        help="Data source: csv or raindrop"
    ),
    input: Optional[Path] = typer.Option(
        None, "--input", "-i",
        help="Input CSV file (required for csv source)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output CSV file (optional for raindrop source)"
    ),
    collection: Optional[str] = typer.Option(
        None, "--collection",
        help="Raindrop.io collection to process"
    ),
    since_last_run: bool = typer.Option(
        False, "--since-last-run",
        help="Only process bookmarks added since last run"
    ),
    since: Optional[str] = typer.Option(
        None, "--since",
        help="Only process bookmarks from this time period (e.g., 7d, 30d)"
    ),
    # ... existing options ...
):
    """Enhance bookmarks from various sources."""

    if source == "csv":
        if not input:
            raise typer.BadParameter("--input required for csv source")
        data_source = CSVDataSource(input, output)
    elif source == "raindrop":
        config = load_config()
        data_source = RaindropMCPDataSource(
            server_url=config.raindrop.mcp_server,
            access_token=config.raindrop.token
        )
    else:
        raise typer.BadParameter(f"Unknown source: {source}")

    # Build filters
    filters = {}
    if collection:
        filters["collection"] = collection
    if since_last_run:
        filters["since_last_run"] = True
    if since:
        filters["since"] = parse_duration(since)

    # Run pipeline with data source
    pipeline = PipelineFactory.create_with_data_source(config, data_source)
    pipeline.execute(filters=filters)
```

**Deliverables**:
- [x] `enhance` command with `--source` option ✅
- [x] `config` subcommand for MCP configuration ✅
- [x] `rollback` command for undo ✅
- [x] Collection filtering ✅
- [x] Time-based filtering ✅
- [x] Help text and examples ✅
- [x] Unit tests for CLI (21 tests) ✅
- [x] End-to-end tests ✅

---

### Phase 5 Parallelization

```
Phase 4 Complete
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 5                                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ 5.1 MCP Client  │                 │ 5.3 CLI Commands            │
│ (35-40k)        │                 │ (15-20k)                    │
│                 │                 │                             │
│ Needs: Nothing  │                 │ Needs: 5.1, 5.2             │
├─────────────────┤                 ├─────────────────────────────┤
│       ↓         │                 │                             │
│ 5.2 Raindrop    │                 │                             │
│ MCP Source      │                 │                             │
│ (40-45k)        │                 │                             │
│ Needs: 4.1, 5.1 │                 │                             │
├─────────────────┴─────────────────┴─────────────────────────────┤
│     5.1 runs first, then 5.2 and 5.3 can partially overlap     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 6: Advanced Features - Export & Monitoring ✅ COMPLETE
**Estimated Tokens**: 70-80k
**Dependencies**: Phase 4 (Data Source Abstraction for consistency)
**Enables**: Broader use cases, ongoing maintenance
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 86 tests passing (3 skipped for network tests)

### Work Items

#### 6.1 Multi-Format Export (35-40k tokens)
**Location**: `bookmark_processor/core/exporters/`

```python
# New directory: core/exporters/
# core/exporters/base.py
class BookmarkExporter(ABC):
    """Base class for bookmark exporters."""

    @abstractmethod
    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> ExportResult: ...

    @property
    @abstractmethod
    def format_name(self) -> str: ...

    @property
    @abstractmethod
    def file_extension(self) -> str: ...

# core/exporters/json_exporter.py
class JSONExporter(BookmarkExporter):
    """Export to JSON format."""

    def export(self, bookmarks: List[Bookmark], output_path: Path) -> ExportResult:
        data = [self._bookmark_to_dict(b) for b in bookmarks]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return ExportResult(path=output_path, count=len(bookmarks))

# core/exporters/markdown_exporter.py
class MarkdownExporter(BookmarkExporter):
    """Export to Markdown files."""
    # Can export to single file or one file per folder

# core/exporters/obsidian_exporter.py
class ObsidianExporter(BookmarkExporter):
    """Export to Obsidian vault format with frontmatter."""

# core/exporters/notion_exporter.py
class NotionExporter(BookmarkExporter):
    """Export to Notion-compatible CSV."""

# core/exporters/opml_exporter.py
class OPMLExporter(BookmarkExporter):
    """Export to OPML for RSS readers."""
```

**CLI**:
```bash
bookmark-processor export --input bookmarks.csv --format json --output bookmarks.json
bookmark-processor export --input bookmarks.csv --format markdown --output bookmarks.md
bookmark-processor export --input bookmarks.csv --format obsidian --output vault/bookmarks/
bookmark-processor export --input bookmarks.csv --format notion --output notion_import.csv
```

**Deliverables**:
- [x] `core/exporters/` directory structure ✅
- [x] Base exporter class ✅
- [x] JSON exporter ✅
- [x] Markdown exporter (single file and directory modes) ✅
- [x] Obsidian exporter with frontmatter ✅
- [x] Notion-compatible CSV exporter ✅
- [x] OPML exporter ✅
- [x] `export` CLI command ✅
- [x] Unit tests for each exporter (48 tests) ✅

---

#### 6.2 Bookmark Health Monitoring (35-40k tokens)
**Location**: `bookmark_processor/core/health_monitor.py`

```python
# New file: core/health_monitor.py
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta

@dataclass
class HealthCheckResult:
    url: str
    status: str  # healthy, redirected, dead, timeout, content_changed
    http_status: Optional[int]
    redirect_url: Optional[str]
    content_changed: bool
    last_checked: datetime
    wayback_url: Optional[str]  # If archived

@dataclass
class HealthReport:
    total: int
    healthy: int
    redirected: int
    dead: int
    timeout: int
    content_changed: int
    newly_dead: int
    recovered: int
    archived: int
    results: List[HealthCheckResult]

class BookmarkHealthMonitor:
    """Monitor bookmark health over time."""

    def __init__(
        self,
        state_tracker: ProcessingStateTracker,
        archive_dead: bool = False
    ):
        self.tracker = state_tracker
        self.archive_dead = archive_dead
        self.wayback = WaybackMachineClient() if archive_dead else None

    async def check_health(
        self,
        bookmarks: List[Bookmark],
        stale_after: Optional[timedelta] = None
    ) -> HealthReport:
        """Check health of bookmarks."""
        results = []

        for bookmark in bookmarks:
            # Skip recently checked
            if stale_after and not self._is_stale(bookmark, stale_after):
                continue

            result = await self._check_single(bookmark)
            results.append(result)

            # Archive dead links if enabled
            if result.status == "dead" and self.archive_dead:
                result.wayback_url = await self._archive_to_wayback(bookmark.url)

        return self._compile_report(results)

    async def _check_single(self, bookmark: Bookmark) -> HealthCheckResult:
        """Check health of a single bookmark."""
        try:
            response = await httpx.head(
                bookmark.url,
                follow_redirects=False,
                timeout=30.0
            )

            if response.status_code == 200:
                return HealthCheckResult(
                    url=bookmark.url,
                    status="healthy",
                    http_status=200,
                    # Check content hash for changes
                    content_changed=await self._content_changed(bookmark)
                )
            elif 300 <= response.status_code < 400:
                return HealthCheckResult(
                    url=bookmark.url,
                    status="redirected",
                    http_status=response.status_code,
                    redirect_url=response.headers.get("Location")
                )
            else:
                return HealthCheckResult(
                    url=bookmark.url,
                    status="dead",
                    http_status=response.status_code
                )
        except httpx.TimeoutException:
            return HealthCheckResult(
                url=bookmark.url,
                status="timeout"
            )
```

**CLI**:
```bash
# Check all bookmarks
bookmark-processor monitor --input bookmarks.csv

# Check stale bookmarks only
bookmark-processor monitor --input bookmarks.csv --stale-after 30d

# Archive dead links
bookmark-processor monitor --input bookmarks.csv --archive-dead

# Report only (no state changes)
bookmark-processor monitor --input bookmarks.csv --report-only
```

**Deliverables**:
- [x] `core/health_monitor.py` ✅
- [x] Wayback Machine integration ✅
- [x] Content change detection ✅
- [x] Health report generation ✅
- [x] `monitor` CLI command ✅
- [x] Unit tests with mocked HTTP (38 tests) ✅
- [x] Integration tests ✅

---

### Phase 6 Parallelization

```
Phase 4 Complete (for consistency with data sources)
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 6                                   │
├───────────────────────────────┬─────────────────────────────────┤
│ 6.1 Multi-Format Export       │ 6.2 Health Monitoring           │
│ (35-40k)                      │ (35-40k)                        │
│                               │                                 │
│ Independent                   │ Uses: State Tracker (Phase 4)   │
├───────────────────────────────┴─────────────────────────────────┤
│                    Can run in parallel                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 7: Interactive Features & Plugins ✅ COMPLETE
**Estimated Tokens**: 90-100k
**Dependencies**: Phase 0 (for report rendering)
**Enables**: Power user workflows, extensibility
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 107 tests passing

### Work Items

#### 7.1 Interactive Processing Mode (40-45k tokens)
**Location**: `bookmark_processor/core/interactive_processor.py`

```python
# New file: core/interactive_processor.py
from rich.console import Console
from rich.prompt import Prompt, Confirm

class InteractiveProcessor:
    """Process bookmarks with user approval."""

    def __init__(
        self,
        pipeline: BookmarkProcessingPipeline,
        confirm_threshold: float = 0.0  # 0 = confirm all
    ):
        self.pipeline = pipeline
        self.confirm_threshold = confirm_threshold
        self.console = Console()

    def process_interactive(
        self,
        bookmarks: List[Bookmark]
    ) -> List[ProcessedBookmark]:
        """Process with interactive approval."""
        results = []

        for i, bookmark in enumerate(bookmarks):
            self._display_bookmark(i, len(bookmarks), bookmark)

            # Get proposed changes
            changes = self.pipeline.propose_changes(bookmark)
            self._display_changes(changes)

            # Skip low-confidence if above threshold
            if changes.confidence >= self.confirm_threshold:
                results.append(self._apply_changes(bookmark, changes))
                continue

            # Get user decision
            action = self._prompt_action()

            if action == "accept_all":
                results.append(self._apply_changes(bookmark, changes))
            elif action == "description_only":
                results.append(self._apply_partial(bookmark, changes, ["description"]))
            elif action == "tags_only":
                results.append(self._apply_partial(bookmark, changes, ["tags"]))
            elif action == "folder_only":
                results.append(self._apply_partial(bookmark, changes, ["folder"]))
            elif action == "skip":
                results.append(bookmark)  # No changes
            elif action == "quit":
                break

        return results

    def _display_bookmark(self, index: int, total: int, bookmark: Bookmark):
        self.console.print(Panel(
            f"[bold]Processing bookmark {index + 1}/{total}[/bold]\n"
            f"URL: {bookmark.url}",
            title="Bookmark"
        ))

    def _display_changes(self, changes: ProposedChanges):
        # Show before/after for description, tags, folder
        ...

    def _prompt_action(self) -> str:
        return Prompt.ask(
            "[A]ccept all | [D]escription only | [T]ags only | "
            "[F]older only | [S]kip | [Q]uit",
            choices=["a", "d", "t", "f", "s", "q"],
            default="a"
        )
```

**CLI**:
```bash
# Full interactive mode
bookmark-processor enhance --input bookmarks.csv --interactive

# Semi-interactive (only confirm low-confidence)
bookmark-processor enhance --input bookmarks.csv --confirm-below 0.7
```

**Deliverables**:
- [x] `core/interactive_processor.py` ✅
- [x] Rich console UI ✅
- [x] Keyboard navigation ✅
- [x] Partial change application ✅
- [x] Progress save on quit ✅
- [x] Unit tests (40 tests) ✅
- [x] Manual testing guide ✅

---

#### 7.2 Plugin Architecture Foundation (50-55k tokens)
**Location**: `bookmark_processor/plugins/`

```python
# New directory: plugins/
# plugins/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BookmarkPlugin(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    def description(self) -> str:
        return ""

    def on_load(self, config: Dict[str, Any]) -> None:
        """Called when plugin is loaded."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass

class ValidatorPlugin(BookmarkPlugin):
    """Plugin for custom URL validation."""

    @abstractmethod
    def validate(
        self,
        url: str,
        content: Optional[str]
    ) -> 'ValidationResult': ...

class AIProcessorPlugin(BookmarkPlugin):
    """Plugin for custom AI processing."""

    @abstractmethod
    def generate_description(
        self,
        bookmark: Bookmark,
        content: str
    ) -> str: ...

class OutputPlugin(BookmarkPlugin):
    """Plugin for custom output formats."""

    @abstractmethod
    def export(
        self,
        bookmarks: List[Bookmark],
        output_path: Path
    ) -> None: ...

# plugins/loader.py
class PluginLoader:
    """Load and manage plugins."""

    def __init__(self, plugin_dir: Path = Path("plugins")):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, BookmarkPlugin] = {}

    def discover_plugins(self) -> List[str]:
        """Find all available plugins."""
        ...

    def load_plugin(self, name: str, config: Dict[str, Any]) -> BookmarkPlugin:
        """Load a plugin by name."""
        ...

    def get_validators(self) -> List[ValidatorPlugin]:
        """Get all loaded validator plugins."""
        ...

    def get_ai_processors(self) -> List[AIProcessorPlugin]:
        """Get all loaded AI processor plugins."""
        ...

# plugins/registry.py
class PluginRegistry:
    """Global plugin registry."""

    _instance = None

    @classmethod
    def instance(cls) -> 'PluginRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, plugin_class: type) -> None: ...
    def get(self, name: str) -> Optional[type]: ...
    def list_all(self) -> List[str]: ...
```

**Example plugins**:
```python
# plugins/examples/paywall_detector.py
class PaywallDetectorPlugin(ValidatorPlugin):
    """Detect paywalled content."""

    name = "paywall-detector"
    version = "1.0.0"

    PAYWALL_INDICATORS = [
        "subscribe to read",
        "premium content",
        "members only"
    ]

    def validate(self, url: str, content: Optional[str]) -> ValidationResult:
        if content:
            is_paywalled = any(
                ind in content.lower()
                for ind in self.PAYWALL_INDICATORS
            )
            return ValidationResult(
                is_valid=True,
                metadata={"is_paywalled": is_paywalled}
            )
        return ValidationResult(is_valid=True)

# plugins/examples/ollama_ai.py
class OllamaPlugin(AIProcessorPlugin):
    """Use local Ollama for AI processing."""

    name = "ollama-ai"
    version = "1.0.0"

    def __init__(self):
        self.model = "llama2"
        self.client = None

    def on_load(self, config: Dict[str, Any]) -> None:
        import ollama
        self.model = config.get("model", "llama2")
        self.client = ollama.Client(
            host=config.get("endpoint", "http://localhost:11434")
        )

    def generate_description(self, bookmark: Bookmark, content: str) -> str:
        response = self.client.generate(
            model=self.model,
            prompt=f"Summarize this webpage in 2 sentences: {content[:2000]}"
        )
        return response["response"]
```

**Configuration** (`config.toml`):
```toml
[plugins]
enabled = ["paywall-detector", "ollama-ai"]

[plugins.ollama-ai]
model = "llama2"
endpoint = "http://localhost:11434"
```

**Deliverables**:
- [x] `plugins/` directory structure ✅
- [x] Base plugin classes ✅
- [x] Plugin loader and registry ✅
- [x] Example validator plugin (PaywallDetector) ✅
- [x] Example AI processor plugin (OllamaAI) ✅
- [x] Configuration support ✅
- [x] CLI `--plugins` option ✅
- [x] Plugin documentation ✅
- [x] Unit tests (67 tests) ✅

---

### Phase 7 Parallelization

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 7                                   │
├───────────────────────────────┬─────────────────────────────────┤
│ 7.1 Interactive Mode          │ 7.2 Plugin Architecture         │
│ (40-45k)                      │ (50-55k)                        │
│                               │                                 │
│ Uses: Report Gen (Phase 0)    │ Independent                     │
├───────────────────────────────┴─────────────────────────────────┤
│                    Can run in parallel                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 8: Architecture Evolution ✅ COMPLETE
**Estimated Tokens**: 90-100k
**Dependencies**: All previous phases (this is a refactor)
**Enables**: Scalability to 100k+ bookmarks
**Status**: ✅ **COMPLETED** (January 2026)
**Tests**: 110 tests passing

### Work Items

#### 8.1 Streaming/Incremental Processing (40-45k tokens)
**Location**: `bookmark_processor/core/streaming/`

Refactor to process bookmarks without loading all into memory:

```python
# New directory: core/streaming/
# core/streaming/reader.py
from typing import Generator, Iterator

class StreamingBookmarkReader:
    """Read bookmarks as a stream instead of loading all into memory."""

    def __init__(self, input_path: Path):
        self.input_path = input_path

    def stream(self) -> Generator[Bookmark, None, None]:
        """Yield bookmarks one at a time."""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Bookmark.from_dict(row)

    def stream_batches(
        self,
        batch_size: int = 100
    ) -> Generator[List[Bookmark], None, None]:
        """Yield bookmarks in batches."""
        batch = []
        for bookmark in self.stream():
            batch.append(bookmark)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# core/streaming/writer.py
class StreamingBookmarkWriter:
    """Write bookmarks incrementally."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._file = None
        self._writer = None

    def __enter__(self) -> 'StreamingBookmarkWriter':
        self._file = open(self.output_path, 'w', newline='', encoding='utf-8')
        return self

    def __exit__(self, *args) -> None:
        if self._file:
            self._file.close()

    def write(self, bookmark: Bookmark) -> None:
        """Write a single bookmark."""
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._get_fieldnames()
            )
            self._writer.writeheader()

        self._writer.writerow(bookmark.to_output_dict())
        self._file.flush()  # Ensure durability

    def write_batch(self, bookmarks: List[Bookmark]) -> None:
        """Write a batch of bookmarks."""
        for bookmark in bookmarks:
            self.write(bookmark)

# core/streaming/pipeline.py
class StreamingPipeline:
    """Process bookmarks in a streaming fashion."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.components = PipelineFactory.create_components(config)

    def execute_streaming(
        self,
        reader: StreamingBookmarkReader,
        writer: StreamingBookmarkWriter
    ) -> PipelineResults:
        """Execute pipeline with streaming I/O."""
        stats = ProcessingStats()

        with writer:
            for batch in reader.stream_batches(self.config.batch_size):
                # Process batch through all stages
                processed = self._process_batch(batch)

                # Write immediately
                writer.write_batch(processed)

                # Update stats
                stats.update(batch=processed)

                # Checkpoint
                self._checkpoint(stats)

        return self._compile_results(stats)
```

**Deliverables**:
- [x] `core/streaming/reader.py` - Generator-based reading ✅
- [x] `core/streaming/writer.py` - Incremental writing ✅
- [x] `core/streaming/pipeline.py` - Streaming pipeline ✅
- [x] Memory usage verification ✅
- [x] Performance benchmarks ✅
- [x] Migration path from existing pipeline ✅
- [x] Unit tests ✅ (35 tests)
- [x] Load tests with large datasets ✅

---

#### 8.2 Enhanced Async Pipeline (35-40k tokens)
**Location**: `bookmark_processor/core/async_pipeline.py`

Improve async processing for better network I/O:

```python
# Enhance core/async_pipeline.py (or create new)
import asyncio
from asyncio import Semaphore
from aiohttp import ClientSession

class AsyncPipelineExecutor:
    """Fully async execution for network-bound operations."""

    def __init__(
        self,
        config: PipelineConfig,
        max_concurrent: int = 20
    ):
        self.config = config
        self.semaphore = Semaphore(max_concurrent)

    async def validate_urls_async(
        self,
        bookmarks: List[Bookmark]
    ) -> Dict[str, ValidationResult]:
        """Validate URLs concurrently."""
        async with ClientSession() as session:
            tasks = [
                self._validate_with_semaphore(session, b.url)
                for b in bookmarks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return dict(zip([b.url for b in bookmarks], results))

    async def _validate_with_semaphore(
        self,
        session: ClientSession,
        url: str
    ) -> ValidationResult:
        async with self.semaphore:
            return await self._validate_url(session, url)

    async def fetch_content_async(
        self,
        urls: List[str]
    ) -> Dict[str, ContentData]:
        """Fetch content concurrently."""
        async with ClientSession() as session:
            tasks = [
                self._fetch_with_semaphore(session, url)
                for url in urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return dict(zip(urls, results))

    async def process_ai_async(
        self,
        bookmarks: List[Bookmark],
        contents: Dict[str, ContentData]
    ) -> Dict[str, AIProcessingResult]:
        """Process AI descriptions concurrently (for cloud APIs)."""
        if self.config.ai_engine == "local":
            # Local AI can't parallelize well
            return self._process_ai_sequential(bookmarks, contents)

        # Cloud APIs can be called in parallel
        tasks = [
            self._process_ai_single(b, contents.get(b.url))
            for b in bookmarks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return dict(zip([b.url for b in bookmarks], results))
```

**Deliverables**:
- [x] `core/async_pipeline.py` with full async support ✅
- [x] Semaphore-based concurrency control ✅
- [x] Async URL validation ✅
- [x] Async content fetching ✅
- [x] Async cloud AI processing ✅
- [x] Performance comparison benchmarks ✅
- [x] Unit tests with async mocks ✅ (35 tests)

---

#### 8.3 Database-Backed State (15-20k tokens)
**Location**: `bookmark_processor/core/database.py`

Enhance the state tracker from Phase 4 with query capabilities:

```python
# Enhance core/data_sources/state_tracker.py → core/database.py
class BookmarkDatabase:
    """Full database backing for processing state and history."""

    ENHANCED_SCHEMA = """
    -- Existing tables from ProcessingStateTracker...

    -- Add query views
    CREATE VIEW IF NOT EXISTS failed_bookmarks AS
        SELECT * FROM processed_bookmarks WHERE status = 'failed';

    CREATE VIEW IF NOT EXISTS recent_bookmarks AS
        SELECT * FROM processed_bookmarks
        ORDER BY processed_at DESC LIMIT 100;

    -- Add full-text search
    CREATE VIRTUAL TABLE IF NOT EXISTS bookmark_fts USING fts5(
        url, title, description, tags
    );
    """

    def query_failed(self) -> List[Bookmark]: ...
    def query_by_date(self, start: datetime, end: datetime) -> List[Bookmark]: ...
    def query_by_status(self, status: str) -> List[Bookmark]: ...
    def search_content(self, query: str) -> List[Bookmark]: ...
    def get_processing_history(self, url: str) -> List[ProcessingRun]: ...
    def compare_runs(self, run1_id: int, run2_id: int) -> RunComparison: ...
```

**Deliverables**:
- [x] Enhanced database schema ✅
- [x] Query methods for common operations ✅
- [x] Full-text search support (FTS5) ✅
- [x] Run comparison utilities ✅
- [x] CLI commands for database queries ✅
- [x] Unit tests ✅ (40 tests)

---

### Phase 8 Parallelization

```
All Previous Phases Complete
          ↓
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 8                                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ 8.1 Streaming   │ 8.2 Async       │ 8.3 Database State          │
│ Processing      │ Pipeline        │                             │
│ (40-45k)        │ (35-40k)        │ (15-20k)                    │
│                 │                 │                             │
│ Independent     │ Can build on    │ Builds on Phase 4           │
│                 │ 8.1 or separate │                             │
├─────────────────┴─────────────────┴─────────────────────────────┤
│     8.1 and 8.3 can run in parallel                             │
│     8.2 can integrate with 8.1 after both complete              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: Full Parallelization Map

```
                              START
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              ┌─────────┐             ┌─────────┐
              │ PHASE 0 │             │ PHASE 3 │  (Independent)
              │ Found-  │             │ AI/Tags │
              │ ation   │             │ Improve │
              │ (60-80k)│             │(90-100k)│
              └────┬────┘             └────┬────┘
                   │                       │
       ┌───────────┼───────────┐           │
       ▼           ▼           ▼           │
┌─────────┐ ┌─────────┐ ┌─────────┐        │
│ PHASE 1 │ │ PHASE 2 │ │ PHASE 4 │        │
│ Quick   │ │ Reports │ │ Data    │        │
│ Wins    │ │ (70-90k)│ │ Source  │        │
│(80-100k)│ │         │ │(80-100k)│        │
└────┬────┘ └────┬────┘ └────┬────┘        │
     │           │           │             │
     │           │           ▼             │
     │           │     ┌─────────┐         │
     │           │     │ PHASE 5 │         │
     │           │     │ MCP     │         │
     │           │     │(90-100k)│         │
     │           │     └────┬────┘         │
     │           │          │              │
     └───────────┼──────────┼──────────────┘
                 │          │
                 ▼          ▼
           ┌─────────┐ ┌─────────┐
           │ PHASE 6 │ │ PHASE 7 │
           │ Export/ │ │ Inter-  │
           │ Monitor │ │ active/ │
           │(70-80k) │ │ Plugins │
           │         │ │(90-100k)│
           └────┬────┘ └────┬────┘
                │           │
                └─────┬─────┘
                      ▼
                ┌─────────┐
                │ PHASE 8 │
                │ Arch    │
                │ Evolve  │
                │(90-100k)│
                └─────────┘
                      │
                      ▼
                    DONE
```

---

## Implementation Order Recommendations

### Fastest Path to User Value
1. **Phase 0** (Foundation) - Required first
2. **Phase 1** (Quick Wins) - Immediate user impact
3. **Phase 2** (Reports) - User visibility
4. **Phase 3** (AI/Tags) - Can run parallel with 1 & 2

### Fastest Path to MCP Integration
1. **Phase 0** (Foundation)
2. **Phase 4** (Data Source Abstraction)
3. **Phase 5** (MCP Integration)

### Recommended Parallel Execution Groups

**Group A** (Can run simultaneously):
- Phase 0 (60-80k)
- Phase 3 (90-100k)

**Group B** (After Group A, can run simultaneously):
- Phase 1 (80-100k) - needs Phase 0
- Phase 2 (70-90k) - needs Phase 0
- Phase 4 (80-100k) - independent

**Group C** (After Phase 4):
- Phase 5 (90-100k)

**Group D** (After Groups A-C basics, can run simultaneously):
- Phase 6 (70-80k)
- Phase 7 (90-100k)

**Group E** (Final):
- Phase 8 (90-100k) - architectural refactor

---

## Total Estimated Tokens

| Phase | Estimated Tokens | Dependencies |
|-------|------------------|--------------|
| 0     | 60-80k           | None |
| 1     | 80-100k          | Phase 0 |
| 2     | 70-90k           | Phase 0 |
| 3     | 90-100k          | None |
| 4     | 80-100k          | None |
| 5     | 90-100k          | Phase 4 |
| 6     | 70-80k           | Phase 4 |
| 7     | 90-100k          | Phase 0 |
| 8     | 90-100k          | All |
| **Total** | **720-850k** | |

---

## Technical Debt Items (Ongoing)

These can be addressed alongside any phase:

- [ ] **Config Consolidation**: Complete Pydantic migration, remove legacy INI
- [ ] **Test Coverage**: Add integration tests, performance regression tests
- [ ] **Documentation**: User guide, troubleshooting, configuration reference
- [ ] **Error Messages**: User-friendly messages with suggested actions

---

*This plan is a living document. Update as implementation progresses and priorities shift.*
