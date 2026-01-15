# Feature Documentation

Comprehensive guide to all features of the Bookmark Validation and Enhancement Tool.

## Table of Contents

- [Core Features](#core-features)
- [URL Validation](#url-validation)
- [AI-Powered Description Enhancement](#ai-powered-description-enhancement)
- [Intelligent Tag Optimization](#intelligent-tag-optimization)
- [Duplicate Detection and Resolution](#duplicate-detection-and-resolution)
- [Checkpoint and Resume](#checkpoint-and-resume)
- [Progress Tracking and Reporting](#progress-tracking-and-reporting)
- [AI-Generated Folder Organization](#ai-generated-folder-organization)
- [Chrome HTML Import/Export](#chrome-html-import-export)
- [Multi-File Processing](#multi-file-processing)
- [Cloud AI Integration](#cloud-ai-integration)
- [Performance Optimization](#performance-optimization)
- [Security Features](#security-features)
- [Advanced Filtering](#advanced-filtering)
- [Processing Modes](#processing-modes)
- [Quality Reporting](#quality-reporting)
- [Hybrid AI Routing](#hybrid-ai-routing)
- [Tag Configuration](#tag-configuration)
- [Multi-Format Export](#multi-format-export)
- [Health Monitoring](#health-monitoring)
- [Interactive Processing](#interactive-processing)
- [Plugin Architecture](#plugin-architecture)
- [MCP Integration](#mcp-integration)
- [Streaming Processing](#streaming-processing)
- [Async Pipeline](#async-pipeline)
- [Database-Backed State](#database-backed-state)

## Core Features

### 📁 CSV Import/Export

The tool processes raindrop.io bookmark exports and creates enhanced imports.

**Input Format (11 columns):**
```csv
id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,GitHub,My note,Auto excerpt,https://github.com,Development,git code,2024-01-01T00:00:00Z,,,false
```

**Output Format (6 columns):**
```csv
url,folder,title,note,tags,created
https://github.com,Development,GitHub,Enhanced AI description,"development, git, collaboration",2024-01-01T00:00:00Z
```

**Key Features:**
- ✅ Automatic format conversion (11→6 columns)
- ✅ UTF-8 encoding support for international characters
- ✅ Folder structure preservation with forward-slash notation
- ✅ Proper tag formatting (quoted for multiple tags)
- ✅ Malformed CSV detection and recovery

**Example Usage:**
```bash
python -m bookmark_processor \
  --input raindrop_export.csv \
  --output raindrop_import.csv
```

### 🔗 Batch Processing

Efficiently process large collections of bookmarks with configurable batch sizes.

**Features:**
- ✅ Configurable batch sizes (1-1000)
- ✅ Memory-efficient processing
- ✅ Parallel network requests (with rate limiting)
- ✅ Progress tracking with estimated completion time

**Configuration:**
```bash
# Small batches for limited memory
python -m bookmark_processor \
  --input large_file.csv \
  --output enhanced.csv \
  --batch-size 25

# Larger batches for faster processing
python -m bookmark_processor \
  --input file.csv \
  --output enhanced.csv \
  --batch-size 100
```

## URL Validation

### 🌐 Comprehensive URL Validation

Advanced URL validation with intelligent retry logic and site-specific handling.

**Validation Features:**
- ✅ HTTP/HTTPS protocol support
- ✅ Status code validation (200 = valid)
- ✅ SSL certificate verification
- ✅ Redirect following (with loop detection)
- ✅ Timeout handling (configurable)
- ✅ DNS resolution checking

**Error Detection:**
- ❌ 404 Not Found
- ❌ 403 Forbidden / 401 Unauthorized
- ❌ 500+ Server errors
- ❌ Connection timeouts
- ❌ DNS resolution failures
- ❌ SSL certificate errors
- ❌ Malformed URLs

**Example Output:**
```bash
Processing URLs... 100%|████████████| 500/500 [05:23<00:00, 1.54it/s]
✓ Valid URLs: 485 (97%)
✗ Invalid URLs: 15 (3%)
```

### 🔄 Intelligent Retry Logic

**Retry Strategies:**
- **Exponential backoff**: Increasing delays between retries
- **Site-specific delays**: Custom delays for major websites
- **Error-type specific**: Different retry logic for different errors

**Site-Specific Handling:**
```ini
# Automatic delays for major sites
google.com: 2.0 seconds
github.com: 1.5 seconds
youtube.com: 2.0 seconds
linkedin.com: 2.0 seconds
```

**Configuration:**
```bash
# Conservative retry strategy
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --max-retries 5
```

### 🛡️ Security Protection

**SSRF Protection:**
- ✅ Private IP address blocking (10.x.x.x, 192.168.x.x, 172.16-31.x.x)
- ✅ Localhost blocking (127.0.0.1, ::1)
- ✅ URL scheme validation (only HTTP/HTTPS)
- ✅ Maximum redirect limits

**User Agent Rotation:**
- ✅ Realistic browser user agents
- ✅ Automatic rotation to avoid detection
- ✅ Configurable user agent lists

## AI-Powered Description Enhancement

### 🤖 Local AI Processing

Uses the facebook/bart-large-cnn model for high-quality description generation.

**Features:**
- ✅ Uses existing notes/excerpts as input context
- ✅ Generates concise, informative descriptions
- ✅ Preserves user intent while adding value
- ✅ Fallback hierarchy for robust processing
- ✅ No external API dependencies

**Fallback Hierarchy:**
1. **AI Enhancement**: AI-generated description using existing content
2. **Existing Content**: Use original note or excerpt
3. **Meta Description**: Extract from webpage metadata
4. **Title-Based**: Create description from bookmark title
5. **Minimal**: Basic description as last resort

**Example Input/Output:**
```
Input Note: "Git repository hosting"
Input Excerpt: "GitHub is where developers collaborate"
→ AI Output: "GitHub provides comprehensive git repository hosting and collaboration tools for software development teams, featuring issue tracking, code review, and project management capabilities."
```

### ☁️ Cloud AI Integration

Premium cloud AI services for enhanced description quality.

**Supported Services:**
- **Claude (Anthropic)**: High-quality, context-aware descriptions
- **OpenAI (GPT)**: Advanced language understanding

**Features:**
- ✅ API key management through configuration
- ✅ Rate limiting and cost tracking
- ✅ Automatic cost confirmations at intervals
- ✅ Fallback to local AI if cloud fails
- ✅ Batch processing optimization

**Setup:**
```ini
# In user_config.ini
[ai]
claude_api_key = your-claude-api-key
openai_api_key = your-openai-api-key
show_running_costs = true
cost_confirmation_interval = 10.0
```

**Usage:**
```bash
# Use Claude AI
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --ai-engine claude

# Use OpenAI
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --ai-engine openai
```

**Cost Tracking:**
```
Processing with Claude AI...
Estimated cost so far: $2.45
Continue processing? (y/n): y
```

## Intelligent Tag Optimization

### 🏷️ Corpus-Aware Tag Generation

Analyzes your entire bookmark collection to create an optimized tagging system.

**How It Works:**
1. **Corpus Analysis**: Analyze all bookmarks to understand content themes
2. **Tag Candidate Generation**: Create 100-200 high-quality tag candidates
3. **Tag Assignment**: Assign 3-5 relevant tags per bookmark
4. **Tag Optimization**: Replace existing tags with optimized ones

**Features:**
- ✅ Intelligent tag vocabulary (100-200 unique tags)
- ✅ Content-based tag assignment
- ✅ Existing tag context preservation
- ✅ Hierarchical tag support
- ✅ Tag frequency balancing

**Example Tag Optimization:**
```
Original tags: "programming, coding, python, dev, development"
Optimized tags: "python, programming, documentation"

Benefits:
- Reduced redundancy (coding = programming)
- Better specificity (python vs general dev)
- Consistent vocabulary across collection
```

**Configuration:**
```ini
[processing]
max_tags_per_bookmark = 5
target_unique_tags = 150
```

### 📊 Tag Quality Metrics

**Quality Indicators:**
- **Specificity**: Tags are specific enough to be useful
- **Coverage**: Important topics are well-represented
- **Consistency**: Similar content gets similar tags
- **Balance**: No single tag dominates the collection

**Example Output:**
```
Tag Optimization Results:
- Original unique tags: 1,247
- Optimized unique tags: 143
- Average tags per bookmark: 3.2
- Tag coverage score: 94%
```

## Duplicate Detection and Resolution

### 🔍 Advanced Duplicate Detection

Intelligent duplicate URL detection with multiple resolution strategies.

**Detection Methods:**
- ✅ URL normalization (case, trailing slashes, protocols)
- ✅ Query parameter handling
- ✅ Redirect resolution
- ✅ Domain canonicalization (www vs non-www)

**Normalization Examples:**
```
https://Example.com/ → https://example.com
http://github.com/user/repo → https://github.com/user/repo
example.com/?utm_source=google → example.com
```

### 🎯 Resolution Strategies

Choose how to handle duplicate URLs:

**1. Highest Quality (Default)**
Intelligently scores bookmarks based on:
- Completeness of metadata (title, note, tags)
- Recency of creation
- Quality of existing descriptions
- Folder organization

**2. Newest**
Keep the most recently created bookmark.

**3. Oldest**
Keep the original bookmark.

**4. Most Complete**
Keep the bookmark with the most metadata fields filled.

**Usage:**
```bash
# Use different strategies
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --duplicate-strategy newest

# Disable duplicate detection
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --no-duplicates
```

**Example Results:**
```
Duplicate Detection Results:
- Total bookmarks: 1,000
- Unique URLs: 847
- Duplicates removed: 153 (15.3%)
- Strategy used: highest_quality
```

## AI-Generated Folder Organization

### 🗂️ Semantic Folder Structure

Automatically organizes bookmarks into logical folder hierarchies using AI analysis.

**Features:**
- ✅ Content-based folder generation using AI analysis
- ✅ Configurable folder size limits (default: 20 bookmarks per folder)
- ✅ Hierarchical folder structure with semantic grouping
- ✅ Original folder hints integration for better categorization
- ✅ Intelligent folder naming based on content themes

**How It Works:**
The AI analyzes bookmark content including:
- Page titles and descriptions
- URL patterns and domains
- Existing folder structures as hints
- Content analysis results
- Tag context

**Example Organization:**
```
Before (Original Folders):
├── Bookmarks/
├── Programming/Python/
├── Programming/JavaScript/
├── Random/
└── Unsorted/

After (AI-Generated):
├── Education/
│   ├── Programming Tutorials/    (20 bookmarks)
│   └── Online Courses/          (15 bookmarks)
├── Development/
│   ├── Documentation/           (18 bookmarks)
│   ├── Tools & Resources/       (20 bookmarks)
│   └── Code Examples/           (12 bookmarks)
└── Technology/
    ├── News & Updates/          (20 bookmarks)
    └── Research Papers/         (8 bookmarks)
```

**Usage:**
```bash
# Enable AI folder generation (default)
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --generate-folders

# Disable AI folders (use original structure)
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --no-folders

# Custom folder size limit
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --max-bookmarks-per-folder 15
```

**Configuration Options:**
- `--generate-folders`: Enable AI folder generation (default: enabled)
- `--no-folders`: Disable AI folders and preserve original structure
- `--max-bookmarks-per-folder N`: Set maximum bookmarks per folder (default: 20)

**Example Results:**
```
AI Folder Generation Results:
- Total folders created: 12
- Maximum folder depth: 2
- Average bookmarks per folder: 16.4
- Folders with optimal size (15-20): 8
- Processing time: 0.3s
```

## Chrome HTML Import/Export

### 🌐 Chrome Bookmark Format Support

Full support for Chrome/Chromium HTML bookmark format (Netscape-Bookmark-file-1).

**Features:**
- ✅ Import Chrome HTML bookmark files
- ✅ Export to Chrome HTML format with AI-generated folders
- ✅ Preserve bookmark hierarchy and folder structure
- ✅ Support for nested folders and bookmark metadata
- ✅ Automatic format detection for mixed file processing

**Supported Import:**
```html
<!DOCTYPE NETSCAPE-Bookmark-file-1>
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DT><H3 FOLDED>Programming</H3>
<DL><p>
<DT><A HREF="https://github.com" ADD_DATE="1640995200">GitHub</A>
<DT><H3>Python</H3>
<DL><p>
<DT><A HREF="https://python.org" ADD_DATE="1640995300">Python.org</A>
</DL><p>
</DL><p>
```

**Usage:**
```bash
# Import Chrome HTML bookmarks
python -m bookmark_processor \
  --input chrome_bookmarks.html \
  --output enhanced.csv

# Generate both CSV and Chrome HTML output
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --chrome-html \
  --html-title "My Enhanced Bookmarks"

# Custom HTML output path
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --chrome-html \
  --html-output "my_bookmarks_20241201.html"
```

## Multi-File Processing

### 📁 Automatic File Detection

Process multiple bookmark files simultaneously with automatic format detection.

**Features:**
- ✅ Auto-detect CSV and HTML files in current directory
- ✅ Combine multiple bookmark collections
- ✅ Cross-file duplicate detection and resolution
- ✅ Unified AI processing across all files
- ✅ Aggregated statistics and reporting

**Auto-Detection Mode:**
```bash
# Process all bookmark files in current directory
python -m bookmark_processor --output combined_bookmarks.csv

# With advanced options
python -m bookmark_processor \
  --output combined_bookmarks.csv \
  --ai-engine claude \
  --generate-folders \
  --max-bookmarks-per-folder 15
```

**Example Multi-File Results:**
```
Multi-file Processing Results:
- Source files: 3 (2 CSV, 1 HTML)
- Total bookmarks found: 2,847
- After deduplication: 2,203
- Valid bookmarks: 2,156
- AI-generated folders: 47
- Processing time: 8.3 minutes
```

## Checkpoint and Resume

### 💾 Automatic Checkpoint Saving

Robust checkpoint system for processing large collections.

**Features:**
- ✅ Automatic saves every 50 processed items (configurable)
- ✅ Secure checkpoint file encryption
- ✅ Automatic resume detection
- ✅ Checkpoint validation and recovery
- ✅ Multiple checkpoint retention

**How It Works:**
```bash
# Start processing
python -m bookmark_processor --input large_file.csv --output enhanced.csv

# Process gets interrupted at item 237...
# Resume automatically
python -m bookmark_processor --input large_file.csv --output enhanced.csv --resume

# Or start fresh
python -m bookmark_processor --input large_file.csv --output enhanced.csv --clear-checkpoints
```

**Checkpoint Information:**
```json
{
  "timestamp": "2024-01-01T12:30:45Z",
  "processed_count": 237,
  "total_count": 1000,
  "last_processed_id": "237",
  "batch_size": 50,
  "current_stage": "url_validation"
}
```

### 🔐 Checkpoint Security

**Security Features:**
- ✅ Automatic cleanup of sensitive data
- ✅ Secure file permissions (600)
- ✅ Checkpoint validation
- ✅ Corruption detection and recovery

**Configuration:**
```ini
[checkpoint]
enabled = true
save_interval = 50
checkpoint_dir = .bookmark_checkpoints
auto_cleanup = true
```

## Progress Tracking and Reporting

### 📈 Real-Time Progress Display

Comprehensive progress tracking for long-running operations.

**Progress Features:**
- ✅ Real-time progress bars with percentage
- ✅ Processing speed (items/second)
- ✅ Estimated time remaining
- ✅ Stage-specific progress indicators
- ✅ Memory usage monitoring

**Example Output:**
```
Stage 1/4: URL Validation
Validating URLs: 75%|████████████▌    | 750/1000 [02:15<00:45, 5.56it/s]

Stage 2/4: AI Description Generation  
Generating descriptions: 45%|██████▊      | 450/1000 [12:30<15:25, 0.59it/s]

Current: Processing batch 18/20
Memory usage: 1.2GB / 4.0GB
Estimated completion: 14:35 (in 25 minutes)
```

### 📊 Detailed Reporting

**Processing Statistics:**
```
Bookmark Processing Complete!

Summary:
- Total bookmarks processed: 1,000
- Valid URLs: 934 (93.4%)
- Invalid URLs: 66 (6.6%)
- Descriptions enhanced: 1,000 (100%)
- Tags optimized: 1,000 (100%)
- Duplicates removed: 23 (2.3%)
- Processing time: 45 minutes 23 seconds
- Average speed: 0.37 bookmarks/second

Error Breakdown:
- 404 Not Found: 45 (68.2%)
- Connection timeout: 12 (18.2%)
- SSL certificate error: 6 (9.1%)
- Other errors: 3 (4.5%)

Performance Metrics:
- Peak memory usage: 1.8GB
- Network efficiency: 94.2%
- AI processing success: 100%
- Tag optimization quality: 96.8%
```

### 📝 Comprehensive Logging

**Log Features:**
- ✅ Timestamped entries
- ✅ Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- ✅ Performance metrics
- ✅ Error diagnostics
- ✅ Security event logging

**Log Files:**
- `logs/bookmark_processor_YYYYMMDD_HHMMSS.log`: Main processing log
- `logs/errors_YYYYMMDD.log`: Error-specific log
- `logs/performance_YYYYMMDD.log`: Performance metrics

## Performance Optimization

### ⚡ Memory Management

Efficient memory usage for large bookmark collections.

**Memory Features:**
- ✅ Streaming CSV processing
- ✅ Batch-based processing
- ✅ Automatic garbage collection
- ✅ Memory usage monitoring
- ✅ Configurable memory limits

**Memory Optimization:**
```bash
# For systems with limited memory
python -m bookmark_processor \
  --input large_file.csv \
  --output enhanced.csv \
  --batch-size 25
```

**Memory Usage Guidelines:**
| Bookmarks | Recommended RAM | Batch Size |
|-----------|----------------|------------|
| 100-500   | 2GB           | 100        |
| 500-2000  | 4GB           | 50         |
| 2000-5000 | 8GB           | 25         |
| 5000+     | 16GB          | 10         |

### 🌐 Network Optimization

**Network Features:**
- ✅ Connection pooling and reuse
- ✅ Configurable concurrent requests
- ✅ Intelligent rate limiting
- ✅ Request/response compression
- ✅ DNS caching

**Network Configuration:**
```ini
[network]
max_concurrent_requests = 10
timeout = 30
default_delay = 0.5
```

### 🚀 Processing Speed

**Speed Optimizations:**
- ✅ Parallel URL validation
- ✅ Batch AI processing
- ✅ Efficient data structures
- ✅ Caching and memoization
- ✅ Database-free operation

**Performance Expectations:**
| Stage | Speed (items/minute) | Notes |
|-------|---------------------|-------|
| URL Validation | 60-120 | Depends on network |
| AI Enhancement | 20-40 | Local AI processing |
| Tag Optimization | 200+ | Fast corpus analysis |
| Duplicate Detection | 500+ | Efficient algorithms |

## Security Features

### 🔒 Data Protection

**Security Measures:**
- ✅ No data transmission to external services (except chosen cloud AI)
- ✅ Secure temporary file handling
- ✅ API key protection in configuration
- ✅ Secure logging (no sensitive data in logs)
- ✅ Input validation and sanitization

### 🛡️ Network Security

**SSRF Protection:**
- ✅ Private IP range blocking
- ✅ Localhost protection
- ✅ Protocol validation
- ✅ Redirect loop detection
- ✅ Maximum redirect limits

**URL Validation Security:**
```bash
Blocked URLs (for security):
- http://localhost:8080/admin
- https://192.168.1.1/config
- ftp://internal.company.com/files
- javascript:alert('xss')
```

### 🔐 Configuration Security

**Best Practices:**
- ✅ Configuration file permissions (600)
- ✅ API key isolation in separate config
- ✅ No secrets in command-line arguments
- ✅ Secure checkpoint file handling

**Security Configuration:**
```bash
# Secure your configuration
chmod 600 user_config.ini

# Verify file permissions
ls -la user_config.ini
# Should show: -rw------- 1 user user 1234 date user_config.ini
```

## Advanced Filtering

### 🔍 Composable Filter System

Filter bookmarks using powerful, composable filters with AND/OR logic.

**Available Filters:**
- **FolderFilter**: Filter by folder path with regex support
- **TagFilter**: Filter by tags (any or all match modes)
- **DateRangeFilter**: Filter by creation date range
- **DomainFilter**: Filter by URL domain patterns
- **StatusFilter**: Filter by processing status

**Usage:**
```bash
# Filter by folder
python -m bookmark_processor --input data.csv --output out.csv --filter-folder "Programming.*"

# Filter by tags (require all)
python -m bookmark_processor --input data.csv --output out.csv --filter-tag "python,ai" --tag-match-mode all

# Filter by date range
python -m bookmark_processor --input data.csv --output out.csv --filter-date "2024-01-01,2024-12-31"

# Filter by domain
python -m bookmark_processor --input data.csv --output out.csv --filter-domain "github.com,gitlab.com"

# Combine filters
python -m bookmark_processor --input data.csv --output out.csv \
  --filter-folder "Tech" --filter-tag "python" --filter-domain "github.com"
```

**Programmatic Usage:**
```python
from bookmark_processor.core.filters import FilterChain, FolderFilter, TagFilter

# Create composable filter chain
chain = FilterChain([
    FolderFilter(pattern="Programming.*"),
    TagFilter(tags=["python"], match_mode="any")
])
filtered = chain.filter(bookmarks)
```

## Processing Modes

### ⚙️ Granular Processing Control

Control exactly which processing stages to run.

**Processing Stages:**
- `VALIDATION`: URL validation
- `CONTENT_FETCH`: Content extraction
- `AI_DESCRIPTION`: AI description generation
- `TAG_GENERATION`: Tag optimization
- `FOLDER_GENERATION`: Folder organization
- `DUPLICATE_DETECTION`: Duplicate removal

**Usage:**
```bash
# Skip validation (for offline testing)
python -m bookmark_processor --input data.csv --output out.csv --skip-validation

# Skip AI processing (faster)
python -m bookmark_processor --input data.csv --output out.csv --skip-ai

# Tags only
python -m bookmark_processor --input data.csv --output out.csv --tags-only

# Validation only
python -m bookmark_processor --input data.csv --output out.csv --validate-only

# Retry only invalid URLs
python -m bookmark_processor --input data.csv --output out.csv --retry-invalid
```

**Preview and Dry-Run:**
```bash
# Preview what would be processed (no changes)
python -m bookmark_processor --input data.csv --output out.csv --preview

# Dry-run mode (simulate processing)
python -m bookmark_processor --input data.csv --output out.csv --dry-run
```

## Quality Reporting

### 📊 Quality Metrics and Analysis

Comprehensive quality scoring and metrics for your bookmark collection.

**Quality Dimensions:**
- **Completeness**: Percentage of fields populated
- **Validity**: URL validation status
- **Description Quality**: AI-enhanced description quality
- **Tag Quality**: Tag relevance and coverage
- **Organization**: Folder structure quality

**Report Formats:**
```bash
# Generate quality report (Rich terminal output)
python -m bookmark_processor --input data.csv --output out.csv --report rich

# JSON report for automation
python -m bookmark_processor --input data.csv --output out.csv --report json

# Markdown report
python -m bookmark_processor --input data.csv --output out.csv --report markdown
```

**Sample Quality Report:**
```
Quality Report
══════════════════════════════════════════════════════
Overall Score: 87.3%

Completeness:     92% ████████████████████░░
Validity:         95% ███████████████████░░░
Description:      85% █████████████████░░░░░
Tags:             82% ████████████████░░░░░░
Organization:     80% ████████████████░░░░░░

Issues Found:
  - 47 bookmarks missing descriptions
  - 23 URLs returning 404 errors
  - 12 bookmarks with no tags
══════════════════════════════════════════════════════
```

## Hybrid AI Routing

### 🧠 Intelligent AI Engine Selection

Automatically route AI requests to local or cloud based on complexity.

**Routing Logic:**
- **Local AI**: Short content, simple descriptions (faster, free)
- **Cloud AI**: Complex content, long pages (better quality)

**Configuration:**
```ini
[ai]
# Enable hybrid routing
enable_hybrid_routing = true

# Complexity threshold for cloud routing
complexity_threshold = 0.7

# Local model for simple tasks
local_model = facebook/bart-large-cnn

# Cloud fallback
cloud_engine = claude
```

**Usage:**
```bash
# Use hybrid routing (auto-select local vs cloud)
python -m bookmark_processor --input data.csv --output out.csv --ai-engine hybrid

# Force cloud AI
python -m bookmark_processor --input data.csv --output out.csv --ai-engine claude

# Force local AI
python -m bookmark_processor --input data.csv --output out.csv --ai-engine local
```

## Tag Configuration

### 🏷️ User-Defined Tag Vocabulary

Define your own tag vocabulary and hierarchy using TOML configuration.

**Configuration File (tags.toml):**
```toml
[vocabulary]
# Define your preferred tags
allowed_tags = [
    "python", "javascript", "rust", "go",
    "web-development", "machine-learning", "devops",
    "tutorial", "documentation", "tool"
]

# Tag aliases (map variations to canonical tags)
[aliases]
"js" = "javascript"
"py" = "python"
"ml" = "machine-learning"
"ai" = "artificial-intelligence"

# Tag hierarchy
[hierarchy]
"programming" = ["python", "javascript", "rust", "go"]
"ai" = ["machine-learning", "deep-learning", "nlp"]

# Forbidden tags (never use these)
[forbidden]
tags = ["misc", "other", "temp", "todo"]
```

**Usage:**
```bash
# Use custom tag configuration
python -m bookmark_processor --input data.csv --output out.csv --tag-config tags.toml
```

## Multi-Format Export

### 📤 Export to Multiple Formats

Export your bookmarks to various formats for different platforms.

**Supported Formats:**

**1. JSON Export:**
```bash
python -m bookmark_processor --input data.csv --export-json bookmarks.json
```

**2. Markdown Export:**
```bash
python -m bookmark_processor --input data.csv --export-markdown bookmarks.md
```

**3. Obsidian Export:**
```bash
# Creates vault-compatible structure with wikilinks
python -m bookmark_processor --input data.csv --export-obsidian ./obsidian_vault/
```

**4. Notion Export:**
```bash
# Creates Notion-compatible markdown with database properties
python -m bookmark_processor --input data.csv --export-notion notion_export/
```

**5. OPML Export:**
```bash
# Standard OPML format for feed readers
python -m bookmark_processor --input data.csv --export-opml bookmarks.opml
```

**Export All Formats:**
```bash
python -m bookmark_processor --input data.csv \
  --export-json bookmarks.json \
  --export-markdown bookmarks.md \
  --export-obsidian ./obsidian/ \
  --export-opml bookmarks.opml
```

## Health Monitoring

### 🏥 Bookmark Health Checks

Monitor the health of your bookmark collection over time.

**Health Check Features:**
- URL validity monitoring
- Wayback Machine integration for dead links
- Domain accessibility tracking
- SSL certificate monitoring
- Content change detection

**Usage:**
```bash
# Run health check on existing bookmarks
python -m bookmark_processor health-check --input bookmarks.csv

# Generate health report
python -m bookmark_processor health-check --input bookmarks.csv --report health_report.json

# Check with Wayback Machine fallback
python -m bookmark_processor health-check --input bookmarks.csv --wayback-fallback
```

**Sample Health Report:**
```json
{
  "total_checked": 1000,
  "healthy": 934,
  "unhealthy": 66,
  "archived_available": 45,
  "issues": {
    "404_not_found": 45,
    "ssl_expired": 8,
    "domain_unreachable": 13
  },
  "recommendations": [
    "45 bookmarks have Wayback Machine archives available",
    "Consider removing 21 permanently dead links"
  ]
}
```

## Interactive Processing

### 🎮 Interactive Approval Mode

Review and approve changes interactively before they're applied.

**Features:**
- Preview proposed changes before applying
- Accept, reject, or modify individual changes
- Batch approval for similar changes
- Undo capability

**Usage:**
```bash
# Enable interactive mode
python -m bookmark_processor --input data.csv --output out.csv --interactive

# Interactive with approval batching
python -m bookmark_processor --input data.csv --output out.csv --interactive --batch-approve
```

**Interactive Session:**
```
Bookmark: https://github.com/user/repo
────────────────────────────────────────────
Current Title:  "user/repo: A project"
Proposed Title: "GitHub - user/repo: Modern CLI Tool"

Current Tags:   git, code
Proposed Tags:  github, cli-tool, development, open-source

Current Folder: Unsorted
Proposed Folder: Development/Tools

[A]ccept  [R]eject  [M]odify  [S]kip  [B]atch approve similar  [Q]uit
>
```

## Plugin Architecture

### 🔌 Extensible Plugin System

Extend functionality with custom plugins.

**Plugin Types:**
- **ValidatorPlugin**: Custom URL validation logic
- **AIProcessorPlugin**: Custom AI backends (e.g., Ollama)
- **OutputPlugin**: Custom output formats

**Example: Ollama AI Plugin:**
```python
from bookmark_processor.plugins import AIProcessorPlugin

class OllamaPlugin(AIProcessorPlugin):
    name = "ollama"

    def process(self, bookmark, content):
        # Call local Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": content}
        )
        return response.json()["response"]
```

**Example: Paywall Detector Plugin:**
```python
from bookmark_processor.plugins import ValidatorPlugin

class PaywallDetectorPlugin(ValidatorPlugin):
    name = "paywall_detector"

    PAYWALL_INDICATORS = ["subscribe to read", "premium content"]

    def validate(self, bookmark, content):
        for indicator in self.PAYWALL_INDICATORS:
            if indicator in content.lower():
                return {"has_paywall": True}
        return {"has_paywall": False}
```

**Plugin Discovery:**
```bash
# List available plugins
python -m bookmark_processor plugins list

# Enable plugins
python -m bookmark_processor --input data.csv --output out.csv \
  --enable-plugin paywall_detector \
  --enable-plugin ollama
```

## MCP Integration

### 🔗 Model Context Protocol (MCP) Support

Direct integration with Raindrop.io via MCP for real-time sync.

**Features:**
- Read bookmarks directly from Raindrop.io
- Write enhanced bookmarks back
- Incremental sync (process only changes)
- Rollback support

**Commands:**
```bash
# Enhance bookmarks directly in Raindrop.io
python -m bookmark_processor enhance --collection "Programming"

# Configure MCP connection
python -m bookmark_processor config --mcp-server "raindrop"

# Rollback last enhancement
python -m bookmark_processor rollback --run-id 12345
```

**Configuration:**
```ini
[mcp]
enabled = true
server = raindrop
api_token = your_raindrop_token
```

## Streaming Processing

### 🌊 Memory-Efficient Streaming

Process large collections with minimal memory footprint.

**Features:**
- Generator-based file reading
- Incremental writing
- Constant memory usage regardless of file size
- Automatic checkpointing

**Usage:**
```bash
# Enable streaming for large files (>10k bookmarks)
python -m bookmark_processor --input huge_file.csv --output out.csv --streaming

# Streaming with custom buffer
python -m bookmark_processor --input huge_file.csv --output out.csv \
  --streaming --buffer-size 1000
```

**Memory Comparison:**
| Bookmarks | Standard Mode | Streaming Mode |
|-----------|---------------|----------------|
| 10,000    | 2.5 GB        | 256 MB         |
| 50,000    | 12 GB         | 256 MB         |
| 100,000   | OOM           | 256 MB         |

## Async Pipeline

### ⚡ Concurrent Processing

Fully async pipeline for maximum throughput.

**Features:**
- Semaphore-based concurrency control
- Async URL validation
- Async content fetching
- Parallel cloud AI processing
- Configurable concurrency limits

**Usage:**
```bash
# Enable async processing
python -m bookmark_processor --input data.csv --output out.csv --async

# Custom concurrency
python -m bookmark_processor --input data.csv --output out.csv \
  --async --max-concurrent 50
```

**Performance Improvement:**
| Stage | Sync Speed | Async Speed | Improvement |
|-------|------------|-------------|-------------|
| URL Validation | 2/sec | 20/sec | 10x |
| Content Fetch | 1/sec | 15/sec | 15x |
| Cloud AI | 0.5/sec | 5/sec | 10x |

## Database-Backed State

### 💾 SQLite State Management

Persistent state tracking with query capabilities.

**Features:**
- Full processing history
- Failed bookmark queries
- Run comparison
- Full-text search across bookmarks
- Processing statistics

**CLI Commands:**
```bash
# Query failed bookmarks
python -m bookmark_processor db query-failed

# Search bookmarks
python -m bookmark_processor db search "python tutorial"

# Compare processing runs
python -m bookmark_processor db compare-runs --run1 123 --run2 456

# View processing history
python -m bookmark_processor db history --url "https://example.com"
```

**Query Examples:**
```python
from bookmark_processor.core.database import BookmarkDatabase

db = BookmarkDatabase("bookmarks.db")

# Query failed bookmarks
failed = db.query_failed()

# Search by content (FTS5)
results = db.search_content("machine learning python")

# Query by date range
recent = db.query_by_date(start=datetime(2024, 1, 1))

# Compare runs
diff = db.compare_runs(run1_id=123, run2_id=456)
```

## Integration Features

### 📥 Raindrop.io Integration

**Export from Raindrop.io:**
1. Go to Raindrop.io Settings → Data → Export
2. Select CSV format
3. Download export file

**Import to Raindrop.io:**
1. Use the enhanced CSV output file
2. Go to Raindrop.io Settings → Data → Import
3. Select CSV format and upload file

**Format Compatibility:**
- ✅ Full compatibility with raindrop.io CSV format
- ✅ Preserves folder structures
- ✅ Maintains creation timestamps
- ✅ Proper tag formatting

### 🔧 API Integration

**Extensibility Features:**
- ✅ Modular architecture for custom integrations
- ✅ Plugin system for custom AI models
- ✅ Configurable data transformations
- ✅ Custom validation rules

## Limitations and Considerations

### ⚠️ Known Limitations

1. **Platform Support**: Linux/WSL only (no native Windows support)
2. **File Size**: Tested up to 10,000 bookmarks
3. **Network Dependency**: Requires internet for URL validation
4. **Processing Time**: Large collections (3000+) can take several hours
5. **Memory Usage**: Large collections require adequate RAM

### 💡 Best Practices

1. **Start Small**: Test with a subset before processing large collections
2. **Use Checkpoints**: Always enable checkpoints for large collections
3. **Monitor Resources**: Watch memory and network usage
4. **Backup Data**: Keep original export files as backup
5. **Regular Updates**: Keep the tool updated for best performance

### 🎯 Optimization Tips

1. **Batch Size**: Adjust based on available memory
2. **Concurrent Requests**: Lower for slower networks
3. **Retry Logic**: Increase retries for unreliable networks
4. **AI Engine**: Use cloud AI for better quality (costs money)
5. **Checkpoint Frequency**: More frequent saves for unreliable systems