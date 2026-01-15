# Bookmark Validation and Enhancement Tool

A powerful Linux/WSL command-line tool that processes raindrop.io bookmark exports to validate URLs, generate AI-enhanced descriptions, and create an optimized tagging system. Perfect for users with large bookmark collections who want to clean, enhance, and better organize their digital bookmarks.

## Features

### Core Capabilities
- **Raindrop.io Format Support**: Transforms 11-column exports into 6-column import format
- **URL Validation**: Validates bookmark accessibility with intelligent retry logic and rate limiting
- **AI-Enhanced Descriptions**: Generates improved descriptions using local AI or cloud APIs (Claude, OpenAI)
- **Smart Tag Optimization**: Creates a coherent tagging system with user-defined vocabulary (100-200 unique tags)
- **Duplicate Detection**: Advanced deduplication with multiple resolution strategies
- **Checkpoint/Resume**: Saves progress automatically and resumes from interruptions
- **Large Dataset Support**: Efficiently processes 3,500+ bookmarks with streaming support
- **Intelligent Rate Limiting**: Site-specific delays for major websites (Google, GitHub, YouTube, etc.)

### Advanced Features (New)
- **Multi-Format Export**: Export to JSON, Markdown, Obsidian, Notion, and OPML formats
- **Composable Filters**: Filter by folder, tags, date range, domain, and status with AND/OR logic
- **Quality Reporting**: Comprehensive quality metrics and scoring in Rich, JSON, or Markdown formats
- **Hybrid AI Routing**: Automatically route to local or cloud AI based on content complexity
- **Tag Configuration**: Define custom tag vocabulary, aliases, and hierarchy via TOML
- **Health Monitoring**: Track bookmark health with Wayback Machine integration for dead links
- **Interactive Mode**: Review and approve changes before applying them
- **Plugin Architecture**: Extend with custom validators, AI processors, and output formats
- **MCP Integration**: Direct integration with Raindrop.io via Model Context Protocol
- **Streaming Processing**: Memory-efficient processing for datasets of any size
- **Async Pipeline**: Concurrent processing with 10x throughput improvement
- **Database State**: SQLite-backed state with full-text search and run comparison

## Quick Start

```bash
# Process raindrop.io CSV export
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv

# Process with resume capability for large datasets
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --resume

# Process with cloud AI (Claude or OpenAI)
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine claude

# Preview changes without processing
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --preview

# Filter by folder and export to multiple formats
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv \
  --filter-folder "Programming" --export-json bookmarks.json --export-markdown bookmarks.md

# Interactive mode for review before applying
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --interactive

# Async processing for maximum speed
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --async --max-concurrent 50
```

📖 **New to the tool?** Check out our [Quick Start Guide](docs/QUICKSTART.md) for a step-by-step walkthrough!

## Installation

### Option 1: Docker (Recommended for Quick Start)

**Easiest way to get started - no Python setup required:**

```bash
# Clone the repository
git clone https://github.com/davistroy/bookmark-validator.git
cd bookmark-validator

# Create data directory and add your CSV
mkdir -p data
cp /path/to/your/raindrop_export.csv data/

# Build and run with Docker Compose
docker-compose build
docker-compose run --rm bookmark-processor \
  --input /app/data/raindrop_export.csv \
  --output /app/data/enhanced_bookmarks.csv
```

**Benefits of Docker:**
- ✅ No Python environment setup required
- ✅ Isolated environment with all dependencies
- ✅ Persistent model cache (no re-downloading)
- ✅ Easy checkpoint/resume functionality
- ✅ Works on Linux, macOS, and Windows

📖 **Docker Documentation:** See [DOCKER.md](DOCKER.md) for complete Docker setup and usage guide

### Option 2: Native Python Installation (Linux/WSL)

**Traditional installation for Linux/WSL:**

```bash
# Clone and set up
git clone https://github.com/davistroy/bookmark-validator.git
cd bookmark-validator
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Test installation
python -m bookmark_processor --version
```

📖 **Detailed Instructions:**
- **[Installation Guide](docs/INSTALLATION.md)** - Complete installation instructions for Linux and WSL
- **[WSL Setup Guide](docs/WSL_SETUP.md)** - Step-by-step WSL setup for Windows users

## Usage

### Basic Processing
```bash
# Process raindrop.io CSV export
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv
```

### Checkpoint and Resume
```bash
# Resume from checkpoint for interrupted processing
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv --resume

# Clear checkpoints and start fresh
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv --clear-checkpoints
```

### Custom Processing Options
```bash
# Custom batch size for memory management
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --batch-size 50

# Verbose logging for debugging
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --verbose

# Custom retry attempts for unreliable networks
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --max-retries 5
```

📖 **Complete Documentation:**
- **[Configuration Guide](docs/CONFIGURATION.md)** - All command-line options and configuration settings
- **[Feature Documentation](docs/FEATURES.md)** - Detailed explanation of all features
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## Input/Output Formats

### Input Format (Raindrop.io Export - 11 columns)
```csv
id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
123,"Example Title","User note","Page excerpt","https://example.com","Tech/AI","ai, research","2024-01-01T00:00:00Z","","","false"
```

### Output Format (Raindrop.io Import - 6 columns)
```csv
url,folder,title,note,tags,created
"https://example.com","Tech/AI","Example Title","AI-enhanced description","ai, research, technology","2024-01-01T00:00:00Z"
```

## Features in Detail

### URL Validation
- Validates each bookmark with HTTP status checking
- Intelligent rate limiting for major sites (Google, GitHub, YouTube, etc.)
- Realistic browser simulation to avoid blocking
- Automatic retry logic with exponential backoff
- Continues processing despite individual failures

### AI Description Generation
- Uses existing notes and excerpts as input context
- Generates concise descriptions (100-150 characters)
- Multi-level fallback strategy for robust processing
- Preserves user intent while enhancing with AI

### Tag Optimization
- Analyzes entire bookmark corpus for optimal tagging
- Generates 100-200 unique tags for your collection
- Replaces inconsistent tags with coherent categories
- Ensures proper formatting for raindrop.io import

### Progress Tracking
- Real-time progress bars with percentage completion
- Stage-specific indicators (validation, AI processing, tagging)
- Accurate time estimation for remaining work
- Comprehensive error logging and reporting
- Memory usage monitoring and health status
- Performance metrics and efficiency tracking

### Error Handling and Fallbacks
- Intelligent error categorization (network, validation)
- Automatic retry logic with exponential backoff
- Graceful fallback cascade: AI enhancement → existing content → meta description → title-based
- Health monitoring with system status alerts
- Comprehensive error statistics and recovery metrics

## Performance

- **Processing Speed**: 3,500+ bookmarks in ≤8 hours
- **Memory Usage**: Peak usage <4GB
- **Network Efficiency**: Maximum 10 concurrent connections
- **Checkpoint Frequency**: Saves progress every 50 items
- **Resume Time**: <30 seconds from checkpoint

## Requirements

- Linux (Ubuntu 20.04+) or WSL2
- Python 3.9+ 
- 8GB RAM (recommended)
- Internet connection for URL validation
- Sufficient disk space for checkpoint files

## AI Configuration

### Local AI Processing (Default)

The tool uses the facebook/bart-large-cnn model for AI-powered description generation:

```bash
# Default local AI processing
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv

# Explicitly specify local AI engine
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine local
```

**Benefits of Local AI:**
- ✅ No API costs or external dependencies
- ✅ Privacy-focused (all processing local)
- ✅ No internet required for AI processing
- ✅ Consistent performance regardless of network

### Future Cloud AI Support

The architecture supports cloud AI integration (currently being developed):

```bash
# Future cloud AI support (in development)
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine claude
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine openai
```

## Configuration

The tool uses intelligent defaults but supports customization through command-line options:

```bash
# Custom network settings
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --timeout 60 --max-retries 5

# Custom processing settings  
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --batch-size 50

# Custom checkpoint settings
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --checkpoint-interval 25
```

Configuration files and advanced settings will be available in future releases.

## Troubleshooting

### Common Issues

**Issue**: "Access denied" errors on certain websites
- **Solution**: The tool uses browser simulation, but some sites may still block. These URLs will be logged for manual review.

**Issue**: Processing seems slow
- **Solution**: Reduce batch size with `--batch-size 25` for better progress visibility. Processing time is normal for large collections.

**Issue**: Out of memory errors
- **Solution**: Process in smaller batches or ensure at least 8GB RAM is available.

**Issue**: Cannot resume from checkpoint
- **Solution**: Check that checkpoint files exist in `.bookmark_checkpoints` directory. Use `--clear-checkpoints` to start fresh.

### Debug Mode
For detailed debugging information:
```bash
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --verbose
```

## Project Status

✅ **FULLY FUNCTIONAL AND ACTIVELY MAINTAINED**

🎯 **Core Features (Complete):**
- URL validation with intelligent rate limiting and progress tracking
- AI-powered description enhancement (local + cloud APIs: Claude, OpenAI)
- Corpus-aware tag optimization with user-defined vocabulary (100-200 unique tags)
- Robust checkpoint/resume functionality for large datasets
- Multi-file processing with auto-detection support
- Advanced progress tracking with real-time metrics
- Comprehensive error handling and recovery
- Cost tracking for cloud AI usage
- Production-ready codebase with full type hints

🆕 **Advanced Features (Complete):**
- **Multi-Format Export**: JSON, Markdown, Obsidian, Notion, OPML
- **Composable Filters**: Filter by folder, tags, date, domain, status
- **Quality Reporting**: Rich, JSON, Markdown quality reports
- **Hybrid AI Routing**: Auto-select local vs cloud based on complexity
- **Tag Configuration**: TOML-based vocabulary and hierarchy
- **Health Monitoring**: Wayback Machine integration for dead links
- **Interactive Mode**: Review/approve changes before applying
- **Plugin Architecture**: Custom validators, AI processors, outputs
- **MCP Integration**: Direct Raindrop.io sync via MCP
- **Streaming Pipeline**: Constant memory for any dataset size
- **Async Pipeline**: 10x throughput with concurrent processing
- **Database State**: SQLite with FTS5 search and run comparison

🧪 **Thoroughly Tested (902+ New Tests):**
- 85%+ test coverage across all modules
- Unit tests for all core components
- Integration tests for end-to-end workflows
- Performance validation with 3,500+ bookmark datasets
- 902 new tests across 9 implementation phases
- GitHub Actions CI/CD pipeline with automated testing

📖 **Complete Documentation (Recently Updated):**
- Installation guides for Linux and WSL environments
- Quick start guide with practical examples
- Comprehensive feature documentation (26+ feature sections)
- Configuration management guide
- Cloud AI setup and optimization guide
- Troubleshooting guide with common solutions
- Technical implementation details

## Recent Updates & Improvements

🚀 **Major Architecture Improvements (9 Phases Complete):**

**Phase 0-2: Foundation & Visibility**
- Report generation infrastructure with Rich/JSON/Markdown output
- Composable filter system with AND/OR operators
- Processing mode abstraction with stage flags
- Quality metrics reporting and enhanced progress tracking

**Phase 3-5: AI & Data Abstraction**
- Hybrid AI router for local/cloud selection
- User-defined tag vocabulary via TOML configuration
- Data source protocol abstraction
- MCP integration for direct Raindrop.io sync
- State tracking with SQLite persistence

**Phase 6-8: Advanced Features & Scalability**
- Multi-format exporters (JSON, Markdown, Obsidian, Notion, OPML)
- Bookmark health monitoring with Wayback Machine
- Interactive processing with approval workflow
- Plugin architecture with loader/registry
- Streaming pipeline for unlimited datasets
- Async pipeline with 10x throughput
- Database-backed state with FTS5 search

## Development

### Project Structure
```
bookmark-validator/
├── bookmark_processor/      # Main application code
├── tests/                   # Test suite
├── docs/                    # Documentation
├── .taskmaster/            # Task management
└── requirements.txt        # Python dependencies
```

### Running Tests
```bash
# Install test dependencies (included in requirements.txt)
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_csv_handler.py -v
python -m pytest tests/test_url_validator.py -v
python -m pytest tests/test_integration.py -v

# Run test runner script
python run_tests.py
```

### Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python and PyInstaller
- AI capabilities powered by Hugging Face Transformers
- Designed for raindrop.io bookmark management

## Documentation

📚 **Complete User Guides:**
- **[Docker Setup Guide](DOCKER.md)** - Docker installation and usage (recommended for quick start)
- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in minutes
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[WSL Setup Guide](docs/WSL_SETUP.md)** - Windows users start here
- **[Configuration Reference](docs/CONFIGURATION.md)** - All options and settings
- **[Feature Documentation](docs/FEATURES.md)** - Complete feature overview
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Solutions to common issues

## Support

For issues, questions, or contributions:
- Create an issue on [GitHub](https://github.com/davistroy/bookmark-validator/issues)
- Check the [documentation](docs/) for detailed guides

---

**Note**: This tool processes bookmarks locally on your machine using the facebook/bart-large-cnn model for AI descriptions. No data is sent to external services except for URL validation requests to the target websites. All AI processing happens offline for complete privacy.