# Bookmark Validation and Enhancement Tool

A powerful Linux/WSL command-line tool that processes both raindrop.io bookmark exports and Chrome HTML bookmark files to validate URLs, generate AI-enhanced descriptions, and create an optimized tagging system. Perfect for users with large bookmark collections who want to clean, enhance, and better organize their digital bookmarks.

## Features

- **Multiple Input Formats**: Supports raindrop.io CSV exports and Chrome HTML bookmark files
- **Auto-Detection Mode**: Automatically finds and processes all bookmark files when no input specified
- **Dual Output Formats**: Generate both raindrop.io CSV and Chrome HTML bookmark files
- **AI-Generated Folder Structure**: Intelligent semantic organization with max 20 bookmarks per folder
- **URL Validation**: Validates bookmark accessibility with intelligent retry logic and rate limiting
- **AI-Enhanced Descriptions**: Generates improved descriptions using local AI or cloud APIs (Claude/OpenAI)
- **Smart Tag Optimization**: Creates a coherent tagging system across your entire bookmark collection
- **Duplicate Detection**: Advanced deduplication with multiple resolution strategies
- **Checkpoint/Resume**: Saves progress automatically and resumes from interruptions
- **Large Dataset Support**: Efficiently processes 3,500+ bookmarks within 8 hours
- **Linux/WSL Only**: Designed specifically for Linux and Windows Subsystem for Linux (WSL2)
- **Multiple AI Engines**: Choose between local processing, Claude API, or OpenAI API
- **Cost Tracking**: Real-time cost monitoring for cloud AI usage with user confirmation

## Quick Start

```bash
# Auto-detect and process all bookmark files in current directory
python -m bookmark_processor --output enhanced.csv

# Process specific file (supports CSV and HTML)
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv
python -m bookmark_processor --input chrome_bookmarks.html --output enhanced.csv

# Generate both CSV and Chrome HTML output
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv \
  --chrome-html --html-title "My Enhanced Bookmarks"

# With AI folder generation and custom settings
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv \
  --generate-folders --max-bookmarks-per-folder 15 --ai-engine openai

# Disable AI folders and use original structure
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv \
  --no-folders --verbose --resume
```

📖 **New to the tool?** Check out our [Quick Start Guide](docs/QUICKSTART.md) for a step-by-step walkthrough!

## Installation

**Quick install for Linux/WSL:**

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

### Auto-Detection Mode (New!)
```bash
# Automatically process all CSV and HTML bookmark files in current directory
python -m bookmark_processor --output enhanced_bookmarks.csv
```

### Single File Processing
```bash
# Process raindrop.io CSV export
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv

# Process Chrome HTML bookmarks
python -m bookmark_processor --input chrome_bookmarks.html --output enhanced_bookmarks.csv
```

### Duplicate Detection
```bash
# Advanced duplicate detection with quality-based resolution
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --duplicate-strategy highest_quality

# Disable duplicate detection
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --no-duplicates
```

### Resume from Checkpoint
```bash
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv --resume
```

### Advanced Options
```bash
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv \
  --batch-size 50 --verbose --ai-engine claude --duplicate-strategy newest
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

### Cost Tracking (Cloud AI)
- Real-time cost monitoring for Claude and OpenAI APIs
- User confirmation prompts at configurable intervals ($10 default)
- Detailed cost breakdowns by provider and operation
- Cost estimation before processing begins
- Historical usage analysis and reporting
- Emergency stop functionality for cost control

### Error Handling and Fallbacks
- Intelligent error categorization (network, API, validation)
- Automatic retry logic with exponential backoff
- Graceful fallback cascade: Cloud AI → Local AI → Basic descriptions
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
- Python 3.12+ 
- 8GB RAM (recommended)
- Internet connection for URL validation
- Sufficient disk space for checkpoint files

## AI Configuration

### Setting Up Cloud AI APIs

1. **Create Configuration File**:
   ```bash
   # Copy default config
   cp bookmark_processor/config/default_config.ini bookmark_processor/config/user_config.ini
   
   # IMPORTANT: Add to .gitignore
   echo "bookmark_processor/config/user_config.ini" >> .gitignore
   ```

2. **Add API Keys** to `user_config.ini`:
   ```ini
   [ai]
   # Uncomment and add your API keys
   claude_api_key = your-claude-api-key-here
   openai_api_key = your-openai-api-key-here
   ```

3. **Use Cloud AI**:
   ```bash
   # Using Claude (Claude 3 Haiku for cost-effectiveness)
   python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine claude
   
   # Using OpenAI (GPT-3.5-turbo for cost-effectiveness)
   python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine openai
   ```

### AI Engine Comparison

| Engine | Model | Cost/1K Tokens | Quality | Speed | Rate Limit | Best For |
|--------|-------|----------------|---------|-------|------------|----------|
| Local | BART | Free | Good | Fast | None | Privacy, Offline |
| Claude | Haiku | ~$0.25/$1.25 | Excellent | Fast | 50 RPM | Quality, Context |
| OpenAI | GPT-3.5 | ~$0.50/$1.50 | Excellent | Fast | 60 RPM | Speed, Efficiency |

> 📖 **Detailed Guide**: See [Cloud AI Integration Guide](docs/CLOUD_AI_GUIDE.md) for comprehensive setup, cost tracking, and optimization tips.

## Configuration

Create a custom configuration file to override default settings:

```ini
[network]
timeout = 30
max_retries = 3
default_delay = 0.5

[processing]
batch_size = 100
max_tags_per_bookmark = 5
target_unique_tags = 150

[checkpoint]
enabled = true
save_interval = 50
```

Save as `config.ini` and use with `--config config.ini`

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

✅ **FULLY COMPLETE AND FUNCTIONAL**

🎯 **Core Features:**
- URL validation with intelligent rate limiting
- AI-powered description enhancement (local + cloud)
- Corpus-aware tag optimization 
- Checkpoint/resume functionality
- Progress tracking and error handling
- Comprehensive documentation

🧪 **Thoroughly Tested:**
- Unit tests for all components
- Integration tests for end-to-end workflows
- Performance validation with large datasets
- Error handling and recovery scenarios

📖 **Complete Documentation:**
- Installation guides for Linux and WSL
- Quick start guide and tutorials
- Feature documentation and configuration
- Troubleshooting guide and FAQ

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
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python run_tests.py --test-type unit
python run_tests.py --test-type integration

# Run tests with coverage
python run_tests.py --coverage
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

**Note**: This tool processes bookmarks locally on your machine. No data is sent to external services except for URL validation requests to the target websites.