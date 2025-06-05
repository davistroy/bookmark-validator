# Bookmark Validation and Enhancement Tool

A powerful Linux/WSL command-line tool that processes raindrop.io bookmark exports to validate URLs, generate AI-enhanced descriptions, and create an optimized tagging system. Perfect for users with large bookmark collections who want to clean, enhance, and better organize their digital bookmarks.

## Features

- **URL Validation**: Validates bookmark accessibility with intelligent retry logic and rate limiting
- **AI-Enhanced Descriptions**: Generates improved descriptions using local AI or cloud APIs (Claude/OpenAI)
- **Smart Tag Optimization**: Creates a coherent tagging system across your entire bookmark collection
- **Checkpoint/Resume**: Saves progress automatically and resumes from interruptions
- **Large Dataset Support**: Efficiently processes 3,500+ bookmarks within 8 hours
- **Linux/WSL Only**: Designed specifically for Linux and Windows Subsystem for Linux (WSL2)
- **Multiple AI Engines**: Choose between local processing, Claude API, or OpenAI API
- **Cost Tracking**: Real-time cost monitoring for cloud AI usage with user confirmation

## Quick Start

```bash
# Using virtual environment
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv

# Using built executable (if available)
./dist/bookmark-processor --input bookmarks.csv --output enhanced.csv
```

## Installation

### Option 1: Using Python Virtual Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/davistroy/bookmark-validator.git
cd bookmark-validator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the tool
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv
```

### Option 2: Build Linux Executable
```bash
# After setting up virtual environment (Option 1)
source venv/bin/activate

# Build executable
./build_linux.sh

# Run executable
./dist/bookmark-processor --input bookmarks.csv --output enhanced.csv
```

## Usage

### Basic Usage
```bash
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv
```

### Resume from Checkpoint
```bash
python -m bookmark_processor --input raindrop_export.csv --output enhanced_bookmarks.csv --resume
```

### Advanced Options
```bash
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --batch-size 50 --verbose
```

### All Command-Line Options
- `--input FILE` - Input CSV file (raindrop.io export format) **[Required]**
- `--output FILE` - Output CSV file (raindrop.io import format) **[Required]**
- `--config FILE` - Custom configuration file
- `--ai-engine ENGINE` - AI engine to use: local (default), claude, or openai
- `--resume` - Resume from existing checkpoint
- `--verbose` - Enable detailed logging
- `--batch-size SIZE` - Processing batch size (default: 100)
- `--max-retries NUM` - Maximum retry attempts (default: 3)
- `--clear-checkpoints` - Clear existing checkpoints and start fresh
- `--help` - Show help message and usage instructions

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

| Engine | Model | Cost/1K Tokens | Quality | Speed | Rate Limit |
|--------|-------|----------------|---------|-------|------------|
| Local | BART | Free | Good | Fast | None |
| Claude | Haiku | ~$0.25/$1.25 | Excellent | Fast | 50 RPM |
| OpenAI | GPT-3.5 | ~$0.50/$1.50 | Excellent | Fast | 60 RPM |

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

## Implementation Status

✅ **Fully Implemented Components:**
- URL Validation Module with intelligent rate limiting
- AI Processing Module with fallback strategies
- Content Analysis Module with BeautifulSoup
- Checkpoint Manager for progress persistence
- Tag Generation with corpus-aware optimization
- Complete Processing Pipeline orchestration
- Progress Tracking with real-time ETA
- Comprehensive Test Suite (unit & integration)
- Linux Build System with PyInstaller
- CSV Handler for raindrop.io format conversion

🧪 **Testing Status:**
- Unit tests: ✅ Complete coverage for core modules
- Integration tests: ✅ End-to-end workflow validation
- Pipeline tests: ✅ Complete processing verification
- Fixture data: ✅ Realistic test scenarios

🏗️ **Build System:**
- Linux executable: ✅ PyInstaller configuration ready
- Virtual environment: ✅ Fully functional
- Dependencies: ✅ All core dependencies installed
- Test runner: ✅ Automated test execution

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

## Support

For issues, questions, or contributions:
- Create an issue on [GitHub](https://github.com/davistroy/bookmark-validator/issues)
- Check the [documentation](docs/) for detailed guides
- Review the task list in `.taskmaster/tasks/` for development progress

---

**Note**: This tool processes bookmarks locally on your machine. No data is sent to external services except for URL validation requests to the target websites.