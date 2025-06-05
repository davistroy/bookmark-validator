# Bookmark Validation and Enhancement Tool

A powerful Windows command-line tool that processes raindrop.io bookmark exports to validate URLs, generate AI-enhanced descriptions, and create an optimized tagging system. Perfect for users with large bookmark collections who want to clean, enhance, and better organize their digital bookmarks.

## Features

- **URL Validation**: Validates bookmark accessibility with intelligent retry logic and rate limiting
- **AI-Enhanced Descriptions**: Generates improved descriptions using existing content and AI analysis
- **Smart Tag Optimization**: Creates a coherent tagging system across your entire bookmark collection
- **Checkpoint/Resume**: Saves progress automatically and resumes from interruptions
- **Large Dataset Support**: Efficiently processes 3,500+ bookmarks within 8 hours
- **Windows Executable**: Standalone .exe file requiring no Python installation

## Quick Start

```cmd
bookmark-processor.exe --input bookmarks.csv --output enhanced.csv
```

## Installation

### Option 1: Download Pre-built Executable (Recommended)
1. Download the latest release from the [Releases](https://github.com/yourusername/bookmark-validator/releases) page
2. Extract `bookmark-processor.exe` to your desired location
3. Open Command Prompt and navigate to the directory
4. Run the tool with your bookmark file

### Option 2: Build from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/bookmark-validator.git
cd bookmark-validator

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build executable
python build_exe.py
```

## Usage

### Basic Usage
```cmd
bookmark-processor.exe --input raindrop_export.csv --output enhanced_bookmarks.csv
```

### Resume from Checkpoint
```cmd
bookmark-processor.exe --input raindrop_export.csv --output enhanced_bookmarks.csv --resume
```

### Advanced Options
```cmd
bookmark-processor.exe --input bookmarks.csv --output enhanced.csv --batch-size 50 --verbose
```

### All Command-Line Options
- `--input FILE` - Input CSV file (raindrop.io export format) **[Required]**
- `--output FILE` - Output CSV file (raindrop.io import format) **[Required]**
- `--config FILE` - Custom configuration file
- `--resume` - Resume from existing checkpoint
- `--verbose` - Enable detailed logging
- `--batch-size SIZE` - Processing batch size (default: 100)
- `--max-retries NUM` - Maximum retry attempts (default: 3)
- `--clear-checkpoints` - Clear existing checkpoints and start fresh
- `--help` - Show help message

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

- Windows 11
- 8GB RAM (recommended)
- Internet connection for URL validation
- Sufficient disk space for checkpoint files

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
```cmd
bookmark-processor.exe --input bookmarks.csv --output enhanced.csv --verbose
```

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
python -m pytest tests/ -v
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
- Create an issue on [GitHub](https://github.com/yourusername/bookmark-validator/issues)
- Check the [documentation](docs/) for detailed guides
- Review the task list in `.taskmaster/tasks/` for development progress

---

**Note**: This tool processes bookmarks locally on your machine. No data is sent to external services except for URL validation requests to the target websites.