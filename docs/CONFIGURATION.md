# Configuration and Command-Line Reference

Complete reference for all command-line options and configuration settings.

## Table of Contents

- [Command-Line Options](#command-line-options)
- [Configuration File](#configuration-file)
- [Environment Variables](#environment-variables)
- [Examples](#examples)
- [Advanced Configuration](#advanced-configuration)

## Command-Line Options

### Required Arguments

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input CSV file (raindrop.io export format) | `--input bookmarks.csv` |
| `--output` | `-o` | Output CSV file (raindrop.io import format) | `--output enhanced.csv` |

### Optional Arguments

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--version` | `-V` | flag | - | Show version and exit |
| `--help` | `-h` | flag | - | Show help message and exit |
| `--config` | `-c` | path | auto | Custom configuration file path |
| `--verbose` | `-v` | flag | false | Enable verbose logging |
| `--resume` | `-r` | flag | false | Resume from existing checkpoint |
| `--clear-checkpoints` | | flag | false | Clear existing checkpoints and start fresh |
| `--batch-size` | `-b` | int | 100 | Processing batch size (1-1000) |
| `--max-retries` | `-m` | int | 3 | Maximum retry attempts (0-10) |
| `--ai-engine` | | choice | local | AI engine: `local`, `claude`, or `openai` |
| `--no-duplicates` | | flag | false | Disable duplicate URL detection |
| `--duplicate-strategy` | | choice | highest_quality | Duplicate resolution strategy |

### Duplicate Resolution Strategies

| Strategy | Description |
|----------|-------------|
| `newest` | Keep the most recently created bookmark |
| `oldest` | Keep the oldest bookmark |
| `most_complete` | Keep the bookmark with the most metadata |
| `highest_quality` | Use intelligent scoring to pick the best bookmark |

### Command-Line Examples

#### Basic Usage
```bash
# Minimal command
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv

# With verbose output
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --verbose
```

#### Processing Options
```bash
# Small batches for limited memory
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --batch-size 25

# More aggressive retries
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --max-retries 5

# Resume interrupted processing
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --resume
```

#### AI Engine Selection
```bash
# Use Claude AI
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --ai-engine claude

# Use OpenAI
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --ai-engine openai

# Local AI (default)
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --ai-engine local
```

#### Duplicate Handling
```bash
# Disable duplicate detection
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --no-duplicates

# Keep newest duplicates
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --duplicate-strategy newest

# Keep most complete duplicates
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --duplicate-strategy most_complete
```

#### Fresh Start
```bash
# Clear checkpoints and start over
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv --clear-checkpoints
```

## Configuration File

The configuration file uses INI format and allows fine-tuning of application behavior.

### Configuration File Locations

1. **User-specified**: `--config /path/to/config.ini`
2. **Project directory**: `./user_config.ini`
3. **Default**: Built-in defaults

### Creating a Configuration File

```bash
# Copy the template
cp bookmark_processor/config/user_config.ini.template user_config.ini

# Edit with your preferred editor
nano user_config.ini
```

### Configuration Sections

#### [network] - Network Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `timeout` | int | 30 | HTTP request timeout (seconds) |
| `max_retries` | int | 3 | Default retry attempts |
| `default_delay` | float | 0.5 | Default delay between requests (seconds) |
| `max_concurrent_requests` | int | 10 | Maximum concurrent HTTP requests |
| `user_agent_rotation` | bool | true | Rotate user agents to avoid detection |
| `google_delay` | float | 2.0 | Special delay for Google domains |
| `github_delay` | float | 1.5 | Special delay for GitHub |
| `youtube_delay` | float | 2.0 | Special delay for YouTube |
| `linkedin_delay` | float | 2.0 | Special delay for LinkedIn |

#### [processing] - Processing Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `batch_size` | int | 100 | Default processing batch size |
| `max_tags_per_bookmark` | int | 5 | Maximum tags to assign per bookmark |
| `target_unique_tags` | int | 150 | Target number of unique tags across collection |
| `ai_model` | str | facebook/bart-large-cnn | Local AI model identifier |
| `max_description_length` | int | 150 | Maximum length for generated descriptions |
| `use_existing_content` | bool | true | Use existing notes/excerpts as AI input |

#### [checkpoint] - Checkpoint Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | bool | true | Enable checkpoint functionality |
| `save_interval` | int | 50 | Save checkpoint every N processed items |
| `checkpoint_dir` | str | .bookmark_checkpoints | Directory for checkpoint files |
| `auto_cleanup` | bool | true | Automatically clean up old checkpoints |

#### [output] - Output Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_format` | str | raindrop_import | Output format (always raindrop_import) |
| `preserve_folder_structure` | bool | true | Maintain folder hierarchies |
| `include_timestamps` | bool | true | Include creation timestamps |
| `error_log_detailed` | bool | true | Detailed error logging |

#### [logging] - Logging Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `log_level` | str | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `log_file` | str | bookmark_processor.log | Log file name |
| `console_output` | bool | true | Show logs in console |
| `performance_logging` | bool | true | Log performance metrics |

#### [ai] - AI Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `default_engine` | str | local | Default AI engine (local, claude, openai) |
| `claude_api_key` | str | - | Claude API key (keep this secret!) |
| `openai_api_key` | str | - | OpenAI API key (keep this secret!) |
| `claude_rpm` | int | 50 | Claude requests per minute limit |
| `openai_rpm` | int | 60 | OpenAI requests per minute limit |
| `claude_batch_size` | int | 10 | Claude processing batch size |
| `openai_batch_size` | int | 20 | OpenAI processing batch size |
| `show_running_costs` | bool | true | Display running cost estimates |
| `cost_confirmation_interval` | float | 10.0 | Confirm costs every $X USD |

#### [executable] - System Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model_cache_dir` | str | ~/.cache/bookmark-processor/models | AI model cache directory |
| `temp_dir` | str | /tmp/bookmark-processor | Temporary files directory |
| `cleanup_on_exit` | bool | true | Clean up temporary files on exit |

### Sample Configuration File

```ini
[network]
timeout = 45
max_retries = 5
default_delay = 1.0
max_concurrent_requests = 5

[processing]
batch_size = 50
max_tags_per_bookmark = 3
target_unique_tags = 100

[ai]
default_engine = claude
claude_api_key = your-api-key-here
show_running_costs = true
cost_confirmation_interval = 5.0

[logging]
log_level = DEBUG
console_output = true

[checkpoint]
save_interval = 25
```

## Environment Variables

While not the primary configuration method, some settings can be overridden with environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `BOOKMARK_PROCESSOR_CONFIG` | Configuration file path | `/path/to/config.ini` |
| `TRANSFORMERS_CACHE` | AI model cache directory | `/path/to/model/cache` |
| `BOOKMARK_CHECKPOINT_DIR` | Checkpoint directory | `/path/to/checkpoints` |

```bash
# Example usage
export BOOKMARK_PROCESSOR_CONFIG=/home/user/my_config.ini
export TRANSFORMERS_CACHE=/mnt/models
python -m bookmark_processor -i bookmarks.csv -o enhanced.csv
```

## Examples

### Performance Optimization

```bash
# For systems with limited memory
python -m bookmark_processor \
  --input large_file.csv \
  --output enhanced.csv \
  --batch-size 25 \
  --max-retries 2 \
  --verbose
```

Configuration file for performance:
```ini
[network]
max_concurrent_requests = 5
timeout = 60

[processing]
batch_size = 25

[checkpoint]
save_interval = 25
```

### Cloud AI Processing

```bash
# Using Claude with cost controls
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --ai-engine claude \
  --batch-size 10 \
  --verbose
```

Configuration file for cloud AI:
```ini
[ai]
default_engine = claude
claude_api_key = your-claude-api-key
claude_batch_size = 5
show_running_costs = true
cost_confirmation_interval = 5.0

[processing]
batch_size = 10
```

### Conservative Processing

```bash
# Slower but more reliable processing
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --batch-size 10 \
  --max-retries 5 \
  --verbose
```

Configuration file for conservative processing:
```ini
[network]
timeout = 60
max_retries = 5
default_delay = 2.0
max_concurrent_requests = 3

[processing]
batch_size = 10

[checkpoint]
save_interval = 10
```

### Development/Testing

```bash
# Fast processing for testing
python -m bookmark_processor \
  --input test_small.csv \
  --output test_output.csv \
  --batch-size 5 \
  --max-retries 1 \
  --clear-checkpoints \
  --verbose
```

## Advanced Configuration

### Custom User Agent Lists

You can customize the user agent rotation by modifying `bookmark_processor/data/user_agents.txt`:

```bash
# Edit user agents
nano bookmark_processor/data/user_agents.txt
```

### Site-Specific Delays

Customize delays for specific domains by editing `bookmark_processor/data/site_delays.json`:

```json
{
  "github.com": 1.5,
  "stackoverflow.com": 1.0,
  "reddit.com": 2.0,
  "your-slow-site.com": 5.0
}
```

### Processing Pipeline Customization

For advanced users, you can modify processing behavior by editing the configuration:

```ini
[processing]
# Adjust AI processing
use_existing_content = false  # Ignore existing notes
max_description_length = 200  # Longer descriptions

# Tag optimization
max_tags_per_bookmark = 7     # More tags per bookmark
target_unique_tags = 200      # Larger tag vocabulary
```

### Logging Customization

```ini
[logging]
log_level = DEBUG             # More detailed logs
log_file = custom_name.log    # Custom log file name
performance_logging = false   # Disable performance logs
console_output = false        # Quiet processing
```

## Security Considerations

### API Key Management

**❌ Never do this:**
```bash
# Don't put API keys in command line or scripts
python -m bookmark_processor --claude-key sk-1234... # WRONG!
```

**✅ Always do this:**
```ini
# Put API keys in configuration file
[ai]
claude_api_key = your-api-key-here
openai_api_key = your-openai-key-here
```

### File Permissions

```bash
# Secure your configuration file
chmod 600 user_config.ini

# Secure checkpoint directory
chmod 700 .bookmark_checkpoints
```

### Network Security

```ini
[network]
# Use reasonable timeouts
timeout = 30

# Limit concurrent requests to be respectful
max_concurrent_requests = 10

# Enable user agent rotation for privacy
user_agent_rotation = true
```

## Troubleshooting Configuration

### Validate Configuration

```bash
# Test configuration loading
python -c "
from bookmark_processor.config.configuration import Configuration
config = Configuration('user_config.ini')
print('Configuration loaded successfully')
print(f'AI engine: {config.get_ai_engine()}')
print(f'Batch size: {config.getint(\"processing\", \"batch_size\")}')
"
```

### Common Configuration Errors

1. **Invalid file path**: Check that paths exist and are readable
2. **Wrong data types**: Ensure integers are integers, booleans are true/false
3. **Missing API keys**: Verify API keys are set correctly for cloud AI
4. **Permission issues**: Check file permissions on config files

### Debug Configuration Issues

```bash
# Run with debug logging to see configuration loading
python -m bookmark_processor \
  --input test.csv \
  --output test_output.csv \
  --verbose \
  --config user_config.ini
```

Check the logs for configuration-related messages in `logs/bookmark_processor_*.log`.