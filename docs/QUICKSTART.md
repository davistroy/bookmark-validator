# Quick Start Guide

Get up and running with the Bookmark Validation and Enhancement Tool in just a few minutes!

## Prerequisites

- Linux or WSL2 environment
- Python 3.8+ installed
- Internet connection

## 1️⃣ Installation (2 minutes)

### Option A: Quick Install
```bash
# Create and activate virtual environment
python3 -m venv bookmark-env
source bookmark-env/bin/activate

# Install from source
git clone https://github.com/davistroy/bookmark-validator.git
cd bookmark-validator
pip install -r requirements.txt
pip install -e .
```

### Option B: If you already have it installed
```bash
# Activate your environment
source bookmark-env/bin/activate  # or wherever you installed it
cd bookmark-validator
```

## 2️⃣ Prepare Your Data (1 minute)

### Export from Raindrop.io
1. Log into raindrop.io
2. Go to Settings → Data → Export
3. Choose CSV format
4. Download your export file

### Sample Data for Testing
```bash
# Create a test file to try the tool
cat > test_bookmarks.csv << 'EOF'
id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,GitHub,Code repository hosting,GitHub is where developers collaborate,https://github.com,Development,coding git,2024-01-01T00:00:00Z,,,false
2,Python Docs,Official Python documentation,Learn Python programming,https://docs.python.org,Programming,python docs,2024-01-02T00:00:00Z,,,false
3,Stack Overflow,Programming Q&A,Get programming help,https://stackoverflow.com,Help,programming help,2024-01-03T00:00:00Z,,,false
EOF
```

## 3️⃣ Basic Usage (30 seconds)

### Simple Processing
```bash
# Process your bookmarks with default settings
python -m bookmark_processor \
  --input test_bookmarks.csv \
  --output enhanced_bookmarks.csv \
  --verbose
```

**What this does:**
- ✅ Validates all URLs (checks if they're accessible)
- ✅ Generates AI-enhanced descriptions
- ✅ Optimizes tags for better organization
- ✅ Removes duplicate URLs
- ✅ Creates a clean CSV for importing back to raindrop.io

## 4️⃣ View Results (30 seconds)

```bash
# Check the output
head -n 5 enhanced_bookmarks.csv

# View processing logs
tail -n 20 logs/bookmark_processor_*.log
```

**Expected output format:**
```csv
url,folder,title,note,tags,created
https://github.com,Development,GitHub,Enhanced description about GitHub as a collaborative development platform,"development, git, collaboration",2024-01-01T00:00:00Z
https://docs.python.org,Programming,Python Docs,Comprehensive Python documentation and tutorials,"python, documentation, programming",2024-01-02T00:00:00Z
```

## 5️⃣ Import Back to Raindrop.io (1 minute)

1. Log into raindrop.io
2. Go to Settings → Data → Import
3. Choose CSV format
4. Upload your `enhanced_bookmarks.csv` file
5. Follow the import wizard

## ⚡ Common Use Cases

### Process Large Collections
```bash
# For large datasets (1000+ bookmarks)
python -m bookmark_processor \
  --input large_export.csv \
  --output enhanced_large.csv \
  --batch-size 50 \
  --resume \
  --verbose
```

### Resume Interrupted Processing
```bash
# If processing was interrupted, just add --resume
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --resume
```

### Use Cloud AI for Better Descriptions
```bash
# First, add API key to config (see Configuration section below)
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --ai-engine claude
```

### Custom Duplicate Handling
```bash
# Keep the newest bookmark when duplicates found
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --duplicate-strategy newest
```

## 🛠️ Configuration (Optional)

### Set up Cloud AI (Claude or OpenAI)
```bash
# Copy configuration template
cp bookmark_processor/config/user_config.ini.template user_config.ini

# Edit the file
nano user_config.ini
```

Add your API keys:
```ini
[ai]
claude_api_key = your-claude-api-key-here
openai_api_key = your-openai-api-key-here
```

Then use cloud AI:
```bash
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --ai-engine claude \
  --verbose
```

## 🚨 Quick Troubleshooting

### Problem: "Command not found"
```bash
# Make sure you're in the right directory and environment
source bookmark-env/bin/activate
cd bookmark-validator
python -m bookmark_processor --version
```

### Problem: "Permission denied"
```bash
# Use virtual environment
python3 -m venv bookmark-env
source bookmark-env/bin/activate
pip install -r requirements.txt
```

### Problem: Out of memory with large files
```bash
# Reduce batch size
python -m bookmark_processor \
  --input large_file.csv \
  --output output.csv \
  --batch-size 25
```

### Problem: Too many failed URLs
```bash
# Check your internet connection and try with more retries
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --max-retries 5 \
  --verbose
```

## 📊 Understanding the Output

### Processing Information
- **Total bookmarks**: Number of bookmarks in your input file
- **Valid bookmarks**: URLs that were successfully validated
- **Invalid bookmarks**: URLs that couldn't be accessed (404, network errors, etc.)
- **Processing time**: How long the enhancement took

### File Outputs
- **enhanced_bookmarks.csv**: Main output file for importing to raindrop.io
- **logs/**: Detailed processing logs
- **.bookmark_checkpoints/**: Checkpoint files for resume functionality

## ⏱️ Performance Expectations

| Bookmark Count | Expected Time | Memory Usage |
|----------------|---------------|--------------|
| 100 bookmarks  | 2-5 minutes   | < 500MB      |
| 1,000 bookmarks| 20-30 minutes | < 1GB        |
| 3,000+ bookmarks| 2-6 hours     | < 4GB        |

## 🎯 Tips for Best Results

1. **Run overnight for large collections** - Processing 3,000+ bookmarks can take several hours
2. **Use --resume** - If processing gets interrupted, you can continue where you left off
3. **Check your network** - URL validation requires internet access
4. **Start small** - Test with a few bookmarks first to understand the process
5. **Use cloud AI sparingly** - It's more accurate but costs money and has rate limits

## 📖 Next Steps

- **Configuration**: Learn about [advanced configuration options](CONFIGURATION.md)
- **Features**: Explore all [available features](FEATURES.md)
- **Troubleshooting**: Check the [comprehensive troubleshooting guide](TROUBLESHOOTING.md)

## 💡 Pro Tips

### Backup First
```bash
# Always backup your original export
cp raindrop_export.csv raindrop_export_backup.csv
```

### Monitor Progress
```bash
# Run with verbose output to see what's happening
python -m bookmark_processor \
  --input bookmarks.csv \
  --output enhanced.csv \
  --verbose
```

### Test Configuration
```bash
# Test with a small subset first
head -n 10 large_export.csv > test_small.csv
python -m bookmark_processor \
  --input test_small.csv \
  --output test_output.csv \
  --verbose
```

---

**Need Help?** Check the [full documentation](README.md) or [create an issue](https://github.com/davistroy/bookmark-validator/issues) on GitHub.