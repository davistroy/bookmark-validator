# Troubleshooting Guide and FAQ

Comprehensive troubleshooting guide for common issues and frequently asked questions.

## Table of Contents

- [Quick Diagnosis](#quick-diagnosis)
- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Network and URL Issues](#network-and-url-issues)
- [AI Processing Issues](#ai-processing-issues)
- [File and Data Issues](#file-and-data-issues)
- [WSL-Specific Issues](#wsl-specific-issues)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Getting Help](#getting-help)

## Quick Diagnosis

### 🔍 Diagnostic Commands

**Check Installation:**
```bash
# Verify the tool is installed
python -m bookmark_processor --version

# Check Python version (requires 3.8+)
python3 --version

# Test basic functionality
python -m bookmark_processor --help
```

**Check Environment:**
```bash
# Verify virtual environment
which python
echo $VIRTUAL_ENV

# Check disk space
df -h .

# Check memory
free -h
```

**Check Configuration:**
```bash
# Test configuration loading
python -c "
from bookmark_processor.config.configuration import Configuration
config = Configuration()
print('Configuration loaded successfully')
"
```

## Installation Issues

### ❌ Python Version Errors

**Problem:** `Python 3.8+ required`

**Solution:**
```bash
# Check current version
python3 --version

# Ubuntu/Debian: Install newer Python
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip

# Use specific Python version
python3.9 -m venv venv
source venv/bin/activate
```

### ❌ Permission Denied Errors

**Problem:** `Permission denied when installing packages`

**Solutions:**

1. **Use Virtual Environment (Recommended):**
```bash
python3 -m venv bookmark-env
source bookmark-env/bin/activate
pip install -r requirements.txt
```

2. **User Installation:**
```bash
pip install --user -r requirements.txt
```

3. **Fix Permissions:**
```bash
# If using system Python (not recommended)
sudo chown -R $USER ~/.local
```

### ❌ Package Installation Failures

**Problem:** `Failed building wheel for package`

**Solutions:**

1. **Update pip and setuptools:**
```bash
pip install --upgrade pip setuptools wheel
```

2. **Install build dependencies:**
```bash
# Ubuntu/Debian
sudo apt install python3-dev build-essential

# CentOS/RHEL/Fedora
sudo dnf install python3-devel gcc gcc-c++ make
```

3. **Use pre-compiled packages:**
```bash
pip install --only-binary=all -r requirements.txt
```

### ❌ Git Clone Issues

**Problem:** `Repository not found` or connection issues

**Solutions:**

1. **Check internet connection:**
```bash
ping github.com
```

2. **Use HTTPS instead of SSH:**
```bash
git clone https://github.com/davistroy/bookmark-validator.git
```

3. **Use alternative download:**
```bash
# Download ZIP instead
wget https://github.com/davistroy/bookmark-validator/archive/main.zip
unzip main.zip
cd bookmark-validator-main
```

## Runtime Errors

### ❌ Module Not Found Errors

**Problem:** `ModuleNotFoundError: No module named 'bookmark_processor'`

**Diagnosis:**
```bash
# Check if in correct directory
ls -la | grep bookmark_processor

# Check if installed
pip list | grep bookmark

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**Solutions:**

1. **Activate virtual environment:**
```bash
source venv/bin/activate  # or your env name
```

2. **Install in development mode:**
```bash
pip install -e .
```

3. **Run from correct directory:**
```bash
cd bookmark-validator
python -m bookmark_processor --help
```

### ❌ Command Not Found

**Problem:** `bookmark-processor: command not found`

**Solutions:**

1. **Use Python module syntax:**
```bash
python -m bookmark_processor --input file.csv --output out.csv
```

2. **Check installation method:**
```bash
# If installed via pip
pip show bookmark-validator

# Add to PATH if using --user
export PATH=$PATH:~/.local/bin
```

### ❌ Configuration Errors

**Problem:** `Configuration file not found` or `Invalid configuration`

**Solutions:**

1. **Create configuration file:**
```bash
cp bookmark_processor/config/user_config.ini.template user_config.ini
```

2. **Check file permissions:**
```bash
ls -la user_config.ini
chmod 600 user_config.ini
```

3. **Validate configuration syntax:**
```bash
python -c "
import configparser
config = configparser.ConfigParser()
config.read('user_config.ini')
print('Configuration file is valid')
"
```

## Performance Issues

### 🐌 Slow Processing

**Problem:** Processing is taking too long

**Diagnosis:**
```bash
# Check system resources
top
htop  # if available
iotop  # if available

# Monitor network
netstat -i
```

**Solutions:**

1. **Optimize batch size:**
```bash
# For limited memory
python -m bookmark_processor \
  --input file.csv --output out.csv \
  --batch-size 25

# For more memory
python -m bookmark_processor \
  --input file.csv --output out.csv \
  --batch-size 100
```

2. **Reduce concurrent requests:**
```bash
# In user_config.ini
[network]
max_concurrent_requests = 5
default_delay = 1.0
```

3. **Disable AI processing temporarily:**
```bash
# In user_config.ini
[processing]
use_existing_content = true
max_description_length = 0  # Use existing descriptions
```

### 💾 Memory Issues

**Problem:** `MemoryError` or system becomes unresponsive

**Diagnosis:**
```bash
# Check memory usage
free -h
ps aux | grep python
```

**Solutions:**

1. **Reduce batch size:**
```bash
python -m bookmark_processor \
  --input file.csv --output out.csv \
  --batch-size 10
```

2. **Close other applications:**
```bash
# Free up memory
sudo systemctl stop unnecessary-service
```

3. **Use swap space:**
```bash
# Add swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

4. **Process in chunks:**
```bash
# Split large files
head -n 1000 large_file.csv > chunk1.csv
tail -n +1001 large_file.csv | head -n 1000 > chunk2.csv
```

### 🕐 Timeout Issues

**Problem:** Processing hangs or times out

**Solutions:**

1. **Increase timeouts:**
```bash
# In user_config.ini
[network]
timeout = 60
max_retries = 5
```

2. **Use verbose logging:**
```bash
python -m bookmark_processor \
  --input file.csv --output out.csv \
  --verbose
```

3. **Check specific URLs:**
```bash
# Test problematic URLs manually
curl -I "https://problem-url.com"
```

## Network and URL Issues

### 🌐 Connection Problems

**Problem:** `Connection failed` or `Network unreachable`

**Diagnosis:**
```bash
# Test internet connectivity
ping google.com
curl -I https://httpbin.org/get

# Check DNS
nslookup google.com
dig google.com
```

**Solutions:**

1. **Check network configuration:**
```bash
# Check network interfaces
ip addr show
route -n
```

2. **Use different DNS:**
```bash
# In /etc/resolv.conf (temporary)
nameserver 8.8.8.8
nameserver 1.1.1.1
```

3. **Configure proxy if needed:**
```bash
# Set proxy environment variables
export http_proxy=http://proxy.company.com:8080
export https_proxy=http://proxy.company.com:8080
```

### 🔒 SSL/TLS Issues

**Problem:** `SSL certificate verification failed`

**Diagnosis:**
```bash
# Test SSL connection
openssl s_client -connect example.com:443
curl -I https://example.com
```

**Solutions:**

1. **Update certificates:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ca-certificates

# CentOS/RHEL/Fedora
sudo dnf update ca-certificates
```

2. **Check system time:**
```bash
# SSL certificates are time-sensitive
date
sudo ntpdate -s time.nist.gov  # if ntpdate available
```

3. **Temporary workaround (not recommended for production):**
```bash
# In user_config.ini (only for testing)
[network]
verify_ssl = false
```

### ❌ High Error Rate

**Problem:** Many URLs failing validation

**Diagnosis:**
```bash
# Check error patterns in logs
grep "ERROR" logs/bookmark_processor_*.log | head -20
```

**Solutions:**

1. **Increase retry attempts:**
```bash
python -m bookmark_processor \
  --input file.csv --output out.csv \
  --max-retries 5
```

2. **Adjust rate limiting:**
```bash
# In user_config.ini
[network]
default_delay = 2.0
max_concurrent_requests = 3
```

3. **Check for systematic issues:**
```bash
# Look for patterns
grep "404\|403\|500" logs/bookmark_processor_*.log | cut -d' ' -f4 | sort | uniq -c
```

## AI Processing Issues

### 🤖 Local AI Problems

**Problem:** `Model loading failed` or AI processing errors

**Diagnosis:**
```bash
# Check model cache
ls -la ~/.cache/huggingface/transformers/

# Test model loading
python -c "
from transformers import pipeline
model = pipeline('summarization', model='facebook/bart-large-cnn')
print('Model loaded successfully')
"
```

**Solutions:**

1. **Clear model cache:**
```bash
rm -rf ~/.cache/huggingface/transformers/
```

2. **Download model manually:**
```bash
python -c "
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
print('Model downloaded successfully')
"
```

3. **Use fallback processing:**
```bash
# In user_config.ini
[processing]
use_existing_content = true
ai_model = none  # Disable AI processing
```

### ☁️ Cloud AI Issues

**Problem:** Cloud AI API errors

**Diagnosis:**
```bash
# Test API keys
python -c "
from bookmark_processor.config.configuration import Configuration
config = Configuration()
print('Claude API key:', 'configured' if config.has_api_key('claude') else 'missing')
print('OpenAI API key:', 'configured' if config.has_api_key('openai') else 'missing')
"
```

**Solutions:**

1. **Verify API keys:**
```bash
# Test Claude API
curl -H "Authorization: Bearer YOUR_CLAUDE_KEY" \
     -H "Content-Type: application/json" \
     https://api.anthropic.com/v1/messages

# Test OpenAI API
curl -H "Authorization: Bearer YOUR_OPENAI_KEY" \
     -H "Content-Type: application/json" \
     https://api.openai.com/v1/models
```

2. **Check rate limits:**
```bash
# In user_config.ini
[ai]
claude_rpm = 30  # Reduce rate
openai_rpm = 40
```

3. **Monitor costs:**
```bash
# In user_config.ini
[ai]
show_running_costs = true
cost_confirmation_interval = 5.0
```

## File and Data Issues

### 📄 CSV Format Problems

**Problem:** `CSV parsing error` or `Invalid format`

**Diagnosis:**
```bash
# Check file structure
head -5 input.csv
wc -l input.csv
file input.csv
```

**Solutions:**

1. **Check CSV structure:**
```bash
# Verify column count
head -1 input.csv | tr ',' '\n' | wc -l
# Should be 11 for raindrop export
```

2. **Fix encoding issues:**
```bash
# Convert to UTF-8
iconv -f ISO-8859-1 -t UTF-8 input.csv > input_utf8.csv
```

3. **Clean malformed CSV:**
```bash
# Basic CSV cleaning
python -c "
import pandas as pd
df = pd.read_csv('input.csv', error_bad_lines=False)
df.to_csv('cleaned_input.csv', index=False)
"
```

### 💾 File Permission Issues

**Problem:** `Permission denied` when reading/writing files

**Solutions:**

1. **Check file permissions:**
```bash
ls -la input.csv output.csv
```

2. **Fix permissions:**
```bash
chmod 644 input.csv
chmod 755 $(dirname output.csv)
```

3. **Use different directory:**
```bash
# Work in home directory
cp input.csv ~/
cd ~
python -m bookmark_processor --input input.csv --output output.csv
```

### 🔄 Checkpoint Issues

**Problem:** `Checkpoint corruption` or resume failures

**Diagnosis:**
```bash
# Check checkpoint files
ls -la .bookmark_checkpoints/
```

**Solutions:**

1. **Clear corrupted checkpoints:**
```bash
python -m bookmark_processor \
  --input file.csv --output out.csv \
  --clear-checkpoints
```

2. **Manual checkpoint cleanup:**
```bash
rm -rf .bookmark_checkpoints/
```

3. **Check disk space:**
```bash
df -h .
```

## WSL-Specific Issues

### 🪟 WSL Installation Problems

**Problem:** WSL not working or not installed

**Solutions:**

1. **Enable WSL features:**
```powershell
# Run as Administrator
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

2. **Install WSL2 kernel:**
```powershell
# Download and install from https://aka.ms/wsl2kernel
wsl --set-default-version 2
```

3. **Install Ubuntu:**
```powershell
wsl --install -d Ubuntu-22.04
```

### 🐌 WSL Performance Issues

**Problem:** Slow performance in WSL

**Solutions:**

1. **Use WSL2:**
```powershell
wsl -l -v
wsl --set-version Ubuntu-22.04 2
```

2. **Work within WSL filesystem:**
```bash
# Don't work on Windows drives (/mnt/c/)
cd ~
mkdir bookmark-processing
cd bookmark-processing
```

3. **Optimize WSL configuration:**
```bash
# Create .wslconfig in Windows user directory
# C:\Users\YourName\.wslconfig
[wsl2]
memory=4GB
processors=4
```

### 📁 File System Issues in WSL

**Problem:** File permission errors between Windows and WSL

**Solutions:**

1. **Work in WSL filesystem:**
```bash
# Copy files to WSL
cp /mnt/c/Users/YourName/Downloads/bookmarks.csv ~/
cd ~
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv
```

2. **Fix file permissions:**
```bash
# In WSL
chmod 644 ~/bookmarks.csv
```

3. **Configure WSL mount options:**
```bash
# Add to /etc/wsl.conf
[automount]
options = "metadata,umask=22,fmask=11"
```

## Frequently Asked Questions

### ❓ General Questions

**Q: What platforms are supported?**
A: Linux and WSL (Windows Subsystem for Linux) only. Native Windows is not supported.

**Q: How long does processing take?**
A: Depends on collection size:
- 100 bookmarks: 2-5 minutes
- 1,000 bookmarks: 20-30 minutes  
- 3,000+ bookmarks: 2-6 hours

**Q: How much memory is needed?**
A: Minimum 4GB RAM, 8GB recommended for large collections (3000+ bookmarks).

**Q: Can I stop and resume processing?**
A: Yes! Use `--resume` to continue from the last checkpoint.

**Q: Will this work with my existing bookmarks?**
A: Yes, if you export them from raindrop.io in CSV format.

### ❓ Feature Questions

**Q: What AI models are used?**
A: 
- Local: facebook/bart-large-cnn (default, free)
- Cloud: Claude (Anthropic) or OpenAI (paid APIs)

**Q: How are duplicates detected?**
A: URLs are normalized (case, protocols, slashes) and compared. Multiple resolution strategies available.

**Q: Can I keep my existing tags?**
A: The tool replaces tags with optimized ones, but uses existing tags as context for generating new ones.

**Q: Does this send my data anywhere?**
A: Only if you choose cloud AI (Claude/OpenAI). Local processing keeps all data on your machine.

**Q: What happens to invalid URLs?**
A: They're logged and excluded from the output file. Check the error logs for details.

### ❓ Configuration Questions

**Q: Where do I put API keys?**
A: In the configuration file:
```bash
cp bookmark_processor/config/user_config.ini.template user_config.ini
# Edit user_config.ini and add your keys
```

**Q: How do I reduce memory usage?**
A: Use smaller batch sizes:
```bash
--batch-size 25
```

**Q: Can I process files from network drives?**
A: Yes, but local files are recommended for better performance.

**Q: How do I speed up processing?**
A: 
- Increase batch size (if you have memory)
- Increase concurrent requests (if network allows)
- Use cloud AI for faster description generation

### ❓ Troubleshooting Questions

**Q: Why are many URLs failing validation?**
A: Common causes:
- Network connectivity issues
- Rate limiting by target sites
- URLs that require authentication
- Dead/moved content

**Q: The process seems stuck. What should I do?**
A: 
1. Check if it's still processing (verbose mode)
2. Look at system resources (memory, CPU)
3. Check network connectivity
4. Review logs for errors

**Q: Can I run this on a server?**
A: Yes, it works on headless Linux servers. Use screen or tmux for long-running processes.

**Q: How do I update the tool?**
A: 
```bash
git pull origin main
pip install -r requirements.txt
```

## Getting Help

### 📞 Support Channels

1. **Check Documentation:**
   - [Installation Guide](INSTALLATION.md)
   - [Quick Start](QUICKSTART.md)
   - [Configuration](CONFIGURATION.md)
   - [Features](FEATURES.md)

2. **Search Issues:**
   - GitHub Issues: https://github.com/davistroy/bookmark-validator/issues
   - Search for similar problems before creating new issues

3. **Create an Issue:**
   - Visit: https://github.com/davistroy/bookmark-validator/issues/new
   - Use the issue template
   - Include all relevant information

### 🐛 Bug Report Template

When reporting bugs, please include:

```
**Environment:**
- OS: Ubuntu 22.04 / WSL2 / etc.
- Python version: 3.9.1
- Tool version: 1.0.0

**Command used:**
python -m bookmark_processor --input file.csv --output out.csv --verbose

**Input data:**
- Number of bookmarks: 1,500
- File size: 2.3MB
- Any special characters or URLs

**Expected behavior:**
Process all bookmarks successfully

**Actual behavior:**
Process hangs at 50% completion

**Error messages:**
[Paste error messages here]

**Log files:**
[Attach or paste relevant log entries]

**Additional context:**
[Any other relevant information]
```

### 📊 Performance Report Template

For performance issues:

```
**System specifications:**
- CPU: Intel i5-8400
- RAM: 8GB
- Storage: SSD
- Network: 100Mbps broadband

**Processing details:**
- Total bookmarks: 3,000
- Processing time: 8 hours (expected 4 hours)
- Memory usage: 6GB peak
- Batch size: 50

**Configuration:**
[Paste relevant config sections]

**Performance logs:**
[Attach performance log files]
```

### 💡 Before Asking for Help

1. **Try the troubleshooting steps** in this guide
2. **Search existing issues** on GitHub
3. **Check the logs** in the `logs/` directory
4. **Try with a smaller dataset** to isolate the problem
5. **Use verbose output** (`--verbose`) to get more information

### 🤝 Contributing

If you fix an issue:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b fix-memory-leak`
3. **Make your changes** and test thoroughly
4. **Submit a pull request** with clear description

### 📈 Feature Requests

For new features:

1. **Check existing feature requests** in GitHub issues
2. **Describe the use case** clearly
3. **Explain how it would help** other users
4. **Consider contributing** the implementation

---

**Remember:** This tool is designed for Linux/WSL environments. For the best experience, ensure you're using a supported platform and have adequate system resources for your bookmark collection size.