# Installation Guide

This guide provides comprehensive installation instructions for the Bookmark Validation and Enhancement Tool on Linux and Windows Subsystem for Linux (WSL).

## Table of Contents

- [System Requirements](#system-requirements)
- [Linux Installation](#linux-installation)
- [WSL Installation (Windows Users)](#wsl-installation-windows-users)
- [Docker Installation](#docker-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+) or WSL2
- **Python**: Python 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended for large datasets)
- **Storage**: 2GB free disk space (additional space for AI model cache)
- **Network**: Internet connection for URL validation and model downloads

### Recommended Requirements
- **Memory**: 8GB RAM or higher
- **Storage**: 10GB free disk space
- **CPU**: Multi-core processor for better performance

## Linux Installation

### Method 1: Installation from Source (Recommended)

1. **Update your system packages**:
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt upgrade -y
   
   # CentOS/RHEL/Fedora
   sudo dnf update -y
   ```

2. **Install Python 3.8+ and pip**:
   ```bash
   # Ubuntu/Debian
   sudo apt install python3 python3-pip python3-venv git -y
   
   # CentOS/RHEL/Fedora
   sudo dnf install python3 python3-pip python3-venv git -y
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/davistroy/bookmark-validator.git
   cd bookmark-validator
   ```

4. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Method 2: Using Package Managers

#### Using pip (PyPI installation):
```bash
# Create virtual environment
python3 -m venv bookmark-validator-env
source bookmark-validator-env/bin/activate

# Install from PyPI
pip install bookmark-validator
```

#### Using pipx (Isolated installation):
```bash
# Install pipx if not already installed
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Install bookmark-validator
pipx install bookmark-validator
```

## WSL Installation (Windows Users)

### Step 1: Install WSL2

1. **Enable WSL feature** (Run as Administrator in PowerShell):
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

2. **Restart your computer**.

3. **Download and install the WSL2 Linux kernel update**:
   - Download from: https://aka.ms/wsl2kernel
   - Run the installer

4. **Set WSL2 as default**:
   ```powershell
   wsl --set-default-version 2
   ```

### Step 2: Install a Linux Distribution

1. **Install Ubuntu 22.04 LTS** (recommended):
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. **Launch Ubuntu and complete setup**:
   - Create a username and password
   - Update the system:
     ```bash
     sudo apt update && sudo apt upgrade -y
     ```

### Step 3: Install the Bookmark Validator

Follow the [Linux Installation](#linux-installation) instructions within your WSL environment.

### WSL File System Access

- **Access Windows files from WSL**: `/mnt/c/Users/YourUsername/`
- **Access WSL files from Windows**: `\\wsl$\Ubuntu-22.04\home\yourusername\`

## Docker Installation

### Using Docker (Advanced Users)

1. **Install Docker**:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   # Log out and back in to apply group changes
   ```

2. **Build the Docker image**:
   ```bash
   git clone https://github.com/davistroy/bookmark-validator.git
   cd bookmark-validator
   docker build -t bookmark-validator .
   ```

3. **Run the container**:
   ```bash
   docker run -v $(pwd)/data:/app/data bookmark-validator \
     --input /app/data/input.csv \
     --output /app/data/output.csv
   ```

## Verification

After installation, verify that the tool is working correctly:

### 1. Check Installation
```bash
# If installed from source
python -m bookmark_processor --version

# If installed via pip/pipx
bookmark-processor --version
```

### 2. Run Help Command
```bash
# If installed from source
python -m bookmark_processor --help

# If installed via pip/pipx
bookmark-processor --help
```

### 3. Test with Sample Data
```bash
# Create a test CSV file
cat > test_bookmarks.csv << EOF
id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,Example Site,Test note,Test excerpt,https://example.com,Test,test,2024-01-01T00:00:00Z,,,false
EOF

# Run the processor
python -m bookmark_processor --input test_bookmarks.csv --output test_output.csv --verbose
```

## Configuration

### Create Configuration File (Optional)

1. **Copy the template**:
   ```bash
   cp bookmark_processor/config/user_config.ini.template user_config.ini
   ```

2. **Edit configuration**:
   ```bash
   nano user_config.ini
   ```

3. **Add API keys for cloud AI** (optional):
   ```ini
   [ai]
   claude_api_key = your-claude-api-key-here
   openai_api_key = your-openai-api-key-here
   ```

## Troubleshooting

### Common Issues

#### 1. Python Version Error
**Problem**: `Python 3.8+ required`
**Solution**:
```bash
# Check Python version
python3 --version

# Install Python 3.8+ if needed (Ubuntu/Debian)
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip
```

#### 2. Permission Errors
**Problem**: Permission denied when installing packages
**Solution**:
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or install with --user flag
pip install --user -r requirements.txt
```

#### 3. SSL Certificate Issues
**Problem**: SSL certificate verification failed
**Solution**:
```bash
# Update certificates (Ubuntu/Debian)
sudo apt update && sudo apt install ca-certificates

# Or upgrade pip
pip install --upgrade pip
```

#### 4. Memory Issues with Large Datasets
**Problem**: Out of memory errors
**Solution**:
```bash
# Reduce batch size
python -m bookmark_processor --input large_file.csv --output output.csv --batch-size 25

# Enable checkpoint/resume
python -m bookmark_processor --input large_file.csv --output output.csv --resume
```

#### 5. WSL-Specific Issues

**Problem**: File permission issues between Windows and WSL
**Solution**:
```bash
# Work within WSL file system
cd ~
mkdir bookmark-processing
cd bookmark-processing
# Copy files here instead of accessing Windows files directly
```

**Problem**: Slow performance in WSL
**Solution**:
```bash
# Ensure WSL2 is being used
wsl -l -v

# Upgrade to WSL2 if needed
wsl --set-version Ubuntu-22.04 2
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**:
   ```bash
   # Logs are created in the logs/ directory
   ls logs/
   cat logs/bookmark_processor_*.log
   ```

2. **Run with verbose output**:
   ```bash
   python -m bookmark_processor --input file.csv --output output.csv --verbose
   ```

3. **Create an issue on GitHub**:
   - Visit: https://github.com/davistroy/bookmark-validator/issues
   - Include your OS, Python version, and error messages

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](QUICKSTART.md)
2. Review [Configuration Options](CONFIGURATION.md)
3. Check [Feature Documentation](FEATURES.md)

## Updates

To update the tool:

```bash
# If installed from source
cd bookmark-validator
git pull origin main
pip install -r requirements.txt

# If installed via pip
pip install --upgrade bookmark-validator
```