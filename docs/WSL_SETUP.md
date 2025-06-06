# WSL Setup Guide for Windows Users

Complete guide to setting up Windows Subsystem for Linux (WSL) to run the Bookmark Validation and Enhancement Tool.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Enable WSL Features](#step-1-enable-wsl-features)
- [Step 2: Install WSL2](#step-2-install-wsl2)
- [Step 3: Install Linux Distribution](#step-3-install-linux-distribution)
- [Step 4: Initial Linux Setup](#step-4-initial-linux-setup)
- [Step 5: Install the Bookmark Tool](#step-5-install-the-bookmark-tool)
- [Working with Files](#working-with-files)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting WSL Issues](#troubleshooting-wsl-issues)
- [WSL vs Native Linux](#wsl-vs-native-linux)

## Overview

### Why WSL?

The Bookmark Validation and Enhancement Tool is designed for Linux environments. Windows users need WSL to run the tool because:

- ✅ **Native Linux compatibility** - Runs all Linux tools and libraries
- ✅ **Better performance** - More efficient than virtual machines
- ✅ **File system integration** - Easy access to Windows files
- ✅ **Command line tools** - Full access to Linux command line
- ✅ **Package management** - Use apt, pip, and other Linux package managers

### What You'll Get

After setup, you'll have:
- A full Linux environment inside Windows
- Access to all Linux command-line tools
- The ability to run the bookmark processor
- Integration between Windows and Linux file systems

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Windows 10 version 1903 or higher, OR Windows 11
- 64-bit processor
- 4GB RAM (8GB recommended)
- 2GB free disk space
- Administrator access

**Check Your Windows Version:**
```cmd
winver
```

**Enable Virtualization (if not already enabled):**
1. Restart your computer
2. Enter BIOS/UEFI settings (usually F2, F12, or Delete during boot)
3. Look for "Virtualization Technology" or "Intel VT-x" or "AMD-V"
4. Enable the setting
5. Save and exit

## Step 1: Enable WSL Features

### Method 1: Using Windows Features (GUI)

1. **Open Windows Features:**
   - Press `Win + R`, type `optionalfeatures`, press Enter
   - OR: Control Panel → Programs → Turn Windows features on or off

2. **Enable required features:**
   - ☑️ Windows Subsystem for Linux
   - ☑️ Virtual Machine Platform

3. **Restart your computer** when prompted.

### Method 2: Using PowerShell (Recommended)

1. **Open PowerShell as Administrator:**
   - Right-click Start button
   - Select "Windows PowerShell (Admin)" or "Terminal (Admin)"

2. **Run these commands:**
   ```powershell
   # Enable WSL feature
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   
   # Enable Virtual Machine Platform
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

3. **Restart your computer:**
   ```powershell
   Restart-Computer
   ```

## Step 2: Install WSL2

### Download and Install WSL2 Kernel

1. **Download the WSL2 Linux kernel update:**
   - Visit: https://aka.ms/wsl2kernel
   - Download and run the installer
   - Follow the installation wizard

2. **Set WSL2 as default version:**
   ```powershell
   # Open PowerShell as Administrator
   wsl --set-default-version 2
   ```

### Verify Installation

```powershell
# Check WSL version
wsl --version

# List available distributions
wsl --list --online
```

Expected output:
```
WSL version: 2.0.9.0
Kernel version: 5.15.79.1
WSLg version: 1.0.47
MSRDC version: 1.2.3575
Direct3D version: 1.606.4
DXCore version: 10.0.25131.1002-220531-1700.rs-onecore-base2-hyp
Windows version: 10.0.22621.963
```

## Step 3: Install Linux Distribution

### Option 1: Quick Install (Recommended)

**Install Ubuntu 22.04 LTS (recommended):**
```powershell
wsl --install -d Ubuntu-22.04
```

**Alternative distributions:**
```powershell
# Ubuntu 20.04
wsl --install -d Ubuntu-20.04

# Debian
wsl --install -d Debian
```

### Option 2: Microsoft Store Installation

1. **Open Microsoft Store**
2. **Search for "Ubuntu"**
3. **Install "Ubuntu 22.04.x LTS"**
4. **Launch Ubuntu from Start Menu**

### Initial Setup

1. **Wait for installation to complete** (may take several minutes)

2. **Create user account when prompted:**
   ```
   Enter new UNIX username: yourusername
   New password: [enter secure password]
   Retype new password: [confirm password]
   ```

3. **Verify installation:**
   ```bash
   # Check Linux version
   cat /etc/os-release
   
   # Check WSL version
   wsl.exe --list --verbose
   ```

## Step 4: Initial Linux Setup

### Update System Packages

```bash
# Update package lists
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install essential tools
sudo apt install -y curl wget git build-essential
```

### Configure Git (Optional but Recommended)

```bash
# Set up git (replace with your information)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Install Python 3.8+

```bash
# Check current Python version
python3 --version

# Install Python development tools
sudo apt install -y python3-pip python3-venv python3-dev

# Verify installation
python3 --version
pip3 --version
```

## Step 5: Install the Bookmark Tool

### Method 1: From Source (Recommended)

```bash
# Navigate to home directory
cd ~

# Clone the repository
git clone https://github.com/davistroy/bookmark-validator.git

# Navigate to project directory
cd bookmark-validator

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Test installation
python -m bookmark_processor --version
```

### Method 2: Using pip

```bash
# Create virtual environment
python3 -m venv bookmark-env
source bookmark-env/bin/activate

# Install from PyPI (when available)
pip install bookmark-validator

# Test installation
bookmark-processor --version
```

### Test with Sample Data

```bash
# Create test file
cat > test_bookmarks.csv << 'EOF'
id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite
1,GitHub,Code hosting,GitHub homepage,https://github.com,Development,git code,2024-01-01T00:00:00Z,,,false
2,Python Docs,Python documentation,Official Python docs,https://docs.python.org,Programming,python docs,2024-01-02T00:00:00Z,,,false
EOF

# Process test file
python -m bookmark_processor \
  --input test_bookmarks.csv \
  --output test_output.csv \
  --verbose

# Check results
cat test_output.csv
```

## Working with Files

### File System Layout

**WSL can access:**
- **WSL files**: `/home/yourusername/` (recommended for processing)
- **Windows files**: `/mnt/c/Users/YourWindowsUsername/`

**Windows can access:**
- **WSL files**: `\\wsl$\Ubuntu-22.04\home\yourusername\`
- **Windows files**: `C:\Users\YourWindowsUsername\`

### Best Practices

1. **Work within WSL file system for better performance:**
   ```bash
   # Good: Work in WSL home directory
   cd ~
   mkdir bookmark-processing
   cd bookmark-processing
   
   # Copy files from Windows to WSL
   cp /mnt/c/Users/YourName/Downloads/bookmarks.csv ./
   ```

2. **Access Windows files when needed:**
   ```bash
   # Access Windows Desktop
   cd /mnt/c/Users/YourName/Desktop
   
   # Access Windows Downloads
   cd /mnt/c/Users/YourName/Downloads
   ```

### File Transfer Examples

**Copy from Windows to WSL:**
```bash
# Copy from Windows Downloads to WSL home
cp /mnt/c/Users/YourName/Downloads/raindrop_export.csv ~/

# Copy to WSL with new name
cp /mnt/c/Users/YourName/Downloads/bookmarks.csv ~/my_bookmarks.csv
```

**Copy from WSL to Windows:**
```bash
# Copy processed file back to Windows Desktop
cp ~/enhanced_bookmarks.csv /mnt/c/Users/YourName/Desktop/
```

### Using Windows File Explorer

1. **Open File Explorer**
2. **In address bar, type:** `\\wsl$\Ubuntu-22.04\home\yourusername`
3. **Bookmark this location** for easy access
4. **Drag and drop files** between Windows and WSL

## Performance Optimization

### WSL Configuration

Create `.wslconfig` file in your Windows user directory:

**Location:** `C:\Users\YourName\.wslconfig`

**Content:**
```ini
[wsl2]
# Memory allocation (adjust based on your system)
memory=4GB

# Number of processors (adjust based on your CPU)
processors=4

# Swap size
swap=2GB

# Disable page reporting (can improve performance)
pageReporting=false

# Network configuration
networkingMode=mirrored
```

### Performance Tips

1. **Use WSL2 (not WSL1):**
   ```powershell
   # Check version
   wsl --list --verbose
   
   # Upgrade to WSL2 if needed
   wsl --set-version Ubuntu-22.04 2
   ```

2. **Work in WSL file system:**
   ```bash
   # Fast: Files in WSL
   cd ~
   python -m bookmark_processor --input bookmarks.csv --output enhanced.csv
   
   # Slower: Files on Windows drive
   cd /mnt/c/Users/YourName/
   python -m bookmark_processor --input bookmarks.csv --output enhanced.csv
   ```

3. **Close unnecessary Windows applications** during processing

4. **Use smaller batch sizes** if memory is limited:
   ```bash
   python -m bookmark_processor \
     --input bookmarks.csv \
     --output enhanced.csv \
     --batch-size 25
   ```

### Resource Monitoring

```bash
# Check memory usage
free -h

# Check CPU usage
top

# Check disk space
df -h

# Monitor real-time system stats
htop  # Install with: sudo apt install htop
```

## Troubleshooting WSL Issues

### Common WSL Problems

#### WSL Command Not Found

**Problem:** `'wsl' is not recognized as an internal or external command`

**Solution:**
1. Check Windows version (needs 1903+)
2. Enable WSL features (see Step 1)
3. Restart computer
4. Update Windows

#### WSL Won't Start

**Problem:** WSL fails to start or shows error 0x80040326

**Solutions:**
```powershell
# Restart WSL service
wsl --shutdown
wsl

# Reset WSL if needed
wsl --unregister Ubuntu-22.04
wsl --install -d Ubuntu-22.04
```

#### Network Issues in WSL

**Problem:** No internet connection in WSL

**Solutions:**
```bash
# Check DNS resolution
nslookup google.com

# Try different DNS servers
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# Restart WSL
exit  # Exit WSL
# In PowerShell: wsl --shutdown
# Then: wsl
```

#### File Permission Issues

**Problem:** Permission denied errors

**Solutions:**
```bash
# Fix file permissions
chmod 644 filename.csv
chmod 755 directory_name

# Work in WSL file system instead of Windows drives
cd ~
cp /mnt/c/path/to/file.csv ./
```

### WSL Performance Issues

#### Slow File Operations

**Solutions:**
1. **Move files to WSL file system:**
   ```bash
   cp /mnt/c/Users/YourName/large_file.csv ~/
   ```

2. **Exclude WSL from Windows Defender:**
   - Open Windows Security
   - Go to Virus & threat protection
   - Add exclusion for: `%USERPROFILE%\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04onWindows_79rhkp1fndgsc`

3. **Optimize WSL configuration** (see Performance Optimization section)

#### Memory Issues

**Problem:** Out of memory errors

**Solutions:**
```bash
# Check memory usage
free -h

# Use smaller batch sizes
python -m bookmark_processor \
  --input file.csv \
  --output out.csv \
  --batch-size 10

# Increase WSL memory limit in .wslconfig
# memory=8GB
```

### WSL Networking Issues

#### Can't Access Internet

```bash
# Test connectivity
ping google.com

# Check DNS
cat /etc/resolv.conf

# Fix DNS if needed
echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf
```

#### Windows Firewall Blocking

1. **Open Windows Defender Firewall**
2. **Allow an app through firewall**
3. **Add vmmem.exe and wsl.exe**

## WSL vs Native Linux

### Advantages of WSL

- ✅ **Easy installation** - No dual boot required
- ✅ **Windows integration** - Access Windows files and apps
- ✅ **No virtual machine overhead** - Better performance than VMs
- ✅ **Development environment** - Perfect for Linux development

### Limitations of WSL

- ❌ **Slight performance overhead** - Not quite as fast as native Linux
- ❌ **Windows dependency** - Requires Windows to run
- ❌ **File system performance** - Cross-system file access is slower

### Performance Comparison

| Metric | Native Linux | WSL2 | WSL1 |
|--------|-------------|------|------|
| CPU Performance | 100% | 95% | 80% |
| Memory Efficiency | 100% | 90% | 85% |
| File I/O (same FS) | 100% | 90% | 70% |
| File I/O (cross FS) | N/A | 60% | 40% |
| Network Performance | 100% | 95% | 85% |

### When to Use What

**Use WSL when:**
- You need Windows and Linux
- Developing on Windows for Linux deployment
- Want easy setup and integration
- Don't want to dual boot

**Use Native Linux when:**
- Maximum performance is critical
- Running production workloads
- Don't need Windows integration
- Want full Linux experience

## Advanced WSL Usage

### Running Background Services

```bash
# Start bookmark processing in background
nohup python -m bookmark_processor \
  --input large_file.csv \
  --output enhanced.csv \
  --verbose > processing.log 2>&1 &

# Check background processes
jobs
ps aux | grep python
```

### Using tmux for Persistent Sessions

```bash
# Install tmux
sudo apt install tmux

# Start new session
tmux new-session -d -s bookmark-processing

# Attach to session
tmux attach-session -t bookmark-processing

# Detach from session (Ctrl+B, then D)

# List sessions
tmux list-sessions
```

### Scheduled Processing

```bash
# Install cron
sudo apt install cron

# Edit crontab
crontab -e

# Add job to run daily at 2 AM
0 2 * * * cd /home/yourusername/bookmark-processing && python -m bookmark_processor --input daily_export.csv --output daily_enhanced.csv
```

### WSL Integration with VS Code

1. **Install WSL extension** in VS Code
2. **Open WSL terminal** in VS Code
3. **Navigate to project directory:**
   ```bash
   cd ~/bookmark-validator
   code .
   ```
4. **VS Code will open in WSL mode** with full Linux support

## Getting Help with WSL

### Microsoft Documentation

- **WSL Documentation**: https://docs.microsoft.com/en-us/windows/wsl/
- **WSL GitHub**: https://github.com/microsoft/WSL
- **WSL Issues**: https://github.com/microsoft/WSL/issues

### Community Support

- **Reddit**: r/bashonubuntuonwindows
- **Stack Overflow**: Questions tagged with 'wsl'
- **Ubuntu Community**: https://askubuntu.com/

### WSL-Specific Commands

```bash
# Check WSL version and status
wsl.exe --status

# List all distributions
wsl.exe --list --all

# Set default distribution
wsl.exe --set-default Ubuntu-22.04

# Terminate all WSL instances
wsl.exe --shutdown

# Export WSL distribution (backup)
wsl.exe --export Ubuntu-22.04 ubuntu-backup.tar

# Import WSL distribution (restore)
wsl.exe --import Ubuntu-22.04 C:\WSL\Ubuntu-22.04 ubuntu-backup.tar
```

---

**Congratulations!** You now have a fully functional Linux environment on Windows. You can proceed with using the Bookmark Validation and Enhancement Tool just like any Linux user. Remember to work within the WSL file system for best performance and refer to the [main documentation](README.md) for usage instructions.