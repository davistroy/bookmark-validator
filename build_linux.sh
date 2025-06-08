#!/bin/bash
# Linux/WSL Build Script for Bookmark Processor

set -e  # Exit on any error

echo "🚀 Building Bookmark Processor for Linux/WSL..."

# Check if we're running in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "📋 Detected WSL environment"
elif [[ "$(uname -s)" == "Linux" ]]; then
    echo "📋 Detected native Linux environment"
else
    echo "❌ This script is designed for Linux/WSL only"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Validate platform compatibility if validation script exists
if [ -f "$PROJECT_ROOT/scripts/validate_platform.py" ]; then
    echo "🔍 Validating platform compatibility..."
    python3 "$PROJECT_ROOT/scripts/validate_platform.py" || {
        echo "❌ Platform validation failed. Exiting."
        exit 1
    }
    echo ""
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "📦 Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Install/upgrade build dependencies
echo "🔧 Installing build dependencies..."
pip install --upgrade pip
pip install pyinstaller

# Install core dependencies (without AI for faster build)
echo "📚 Installing core dependencies..."
pip install pandas requests beautifulsoup4 tqdm validators chardet lxml configparser

# Run the Python build script
echo "🏗️ Running build script..."
cd "$PROJECT_ROOT"
python build/build_linux.py

echo "✅ Build completed!"
echo ""
echo "📁 Outputs:"
echo "   Executable: dist/bookmark-processor"
echo "   Package: dist/bookmark-processor-linux/"
echo "   Tarball: dist/bookmark-processor-linux.tar.gz"
echo ""
echo "🧪 Quick test:"
echo "   ./dist/bookmark-processor --help"