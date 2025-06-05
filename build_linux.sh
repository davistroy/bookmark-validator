#!/bin/bash
# Linux Build Script for Bookmark Processor

set -e  # Exit on any error

echo "🚀 Building Bookmark Processor for Linux..."

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

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