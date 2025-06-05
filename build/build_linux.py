#!/usr/bin/env python3
"""
Linux Build Script for Bookmark Processor

This script creates a standalone Linux executable using PyInstaller,
optimized for WSL and Linux environments.
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path


def run_command(cmd, cwd=None, capture_output=False):
    """Run a command and handle errors"""
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(
        cmd, 
        cwd=cwd, 
        capture_output=capture_output,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        if capture_output:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    return result


def clean_build_artifacts():
    """Clean previous build artifacts"""
    print("Cleaning build artifacts...")
    
    project_root = Path(__file__).parent.parent
    
    # Directories to clean
    dirs_to_clean = [
        project_root / "dist",
        project_root / "build" / "pyinstaller",
        project_root / "__pycache__",
    ]
    
    # Files to clean
    files_to_clean = [
        project_root / "bookmark_processor.spec"
    ]
    
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"Removing {dir_path}")
            shutil.rmtree(dir_path)
    
    for file_path in files_to_clean:
        if file_path.exists():
            print(f"Removing {file_path}")
            file_path.unlink()


def check_dependencies():
    """Check and install required dependencies"""
    print("Checking dependencies...")
    
    required_packages = [
        "pyinstaller",
        "pandas", 
        "requests",
        "beautifulsoup4",
        "tqdm",
        "validators",
        "chardet",
        "lxml"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing, installing...")
            run_command(f"pip install {package}")


def create_pyinstaller_spec():
    """Create PyInstaller spec file for Linux build"""
    print("Creating PyInstaller spec file...")
    
    project_root = Path(__file__).parent.parent
    
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

project_root = Path("{project_root}")

a = Analysis(
    [str(project_root / "bookmark_processor" / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "bookmark_processor" / "config" / "default_config.ini"), "bookmark_processor/config"),
        (str(project_root / "bookmark_processor" / "data" / "*.txt"), "bookmark_processor/data"),
        (str(project_root / "bookmark_processor" / "data" / "*.json"), "bookmark_processor/data"),
    ],
    hiddenimports=[
        "bookmark_processor",
        "bookmark_processor.cli",
        "bookmark_processor.core.bookmark_processor",
        "bookmark_processor.core.pipeline",
        "bookmark_processor.core.csv_handler",
        "bookmark_processor.core.url_validator",
        "bookmark_processor.core.content_analyzer",
        "bookmark_processor.core.ai_processor",
        "bookmark_processor.core.tag_generator",
        "bookmark_processor.core.checkpoint_manager",
        "bookmark_processor.core.data_models",
        "bookmark_processor.utils.intelligent_rate_limiter",
        "bookmark_processor.utils.browser_simulator",
        "bookmark_processor.utils.retry_handler",
        "bookmark_processor.utils.progress_tracker",
        "bookmark_processor.utils.logging_setup",
        "bookmark_processor.utils.validation",
        "bookmark_processor.config.configuration",
        "pandas",
        "numpy",
        "requests",
        "urllib3",
        "beautifulsoup4",
        "bs4",
        "tqdm",
        "validators",
        "chardet",
        "lxml",
        "datetime",
        "json",
        "csv",
        "logging",
        "configparser",
        "pathlib",
        "dataclasses",
        "typing"
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PIL",
        "IPython",
        "jupyter",
        "notebook",
        "test",
        "tests",
        "testing"
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='bookmark-processor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)
'''
    
    spec_file = project_root / "bookmark_processor.spec"
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print(f"Created spec file: {spec_file}")
    return spec_file


def build_executable():
    """Build the Linux executable"""
    print("Building Linux executable...")
    
    project_root = Path(__file__).parent.parent
    spec_file = create_pyinstaller_spec()
    
    # Build with PyInstaller (use python -m to ensure we use the right environment)
    cmd = [
        sys.executable, "-m", "PyInstaller", 
        "--clean",
        "--noconfirm",
        str(spec_file)
    ]
    
    run_command(cmd, cwd=project_root)
    
    # Check if build was successful
    exe_path = project_root / "dist" / "bookmark-processor"
    if not exe_path.exists():
        print("❌ Build failed - executable not found")
        sys.exit(1)
    
    print(f"✅ Build successful: {exe_path}")
    return exe_path


def test_executable():
    """Test the built executable"""
    print("Testing executable...")
    
    project_root = Path(__file__).parent.parent
    exe_path = project_root / "dist" / "bookmark-processor"
    
    if not exe_path.exists():
        print("❌ Executable not found")
        sys.exit(1)
    
    # Make executable
    os.chmod(exe_path, 0o755)
    
    # Test basic functionality
    try:
        result = run_command([str(exe_path), "--help"], capture_output=True)
        if "Bookmark Validation and Enhancement Tool" in result.stdout:
            print("✅ Executable help test passed")
        else:
            print("❌ Executable help test failed")
            print(result.stdout)
            sys.exit(1)
    except Exception as e:
        print(f"❌ Executable test failed: {e}")
        sys.exit(1)


def create_distribution():
    """Create distribution package"""
    print("Creating distribution package...")
    
    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"
    exe_path = dist_dir / "bookmark-processor"
    
    if not exe_path.exists():
        print("❌ Executable not found")
        sys.exit(1)
    
    # Create distribution structure
    package_dir = dist_dir / "bookmark-processor-linux"
    package_dir.mkdir(exist_ok=True)
    
    # Copy executable
    shutil.copy2(exe_path, package_dir / "bookmark-processor")
    
    # Copy documentation
    docs_to_copy = [
        "README.md",
        "LICENSE", 
        "CLAUDE.md"
    ]
    
    for doc in docs_to_copy:
        doc_path = project_root / doc
        if doc_path.exists():
            shutil.copy2(doc_path, package_dir)
    
    # Create sample data directory
    samples_dir = package_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Copy test data as samples
    test_data_dir = project_root / "test_data"
    if test_data_dir.exists():
        for sample_file in test_data_dir.glob("*.csv"):
            if "test_input" in sample_file.name:
                shutil.copy2(sample_file, samples_dir / "sample_bookmarks.csv")
    
    # Create usage instructions
    usage_file = package_dir / "USAGE.txt"
    with open(usage_file, 'w') as f:
        f.write("""Bookmark Processor - Linux Distribution

USAGE:
    ./bookmark-processor --input input.csv --output output.csv

EXAMPLES:
    # Basic processing
    ./bookmark-processor --input samples/sample_bookmarks.csv --output enhanced_bookmarks.csv
    
    # Verbose processing with small batch
    ./bookmark-processor --input bookmarks.csv --output enhanced.csv --verbose --batch-size 25
    
    # Resume interrupted processing
    ./bookmark-processor --input bookmarks.csv --output enhanced.csv --resume
    
    # Clear checkpoints and start fresh
    ./bookmark-processor --input bookmarks.csv --output enhanced.csv --clear-checkpoints

INPUT FORMAT:
    11-column raindrop.io export CSV format:
    id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite

OUTPUT FORMAT:
    6-column raindrop.io import CSV format:
    url,folder,title,note,tags,created

For more information, see README.md and CLAUDE.md

""")
    
    print(f"✅ Distribution created: {package_dir}")
    
    # Create tarball
    import tarfile
    
    tarball_path = dist_dir / "bookmark-processor-linux.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(package_dir, arcname="bookmark-processor-linux")
    
    print(f"✅ Tarball created: {tarball_path}")
    
    return package_dir, tarball_path


def main():
    """Main build process"""
    print("🚀 Starting Linux build process...")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    try:
        # Build steps
        clean_build_artifacts()
        check_dependencies()
        exe_path = build_executable()
        test_executable()
        package_dir, tarball_path = create_distribution()
        
        print("\\n🎉 Build completed successfully!")
        print(f"   Executable: {exe_path}")
        print(f"   Package: {package_dir}")
        print(f"   Tarball: {tarball_path}")
        print("\\n📋 Next steps:")
        print("   1. Test the executable with your bookmark data")
        print("   2. Copy the distribution to your target Linux system")
        print("   3. Extract and run: tar -xzf bookmark-processor-linux.tar.gz")
        
    except KeyboardInterrupt:
        print("\\n❌ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()