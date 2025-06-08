#!/usr/bin/env python3
"""
Platform Validation Script

This script validates that the current environment is compatible
with the Bookmark Processor build requirements (Linux/WSL only).
"""

import platform
import sys
from pathlib import Path


def check_platform():
    """Check if current platform is Linux/WSL"""
    system = platform.system()
    
    if system != "Linux":
        return False, f"Unsupported platform: {system}. This tool requires Linux/WSL."
    
    return True, f"Platform compatible: {system}"


def check_wsl():
    """Check if running in WSL"""
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            if "microsoft" in version_info:
                return True, "WSL environment detected"
            else:
                return True, "Native Linux environment detected"
    except:
        return True, "Linux environment (WSL detection unavailable)"


def check_python_version():
    """Check Python version compatibility"""
    min_version = (3, 9)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        return False, f"Python {current_version[0]}.{current_version[1]} is too old. Minimum required: {min_version[0]}.{min_version[1]}"
    
    return True, f"Python {current_version[0]}.{current_version[1]} is compatible"


def check_required_commands():
    """Check if required system commands are available"""
    required_commands = ["bash", "chmod", "tar", "gzip"]
    missing_commands = []
    
    import shutil
    for cmd in required_commands:
        if not shutil.which(cmd):
            missing_commands.append(cmd)
    
    if missing_commands:
        return False, f"Missing required commands: {', '.join(missing_commands)}"
    
    return True, "All required system commands are available"


def check_build_dependencies():
    """Check if build dependencies are available"""
    try:
        import pip
        return True, "Pip is available for dependency installation"
    except ImportError:
        return False, "Pip is not available. Please install pip first."


def main():
    """Run all platform validation checks"""
    print("🔍 Validating platform compatibility for Bookmark Processor...")
    print()
    
    checks = [
        ("Platform Check", check_platform),
        ("Environment Check", check_wsl),
        ("Python Version", check_python_version),
        ("System Commands", check_required_commands),
        ("Build Dependencies", check_build_dependencies),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}: {message}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"❌ {check_name}: Error during check - {e}")
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All platform checks passed! Environment is ready for building.")
        return 0
    else:
        print("⚠️  Some platform checks failed. Please address the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())