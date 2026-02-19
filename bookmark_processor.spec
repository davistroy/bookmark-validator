# -*- mode: python ; coding: utf-8 -*-
# Auto-generated PyInstaller spec file for Linux/WSL build

import sys
import os
from pathlib import Path

# Use relative path to avoid hardcoded system paths
project_root = Path(os.getcwd())

a = Analysis(
    [str(project_root / "bookmark_processor" / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "bookmark_processor" / "data" / "user_agents.txt"), "bookmark_processor/data"),
        (str(project_root / "bookmark_processor" / "data" / "site_delays.json"), "bookmark_processor/data"),
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
    hooksconfig={},
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
