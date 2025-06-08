# -*- mode: python ; coding: utf-8 -*-
# Simplified PyInstaller spec file for Linux/WSL build

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
        (str(project_root / "bookmark_processor" / "config" / "default_config.ini"), "bookmark_processor/config"),
        (str(project_root / "bookmark_processor" / "data" / "user_agents.txt"), "bookmark_processor/data"),
        (str(project_root / "bookmark_processor" / "data" / "site_delays.json"), "bookmark_processor/data"),
    ],
    hiddenimports=[
        "bookmark_processor",
        "bookmark_processor.cli",
        "bookmark_processor.core.csv_handler",
        "bookmark_processor.core.url_validator",
        "bookmark_processor.core.ai_processor",
        "bookmark_processor.core.pipeline",
        "bookmark_processor.core.data_models",
        "bookmark_processor.config.configuration",
        "pandas",
        "requests",
        "beautifulsoup4",
        "bs4",
        "tqdm",
        "validators",
        "chardet",
        "lxml",
        "configparser"
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PIL", 
        "pillow",
        "IPython",
        "jupyter",
        "notebook",
        "test",
        "tests",
        "testing",
        "torch",
        "tensorflow",
        "transformers",
        "sklearn",
        "scipy",
        "sympy",
        "nltk",
        "spacy",
        "jinja2",
        "plotly",
        "seaborn",
        "statsmodels"
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
