# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Gemma4-LLM single-file EXE

import sys
from pathlib import Path

ROOT = Path(SPECPATH)

a = Analysis(
    ['app.py'],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Bundled llama-server + engine DLLs + CUDA runtime
        (str(ROOT / 'bin_exe' / '*'), 'bin'),
        # WebUI assets (HTML/CSS/JS + fonts)
        (str(ROOT / 'static' / '*'), 'static'),
        (str(ROOT / 'static' / 'fonts' / '*'), 'static/fonts'),
        # Models NOT bundled (4 GB+ exceeds PyInstaller onefile limit).
        # The app's setup UI downloads them on first run (~3.1 GB + ~1 GB).
        # App icon
        (str(ROOT / 'gemma4.ico'), '.'),
    ],
    hiddenimports=[
        'flask',
        'flask_cors',
        'requests',
        'werkzeug',
        'jinja2',
        'webview',
        'webview.platforms.edgechromium',
        'clr_loader',
        'pythonnet',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'PIL',
        'pandas',
        'pytest',
        'setuptools',
        'unittest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Gemma4-LLM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / 'gemma4.ico'),
)
