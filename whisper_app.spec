# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Whisper Transcription app.
Creates a fully self-contained macOS app bundle.
"""

import sys
from pathlib import Path

block_cipher = None

# Get the project directory
project_dir = Path(SPECPATH)

a = Analysis(
    ['app.py'],
    pathex=[str(project_dir)],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('venv/lib/python3.14/site-packages/whisper/assets', 'whisper/assets'),
    ],
    hiddenimports=[
        'whisper',
        'whisper.tokenizer',
        'whisper.audio',
        'whisper.model',
        'whisper.decoding',
        'whisper.transcribe',
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchaudio',
        'numpy',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'numba',
        'llvmlite',
        'flask',
        'jinja2',
        'werkzeug',
        'webview',
        'webview.platforms',
        'webview.platforms.cocoa',
        'bottle',
        'proxy_tools',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'PIL',
        'tkinter',
        'PyQt5',
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperTranscription',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WhisperTranscription',
)

app = BUNDLE(
    coll,
    name='Bulk Transcribe.app',
    icon='AppIcon.icns',
    bundle_identifier='com.bulktranscribe.app',
    info_plist={
        'CFBundleName': 'Bulk Transcribe',
        'CFBundleDisplayName': 'Bulk Transcribe',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
    },
)
