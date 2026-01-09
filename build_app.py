#!/usr/bin/env python3
"""
Build script to create a native macOS app bundle for Whisper Transcription.
Supports both lightweight (requires system Python/Whisper) and 
fully self-contained (bundles all dependencies) builds.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

APP_NAME = "Bulk Transcribe"
BUNDLE_ID = "com.bulktranscribe.app"
VERSION = "1.0.0"


def build_standalone():
    """Build a fully self-contained app using PyInstaller."""
    project_dir = Path(__file__).parent
    venv_pyinstaller = project_dir / "venv" / "bin" / "pyinstaller"
    
    if not venv_pyinstaller.exists():
        print("Error: PyInstaller not found in venv. Run:")
        print("  ./venv/bin/pip install pyinstaller openai-whisper")
        return None
    
    print("Building self-contained app with PyInstaller...")
    print("This may take several minutes due to bundling PyTorch and Whisper...")
    print()
    
    # Run PyInstaller
    result = subprocess.run([
        str(venv_pyinstaller),
        'whisper_app.spec',
        '--clean',
        '--noconfirm',
    ], cwd=str(project_dir))
    
    if result.returncode != 0:
        print("Error: PyInstaller build failed")
        return None
    
    app_path = project_dir / "dist" / f"{APP_NAME}.app"
    if app_path.exists():
        print(f"\nStandalone app created: {app_path}")
        print(f"\nThis app is fully self-contained (~2-3GB) and includes:")
        print("  - Python runtime")
        print("  - Flask web framework")
        print("  - OpenAI Whisper")
        print("  - PyTorch")
        print("\nNo additional installation required!")
        return app_path
    
    return None


def create_app_bundle():
    """Create a macOS .app bundle."""
    
    project_dir = Path(__file__).parent
    dist_dir = project_dir / "dist"
    app_dir = dist_dir / f"{APP_NAME}.app"
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    # Clean previous build
    if app_dir.exists():
        shutil.rmtree(app_dir)
    
    # Create directory structure
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Info.plist
    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>{APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>{APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>{BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>{VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>{VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
"""
    
    with open(contents_dir / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create launcher script
    launcher_script = f"""#!/bin/bash

# Get the directory where the app bundle is located
APP_DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    osascript -e 'display alert "Python 3 Required" message "Please install Python 3 to use this application.\\n\\nInstall via: brew install python3"'
    exit 1
fi

# Check if whisper is available
if ! command -v whisper &> /dev/null; then
    osascript -e 'display alert "Whisper Required" message "OpenAI Whisper is not installed.\\n\\nInstall via: pipx install openai-whisper"'
    exit 1
fi

# Check if Flask is installed, if not install it
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    osascript -e 'display notification "Installing Flask dependency..." with title "Whisper Transcription"'
    pip3 install --user flask 2>/dev/null || python3 -m pip install --user flask
fi

# Start the Flask server
cd "$APP_DIR"
python3 app.py &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Open browser
open "http://localhost:8765"

# Wait for the server process
wait $SERVER_PID
"""
    
    launcher_path = macos_dir / "launcher"
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    # Make launcher executable
    os.chmod(launcher_path, 0o755)
    
    # Copy application files to Resources
    files_to_copy = [
        "app.py",
        "transcribe.py",
        "gui.py",
    ]
    
    for filename in files_to_copy:
        src = project_dir / filename
        if src.exists():
            shutil.copy(src, resources_dir / filename)
    
    # Copy templates folder
    templates_src = project_dir / "templates"
    if templates_src.exists():
        shutil.copytree(templates_src, resources_dir / "templates")
    
    # Create a simple icon (placeholder - you can replace with a real .icns file)
    create_placeholder_icon(resources_dir)
    
    print(f"App bundle created: {app_dir}")
    print(f"\nTo install:")
    print(f"  1. Drag '{APP_NAME}.app' to your Applications folder")
    print(f"  2. Double-click to launch")
    print(f"\nPrerequisites:")
    print(f"  - Python 3 (brew install python3)")
    print(f"  - OpenAI Whisper (pipx install openai-whisper)")
    
    return app_dir


def create_placeholder_icon(resources_dir):
    """Create a simple placeholder icon."""
    # For a real app, you'd want to create a proper .icns file
    # This creates a minimal placeholder
    pass


def create_dmg(app_dir):
    """Create a DMG installer (optional)."""
    dist_dir = app_dir.parent
    dmg_path = dist_dir / f"{APP_NAME}.dmg"
    
    # Remove existing DMG
    if dmg_path.exists():
        dmg_path.unlink()
    
    try:
        subprocess.run([
            "hdiutil", "create",
            "-volname", APP_NAME,
            "-srcfolder", str(app_dir),
            "-ov",
            "-format", "UDZO",
            str(dmg_path)
        ], check=True)
        print(f"\nDMG installer created: {dmg_path}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not create DMG: {e}")
    except FileNotFoundError:
        print("Warning: hdiutil not found, skipping DMG creation")


if __name__ == "__main__":
    print(f"Building {APP_NAME} macOS app...")
    print("=" * 50)
    
    if "--standalone" in sys.argv:
        # Build fully self-contained app with all dependencies
        app_dir = build_standalone()
        if app_dir and "--dmg" in sys.argv:
            create_dmg(app_dir)
    else:
        # Build lightweight app (requires system Python/Whisper)
        app_dir = create_app_bundle()
        if "--dmg" in sys.argv:
            create_dmg(app_dir)
    
    print("\nBuild complete!")
    print("\nUsage:")
    print("  python3 build_app.py              # Lightweight build (requires system deps)")
    print("  python3 build_app.py --standalone # Self-contained build (bundles everything)")
    print("  python3 build_app.py --standalone --dmg  # Self-contained + DMG installer")
