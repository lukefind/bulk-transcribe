# Bulk Transcribe

<p align="center">
  <img src="logo-assets/apple-icon-180x180.png" alt="Bulk Transcribe" width="128">
</p>

A native macOS app for batch transcribing audio files to markdown using OpenAI Whisper. Features a native app window, real-time progress tracking, and flexible output options.

## Features

- **Native macOS App** - Runs in its own window, no browser required
- **Metal GPU Acceleration** - ~5-6x realtime on Apple Silicon (M1/M2/M3)
- **Batch Processing** - Transcribe entire folders of audio files
- **Skip Existing Files** - Resume interrupted jobs without re-transcribing
- **Multiple Formats** - MP3, WAV, M4A, FLAC, OGG, WebM, MP4, MOV, AVI
- **Model Management** - Download, delete, and switch between Whisper models
- **Smart Defaults** - Large-v3 model with Metal GPU for best quality
- **Language Support** - Auto-detect or specify from 12+ languages
- **Flexible Output** - Segmented, full text, or both with optional timestamps
- **Real-time Progress** - Per-file progress bar, ETA, and elapsed time
- **Persistent Settings** - Folder selections and preferences saved across sessions
- **Cancel Options** - Force stop or cancel after current file

## Screenshots

### Main Interface
<p align="center">
  <img src="screenshots/main-interface.png" alt="Bulk Transcribe Main Interface" width="600">
</p>

## Installation

### Option 1: Download DMG (Recommended)

1. Download `Bulk Transcribe.dmg` from the [releases](https://github.com/lukefind/bulk-transcribe/releases)
2. Open the DMG and drag `Bulk Transcribe.app` to Applications
3. Double-click to launch

**This is a fully self-contained app** - no additional installation required.

### Option 2: Run Locally (Development)

**Prerequisites:**
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

```bash
# Clone the repository
git clone https://github.com/lukefind/bulk-transcribe.git
cd bulk-transcribe

# Create virtual environment and install dependencies
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Run the app
./venv/bin/python app.py
```

### Option 3: Build from Source

```bash
# After completing Option 2 setup, build the standalone app
./venv/bin/pip install pyinstaller
./venv/bin/python -m PyInstaller --clean --noconfirm whisper_app.spec

# Create DMG (optional)
hdiutil create -volname "Bulk Transcribe" -srcfolder "dist/Bulk Transcribe.app" -ov -format UDZO "dist/Bulk Transcribe.dmg"
```

## Usage

1. Launch **Bulk Transcribe.app**
2. Click **Browse** to select your input folder (containing audio files)
3. Click **Browse** to select your output folder
4. Choose model, language, and output options
5. Click **Start Transcription**

### Skip Existing Files

By default, files that already have a `_transcription.md` output are skipped. This allows you to resume interrupted jobs. Check **Overwrite existing files** to re-transcribe all files.

## Models

| Model | Speed | Accuracy | RAM | Recommendation |
|-------|-------|----------|-----|----------------|
| tiny | Fastest | Basic | ~1 GB | Quick tests |
| base | Fast | Good | ~1 GB | Light usage |
| small | Balanced | Better | ~2 GB | General use |
| medium | Slow | High | ~5 GB | Better quality |
| large | Slowest | Best | ~10 GB | Best quality |
| **large-v3** | **Slowest** | **Best** | **~10 GB** | **Recommended** |

## Output Options

- **Segmented Transcription** - Timestamped segments (HH:MM:SS format)
- **Full Transcription** - Complete text block
- **Include Timestamps** - Show start/end times for segments
- **Word-level Timestamps** - Detailed per-word timing (slower)
- **Overwrite existing files** - Re-transcribe files that already have output

### JSON Output Format

Each transcription produces a JSON file with segment-level data. Segments include:

- **`id`** - App-assigned stable identifier (`seg_001`, `seg_002`, etc.) that remains contiguous after post-processing (merging/splitting)
- **`index`** - Zero-based contiguous integer index (0, 1, 2, ...)
- **`source_id`** - Original Whisper segment ID(s) as string, merged with `+` when segments are combined (e.g., `"0+1"`)
- **`start`** / **`end`** - Timestamp boundaries in seconds
- **`text`** - Transcribed text
- **`words`** - Optional word-level timestamps (when enabled)

## Processing Modes

| Mode | Speed | Workers | Best For |
|------|-------|---------|----------|
| **Metal GPU** | ~5-6x realtime | 1 | Apple Silicon Macs (recommended) |
| CPU | ~0.5-1x realtime | 1-2 | Compatibility, older Macs |

- **Metal GPU (default)** - Uses Apple's Metal Performance Shaders for GPU acceleration. Single worker only, but very fast (~5-6x realtime means a 20 min file takes ~3-4 min).
- **CPU** - Falls back to CPU processing. Slower but more compatible. Supports up to 2 parallel workers for batch processing.

## System Requirements

- **macOS** 10.15 (Catalina) or later
- **Apple Silicon** (M1/M2/M3) recommended for Metal GPU acceleration
- **RAM** 8GB minimum, 16GB+ recommended for large-v3 model
- **Storage** ~500MB for app, plus model cache (~3GB for large-v3 model)

## Server Deployment

For Ubuntu server deployment with Docker, see the [Server Deployment Guide](docs/server-deploy-ubuntu.md). The guide includes:

- Docker Compose setup for CPU and NVIDIA GPU
- Environment configuration (copy `.env.example` to `.env`)
- Reverse proxy configuration
- Management commands

## Documentation

- [User Guide](USER_GUIDE.md) - Detailed usage instructions and tips
- [Server Deployment Guide](docs/server-deploy-ubuntu.md) - Ubuntu server deployment with Docker

## Privacy

All processing happens locally on your Mac. No audio is sent to external servers.

## License

MIT License
