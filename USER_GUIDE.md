# Bulk Transcribe User Guide

A simple, powerful tool for batch transcribing audio files to markdown using OpenAI Whisper.

## Getting Started

### Launching the App

1. Double-click **Bulk Transcribe.app** in your Applications folder
2. The app opens in its own native window (no browser required)

### Quick Start

1. Click **Browse** next to "Input Folder" and select a folder containing audio files
2. Click **Browse** next to "Output Folder" and select where to save transcriptions
3. Choose your preferred model and language
4. Click **Start Transcription**

## Interface Overview

### Input Folder
Select the folder containing your audio files. Supported formats:
- MP3, WAV, M4A, FLAC, OGG
- WebM, MP4, MOV, AVI

The app will automatically find all audio files in the folder and subfolders.

### Output Folder
Select where to save the transcription markdown files. Each audio file will generate a corresponding `_transcription.md` file.

### Model Selection

| Model | Speed | Accuracy | RAM Required | Best For |
|-------|-------|----------|--------------|----------|
| **tiny** | Fastest | Basic | ~1 GB | Quick tests |
| **base** | Fast | Good | ~1 GB | Light usage |
| **small** | Balanced | Better | ~2 GB | General use |
| **medium** | Slow | High | ~5 GB | Better quality |
| **large** | Slowest | Best | ~10 GB | Best quality |
| **turbo** | Fast | Good | ~6 GB | **Recommended** |

**Recommendation:** The `turbo` model offers the best balance of speed and quality for most use cases. It's selected by default and marked as "recommended" in the UI.

### Model Management

Click **Manage** next to the model dropdown to:
- **View downloaded models** - See which models are available locally
- **Download new models** - Click "Download" to add a model
- **Delete models** - Remove models you no longer need to free disk space

### Language Selection

- **Auto-detect**: Whisper will automatically detect the spoken language
- **Specific language**: Select a language for better accuracy and faster processing

### Output Options

#### Segmented Transcription
Splits the transcription into timestamped segments. Useful for:
- Long recordings
- Finding specific parts of audio
- Creating subtitles

#### Full Transcription
Includes the complete text as a single block. Useful for:
- Reading the full content
- Copy/pasting text
- Document creation

#### Include Timestamps
Adds start/end times to each segment in HH:MM:SS format.

#### Word-level Timestamps
Provides more detailed timing for each word. Note: This significantly increases processing time.

#### Overwrite Existing Files
By default, files that already have output are **skipped**. This allows you to:
- Resume interrupted transcription jobs
- Add new files to a folder without re-transcribing existing ones

Check this option to re-transcribe all files, even if output already exists.

## Processing Modes

### Metal GPU (Recommended)
- **Speed**: ~5-6x realtime (20 min audio in ~3-4 min)
- **Workers**: Single worker only
- **Best for**: Apple Silicon Macs (M1/M2/M3)

Metal GPU uses Apple's Metal Performance Shaders for hardware-accelerated transcription. This is the fastest option on Apple Silicon and is selected by default.

### CPU Mode
- **Speed**: ~0.5-1x realtime (20 min audio in ~20-40 min)
- **Workers**: 1-2 parallel workers
- **Best for**: Compatibility, older Intel Macs

CPU mode is slower but more compatible. Use this if you experience errors with Metal GPU mode. With 2 workers, you can process 2 files simultaneously.

## During Transcription

### Progress Indicators

- **Overall progress bar**: Shows how many files have been completed (e.g., "3/47")
- **Current file**: Shows which file is being processed
- **Per-file progress bar**: Shows percentage completion for the current file
- **Elapsed time**: Shows how long the job has been running
- **ETA**: Estimated time remaining based on completed files
- **Status messages**: Shows current activity (loading model, transcribing)
- **Skipped files**: Shown in yellow if output already exists
- **Completed files**: Shows processing time and speed (e.g., "3m 27s · 5.8x")

### Cancelling

Two cancel options are available:

- **Cancel After File** (yellow button) - Completes the current file, then stops. Use this to safely stop without losing progress on the current file.
- **Force Stop** (red button) - Stops immediately. The current file may be incomplete, but all previously completed files are saved.

### Persistent Settings

Your folder selections and language preference are automatically saved and restored when you reopen the app.

## Output Format

Each transcription is saved as a markdown file with the following structure:

```markdown
# Transcription: audio_file.mp3

**Source File:** `audio_file.mp3`
**Model Used:** `turbo`
**Detected Language:** `en`

---

## Segmented Transcription

### Segment 1 (00:00.00 - 00:15.23)

[Transcribed text for this segment]

### Segment 2 (00:15.23 - 00:32.45)

[Transcribed text for this segment]
```

## Tips for Best Results

### Audio Quality
- Clear audio with minimal background noise produces best results
- Higher quality recordings (44.1kHz or higher) are preferred
- Mono or stereo both work well

### Model Selection
- **For most use cases**: `turbo` (recommended - best speed/quality balance)
- For quick drafts: `tiny` or `base`
- For important content: `small` or `medium`
- For critical accuracy: `large`

### Language
- Always specify the language if you know it - this improves accuracy and speed
- Auto-detect works well for single-language content
- For mixed-language content, use auto-detect

### Large Files
- Very long files (>1 hour) may take significant time
- Consider splitting large files before transcription
- The `medium` or `large` models handle long content better

## Troubleshooting

### App Won't Launch
- Ensure you have macOS 10.15 or later
- Try right-clicking and selecting "Open" if blocked by Gatekeeper

### Transcription Fails
- **"Out of memory"**: Try a smaller model
- **"Audio file not found"**: Check the file isn't corrupted

### Poor Accuracy
- Try a larger model
- Specify the correct language
- Check audio quality
- Ensure clear speech without heavy accents or background noise

### Slow Performance
- **Use Metal GPU mode** (default) - ~5-6x faster than CPU on Apple Silicon
- Use a smaller model (tiny or base for quick drafts)
- Close other applications to free RAM
- For CPU mode, try 2 workers for parallel processing

## Privacy

- All processing happens locally on your Mac
- No audio is sent to external servers
- Whisper models are downloaded once and cached locally

## System Requirements

- **macOS**: 10.15 (Catalina) or later
- **Processor**: Apple Silicon (M1/M2/M3) recommended for Metal GPU acceleration
- **RAM**: 8GB minimum, 16GB+ recommended for turbo model
- **Storage**: ~500MB for app, plus model cache (~6GB for turbo model)
