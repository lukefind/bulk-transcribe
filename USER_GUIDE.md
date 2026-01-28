# Bulk Transcribe User Guide

Local-first transcription with a review workflow built for 100% accuracy. The app runs as a local web UI and can optionally offload processing to a remote GPU worker.

---

## Getting Started

### Launch (Local)

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python app.py
```

Open: http://localhost:8476

### Launch (Docker)

```bash
docker compose up -d app
```

Open: http://localhost:8476

---

## Core Workflow

1. Upload audio files (drag/drop or folder selection)
2. Choose model and language
3. Configure output options
4. Enable diarization if needed
5. Start transcription
6. Review and correct in the Review workspace
7. Export final outputs

---

## Models

Supported Whisper models:

- `tiny`, `base`, `small`, `medium`, `large`, `large-v3`

**Recommendation**: `large-v3` for best accuracy.

---

## Output Options

- **Segmented transcription** — timestamped chunks for review
- **Full transcription** — single block of text
- **Include timestamps** — add timecodes to segments
- **Word-level timestamps** — slower, but enables max segment length splitting
- **Overwrite existing files** — re-transcribe even if outputs exist
- **Silence threshold** — now includes **0.4** (very permissive) up to 0.8 (strict)
- **VAD pre-filter** — optional Silero VAD to remove silence before Whisper
- **Hallucination detection** — flags likely hallucinations (repeated phrases, foreign text); flag-only, no removal

---

## Speaker Diarization

Enable diarization to assign speakers.

### Key controls

- **Min speakers / Max speakers / Fixed speakers**
- **Max duration per segment** (auto-split long audio)
- **Auto-split long audio**
- **Fast Switching** (shorter segments, higher overlap)

### Recommended settings (adult + 2 kids)

- **Fixed speakers**: 3
- **Min speakers**: 2
- **Max speakers**: 3
- **Max duration per segment**: 3–5 min
- **Auto-split**: ON
- **Fast Switching**: ON (slower, but better turn-taking)

> **Note:** Fast Switching increases diarization time.

---

## Review Workspace

The Review UI is designed for fast correction.

### Merged vs Raw Timeline Toggle

The review toolbar includes a **Merged** checkbox:

- **Merged view (default)**: Combines adjacent same-speaker chunks for readability. Best for most review tasks.
- **Raw view**: Shows diarization-aligned chunks with more speaker switches. Useful when you need to see the original segmentation or debug speaker assignment issues.

Toggle between views at any time — your preference is saved per session.

### Info Modal

Click the **ⓘ** button in the review toolbar to see a quick reference for:
- View modes (Merged/Raw)
- Fast Switching explanation
- Export formats
- Keyboard shortcuts

### Speaker relabeling
- Click a speaker label in the sidebar to rename
- Colors can be changed via the palette
- Labels + colors persist per session

### Speaker hotkey numbers

In **Edit Speakers**, assign hotkey numbers (1–9) to each speaker:
- Numbers affect shortcuts **in this file only**
- Reassigning a number clears it from the previous speaker
- Use **Copy from previous** to carry assignments to the next file
- Use **Reset** to clear all assigned numbers

### Editing text
- Click the pencil icon or press **E** on a selected chunk
- **Enter** saves and exits
- **Shift+Enter** inserts a newline
- **Esc** cancels edit

### Split mid‑segment interjections
- While editing:
  - **Cmd/Ctrl + S** → split at cursor
  - **Cmd/Ctrl + Shift + S** → split by selection (3‑way)

This lets you insert short interjections as separate chunks and assign a different speaker.

### Assign speaker
- Use number keys **1–9** to assign speaker
- Speaker buttons appear on hover or when chunk is selected

### Delete / Undo
- **Backspace/Delete**: delete selected chunk (when not editing)
- **Cmd/Ctrl + Z**: undo last change (per file, in-memory)
- Inline **Undo** button on each chunk (disabled when no history)

---

## Exports

From Review:

- **Project (.btproj)** — full archive
- **Markdown (.md)** — timestamped speaker transcript
- **Timeline (.json)** — structured review data
- **Word (.docx)** — timestamped transcript with speaker colors
- **PDF (.pdf)** — timestamped transcript with speaker colors

### Session Export/Import

Export your entire session (all jobs with editor state) as a zip:
- Click **Export Session** in Recent Jobs header
- Includes job manifests, review state, timelines, and outputs
- Export runs in the background with a progress modal

Import a session export to restore all jobs:
- Click **Import Session** and select a session zip
- All editor state (speaker labels, colors, edits) is preserved

### Session Sharing

Share a live server session without exporting:
- Click **Share Session**
- Choose **Read-only** or **Full edit**
- Copy the token (and optional password) to another user

Join a shared session:
- Click **Join Session** and paste the token
- Read-only sessions show a banner and disable edits

---

## Remote GPU Worker (Optional)

Use a remote worker for large batches or diarization-heavy runs.

Guides:
- [Remote Worker Setup](docs/remote-worker.md)
- [RunPod GPU Worker](docs/gpu-worker-runpod.md)

---

## Performance Tips

- Use `large-v3` for accuracy
- Enable diarization only when needed (it adds time)
- For large batches, keep a GPU worker running and queue all files

---

## Troubleshooting

### Remote worker required but offline
- Verify worker URL and token in UI
- RunPod proxy URL may rotate — re-copy if needed
- Restart pod if `/v1/ping` times out

### Diarization missing switches
- Use **Fixed speakers** (e.g., 3)
- Turn on **Fast Switching**
- Reduce max duration (3–5 min)

### Slow performance
- First run is slower due to model download
- Diarization increases runtime
- For heavy batches, use a GPU worker (L40/L40S/A40 recommended)

---

## Privacy

- All processing is local by default
- Remote GPU is explicit and opt‑in
- Treat all inputs/outputs as sensitive
