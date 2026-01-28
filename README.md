# Bulk Transcribe

<p align="center">
  <img src="logo-assets/apple-icon-180x180.png" alt="Bulk Transcribe" width="128">
</p>

Local-first transcription with a review workflow built for 100% accuracy. Bulk Transcribe runs entirely on your machine by default, with an optional remote GPU worker for heavy batches. It is designed for **human review**: machine output is a first-pass annotation layer, not final truth.

## Highlights

- **Local-first**: all processing stays on your machine by default
- **Optional remote GPU worker**: offload transcription/diarization to RunPod/Lambda/Vast
- **Review-first UX**: fast speaker relabeling, per-chunk edits, keyboard shortcuts
- **Speaker diarization**: optional, with fast-switching mode for rapid turn-taking
- **Batch processing**: folders, large lists, queue-aware dispatch
- **Exports**: Project archive (.btproj), Markdown, JSON timeline, Word (DOCX), PDF
- **Bulk ops**: bulk export/import, re-run selected jobs, clear duplicates

## Quick Start (Local)

```bash
# Install deps
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Run server
./venv/bin/python app.py
```

Open: http://localhost:8476

## Quick Start (Docker)

```bash
docker compose up -d app
```

Open: http://localhost:8476

## Remote GPU Worker (Optional)

Use a remote worker for large batches or diarization-heavy jobs. See:

- [Remote Worker Setup](docs/remote-worker.md)
- [RunPod GPU Worker Guide](docs/gpu-worker-runpod.md)
- [Pod-per-session workflow](docs/pod-per-session-workflow.md)

## Models

Supported Whisper models:

- `tiny`, `base`, `small`, `medium`, `large`, `large-v3`

`large-v3` is the default recommendation for best accuracy.

## Review Exports

The review interface supports:

- **Project (.btproj)** — full project archive for backup/transfer
- **Markdown (.md)** — speaker-attributed transcript with timestamps
- **Timeline (.json)** — structured review timeline
- **Word (.docx)** — timestamped transcript (requires `python-docx`)
- **PDF (.pdf)** — timestamped transcript (requires `reportlab`)

### Session Export/Import

Export your entire session (all jobs with editor state) as a zip:
- Click **Export Session** in Recent Jobs header
- Includes job manifests, review state, timelines, and outputs

Import a session export to restore all jobs:
- Click **Import Session** and select a session zip
- All editor state (speaker labels, colors, edits) is preserved

See [Review Workflow](docs/REVIEW_WORKFLOW.md) for details.

## Privacy & Security

- **Local by default** — no cloud APIs
- **Remote GPU is explicit** — only used when configured
- **Sensitive inputs** — treat all audio/transcripts as private

## Documentation

- [User Guide](USER_GUIDE.md)
- [Install & Setup](docs/INSTALL.md)
- [Review Workflow](docs/REVIEW_WORKFLOW.md)
- [Transcript Schema](docs/TRANSCRIPT_SCHEMA.md)
- [Server Deployment (Ubuntu)](docs/server-deploy-ubuntu.md)
- [RunPod Worker Setup](docs/gpu-worker-runpod.md)
- [Remote Worker Architecture](docs/remote-worker.md)
- [Heavy Batch Operations](docs/heavy-batch-operations.md)
- [Performance & Cost](docs/performance-and-cost.md)

## License

MIT License
