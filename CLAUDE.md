# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bulk Transcribe is a local-first audio transcription tool with a review-focused UI for achieving 100% accuracy. Uses OpenAI Whisper for transcription with optional speaker diarization (pyannote). Can optionally offload processing to remote GPU workers.

**Key principle**: Machine output is a first-pass annotation layer, not ground truth. Human review is first-class.

## Development Commands

```bash
# Setup
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Run server (opens http://localhost:8476)
./venv/bin/python app.py

# Run tests
./venv/bin/pytest tests/

# Smoke tests
./scripts/smoke_diarization.sh
./scripts/smoke_remote_worker.sh

# Docker (CPU)
docker compose up -d app

# Docker (GPU)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# macOS app build
./venv/bin/pyinstaller whisper_app.spec
```

## Architecture

### Pipeline
```
Audio → (optional VAD) → Transcription (Whisper) → (optional Diarization) → Review UI → Exports
```

### Data Model (Critical)
- **Machine transcript** (`transcript.json`): Immutable after generation, never modify
- **Human edits** (`review_state.json`): Append-only corrections, preserves audit trail
- **Exports** (MD, JSON, DOCX, PDF): Derived views, regenerable from source

### Key Modules
| File | Purpose |
|------|---------|
| `app.py` | Flask server, all `/api/*` endpoints |
| `transcribe.py` | Whisper transcription logic |
| `diarization.py` | Speaker diarization (pyannote) |
| `session_store.py` | Session/job management, multi-user file isolation |
| `remote_worker.py` | GPU worker dispatch and polling |
| `worker/app.py` | GPU worker Flask service |

### Session Structure
```
/data/sessions/<sessionId>/jobs/<jobId>/
├── manifest.json
├── inputs/
├── outputs/
│   ├── transcript.json (immutable)
│   ├── diarization.json
│   └── speaker.md
└── review_state.json
```

## Non-Negotiables

1. **No cloud APIs** — all processing local by default
2. **No telemetry/phone-home** — strict privacy
3. **Machine transcripts immutable** — never overwrite after generation
4. **Human edits append-only** — preserve audit trail
5. **Remote GPU explicit** — never used implicitly, requires `REMOTE_WORKER_MODE`
6. **Reviewer speed first** — optimize for fast human correction

## API Conventions

- `GET /api/*` for reads, `POST /api/*` for mutations
- Mutations require CSRF token from `/api/runtime`
- Session via cookie: `bt_session`
- Error codes: uppercase snake_case (`REMOTE_WORKER_UNREACHABLE`, `DIARIZATION_TOO_LONG_CPU`)

## Before Merging

- [ ] Run: `./venv/bin/pytest tests/`
- [ ] For remote work: `./scripts/smoke_remote_worker.sh`
- [ ] For diarization: `./scripts/smoke_diarization.sh`
- [ ] Does this optimize for reviewer speed?
- [ ] Are machine transcripts still immutable?
- [ ] Are human edits still append-only?
- [ ] Works offline (no cloud APIs)?

## Key Environment Variables

```bash
PORT=8476
APP_MODE=server          # server (multi-user) or local (single-user)
DEVICE=cpu               # Auto-detected; check via /api/runtime
HF_TOKEN=                # Required for speaker diarization
REMOTE_WORKER_URL=       # GPU worker endpoint
REMOTE_WORKER_TOKEN=     # Shared secret
REMOTE_WORKER_MODE=off   # off|optional|required
```

See `.env.example` for complete reference.
