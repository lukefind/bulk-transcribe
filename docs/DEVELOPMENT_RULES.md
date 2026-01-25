# Development Rules for Bulk Transcribe

This document supplements `AGENTS.md` with practical development guidelines.

---

## Core Principles (from AGENTS.md)

1. **Reviewer speed** - Every feature must optimize for human review workflow
2. **Auditability** - Machine transcript immutable, human edits append-only
3. **Determinism** - Same inputs → same outputs (where feasible)
4. **Simplicity** - No magic, explicit data flow

---

## Code Organization

### Key Files
| File | Purpose |
|------|---------|
| `app.py` | Main Flask server, all API endpoints |
| `transcribe.py` | Whisper transcription logic |
| `diarization.py` | Speaker diarization (pyannote) |
| `session_store.py` | Session/job management, file storage |
| `remote_worker.py` | Controller-side remote GPU dispatch |
| `worker/app.py` | GPU worker service |
| `review_timeline.py` | Review UI timeline logic |

### Adding New Features
1. Identify insertion point in pipeline: `Audio → Transcription → Diarization → Export`
2. Write pure functions where possible
3. Add to existing modules before creating new files
4. Update relevant docs (`docs/TRANSCRIPT_SCHEMA.md`, etc.)

---

## API Conventions

### Endpoints
- `GET /api/*` - Read operations
- `POST /api/*` - Mutations
- All mutations require CSRF token (from `/api/runtime`)
- Session ID via cookie (`bt_session`)

### Response Format
```json
{
  "success": true,
  "data": { ... }
}
// or
{
  "error": "Human-readable message",
  "code": "MACHINE_READABLE_CODE"
}
```

### Error Codes
Use uppercase snake_case: `REMOTE_WORKER_UNREACHABLE`, `DIARIZATION_TOO_LONG_CPU`

---

## Testing Requirements

### Before Merging
1. Run existing tests: `./venv/bin/pytest tests/`
2. For remote worker changes: `./scripts/smoke_remote_worker.sh`
3. For diarization changes: `./scripts/smoke_diarization.sh`

### Adding Tests
- Unit tests for pure logic (segment splitting, edit application)
- Integration tests for API endpoints
- Use `test.wav` fixture for audio tests

### Test Files
```
tests/
├── test_diarization.py         # Diarization logic
├── test_diarization_policy.py  # Chunking/policy
├── test_remote_worker.py       # GPU worker integration
├── test_review_mode.py         # Review UI
├── test_session_isolation.py   # Multi-user isolation
└── test_worker_ping.py         # Worker health checks
```

---

## Transcript Data Model (Critical)

### Machine Transcript (Immutable)
- JSON file with `schema_version`
- Segments have: `id`, `index`, `source_id`, `start`, `end`, `text`, `words`
- **Never modify after generation**

### Human Edits (Append-Only)
- JSONL edits log
- Final transcript = `apply(machine, edits)`
- Preserves full audit trail

### Exports (Derived Views)
- Markdown, SRT, VTT are regenerable
- Never store as source of truth

---

## Remote GPU Worker

### Architecture
```
Controller (CPU)          GPU Worker (RunPod/Lambda)
├── Web UI               ├── Whisper inference
├── File storage         ├── Diarization
├── Job queue            └── Returns results
└── Review workspace
```

### Configuration
```bash
# Controller .env
REMOTE_WORKER_URL=https://xxx-8477.proxy.runpod.net
REMOTE_WORKER_TOKEN=<shared-secret>
REMOTE_WORKER_MODE=optional  # off|optional|required

# Worker env
WORKER_TOKEN=<same-shared-secret>
HF_TOKEN=<for-diarization>
```

### Cost Control
- **Never leave GPU pods running idle**
- Use `REMOTE_WORKER_MODE=optional` to choose per-job
- Batch jobs: start pod → run all → stop pod
- Monitor with `/api/runtime` → `remoteWorker.workerCapabilities`

---

## Diarization Guidelines

### Limits (CPU Safety)
- `MAX_DIARIZATION_DURATION_SECONDS=180` (3 min default)
- Longer files auto-chunk or require GPU
- Always set `HF_TOKEN` for pyannote models

### When to Enable
- Multi-speaker content (interviews, meetings)
- Adds 20-40% processing time on GPU
- Adds 50-100% on CPU

---

## UI/UX Standards

### Review Workflow First-Class
- One-click jump to audio moment
- Quick speaker relabeling
- Split/merge segments without losing audit trail
- Flag low-confidence/ambiguous parts

### Defaults
- Short atomic segments (utterances) over long paragraphs
- "Unknown/Ambiguous" labels when evidence is weak
- Timestamps in HH:MM:SS format

---

## Security

### Non-Negotiables
- No cloud APIs, no telemetry, no phone-home
- All processing local (or explicit remote GPU)
- Tokens via env vars, never in code
- Signed URLs for worker file access

### Session Isolation
- Each session has isolated storage
- Worker only sees hashed session IDs
- HTTPS required in production

---

## Dependency Management

### Adding Dependencies
1. Ask before adding any dependency
2. Prefer: stdlib → existing utils → already-installed
3. Pin versions in requirements.txt
4. ML deps (VAD, diarization) must be optional/feature-flagged

### Current Requirements
- `requirements.txt` - Core app
- `requirements-server.txt` - Server deployment
- `worker/requirements.txt` - GPU worker

---

## Deployment

### Local Development
```bash
./venv/bin/python app.py
```

### Docker (Server)
```bash
docker-compose up -d
```

### GPU Worker (RunPod)
```bash
./scripts/build_worker.sh latest
./scripts/push_worker.sh latest <dockerhub-user>
# Deploy on RunPod with template
```

---

## Debugging

### Logs
- `LOG_LEVEL=debug` for verbose output
- `SUPPRESS_THIRD_PARTY_WARNINGS=0` to see all warnings

### Common Issues
| Issue | Check |
|-------|-------|
| Worker unreachable | `curl $REMOTE_WORKER_URL/health` |
| Auth errors | Token match between controller/worker |
| Jobs stuck | Check `lastErrorCode` in job status |
| OOM | Reduce model size or enable chunking |

---

## Checklist for Changes

- [ ] Does this optimize for reviewer speed?
- [ ] Is the data flow explicit and testable?
- [ ] Are machine transcripts still immutable?
- [ ] Are human edits still append-only?
- [ ] Did I add/update tests?
- [ ] Did I update relevant docs?
- [ ] Does this work offline (no cloud APIs)?
