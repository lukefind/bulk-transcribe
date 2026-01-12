# Release Checklist

Pre-release verification checklist for Bulk Transcribe server deployments.

## Prerequisites

```bash
# Build and start
docker compose up -d --build

# Verify health
curl http://localhost:8476/healthz
# Expected: {"ok":true}
```

## Automated Tests

```bash
# Install dev dependencies (in venv)
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

## Manual Verification

### 1. Session Isolation

**Test**: Two browsers cannot access each other's data.

1. Open browser A (normal window)
2. Open browser B (incognito/private window)
3. In browser A: upload a file and start transcription
4. In browser B: check `/api/jobs` - should return empty list
5. In browser B: try to access browser A's job ID via `/api/jobs/<id>` - should return 404

**Pass criteria**: Browser B cannot see or download browser A's files.

### 2. CSRF Protection

**Test**: POST requests require valid CSRF token.

```bash
# Get a session cookie
curl -c cookies.txt http://localhost:8476/

# Try upload without CSRF token - should fail
curl -b cookies.txt -X POST http://localhost:8476/api/uploads
# Expected: {"error":"Invalid or missing CSRF token"}

# Get CSRF token
curl -b cookies.txt http://localhost:8476/api/session
# Returns: {"csrfToken":"..."}

# Upload with CSRF token - should work
curl -b cookies.txt -X POST \
  -H "X-CSRF-Token: <token>" \
  -F "files=@test.mp3" \
  http://localhost:8476/api/uploads
```

**Pass criteria**: Requests without valid CSRF token return 403.

### 3. Security Headers

**Test**: API responses include security headers.

```bash
curl -D - http://localhost:8476/api/mode
```

**Expected headers**:
- `Cache-Control: no-store`
- `Pragma: no-cache`
- `X-Content-Type-Options: nosniff`
- `Set-Cookie: ... HttpOnly; ... SameSite=Lax`

### 4. Legacy Endpoints Blocked

**Test**: Server mode blocks folder-based endpoints.

```bash
curl -X POST http://localhost:8476/browse -H "Content-Type: application/json" -d '{}'
# Expected: {"error":"Folder browser not available in server mode."}

curl -X POST http://localhost:8476/preview -H "Content-Type: application/json" -d '{}'
# Expected: 404
```

### 5. Stale Job Detection

**Test**: Jobs stuck in "running" state are auto-marked as failed.

1. Start a job
2. Kill the container mid-transcription: `docker compose kill`
3. Restart: `docker compose up -d`
4. Wait for `JOB_STALE_MINUTES` (default 30, or set lower for testing)
5. Check job status - should be `failed` with error code `STALE_JOB`

**Pass criteria**: Stale jobs don't stay "running" forever.

### 6. Duplicate Filename Handling

**Test**: Uploading two files with the same name produces distinct outputs.

1. Upload `audio.mp3` (file 1)
2. Upload `audio.mp3` (file 2, different content)
3. Start transcription
4. Check outputs - should have distinct filenames (e.g., `audio_abc123.md`, `audio_def456.md`)

**Pass criteria**: No output file overwrites.

### 7. Partial Failure Handling

**Test**: Job with some failed files still produces downloadable outputs.

1. Upload a valid audio file and an invalid/corrupted file
2. Start transcription
3. Job should complete with status `complete_with_errors`
4. Valid file's outputs should be downloadable
5. Failed file should show error in outputs list

**Pass criteria**: Partial success yields usable downloads.

### 8. Rerun Functionality

**Test**: Completed jobs can be re-run without re-uploading.

1. Complete a transcription job
2. Call `POST /api/jobs/<job_id>/rerun` with CSRF token
3. New job should be created using same input files
4. Original job remains unchanged

**Pass criteria**: Rerun creates new job from existing inputs.

### 9. Upload Limits

**Test**: Large uploads are rejected with friendly error.

```bash
# Create a file larger than MAX_UPLOAD_MB
dd if=/dev/zero of=large.bin bs=1M count=600

curl -b cookies.txt -X POST \
  -H "X-CSRF-Token: <token>" \
  -F "files=@large.bin" \
  http://localhost:8476/api/uploads
# Expected: {"error":"File too large. Maximum upload size is 500MB."}
```

### 10. Admin Stats Endpoint

**Test**: Admin endpoint requires token and returns stats.

```bash
# Without token - should fail
curl http://localhost:8476/api/admin/stats
# Expected: 404 (if ADMIN_TOKEN not set) or 401 (if set but no header)

# With token
curl -H "X-Admin-Token: your-secret-token" http://localhost:8476/api/admin/stats
# Expected: {"sessionCount":..., "jobsByStatus":..., "totalDiskUsageMB":...}
```

**Setup**: Set `ADMIN_TOKEN` environment variable in `.env` or docker-compose.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_MODE` | `server` | `server` or `local` |
| `MAX_UPLOAD_MB` | `500` | Max upload size per request |
| `MAX_TOTAL_SESSION_MB` | `2000` | Max storage per session |
| `MAX_FILES_PER_JOB` | `50` | Max files per job |
| `SESSION_TTL_HOURS` | `24` | Session expiry time |
| `JOB_STALE_MINUTES` | `30` | Time before running job is marked stale |
| `MAX_JOB_RUNTIME_MINUTES` | `120` | Max job runtime before auto-cancel |
| `ADMIN_TOKEN` | (none) | Token for `/api/admin/stats` |
| `COOKIE_SECURE` | `0` | Set to `1` behind HTTPS |

## API Endpoints Reference

### Session Management
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/session` | GET | Cookie | Get CSRF token |
| `/api/mode` | GET | - | Get app mode |

### Uploads
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/uploads` | POST | Cookie + CSRF | Upload files |
| `/api/uploads` | GET | Cookie | List uploads |

### Jobs
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/jobs` | GET | Cookie | List jobs |
| `/api/jobs` | POST | Cookie + CSRF | Create job |
| `/api/jobs/<id>` | GET | Cookie | Get job status |
| `/api/jobs/<id>/cancel` | POST | Cookie + CSRF | Cancel job |
| `/api/jobs/<id>/rerun` | POST | Cookie + CSRF | Rerun job |
| `/api/jobs/<id>/outputs` | GET | Cookie | List outputs |
| `/api/jobs/<id>/download` | GET | Cookie | Download zip |
| `/api/jobs/<id>/outputs/<id>` | GET | Cookie | Download file |

### Admin
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/admin/stats` | GET | X-Admin-Token | Server stats |

### Runtime
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/runtime` | GET | - | Get compute backend capabilities |

## Compute Backend Behavior

The system detects available compute backends at startup and enforces them at job creation.

### Backend Availability by Environment

| Environment | Docker | Available Backends |
|-------------|--------|-------------------|
| macOS arm64 (Apple Silicon) | No | cpu, metal (if torch+mps works) |
| macOS arm64 (Apple Silicon) | Yes | cpu only |
| macOS x86_64 | No | cpu |
| macOS x86_64 | Yes | cpu only |
| Linux x86_64 | No | cpu |
| Linux x86_64 + NVIDIA | No | cpu, cuda |
| Linux x86_64 + NVIDIA | Yes | cpu, cuda |

### Backend Verification

**Test**: Runtime endpoint reports correct capabilities.

```bash
curl http://localhost:8476/api/runtime
```

**Expected (macOS Docker)**:
```json
{
  "os": "linux",
  "arch": "arm64",
  "inDocker": true,
  "cudaAvailable": false,
  "metalAvailable": false,
  "supportedBackends": ["cpu"],
  "recommendedBackend": "cpu"
}
```

**Test**: Invalid backend is rejected.

```bash
# Get session and CSRF token
curl -c cookies.txt http://localhost:8476/
CSRF=$(curl -s -b cookies.txt http://localhost:8476/api/session | jq -r .csrfToken)

# Try to create job with unsupported backend
curl -b cookies.txt -X POST http://localhost:8476/api/jobs \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $CSRF" \
  -d '{"uploadIds":["test"], "options":{"backend":"metal"}}'
```

**Expected**:
```json
{
  "error": {
    "code": "BACKEND_UNSUPPORTED",
    "message": "Metal backend is not available in this environment. Available: cpu."
  }
}
```

**Pass criteria**:
- `/api/runtime` returns accurate environment info
- UI only shows supported backends
- Unsupported backend requests return 400 with BACKEND_UNSUPPORTED
- Job manifest includes `backend` and `environment` fields
- Progress display shows which backend is being used

## Sign-off

- [ ] All automated tests pass
- [ ] Session isolation verified
- [ ] CSRF protection verified
- [ ] Security headers present
- [ ] Legacy endpoints blocked
- [ ] Stale job detection works
- [ ] Duplicate filenames handled
- [ ] Partial failures handled
- [ ] Backend detection accurate (no Metal in Docker)
- [ ] Unsupported backend rejected with BACKEND_UNSUPPORTED
- [ ] Rerun functionality works
- [ ] Upload limits enforced
- [ ] Admin stats accessible

### 8. Diarization (Pyannote) Verification

**Prerequisites**: 
- HF_TOKEN set in environment with access to pyannote models
- Container memory limit of at least 8GB (diarization is memory-intensive)
- For audio files >5 minutes, consider using GPU or increasing memory to 16GB

**Test**: HF access verification via /api/runtime

```bash
# Check diarization readiness
curl -s http://localhost:8476/api/runtime | jq '.hfTokenPresent, .hfAccessOk, .diarizationAvailable'
# Expected: true, true, true
```

**Test**: End-to-end diarization smoke test

```bash
# Run smoke test with a real audio file
./scripts/smoke_diarization.sh ./samples/test.wav
```

**Expected logs** (structured JSON):
```json
{"event":"job_started","jobId":"...","sessionId":"...","backend":"cpu",...}
{"event":"diarization_started","jobId":"...","file":"test.wav",...}
{"event":"diarization_model_loading_started","jobId":"...",...}
{"event":"diarization_model_loading_finished","jobId":"...","durationMs":...}
{"event":"diarization_finished","jobId":"...","numSegments":...,"numSpeakers":...}
{"event":"merge_started","jobId":"...",...}
{"event":"merge_finished","jobId":"...","numMergedSegments":...}
{"event":"job_finished","jobId":"...","status":"complete",...}
```

**Expected outputs**:
- `.speaker.md` - Transcript with speaker labels
- `.diarization.json` - Raw diarization data
- `.rttm` - Standard RTTM format

**Test**: Diarization timeout (optional)

```bash
# Set very short timeout and test
export MAX_DIARIZATION_MINUTES=0.01
# Run job - should fail with DIARIZATION_TIMEOUT
```

**Test: UI feedback**

1. Load UI with invalid HF_TOKEN
   - Diarization checkbox should be disabled
   - Warning message should show reason

2. Load UI with valid HF_TOKEN
   - Diarization checkbox should be enabled
   - No warning message

3. Run diarization job
   - Progress should show: "loading diarization pipeline..."
   - Progress should show: "running diarization..."
   - Progress should show: "merging transcript + speakers..."

**Pass criteria**:
- `/api/runtime` accurately reports diarization readiness
- HF access check works without downloading models
- Diarization job completes and produces all required outputs
- Structured logs show all stages with durations
- Jobs timeout after MAX_DIARIZATION_MINUTES if stuck
- UI shows clear feedback for availability and progress
- Error reports are downloadable when diarization fails

- [ ] HF access verification works
- [ ] Diarization smoke test passes
- [ ] All required outputs generated
- [ ] Structured logs present with durations
- [ ] Timeout enforcement works
- [ ] UI shows availability feedback
- [ ] UI shows progress updates
- [ ] Error reports downloadable

**Verified by**: _______________  
**Date**: _______________  
**Version**: _______________
