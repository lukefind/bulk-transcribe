# Remote GPU Worker Setup Guide

This guide explains how to set up and configure the Remote GPU Worker for Bulk Transcribe, enabling you to offload transcription and diarization to a dedicated GPU server. Remote GPU is **optional** and only used when explicitly configured in the UI.

## Architecture Overview

```
┌─────────────────┐         ┌─────────────────┐
│   Controller    │◄───────►│   GPU Worker    │
│  (Main Server)  │         │   (Compute)     │
├─────────────────┤         ├─────────────────┤
│ - UI/Review     │         │ - Whisper       │
│ - Job Storage   │         │ - Diarization   │
│ - Session Mgmt  │         │ - GPU Compute   │
│ - File Storage  │         │                 │
└─────────────────┘         └─────────────────┘
```

**Controller**: Your main server that handles the web UI, file uploads, job management, and review workspace. Can run on CPU.

**GPU Worker**: A compute-only service that runs Whisper and diarization on GPU. Can be deployed on RunPod, Lambda Labs, Vast.ai, or any NVIDIA GPU server.

## Configuration

### Controller Environment Variables

Add these to your controller's `.env` file:

```bash
# Remote Worker Configuration
REMOTE_WORKER_URL=https://your-gpu-worker.example.com
REMOTE_WORKER_TOKEN=your-shared-secret-token
REMOTE_WORKER_MODE=optional  # off|optional|required

# Optional tuning
REMOTE_WORKER_TIMEOUT_SECONDS=7200  # 2 hours default
REMOTE_WORKER_POLL_SECONDS=2
REMOTE_WORKER_UPLOAD_MODE=pull  # pull|push (pull recommended)

# Required for signed URLs
SECRET_KEY=your-secret-key-for-signing
CONTROLLER_BASE_URL=https://your-controller.example.com
```

#### Mode Options

- `off`: All jobs run locally (default)
- `optional`: UI shows "Run on GPU" toggle; user chooses per-job
- `required`: All jobs automatically run on remote worker

### Worker Environment Variables

```bash
# Authentication
WORKER_TOKEN=your-shared-secret-token  # Must match REMOTE_WORKER_TOKEN

# Configuration
WORKER_PORT=8477
WORKER_TMP_DIR=/tmp/bt-worker
WORKER_MAX_FILE_MB=2000
WORKER_MAX_CONCURRENT_JOBS=1
WORKER_MODEL=large-v3

# For diarization
HF_TOKEN=your-huggingface-token
DIARIZATION_DEVICE=cuda  # or auto
```

## Deployment Options

### Option 1: Docker Compose (Same Host)

Add to your `docker-compose.yml`:

```yaml
services:
  controller:
    build: .
    ports:
      - "8476:8476"
    environment:
      - REMOTE_WORKER_URL=http://worker:8477
      - REMOTE_WORKER_TOKEN=${WORKER_TOKEN}
      - REMOTE_WORKER_MODE=optional
    volumes:
      - ./data:/data

  worker:
    build:
      context: .
      dockerfile: worker/Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - WORKER_TOKEN=${WORKER_TOKEN}
      - HF_TOKEN=${HF_TOKEN}
    ports:
      - "8477:8477"
```

### Option 2: RunPod Deployment

1. Create a RunPod GPU pod with NVIDIA GPU (L40/L40S or A40 recommended; RTX 4090 is a good fallback)

2. SSH into the pod and clone the repo:
   ```bash
   git clone https://github.com/your-repo/bulk-transcribe.git
   cd bulk-transcribe
   ```

3. Install dependencies:
   ```bash
   pip install -r worker/requirements.txt
   ```

4. Set environment variables:
   ```bash
   export WORKER_TOKEN=your-shared-secret
   export HF_TOKEN=your-huggingface-token
   export WORKER_PORT=8477
   ```

5. Run the worker:
   ```bash
   python -m worker.app
   ```

6. Configure your controller with the RunPod URL:
   ```bash
   REMOTE_WORKER_URL=https://your-pod-id-8477.proxy.runpod.net
   ```

### Option 3: Vast.ai / Lambda Labs

Similar to RunPod - provision a GPU instance, install dependencies, and run the worker service. Use the instance's public URL as `REMOTE_WORKER_URL`.

## Security Considerations

1. **Token Authentication**: Both controller and worker use a shared secret token. Generate a strong random token:
   ```bash
   openssl rand -hex 32
   ```

2. **Signed URLs**: Input file downloads use HMAC-signed URLs with expiry. The worker cannot access files without valid signatures.

3. **No Session Leakage**: Worker only receives a hashed session ID for logging; full session IDs are never exposed.

4. **HTTPS**: Always use HTTPS in production for both controller and worker.

## API Reference

### Worker Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/jobs` | POST | Create job |
| `/v1/jobs/<id>` | GET | Get job status |
| `/v1/jobs/<id>/cancel` | POST | Cancel job |

### Controller Worker Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/jobs/<id>/inputs/<input_id>` | GET | Signed URL | Download input file |
| `/api/jobs/<id>/worker/outputs` | POST | Bearer Token | Upload output file |
| `/api/jobs/<id>/worker/complete` | POST | Bearer Token | Mark job complete |
| `/api/remote-worker/status` | GET | None | Check worker status |

## Troubleshooting

### Worker not reachable

1. Check the worker is running: `curl https://worker-url/health`
2. Verify firewall allows port 8477
3. Check `REMOTE_WORKER_URL` is correct (include https://)

### Authentication errors

1. Verify `WORKER_TOKEN` matches `REMOTE_WORKER_TOKEN`
2. Check token has no extra whitespace

### Jobs stuck in "dispatching"

1. Check controller logs for connection errors
2. Verify `CONTROLLER_BASE_URL` is accessible from worker
3. Check `SECRET_KEY` is set for signed URLs

### Pod destroyed / worker restarted mid-job

Remote worker jobs are stored in worker memory by default. If you destroy the pod (or the worker restarts) while a job is running, the worker may lose the job.

Controller behavior:

- If the controller polls `/v1/jobs/<workerJobId>` and receives `404`, it will mark the controller job as failed with:
  - `error.code = REMOTE_JOB_NOT_FOUND`
  - `error.message = "Remote worker restarted or job expired; please retry."`
  - `remote.lastError` populated for debugging

UI behavior:

- The UI will show a human-readable error.
- For `REMOTE_JOB_NOT_FOUND`, the UI will show a **Retry Job** button (server mode).

### Diarization not working on worker

1. Verify `HF_TOKEN` is set on worker
2. Accept pyannote model agreements on HuggingFace
3. Check GPU memory (diarization needs ~4GB VRAM)

## Performance Tips

1. **Use large-v3 model** on GPU for best accuracy
2. **Enable diarization** - GPU makes it practical for longer files
3. **Batch uploads** - upload multiple files, then start job
4. **Monitor GPU memory** - large files may need chunking

## Monitoring

Worker logs include:
- Job creation/completion
- Download/upload progress
- Model load times
- Processing stages

Controller logs include:
- Dispatch events
- Poll status updates
- Worker communication errors
