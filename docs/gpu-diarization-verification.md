# GPU Diarization Verification Guide

This document provides step-by-step verification that GPU diarization is working correctly.

## Prerequisites

1. Controller deployed with latest commits
2. Worker deployed on RunPod with A40 or L40 GPU
3. `DIARIZATION_DEVICE=cuda` set on worker (or `auto` with CUDA available)
4. `WORKER_TOKEN` matching between controller and worker

## Quick Verification Commands

### 1. Verify Controller is Running

```bash
# Check controller health
curl -s https://transcribe.lukus.cloud/api/runtime | jq '{
  backend: .backend,
  remoteWorker: .remoteWorker,
  buildCommit: .buildCommit
}'
```

Expected output:
```json
{
  "backend": "cpu",
  "remoteWorker": {
    "configured": true,
    "available": true,
    "mode": "optional"
  },
  "buildCommit": "<latest-commit>"
}
```

### 2. Verify Worker is Running and CUDA Available

```bash
# Check worker health (requires auth)
curl -s -H "Authorization: Bearer $WORKER_TOKEN" \
  https://<worker-url>/v1/ping | jq '{
  status: .status,
  gpu: .gpu,
  gpuName: .gpuName,
  cuda: .cuda,
  diarization: .diarization,
  models: .models,
  buildCommit: .buildCommit
}'
```

Expected output:
```json
{
  "status": "ok",
  "gpu": true,
  "gpuName": "NVIDIA A40",
  "cuda": "12.x",
  "diarization": true,
  "models": ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"],
  "buildCommit": "<latest-commit>"
}
```

### 3. Verify Worker Health Check

```bash
curl -s -H "Authorization: Bearer $WORKER_TOKEN" \
  https://<worker-url>/v1/health | jq '.checks'
```

Expected: All checks should be `true`:
```json
{
  "numpy": true,
  "torch": true,
  "torch_cuda": true,
  "whisper": true,
  "diarization": true
}
```

## During Job Execution

### 4. Check Worker Logs for GPU Diarization

Look for this log entry in worker stdout:

```json
{
  "event": "diarization_device_selected",
  "device": "cuda",
  "cudaAvailable": true,
  "cudaVersion": "12.1",
  "gpu": "NVIDIA A40",
  "pyannoteVersion": "3.1.1",
  "configuredDevice": "auto"
}
```

**If `device` is `cpu` instead of `cuda`, diarization is NOT using GPU.**

### 5. Check Controller Logs for Dispatch Details

Look for this log entry:

```json
{
  "event": "remote_job_created",
  "jobId": "...",
  "workerJobId": "wk_...",
  "mappedModel": "large-v3",
  "originalModel": null,
  "uploadMode": "push",
  "outputsUploadUrl": "https://transcribe.lukus.cloud/api/jobs/.../worker/outputs",
  "completeUrl": "https://transcribe.lukus.cloud/api/jobs/.../worker/complete",
  "diarizationEnabled": true,
  "gpuOptimizedChunking": true,
  "chunkSeconds": 300,
  "overlapSeconds": 8
}
```

## After Job Completion

### 6. Verify Outputs Exist on Controller

```bash
# SSH into controller or use docker exec
docker compose exec app sh -c "find /data/sessions/*/jobs/<JOB_ID> -type f"
```

Expected output:
```
/data/sessions/.../jobs/<JOB_ID>/job.json
/data/sessions/.../jobs/<JOB_ID>/outputs/<input>_transcript.json
/data/sessions/.../jobs/<JOB_ID>/outputs/<input>_transcript.md
/data/sessions/.../jobs/<JOB_ID>/outputs/<input>_diarization.json
/data/sessions/.../jobs/<JOB_ID>/outputs/<input>_speaker.md
```

### 7. Verify job.json Contains Remote Metadata

```bash
docker compose exec app sh -c "cat /data/sessions/*/jobs/<JOB_ID>/job.json | jq '.remote'"
```

Expected:
```json
{
  "workerJobId": "wk_...",
  "workerUrl": "https://...",
  "uploadMode": "push",
  "outputsUploadUrl": "https://.../api/jobs/.../worker/outputs",
  "completeUrl": "https://.../api/jobs/.../worker/complete",
  "startedAt": "...",
  "completedAt": "...",
  "outputsReceived": 4
}
```

## Model Validation

### 8. Verify Model is Sent Exactly

Start a job with model "large-v3" and check controller logs:

```json
{
  "event": "remote_job_created",
  "model": "large-v3"
}
```

The exact model name should be sent to the worker with no aliasing or mapping.

## Troubleshooting

### Diarization Using CPU Instead of GPU

1. Check `DIARIZATION_DEVICE` env var on worker:
   ```bash
   echo $DIARIZATION_DEVICE  # Should be 'auto' or 'cuda'
   ```

2. Check CUDA is available:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. Check worker logs for `diarization_device_selected` event

### Outputs Not Appearing on Controller

1. Check worker logs for `upload_failed` events
2. Check controller logs for `worker_output_received` events
3. Verify `CONTROLLER_BASE_URL` is set correctly on controller
4. Verify `WORKER_TOKEN` matches on both sides

### Job Completes But No Outputs

If job shows complete but `outputsReceived: 0`:
- Job should fail with `REMOTE_OUTPUTS_MISSING` error
- Check worker logs for upload errors
- Check controller is accessible from worker (Cloudflare, firewall)

## Optimal Settings for A40

For best diarization performance on NVIDIA A40/L40:

| Setting | Value | Reason |
|---------|-------|--------|
| `DIARIZATION_DEVICE` | `cuda` or `auto` | Force GPU usage |
| Whisper model | `large-v3` | Best accuracy |
| Chunk size | 300s | Reduces overhead |
| Overlap | 8s | Preserves accuracy at boundaries |

These are automatically applied when dispatching to GPU worker (if not user-specified).

## Environment Variables Reference

### Controller
- `CONTROLLER_BASE_URL`: Public URL for worker callbacks (required for push mode)
- `REMOTE_WORKER_URL`: Worker endpoint URL
- `REMOTE_WORKER_TOKEN` or `WORKER_TOKEN`: Shared auth token
- `REMOTE_WORKER_MODE`: `off` | `optional` | `required`

### Worker
- `WORKER_TOKEN`: Shared auth token (must match controller)
- `DIARIZATION_DEVICE`: `auto` | `cuda` | `cpu` (default: auto)
- `HF_TOKEN`: HuggingFace token for pyannote models
- `WORKER_MODEL`: Default Whisper model (default: large-v3)
