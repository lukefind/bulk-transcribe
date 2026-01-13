# GPU Worker Image Build Guide

This document covers building and pushing the GPU worker Docker image for multi-architecture deployment.

## Overview

The worker image supports two architectures:
- **linux/amd64**: Production deployment on RunPod/cloud GPU instances (CUDA-enabled)
- **linux/arm64**: Local development/testing on Apple Silicon Macs (CPU-only)

## Prerequisites

- Docker with buildx enabled
- GHCR (GitHub Container Registry) access
- Logged in to GHCR: `docker login ghcr.io`

## Build Commands

### Multi-arch Build and Push to GHCR

```bash
# From repo root
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --build-arg BUILD_COMMIT=$(git rev-parse --short HEAD) \
    --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
    -t ghcr.io/lukefind/bulk-transcribe-worker:latest \
    -t ghcr.io/lukefind/bulk-transcribe-worker:$(git rev-parse --short HEAD) \
    --push \
    -f worker/Dockerfile \
    .
```

### Local Build (single arch, no push)

```bash
# Build for current architecture only
docker build \
    --build-arg BUILD_COMMIT=$(git rev-parse --short HEAD) \
    --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
    -t bulk-transcribe-worker:local \
    -f worker/Dockerfile \
    .
```

## Architecture Notes

### linux/amd64 (Production)

- Installs CUDA-enabled torch/torchaudio from `https://download.pytorch.org/whl/cu121`
- Requires NVIDIA GPU runtime for GPU acceleration
- Deploy on RunPod, Lambda Labs, or any NVIDIA GPU cloud

### linux/arm64 (Development)

- Falls back to CPU-only torch/torchaudio wheels
- Works on Apple Silicon Macs for local testing
- No GPU acceleration (transcription will be slower)

## Verification

After building, verify the image works:

```bash
# Run container
docker run -d -p 8477:8477 \
    -e WORKER_TOKEN=testtoken \
    --name worker-test \
    bulk-transcribe-worker:local

# Test ping endpoint
curl -H "Authorization: Bearer testtoken" http://localhost:8477/v1/ping

# Expected response (JSON with status: ok)
# {
#   "status": "ok",
#   "version": "...",
#   "gpu": true/false,
#   "diarization": false,
#   "activeJobs": 0,
#   "maxConcurrentJobs": 1
# }

# Check torch versions inside container
docker exec worker-test python3 -c "import torch, torchaudio; print(f'torch={torch.__version__} torchaudio={torchaudio.__version__}')"

# Expected on amd64: torch=2.2.2+cu121 torchaudio=2.2.2+cu121
# Expected on arm64: torch=2.2.2 torchaudio=2.2.2

# Cleanup
docker stop worker-test && docker rm worker-test
```

## Dependency Pinning

The worker uses exact version pins to ensure reproducible builds:

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.2.2 | Installed via Dockerfile for CUDA control |
| torchaudio | 2.2.2 | Must match torch version |
| pyannote.audio | 3.1.1 | Pinned for compatibility |
| openai-whisper | 20231117 | Stable release |

**Important**: Do not use `>=` constraints for these packages. Version mismatches between torch/torchaudio/pyannote cause runtime crashes.

## Troubleshooting

### `/v1/ping` returns 500

Check container logs:
```bash
docker logs worker-test
```

Common causes:
- torch/torchaudio version mismatch
- pyannote.audio import failure (check `diarizationError` in ping response)

### Build fails on arm64

The Dockerfile uses a fallback pattern for torch installation. If CUDA wheels fail (expected on arm64), it falls back to default PyPI wheels.

### Errno 5 I/O errors during build

The Dockerfile sets `TMPDIR=/var/tmp/pip` to avoid I/O errors during pip package extraction. If issues persist, try building with more disk space.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_TOKEN` | (required) | Authentication token |
| `WORKER_PORT` | 8477 | HTTP port |
| `WORKER_MODEL` | large-v3 | Default Whisper model |
| `WORKER_MAX_FILE_MB` | 2000 | Max upload size |
| `WORKER_MAX_CONCURRENT_JOBS` | 1 | Max parallel jobs |
| `HF_TOKEN` | (optional) | HuggingFace token for diarization |
