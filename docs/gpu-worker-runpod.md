# GPU Worker Deployment on RunPod

Step-by-step guide to deploy the Bulk Transcribe GPU Worker on RunPod.

## Quick Reference (main branch workflow)

**Standard image tag**: `ghcr.io/lukefind/bulk-transcribe-worker:main-amd64`

```bash
# Build and push from main branch
git checkout main
./scripts/build_worker.sh main-amd64
./scripts/push_worker.sh main-amd64 ghcr.io/lukefind

# Redeploy RunPod pod (uses same image tag, pulls latest)
# Then verify:
curl $CONTROLLER_URL/api/runtime | jq '.remoteWorker.identity'
```

---

## Prerequisites

- RunPod account with credits
- Docker image pushed to a registry (Docker Hub, GHCR, etc.)
- Your controller running and accessible from the internet

## Step 1: Push Your Worker Image

**Recommended: Use `main-amd64` tag for stable deployments**

```bash
# Build for linux/amd64 (required for RunPod)
./scripts/build_worker.sh main-amd64

# Push to GHCR
./scripts/push_worker.sh main-amd64 ghcr.io/lukefind
```

Your image will be at: `ghcr.io/lukefind/bulk-transcribe-worker:main-amd64`

The push script will output the `IMAGE_DIGEST` - save this for provable identity.

## Step 2: Create RunPod Template

1. Go to [RunPod Templates](https://www.runpod.io/console/user/templates)
2. Click "New Template"
3. Fill in:

| Field | Value |
|-------|-------|
| Template Name | `bulk-transcribe-worker` |
| Container Image | `ghcr.io/lukefind/bulk-transcribe-worker:main-amd64` |
| Container Start Command | (leave empty) |
| Docker Command | (leave empty) |

4. Add Environment Variables:

| Variable | Value |
|----------|-------|
| `WORKER_TOKEN` | Your shared secret (generate with `openssl rand -hex 32`) |
| `HF_TOKEN` | Your HuggingFace token (for diarization) |
| `WORKER_PORT` | `8477` |

5. Expose HTTP Ports: `8477`

6. Click "Save Template"

## Step 3: Deploy a Pod

1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
2. Click "Deploy"
3. Select your template
4. Choose GPU:

| GPU | VRAM | Cost | Recommendation |
|-----|------|------|----------------|
| RTX 3090 | 24GB | ~$0.40/hr | Good for testing |
| RTX 4090 | 24GB | ~$0.70/hr | Fast, good value |
| A10 | 24GB | ~$0.50/hr | Reliable |
| A40 | 48GB | ~$1.00/hr | Long files + diarization |
| A100 | 80GB | ~$1.50-2.50/hr | Maximum speed |

5. Select "On-Demand" (not Spot for production)
6. Click "Deploy"

## Step 4: Get Your Worker URL

1. Wait for pod to start (1-2 minutes)
2. Click on your pod
3. Find the "Connect" section
4. Copy the HTTP URL for port 8477

It will look like: `https://abc123-8477.proxy.runpod.net`

## Step 5: Test the Worker

```bash
# Test health endpoint
curl https://abc123-8477.proxy.runpod.net/health

# Test capabilities
curl https://abc123-8477.proxy.runpod.net/v1/ping
```

Expected response:
```json
{
  "status": "ok",
  "version": "abc123",
  "gpu": true,
  "gpuName": "NVIDIA A100-SXM4-80GB",
  "cuda": "12.1",
  "models": ["large-v3"],
  "diarization": true
}
```

## Step 6: Configure Your Controller

Add to your controller's `.env`:

```bash
REMOTE_WORKER_URL=https://abc123-8477.proxy.runpod.net
REMOTE_WORKER_TOKEN=your-shared-secret
REMOTE_WORKER_MODE=optional
CONTROLLER_BASE_URL=https://your-controller.example.com
SECRET_KEY=another-random-secret
```

Restart your controller.

## Step 7: Run Smoke Test

```bash
export REMOTE_WORKER_URL=https://abc123-8477.proxy.runpod.net
export REMOTE_WORKER_TOKEN=your-shared-secret
./scripts/smoke_remote_worker.sh
```

## Troubleshooting

### Pod won't start

- Check RunPod logs in the pod details
- Verify your Docker image is public or you've configured registry auth
- Ensure WORKER_TOKEN is set

### Worker unreachable

- Wait 1-2 minutes after pod starts
- Check the HTTP URL is correct (includes port 8477)
- Verify pod is in "Running" state

### Authentication errors

- Verify WORKER_TOKEN matches REMOTE_WORKER_TOKEN exactly
- Check for extra whitespace in tokens

### Jobs fail with "Cannot connect to worker"

- Ensure CONTROLLER_BASE_URL is set and accessible from RunPod
- Check your controller's firewall allows incoming connections
- Verify SECRET_KEY is set for signed URLs

### Out of memory

- Use a GPU with more VRAM
- Reduce file size or split long audio
- Disable diarization for very long files

## Cost Optimization

1. **Use Spot instances** for non-critical batch jobs (50-70% cheaper)
2. **Stop pods when idle** - RunPod charges by the minute
3. **Use smaller GPUs** for testing (RTX 3090 is fine for development)
4. **Batch your jobs** - start pod, run all jobs, stop pod

## Verifying Worker Identity

After deploying or redeploying, verify the correct image is running:

```bash
# Check worker identity via controller
curl $CONTROLLER_URL/api/runtime | jq '.remoteWorker.identity'
```

Expected output:
```json
{
  "gitCommit": "b4b98e5",
  "imageDigest": "sha256:abc123...",
  "buildTime": "2026-01-25T15:30:00Z"
}
```

- **gitCommit**: Should match the commit you built from
- **imageDigest**: Only present if you set `IMAGE_DIGEST` env var in RunPod
- **buildTime**: When the image was built

### Updating the Worker

When you push a new image to `:main-amd64`:

1. Push the new image: `./scripts/push_worker.sh main-amd64 ghcr.io/lukefind`
2. In RunPod, click "Restart" on your pod (or redeploy)
3. Verify the new commit appears in `/api/runtime`

The `:main-amd64` tag is mutable - redeploying the pod pulls the latest image with that tag.

## Security Notes

- Never commit tokens to git
- Use HTTPS for both controller and worker
- Rotate tokens periodically
- RunPod pods are isolated but not private - don't store sensitive data
