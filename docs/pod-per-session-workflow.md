# Pod-Per-Session Workflow

This guide explains how to use the UI-managed remote worker configuration for a "destroy pod each session" workflow with RunPod.

## Overview

Instead of editing `.env` files and restarting the controller container each time you create a new RunPod pod, you can configure the remote worker directly from the admin UI.

**Benefits:**
- No container restarts needed
- Paste new RunPod URL directly in browser
- Test connection before saving
- See worker identity and status in real-time

## Prerequisites

1. Controller deployed with `ADMIN_TOKEN` set
2. RunPod account with credits
3. Worker image pushed to registry (see [gpu-worker-runpod.md](gpu-worker-runpod.md))

## Setup

### 1. Set ADMIN_TOKEN on Controller

Add to your controller's `.env` or Docker environment:

```bash
ADMIN_TOKEN=your-secure-admin-token
```

Generate a secure token:
```bash
openssl rand -hex 32
```

**Important:** Without `ADMIN_TOKEN`, the admin endpoints are disabled.

### 2. Access Admin Panel

Navigate to: `https://your-controller.example.com/admin`

Enter your admin token to authenticate.

## Workflow

### Starting a Session

1. **Create RunPod Pod**
   - Use template with `ghcr.io/lukefind/bulk-transcribe-worker:main-amd64`
   - Set `WORKER_TOKEN` in pod environment
   - Wait for pod to start (~1-2 min)

2. **Get Worker URL**
   - In RunPod console, find the HTTP proxy URL for port 8477
   - Example: `https://abc123-8477.proxy.runpod.net`

3. **Configure in Admin UI**
   - Go to `/admin` on your controller
   - Paste the RunPod URL
   - Enter the worker token (same as `WORKER_TOKEN` in pod)
   - Set mode to "Optional" or "Required"
   - Click "Test Connection" to verify
   - Click "Save"

4. **Start Transcribing**
   - Return to main UI (`/`)
   - Jobs will now use the remote GPU worker

### Ending a Session

1. **Stop/Destroy RunPod Pod**
   - In RunPod console, stop or terminate the pod
   - This stops billing immediately

2. **Update Controller (Optional)**
   - Go to `/admin`
   - Set mode to "Off" or leave as-is
   - The controller will gracefully fall back to CPU if worker is unreachable

## Configuration Precedence

The controller uses this precedence for remote worker config:

1. **Environment variables** (highest priority)
   - `REMOTE_WORKER_URL`, `WORKER_TOKEN`, `REMOTE_WORKER_MODE`
   - If these are set, they override UI config

2. **Saved config** (UI-configured)
   - Stored in `/data/config/remote_worker.json`
   - Persists across container restarts

3. **Defaults** (disabled)
   - Mode: off
   - No URL or token

**Tip:** For pod-per-session workflow, don't set `REMOTE_WORKER_URL` in environment. Let the UI config take effect.

## Security Notes

- **Token is never exposed** after saving - the UI only shows "Set" or "Not set"
- Admin endpoints require `X-Admin-Token` header
- Config file has restricted permissions (0600)
- Token is never logged

## API Reference

### GET /api/admin/remote-worker

Get current saved configuration (without token).

```bash
curl -H "X-Admin-Token: $ADMIN_TOKEN" \
  https://your-controller/api/admin/remote-worker
```

Response:
```json
{
  "url": "https://abc123-8477.proxy.runpod.net",
  "mode": "optional",
  "tokenSet": true,
  "updatedAt": "2026-01-25T16:00:00Z",
  "configSource": "saved"
}
```

### POST /api/admin/remote-worker

Update configuration.

```bash
curl -X POST \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://new-url-8477.proxy.runpod.net", "token": "new-token", "mode": "optional"}' \
  https://your-controller/api/admin/remote-worker
```

- Omit `token` to keep existing
- Set `token` to empty string to clear

### POST /api/admin/remote-worker/test

Test connection to worker.

```bash
curl -X POST \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://abc123-8477.proxy.runpod.net", "token": "test-token"}' \
  https://your-controller/api/admin/remote-worker/test
```

Response (success):
```json
{
  "ok": true,
  "latencyMs": 150,
  "identity": {
    "gitCommit": "f22e342",
    "buildTime": "2026-01-25T15:30:00Z"
  },
  "gpu": true,
  "gpuName": "NVIDIA A100-SXM4-80GB",
  "diarization": true
}
```

## Troubleshooting

### "Admin endpoint not configured"

Set `ADMIN_TOKEN` in controller environment and restart.

### "Unauthorized"

Check that you're using the correct admin token.

### Connection test fails but curl works

- Ensure URL includes the full proxy path (e.g., `https://xxx-8477.proxy.runpod.net`)
- Check token matches exactly (no extra whitespace)
- Verify pod is in "Running" state

### Config not taking effect

If env vars are set, they override UI config. Check:
```bash
# In controller container
echo $REMOTE_WORKER_URL
echo $WORKER_TOKEN
```

If these are set, either unset them or use them instead of UI config.

## Cost Optimization

With this workflow:

1. **Start pod** when you have work
2. **Configure via UI** (takes 30 seconds)
3. **Run your batch**
4. **Stop pod immediately** when done

You only pay for actual GPU time, not idle time waiting for work.
