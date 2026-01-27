# Install & Setup

This guide covers local setup, Docker setup, and optional remote GPU worker configuration.

---

## Local (Python)

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python app.py
```

Open: http://localhost:8476

---

## Docker (Local)

```bash
docker compose up -d app
```

Open: http://localhost:8476

---

## Optional: Remote GPU Worker (RunPod)

Use a remote GPU worker for large batches or diarization-heavy jobs.

### 1) Build & Push Worker Image

```bash
./scripts/build_worker.sh main-amd64
./scripts/push_worker.sh main-amd64 ghcr.io/lukefind
```

### 2) Create RunPod Template

Container image:
```
ghcr.io/lukefind/bulk-transcribe-worker:main-amd64
```

Environment variables (recommended):

```
WORKER_TOKEN=<shared-secret>
WORKER_PORT=8477
WORKER_MODEL=large-v3
WORKER_MAX_CONCURRENT_JOBS=1
WORKER_MAX_FILE_MB=2000
WORKER_TMP_DIR=/tmp/bt-worker
DIARIZATION_DEVICE=cuda
HF_TOKEN=<huggingface-token>
WORKER_PING_PUBLIC=false
```

### 3) Start a Pod

Recommended GPUs:
- **L40 / L40S** (fastest)
- **A40** (strong balance)
- **RTX 4090** (fallback)

### 4) Connect the Worker

In the app UI:
- Paste the RunPod proxy URL
- Enter `WORKER_TOKEN`
- Set mode to **Required** or **Optional**

---

## Verify

From the controller:

```bash
curl -sS http://localhost:8476/api/runtime | jq '.remoteWorker'
```

From the worker:

```bash
curl -sS https://<worker-url>/v1/ping -H "Authorization: Bearer $WORKER_TOKEN" | jq
```

---

## Notes

- First run is slower (model download + cache warm-up)
- Diarization adds time; use Fast Switching only when needed
