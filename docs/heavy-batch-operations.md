# Heavy Batch Operations Guide

This guide covers running large transcription workloads (e.g., 48 files x 20 minutes each) with the remote GPU worker.

## Overview

The system is designed for heavy batch workloads with:
- **Capacity-aware dispatch**: Controller waits for worker capacity before sending jobs
- **Exponential backoff**: Polling uses backoff with jitter to avoid thundering herd
- **Resumable batch runner**: Script tracks state and can resume interrupted runs
- **Operator visibility**: Clear error codes and status in UI and API

## Worker Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_MAX_CONCURRENT_JOBS` | `1` | Max simultaneous jobs on worker |
| `WORKER_MODEL` | `large-v3` | Default Whisper model |
| `WORKER_MAX_FILE_MB` | `2000` | Max file size in MB |
| `WORKER_TOKEN` | (required) | Authentication token |
| `WORKER_PING_PUBLIC` | `false` | Allow unauthenticated `/v1/ping` |

### Recommended Settings by GPU

| GPU | VRAM | Recommended `WORKER_MAX_CONCURRENT_JOBS` | Notes |
|-----|------|------------------------------------------|-------|
| RTX 4090 | 24GB | 2-3 | Good fallback |
| A40 | 48GB | 2-3 | Strong balance |
| L40 / L40S | 48GB | 2-3 | Best throughput |
| RTX 3090 | 24GB | 2 | Slightly slower than 4090 |
| RTX 3080 | 10GB | 1 | Limited VRAM for large models |
| A100 40GB | 40GB | 4-6 | Data center GPU, excellent throughput |
| A100 80GB | 80GB | 8-10 | Maximum throughput |

### Diarization Considerations

When diarization is enabled:
- VRAM usage increases significantly
- Reduce `WORKER_MAX_CONCURRENT_JOBS` by 1
- Consider using `--diarization-max-duration` to limit segment length
- **Fast Switching** (shorter chunks + higher overlap) improves turn-taking but slows runtime

## Controller Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REMOTE_WORKER_URL` | (none) | Worker URL (e.g., `http://gpu-server:8477`) |
| `REMOTE_WORKER_TOKEN` | (none) | Authentication token (must match worker) |
| `REMOTE_WORKER_MODE` | `off` | `off`, `optional`, or `required` |
| `REMOTE_WORKER_TIMEOUT_SECONDS` | `3600` | Job timeout (1 hour default) |
| `REMOTE_WORKER_POLL_SECONDS` | `5` | Status poll interval |

## Batch Runner

The `scripts/run_batch.sh` script automates batch processing with resumability.

### Usage

```bash
./scripts/run_batch.sh --input ./myfiles [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--input DIR` | Input directory containing audio files (required) |
| `--controller URL` | Controller URL (default: `http://localhost:8476`) |
| `--remote` | Request remote GPU worker execution |
| `--diarization` | Enable speaker diarization |
| `--model MODEL` | Whisper model (default: `large-v3`) |
| `--state FILE` | State file path (default: `batch_state.json` in input dir) |
| `--parallel N` | Max parallel jobs (default: 1) |

### Example

```bash
# Process 48 files with remote GPU and diarization
./scripts/run_batch.sh \
    --input ./interview-recordings \
    --remote \
    --diarization \
    --model large-v3
```

### Resumability

The script writes `batch_state.json` to track:
- Upload IDs for each file
- Job IDs and status
- Execution mode (local/remote)

If interrupted, re-run the same command to resume. Completed jobs are skipped.

### State File Format

```json
{
  "files": {
    "interview-001.mp3": {
      "uploadId": "abc123",
      "jobId": "xyz789",
      "status": "complete",
      "executionMode": "remote"
    }
  },
  "session": "...",
  "csrfToken": "..."
}
```

## Job States

### Remote Job Lifecycle

```
queued -> queued_remote -> running -> complete/failed/canceled
```

| State | Description |
|-------|-------------|
| `queued` | Job created, waiting to start |
| `queued_remote` | Waiting for worker capacity |
| `running` | Actively processing on worker |
| `complete` | Successfully finished |
| `failed` | Error occurred |
| `canceled` | User canceled |

### Cancellation

Jobs can be canceled at any state:
- **From `queued_remote`**: Immediately marked canceled, no worker contact needed
- **From `running`**: Cancel request sent to worker, job marked canceled

## Error Codes

Standard error codes for remote worker failures:

| Code | Description | Action |
|------|-------------|--------|
| `REMOTE_WORKER_UNAUTHORIZED` | Token mismatch | Check `REMOTE_WORKER_TOKEN` matches worker |
| `REMOTE_WORKER_UNREACHABLE` | Worker temporarily unreachable | Job stays queued; wait for recovery or cancel |
| `REMOTE_WORKER_TIMEOUT` | Job or capacity wait timed out | Increase timeout or check worker health |
| `REMOTE_WORKER_CAPACITY` | Timed out waiting for capacity | Reduce batch size or increase worker capacity |
| `REMOTE_DISPATCH_FAILED` | Failed to create job on worker | Check worker logs |
| `REMOTE_FAILED` | Worker reported job failure | Check worker logs for details |
| `USER_CANCELED` | User canceled the job | No action needed |

## Monitoring

### API Endpoints

```bash
# Check worker status and capacity
curl -s http://localhost:8476/api/runtime | jq '.remoteWorker'

# Check job status
curl -s -b cookies.txt http://localhost:8476/api/jobs/JOB_ID
```

### Worker Capacity Response

```json
{
  "connected": true,
  "workerCapabilities": {
    "activeJobs": 1,
    "maxConcurrentJobs": 2,
    "gpu": true,
    "gpuName": "NVIDIA GeForce RTX 4090"
  }
}
```

### Job Status Response

```json
{
  "status": "running",
  "executionMode": "remote",
  "progress": {
    "stage": "transcribing",
    "percent": 45
  },
  "worker": {
    "workerJobId": "wk_abc123",
    "lastSeenAt": "2024-01-15T10:30:00Z",
    "gpu": true
  },
  "lastErrorCode": null,
  "lastErrorMessage": null
}
```

## Smoke Test

Before running a large batch:

```bash
# 1. Verify worker is reachable
curl -sS -H "Authorization: Bearer $WORKER_TOKEN" \
    http://gpu-server:8477/v1/ping | jq

# 2. Check capacity
curl -sS http://localhost:8476/api/runtime | jq '.remoteWorker.workerCapabilities'

# 3. Test with single file
./scripts/run_batch.sh --input ./test-single-file --remote

# 4. Check logs
tail -f logs/app.log | grep -E 'remote_|capacity'
```

## Failure Modes

### Worker Unreachable

**Symptoms**: Jobs stuck in `queued_remote`, `REMOTE_WORKER_UNREACHABLE` errors

**Causes**:
- Network connectivity issue
- Worker not running
- Firewall blocking port

**Resolution**:
1. Check worker is running: `curl http://gpu-server:8477/health`
2. Check network: `ping gpu-server`
3. Check firewall rules

### Capacity Exhausted

**Symptoms**: Jobs stuck in `queued_remote` with "Waiting for worker capacity" message

**Causes**:
- Too many concurrent jobs
- Long-running jobs blocking capacity

**Resolution**:
1. Wait for current jobs to complete
2. Increase `WORKER_MAX_CONCURRENT_JOBS` if GPU has headroom
3. Reduce batch parallelism

### Authentication Failures

**Symptoms**: `REMOTE_WORKER_UNAUTHORIZED` error, 401/403 responses

**Causes**:
- Token mismatch between controller and worker
- Token not set

**Resolution**:
1. Verify `REMOTE_WORKER_TOKEN` matches `WORKER_TOKEN`
2. Restart both services after changing tokens

### Timeout

**Symptoms**: `REMOTE_WORKER_TIMEOUT` error after long wait

**Causes**:
- Job taking longer than timeout
- Worker crashed mid-job

**Resolution**:
1. Increase `REMOTE_WORKER_TIMEOUT_SECONDS`
2. Check worker logs for crashes
3. Consider splitting large files

## Best Practices

1. **Start small**: Test with 2-3 files before running full batch
2. **Monitor capacity**: Watch `activeJobs` vs `maxConcurrentJobs`
3. **Use resumability**: Always use the batch runner for large jobs
4. **Check logs**: Monitor both controller and worker logs
5. **Set appropriate timeouts**: Large files with diarization can take 30+ minutes
6. **Backup state file**: Copy `batch_state.json` periodically for very large batches
