"""
Remote Worker Client for Bulk Transcribe Controller.

This module handles communication with remote GPU workers for offloading
transcription and diarization workloads.

Configuration (environment variables):
- REMOTE_WORKER_URL: Base URL of the worker (e.g., https://gpu-worker.example.com)
- REMOTE_WORKER_TOKEN: Shared secret for authentication
- REMOTE_WORKER_MODE: off|optional|required (default: off)
- REMOTE_WORKER_TIMEOUT_SECONDS: Job timeout (default: 7200)
- REMOTE_WORKER_POLL_SECONDS: Poll interval (default: 2)
- REMOTE_WORKER_UPLOAD_MODE: pull|push (default: pull)
"""

import os
import time
import hmac
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any
from urllib.parse import urljoin

import requests
from logger import log_event


# Configuration
def get_worker_config() -> Dict[str, Any]:
    """Get remote worker configuration from environment."""
    return {
        'url': os.environ.get('REMOTE_WORKER_URL', ''),
        'token': os.environ.get('REMOTE_WORKER_TOKEN', ''),
        'mode': os.environ.get('REMOTE_WORKER_MODE', 'off'),  # off|optional|required
        'timeoutSeconds': int(os.environ.get('REMOTE_WORKER_TIMEOUT_SECONDS', '7200')),
        'pollSeconds': int(os.environ.get('REMOTE_WORKER_POLL_SECONDS', '2')),
        'uploadMode': os.environ.get('REMOTE_WORKER_UPLOAD_MODE', 'pull'),  # pull|push
        # Expected identity for mismatch detection (optional, for auditing)
        'expectedGitCommit': os.environ.get('EXPECTED_WORKER_GIT_COMMIT', ''),
        'expectedImageDigest': os.environ.get('EXPECTED_WORKER_IMAGE_DIGEST', ''),
    }


def is_remote_worker_available() -> bool:
    """Check if remote worker is configured and reachable."""
    config = get_worker_config()
    if not config['url'] or not config['token']:
        return False
    
    try:
        response = requests.get(
            urljoin(config['url'], '/health'),
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


# Cache for worker status - keyed by config to avoid stale data on config change
_worker_status_cache: Dict[str, Any] = {}
_worker_status_cache_time: Dict[str, float] = {}
_WORKER_STATUS_CACHE_TTL = 30  # seconds


def _get_cache_key(config: Dict[str, Any]) -> str:
    """Generate cache key from config to invalidate on config change."""
    token_prefix = config['token'][:8] if config['token'] else ''
    return f"{config['url']}|{token_prefix}|{config['mode']}"


def get_remote_worker_status(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get detailed status of remote worker including capabilities.
    
    Results are cached for 30 seconds, keyed by config to avoid stale data.
    Use force_refresh=True to bypass cache.
    """
    config = get_worker_config()
    cache_key = _get_cache_key(config)
    
    # Return cached result if fresh and config unchanged
    if not force_refresh and cache_key in _worker_status_cache:
        if time.time() - _worker_status_cache_time.get(cache_key, 0) < _WORKER_STATUS_CACHE_TTL:
            return _worker_status_cache[cache_key]
    
    # Build expected identity from config (if set)
    expected_identity = None
    if config['expectedGitCommit'] or config['expectedImageDigest']:
        expected_identity = {
            'gitCommit': config['expectedGitCommit'] or None,
            'imageDigest': config['expectedImageDigest'] or None
        }
    
    result = {
        'enabled': config['mode'] != 'off',
        'configured': bool(config['url'] and config['token']),
        'mode': config['mode'],
        'url': config['url'],  # Full URL - UI can truncate for display
        'connected': False,
        'lastPingAt': None,
        'latencyMs': None,
        'workerVersion': None,
        # Worker-reported identity (gitCommit, buildTime, declaredImageDigest)
        'identity': None,
        # Controller-configured expected identity (for mismatch detection)
        'expectedIdentity': expected_identity,
        # Mismatch detection results
        'identityMatches': None,  # True/False/None (None if no expected identity)
        'identityMismatchReason': None,
        'workerCapabilities': None,
        'error': None
    }
    
    if not result['configured']:
        if config['mode'] == 'off':
            result['error'] = None  # Not an error if mode is off
        else:
            result['error'] = 'Remote worker not configured (missing URL or token)'
        _worker_status_cache[cache_key] = result
        _worker_status_cache_time[cache_key] = time.time()
        return result
    
    # Ping with auth token - short timeout to avoid blocking UI
    start_time = time.time()
    headers = {'Authorization': f'Bearer {config["token"]}'}
    
    try:
        response = requests.get(
            urljoin(config['url'], '/v1/ping'),
            headers=headers,
            timeout=2
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        if response.status_code == 200:
            result['connected'] = True
            result['lastPingAt'] = datetime.now(timezone.utc).isoformat()
            result['latencyMs'] = latency_ms
            
            ping_data = response.json()
            result['workerVersion'] = ping_data.get('version')
            # Capture worker-reported identity
            result['identity'] = ping_data.get('identity')
            
            # Perform identity mismatch detection
            identity = result['identity']
            if identity and expected_identity:
                mismatches = []
                
                # Check gitCommit mismatch
                if expected_identity.get('gitCommit'):
                    actual_commit = identity.get('gitCommit')
                    if actual_commit != expected_identity['gitCommit']:
                        mismatches.append(f"gitCommit: expected '{expected_identity['gitCommit']}', got '{actual_commit}'")
                
                # Check imageDigest mismatch (compare with declaredImageDigest)
                if expected_identity.get('imageDigest'):
                    actual_digest = identity.get('declaredImageDigest') or identity.get('imageDigest')
                    if actual_digest != expected_identity['imageDigest']:
                        mismatches.append(f"imageDigest: expected '{expected_identity['imageDigest'][:20]}...', got '{(actual_digest or 'none')[:20]}...'")
                
                if mismatches:
                    result['identityMatches'] = False
                    result['identityMismatchReason'] = '; '.join(mismatches)
                    log_event('warning', 'remote_worker_identity_mismatch',
                              reason=result['identityMismatchReason'],
                              expectedGitCommit=expected_identity.get('gitCommit'),
                              actualGitCommit=identity.get('gitCommit'))
                else:
                    result['identityMatches'] = True
            elif expected_identity:
                # Expected identity set but worker didn't report identity
                result['identityMatches'] = False
                result['identityMismatchReason'] = 'Worker did not report identity'
            # else: no expected identity configured, leave identityMatches as None
            
            # Log identity for debugging
            if identity:
                log_event('debug', 'remote_worker_identity',
                          gitCommit=identity.get('gitCommit'),
                          declaredImageDigest=(identity.get('declaredImageDigest') or 'not set')[:20],
                          buildTime=identity.get('buildTime'),
                          identityMatches=result['identityMatches'])
            
            result['workerCapabilities'] = {
                'whisperModels': ping_data.get('models', []),
                'diarization': ping_data.get('diarization', False),
                'gpu': ping_data.get('gpu', False),
                'gpuName': ping_data.get('gpuName'),
                'cuda': ping_data.get('cuda'),
                'maxFileMB': ping_data.get('maxFileMB'),
                'activeJobs': ping_data.get('activeJobs', 0),
                'maxConcurrentJobs': ping_data.get('maxConcurrentJobs', 1)
            }
        elif response.status_code in (401, 403):
            result['error'] = 'Unauthorized (check REMOTE_WORKER_TOKEN)'
        else:
            result['error'] = f'Worker returned status {response.status_code}'
            
    except requests.exceptions.Timeout:
        result['error'] = 'Worker connection timed out'
    except requests.exceptions.ConnectionError:
        result['error'] = 'Cannot connect to worker'
    except Exception as e:
        result['error'] = str(e)[:100]
    
    # Cache the result keyed by config
    _worker_status_cache[cache_key] = result
    _worker_status_cache_time[cache_key] = time.time()
    
    return result


def should_use_remote_worker(user_requested: bool = False) -> bool:
    """
    Determine if a job should use remote worker.
    
    Args:
        user_requested: True if user explicitly requested remote execution
    
    Returns:
        True if job should run on remote worker
    """
    config = get_worker_config()
    mode = config['mode']
    
    if mode == 'off':
        return False
    elif mode == 'required':
        return True
    elif mode == 'optional':
        return user_requested and is_remote_worker_available()
    
    return False


def check_worker_capacity() -> Dict[str, Any]:
    """
    Check if worker has capacity for new jobs.
    
    Returns:
        Dict with:
        - hasCapacity: bool
        - activeJobs: int
        - maxConcurrentJobs: int
        - error: str or None
    """
    status = get_remote_worker_status(force_refresh=True)
    
    if not status['connected']:
        return {
            'hasCapacity': False,
            'activeJobs': 0,
            'maxConcurrentJobs': 0,
            'error': status.get('error') or 'Worker not connected'
        }
    
    caps = status.get('workerCapabilities', {})
    active = caps.get('activeJobs', 0)
    max_concurrent = caps.get('maxConcurrentJobs', 1)
    
    return {
        'hasCapacity': active < max_concurrent,
        'activeJobs': active,
        'maxConcurrentJobs': max_concurrent,
        'error': None
    }


import random

# Backoff configuration
_BACKOFF_MIN_SECONDS = 2
_BACKOFF_MAX_SECONDS = 30
_BACKOFF_BASE = 2
_BACKOFF_JITTER_FACTOR = 0.3


def calculate_backoff(attempt: int, seed: int = None) -> float:
    """
    Calculate exponential backoff with jitter.
    
    Args:
        attempt: Attempt number (0-indexed)
        seed: Optional seed for deterministic jitter (for testing)
    
    Returns:
        Backoff duration in seconds
    """
    if seed is not None:
        random.seed(seed)
    
    # Exponential backoff: base^attempt
    base_delay = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE ** attempt)
    
    # Add jitter: +/- jitter_factor
    jitter = base_delay * _BACKOFF_JITTER_FACTOR * (2 * random.random() - 1)
    delay = base_delay + jitter
    
    # Clamp to min/max
    return max(_BACKOFF_MIN_SECONDS, min(_BACKOFF_MAX_SECONDS, delay))


def generate_signed_url(base_url: str, job_id: str, input_id: str, 
                       secret: str, expires_in_seconds: int = 3600) -> str:
    """
    Generate a signed URL for input file download.
    
    Args:
        base_url: Controller base URL
        job_id: Job ID
        input_id: Input/upload ID
        secret: Server secret for signing
        expires_in_seconds: URL validity duration
    
    Returns:
        Signed URL string
    """
    expires_at = int(time.time()) + expires_in_seconds
    
    # Create signature payload
    payload = f"{job_id}:{input_id}:{expires_at}"
    signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()[:32]
    
    # Build URL
    url = f"{base_url}/api/jobs/{job_id}/inputs/{input_id}"
    url += f"?expires={expires_at}&sig={signature}"
    
    return url


def verify_signed_url(job_id: str, input_id: str, expires: str, 
                     signature: str, secret: str) -> bool:
    """
    Verify a signed URL.
    
    Returns:
        True if signature is valid and not expired
    """
    try:
        expires_at = int(expires)
        if time.time() > expires_at:
            return False
        
        payload = f"{job_id}:{input_id}:{expires_at}"
        expected_sig = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()[:32]
        
        return hmac.compare_digest(signature, expected_sig)
    except Exception:
        return False


class RemoteWorkerClient:
    """Client for communicating with remote GPU worker."""
    
    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        self.session.headers['Authorization'] = f'Bearer {token}'
    
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make authenticated request to worker."""
        url = f"{self.url}{path}"
        kwargs.setdefault('timeout', 30)
        return self.session.request(method, url, **kwargs)
    
    def create_job(self, controller_job_id: str, controller_session_hash: str,
                   inputs: list, options: dict, callback_url: str,
                   outputs_upload_url: str) -> Dict[str, Any]:
        """
        Create a job on the remote worker.
        
        Args:
            controller_job_id: Job ID on controller
            controller_session_hash: Hashed session ID (for logging only)
            inputs: List of input file descriptors with downloadUrl
            options: Job options (model, diarization settings, etc.)
            callback_url: URL for worker to POST completion status
            outputs_upload_url: URL for worker to upload output files
        
        Returns:
            Dict with workerJobId and status
        """
        payload = {
            'controllerJobId': controller_job_id,
            'controllerSessionHash': controller_session_hash,
            'inputs': inputs,
            'options': options,
            'callbackUrl': callback_url,
            'outputsUploadUrl': outputs_upload_url
        }
        
        response = self._request(
            'POST', '/v1/jobs',
            json=payload,
            headers={'Idempotency-Key': controller_job_id}
        )
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, worker_job_id: str) -> Dict[str, Any]:
        """Get job status from worker."""
        response = self._request('GET', f'/v1/jobs/{worker_job_id}')
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, worker_job_id: str) -> Dict[str, Any]:
        """Request job cancellation on worker."""
        response = self._request('POST', f'/v1/jobs/{worker_job_id}/cancel')
        response.raise_for_status()
        return response.json()


def dispatch_to_remote_worker(
    session_id: str,
    job_id: str,
    inputs: list,
    options: dict,
    controller_base_url: str,
    update_manifest_callback: Callable,
    is_cancelled_callback: Callable
) -> bool:
    """
    Dispatch a job to remote worker and poll for completion.
    
    This function runs in a background thread and:
    1. Waits for worker capacity (queued_remote state)
    2. Creates job on remote worker
    3. Polls for status updates with exponential backoff on errors
    4. Updates local manifest with progress
    5. Handles completion/failure/cancellation
    
    Args:
        session_id: Session ID
        job_id: Job ID
        inputs: List of input file info (with paths)
        options: Job options
        controller_base_url: Base URL of this controller
        update_manifest_callback: Function to update job manifest
        is_cancelled_callback: Function to check if job is cancelled
    
    Returns:
        True if job completed successfully
    """
    config = get_worker_config()
    client = RemoteWorkerClient(config['url'], config['token'])
    
    # Generate session hash for logging (don't expose full session ID)
    session_hash = hashlib.sha256(session_id.encode()).hexdigest()[:8]
    
    # Wait for worker capacity before dispatching
    capacity_wait_start = time.time()
    capacity_wait_timeout = 3600  # 1 hour max wait for capacity
    capacity_check_attempt = 0
    
    while True:
        if is_cancelled_callback():
            log_event('info', 'remote_job_cancelled_while_queued', jobId=job_id)
            update_manifest_callback(
                status='canceled',
                finishedAt=datetime.now(timezone.utc).isoformat(),
                error={'code': 'USER_CANCELED', 'message': 'Job canceled while waiting for worker capacity'}
            )
            return False
        
        capacity = check_worker_capacity()
        
        if capacity['error']:
            # Worker unreachable - use backoff
            backoff = calculate_backoff(capacity_check_attempt)
            log_event('warning', 'remote_capacity_check_failed',
                      jobId=job_id, error=capacity['error'], backoffSeconds=backoff)
            update_manifest_callback(
                status='queued_remote',
                progress={'stage': 'queued_remote', 'currentFile': f'Worker unreachable, retrying in {int(backoff)}s...'},
                lastErrorCode='REMOTE_WORKER_UNREACHABLE',
                lastErrorMessage=capacity['error'][:200],
                lastErrorAt=datetime.now(timezone.utc).isoformat()
            )
            capacity_check_attempt += 1
            time.sleep(backoff)
            
            # Check timeout
            if time.time() - capacity_wait_start > capacity_wait_timeout:
                log_event('error', 'remote_capacity_wait_timeout', jobId=job_id)
                update_manifest_callback(
                    status='failed',
                    finishedAt=datetime.now(timezone.utc).isoformat(),
                    error={'code': 'REMOTE_WORKER_TIMEOUT', 'message': 'Timed out waiting for worker capacity'}
                )
                return False
            continue
        
        if capacity['hasCapacity']:
            log_event('info', 'remote_capacity_available',
                      jobId=job_id, activeJobs=capacity['activeJobs'], 
                      maxConcurrentJobs=capacity['maxConcurrentJobs'])
            break
        
        # No capacity - wait with backoff
        backoff = calculate_backoff(min(capacity_check_attempt, 4))  # Cap at ~16s base
        log_event('info', 'remote_waiting_for_capacity',
                  jobId=job_id, activeJobs=capacity['activeJobs'],
                  maxConcurrentJobs=capacity['maxConcurrentJobs'], backoffSeconds=backoff)
        update_manifest_callback(
            status='queued_remote',
            progress={
                'stage': 'queued_remote',
                'currentFile': f'Waiting for worker capacity ({capacity["activeJobs"]}/{capacity["maxConcurrentJobs"]} active)...'
            }
        )
        capacity_check_attempt += 1
        time.sleep(backoff)
        
        # Check timeout
        if time.time() - capacity_wait_start > capacity_wait_timeout:
            log_event('error', 'remote_capacity_wait_timeout', jobId=job_id)
            update_manifest_callback(
                status='failed',
                finishedAt=datetime.now(timezone.utc).isoformat(),
                error={'code': 'REMOTE_WORKER_CAPACITY', 'message': 'Timed out waiting for worker capacity'}
            )
            return False
    
    # Build signed download URLs for inputs
    secret = os.environ.get('SECRET_KEY', 'default-secret-key')
    remote_inputs = []
    for inp in inputs:
        download_url = generate_signed_url(
            controller_base_url,
            job_id,
            inp['uploadId'],
            secret,
            expires_in_seconds=config['timeoutSeconds']
        )
        remote_inputs.append({
            'inputId': inp['uploadId'],
            'filename': inp.get('originalFilename', inp.get('filename', 'input')),
            'contentType': 'audio/*',
            'sizeBytes': inp.get('sizeBytes', 0),
            'downloadUrl': download_url
        })
    
    # Build callback URLs
    callback_url = f"{controller_base_url}/api/jobs/{job_id}/worker/complete"
    outputs_upload_url = f"{controller_base_url}/api/jobs/{job_id}/worker/outputs"
    
    log_event('info', 'remote_dispatch_started',
              jobId=job_id, sessionHash=session_hash,
              workerUrl=config['url'][:50])
    
    try:
        # Create job on worker
        result = client.create_job(
            controller_job_id=job_id,
            controller_session_hash=session_hash,
            inputs=remote_inputs,
            options=options,
            callback_url=callback_url,
            outputs_upload_url=outputs_upload_url
        )
        
        worker_job_id = result.get('workerJobId')
        if not worker_job_id:
            raise ValueError('Worker did not return workerJobId')
        
        # Update manifest with worker info
        update_manifest_callback(
            worker={
                'workerJobId': worker_job_id,
                'url': config['url'],
                'createdAt': datetime.now(timezone.utc).isoformat(),
                'lastSeenAt': datetime.now(timezone.utc).isoformat()
            },
            executionMode='remote'
        )
        
        log_event('info', 'remote_job_created',
                  jobId=job_id, workerJobId=worker_job_id)
        
        # Poll for completion with exponential backoff on errors
        poll_interval = config['pollSeconds']
        timeout = config['timeoutSeconds']
        start_time = time.time()
        consecutive_errors = 0
        
        while True:
            # Check for cancellation
            if is_cancelled_callback():
                log_event('info', 'remote_job_cancelling', jobId=job_id, workerJobId=worker_job_id)
                try:
                    client.cancel_job(worker_job_id)
                except Exception as e:
                    log_event('warning', 'remote_cancel_failed', jobId=job_id, error=str(e))
                update_manifest_callback(
                    status='canceled',
                    finishedAt=datetime.now(timezone.utc).isoformat(),
                    error={'code': 'USER_CANCELED', 'message': 'Job canceled by user'}
                )
                return False
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                log_event('error', 'remote_job_timeout', jobId=job_id, workerJobId=worker_job_id)
                update_manifest_callback(
                    status='failed',
                    finishedAt=datetime.now(timezone.utc).isoformat(),
                    error={'code': 'REMOTE_WORKER_TIMEOUT', 'message': f'Worker job timed out after {timeout}s'}
                )
                return False
            
            # Poll worker status
            try:
                status = client.get_job_status(worker_job_id)
                consecutive_errors = 0  # Reset on success
                
                # Update manifest with progress
                update_manifest_callback(
                    progress={
                        'stage': status.get('stage', 'unknown'),
                        'currentFileIndex': status.get('progress', {}).get('currentFileIndex', 0),
                        'totalFiles': status.get('progress', {}).get('totalFiles', len(inputs)),
                        'chunkIndex': status.get('progress', {}).get('chunkIndex', 0),
                        'totalChunks': status.get('progress', {}).get('totalChunks', 0),
                        'percent': status.get('progress', {}).get('percent', 0),
                        'remoteStatus': status.get('status')
                    },
                    worker={
                        'workerJobId': worker_job_id,
                        'url': config['url'],
                        'lastSeenAt': datetime.now(timezone.utc).isoformat(),
                        'gpu': status.get('gpu')
                    }
                )
                
                # Check for completion
                worker_status = status.get('status')
                if worker_status == 'complete':
                    log_event('info', 'remote_job_complete', jobId=job_id, workerJobId=worker_job_id)
                    update_manifest_callback(
                        status='complete',
                        finishedAt=datetime.now(timezone.utc).isoformat()
                    )
                    return True
                elif worker_status == 'failed':
                    error = status.get('error', {})
                    log_event('error', 'remote_job_failed', jobId=job_id, 
                              workerJobId=worker_job_id, error=error)
                    update_manifest_callback(
                        status='failed',
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                        error=error or {'code': 'REMOTE_FAILED', 'message': 'Worker job failed'}
                    )
                    return False
                elif worker_status == 'canceled':
                    log_event('info', 'remote_job_canceled', jobId=job_id, workerJobId=worker_job_id)
                    update_manifest_callback(
                        status='canceled',
                        finishedAt=datetime.now(timezone.utc).isoformat()
                    )
                    return False
                
                # Normal poll interval on success
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                consecutive_errors += 1
                backoff = calculate_backoff(consecutive_errors)
                log_event('warning', 'remote_poll_error', 
                          jobId=job_id, error=str(e), 
                          consecutiveErrors=consecutive_errors, backoffSeconds=backoff)
                
                # Update manifest with error info
                update_manifest_callback(
                    lastErrorCode='REMOTE_WORKER_UNREACHABLE',
                    lastErrorMessage=str(e)[:200],
                    lastErrorAt=datetime.now(timezone.utc).isoformat()
                )
                
                # Backoff on errors
                time.sleep(backoff)
        
    except Exception as e:
        log_event('error', 'remote_dispatch_failed', jobId=job_id, error=str(e))
        update_manifest_callback(
            status='failed',
            error={'code': 'REMOTE_DISPATCH_FAILED', 'message': str(e)[:200]}
        )
        return False
