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


# Cache for worker status (avoid spamming worker on every /api/runtime call)
_worker_status_cache = None
_worker_status_cache_time = 0
_WORKER_STATUS_CACHE_TTL = 30  # seconds


def get_remote_worker_status(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get detailed status of remote worker including capabilities.
    
    Results are cached for 30 seconds to avoid spamming the worker.
    Use force_refresh=True to bypass cache.
    """
    global _worker_status_cache, _worker_status_cache_time
    
    # Return cached result if fresh
    if not force_refresh and _worker_status_cache is not None:
        if time.time() - _worker_status_cache_time < _WORKER_STATUS_CACHE_TTL:
            return _worker_status_cache
    
    config = get_worker_config()
    
    result = {
        'enabled': config['mode'] != 'off',
        'configured': bool(config['url'] and config['token']),
        'mode': config['mode'],
        'url': config['url'][:50] + '...' if len(config['url']) > 50 else config['url'],
        'connected': False,
        'lastPingAt': None,
        'latencyMs': None,
        'workerVersion': None,
        'workerCapabilities': None,
        'error': None
    }
    
    if not result['configured']:
        if config['mode'] == 'off':
            result['error'] = None  # Not an error if mode is off
        else:
            result['error'] = 'Remote worker not configured (missing URL or token)'
        _worker_status_cache = result
        _worker_status_cache_time = time.time()
        return result
    
    # Try the detailed /v1/ping endpoint - short timeout to avoid blocking UI
    start_time = time.time()
    try:
        response = requests.get(
            urljoin(config['url'], '/v1/ping'),
            timeout=2  # Short timeout - UI should not block
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        if response.status_code == 200:
            result['connected'] = True
            result['lastPingAt'] = datetime.now(timezone.utc).isoformat()
            result['latencyMs'] = latency_ms
            
            ping_data = response.json()
            result['workerVersion'] = ping_data.get('version')
            result['workerCapabilities'] = {
                'whisperModels': ping_data.get('models', []),
                'diarization': ping_data.get('diarization', False),
                'gpu': ping_data.get('gpu', False),
                'gpuName': ping_data.get('gpuName'),
                'cuda': ping_data.get('cuda'),
                'maxFileMB': ping_data.get('maxFileMB')
            }
        else:
            result['error'] = f'Worker returned status {response.status_code}'
            
    except requests.exceptions.Timeout:
        result['error'] = 'Worker connection timed out'
    except requests.exceptions.ConnectionError:
        result['error'] = 'Cannot connect to worker'
    except Exception as e:
        result['error'] = str(e)[:100]
    
    # Cache the result
    _worker_status_cache = result
    _worker_status_cache_time = time.time()
    
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
    1. Creates job on remote worker
    2. Polls for status updates
    3. Updates local manifest with progress
    4. Handles completion/failure/cancellation
    
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
        
        # Poll for completion
        poll_interval = config['pollSeconds']
        timeout = config['timeoutSeconds']
        start_time = time.time()
        
        while True:
            # Check for cancellation
            if is_cancelled_callback():
                log_event('info', 'remote_job_cancelling', jobId=job_id, workerJobId=worker_job_id)
                try:
                    client.cancel_job(worker_job_id)
                except Exception as e:
                    log_event('warning', 'remote_cancel_failed', jobId=job_id, error=str(e))
                return False
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                log_event('error', 'remote_job_timeout', jobId=job_id, workerJobId=worker_job_id)
                update_manifest_callback(
                    status='failed',
                    error={'code': 'REMOTE_TIMEOUT', 'message': f'Worker job timed out after {timeout}s'}
                )
                return False
            
            # Poll worker status
            try:
                status = client.get_job_status(worker_job_id)
                
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
                        'lastSeenAt': datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # Check for completion
                worker_status = status.get('status')
                if worker_status == 'complete':
                    log_event('info', 'remote_job_complete', jobId=job_id, workerJobId=worker_job_id)
                    return True
                elif worker_status == 'failed':
                    error = status.get('error', {})
                    log_event('error', 'remote_job_failed', jobId=job_id, 
                              workerJobId=worker_job_id, error=error)
                    update_manifest_callback(
                        status='failed',
                        error=error or {'code': 'REMOTE_FAILED', 'message': 'Worker job failed'}
                    )
                    return False
                elif worker_status == 'canceled':
                    log_event('info', 'remote_job_canceled', jobId=job_id, workerJobId=worker_job_id)
                    return False
                
            except requests.exceptions.RequestException as e:
                log_event('warning', 'remote_poll_error', jobId=job_id, error=str(e))
                # Continue polling - transient network errors are expected
            
            time.sleep(poll_interval)
        
    except Exception as e:
        log_event('error', 'remote_dispatch_failed', jobId=job_id, error=str(e))
        update_manifest_callback(
            status='failed',
            error={'code': 'REMOTE_DISPATCH_FAILED', 'message': str(e)[:200]}
        )
        return False
