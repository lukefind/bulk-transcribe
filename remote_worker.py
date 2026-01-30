"""
Remote Worker Client for Bulk Transcribe Controller.

This module handles communication with remote GPU workers for offloading
transcription and diarization workloads.

Configuration (environment variables):
- REMOTE_WORKER_URL: Base URL of the worker (e.g., https://gpu-worker.example.com)
- REMOTE_WORKER_TOKEN or WORKER_TOKEN: Shared secret for authentication
  (REMOTE_WORKER_TOKEN takes precedence if both are set)
- REMOTE_WORKER_MODE: off|optional|required (default: off)
- REMOTE_WORKER_TIMEOUT_SECONDS: Job timeout (default: 7200)
- REMOTE_WORKER_POLL_SECONDS: Poll interval (default: 2)
- REMOTE_WORKER_UPLOAD_MODE: pull|push (default: pull)
- EXPECTED_WORKER_GIT_COMMIT: Expected git commit for identity mismatch detection
- EXPECTED_WORKER_IMAGE_DIGEST: Expected image digest for identity mismatch detection
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


# =============================================================================
# Model Mapping (UI labels → actual Whisper models)
# =============================================================================

# Models that workers actually support (from worker /v1/ping)
SUPPORTED_WORKER_MODELS = [
    'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
    'medium', 'medium.en', 'large', 'large-v1', 'large-v2', 'large-v3'
]

def validate_model_for_worker(model: str) -> None:
    """
    Validate that model is supported by the worker.
    No aliasing or mapping - model must be exact.
    
    Args:
        model: Model name (must be exact, e.g., 'large-v3', not 'turbo')
    
    Raises:
        ValueError: If model is not supported
    """
    if model not in SUPPORTED_WORKER_MODELS:
        raise ValueError(
            f"Model '{model}' is not supported by worker. "
            f"Supported models: {', '.join(SUPPORTED_WORKER_MODELS)}"
        )


def _classify_poll_http_error(status_code: int) -> Optional[Dict[str, str]]:
    """Classify worker poll HTTP errors into actionable controller failure codes."""
    if status_code == 404:
        return {
            'code': 'REMOTE_JOB_NOT_FOUND',
            'message': 'Remote worker restarted or lost the job. Please retry.'
        }
    if status_code in (401, 403):
        return {
            'code': 'REMOTE_AUTH_FAILED',
            'message': 'Remote worker authentication failed; check WORKER_TOKEN/REMOTE_WORKER_TOKEN.'
        }
    return None


def _should_auto_retry_on_404(manifest: dict) -> bool:
    """Check if job should be auto-retried after 404 (worker lost job)."""
    remote = manifest.get('remote', {})
    retry_count = remote.get('retryCount', 0)
    return retry_count < 1


# =============================================================================
# HTTP Request Helpers (must be defined before use)
# =============================================================================

def _get_build_commit() -> str:
    """Get build commit for User-Agent header."""
    return os.environ.get('BUILD_COMMIT', 'unknown')


def _get_default_headers(token: str) -> Dict[str, str]:
    """
    Get default headers for all worker requests.
    
    These headers are required to avoid Cloudflare 403 blocks:
    - User-Agent: Identifies the controller
    - Accept: Signals we expect JSON
    - Authorization: Bearer token for auth
    """
    return {
        'Authorization': f'Bearer {token}',
        'User-Agent': f'bulk-transcribe-controller/{_get_build_commit()}',
        'Accept': 'application/json',
    }


def _format_request_error(e: requests.exceptions.RequestException, 
                          response: requests.Response = None) -> str:
    """
    Format request error for logging (without exposing token).
    
    Includes status code and first 300 chars of body if available.
    """
    if response is not None:
        body_preview = response.text[:300] if response.text else ''
        return f"HTTP {response.status_code}: {body_preview}"
    return str(e)[:300]


# =============================================================================
# Configuration
# =============================================================================

def get_worker_config() -> Dict[str, Any]:
    """
    Get remote worker configuration.
    
    Precedence:
    1. Environment variables (highest priority)
    2. Saved config from config_store (UI-configured)
    3. Defaults (disabled)
    
    Token precedence within env: REMOTE_WORKER_TOKEN > WORKER_TOKEN
    """
    # Check environment variables first
    env_url = os.environ.get('REMOTE_WORKER_URL', '')
    env_token = os.environ.get('REMOTE_WORKER_TOKEN') or os.environ.get('WORKER_TOKEN', '')
    env_mode = os.environ.get('REMOTE_WORKER_MODE', '')
    
    # If env vars are set, use them (original behavior)
    if env_url and env_token:
        return {
            'url': env_url,
            'token': env_token,
            'mode': env_mode or 'off',
            'timeoutSeconds': int(os.environ.get('REMOTE_WORKER_TIMEOUT_SECONDS', '7200')),
            'pollSeconds': int(os.environ.get('REMOTE_WORKER_POLL_SECONDS', '2')),
            'uploadMode': os.environ.get('REMOTE_WORKER_UPLOAD_MODE', 'pull'),
            'expectedGitCommit': os.environ.get('EXPECTED_WORKER_GIT_COMMIT', ''),
            'expectedImageDigest': os.environ.get('EXPECTED_WORKER_IMAGE_DIGEST', ''),
            'configSource': 'env',
        }
    
    # Fall back to saved config (UI-configured)
    try:
        from config_store import get_remote_worker_url, get_remote_worker_token, get_remote_worker_mode
        saved_url = get_remote_worker_url()
        saved_token = get_remote_worker_token()
        saved_mode = get_remote_worker_mode()
        
        if saved_url and saved_token:
            return {
                'url': saved_url,
                'token': saved_token,
                'mode': saved_mode or 'off',
                'timeoutSeconds': int(os.environ.get('REMOTE_WORKER_TIMEOUT_SECONDS', '7200')),
                'pollSeconds': int(os.environ.get('REMOTE_WORKER_POLL_SECONDS', '2')),
                'uploadMode': os.environ.get('REMOTE_WORKER_UPLOAD_MODE', 'pull'),
                'expectedGitCommit': os.environ.get('EXPECTED_WORKER_GIT_COMMIT', ''),
                'expectedImageDigest': os.environ.get('EXPECTED_WORKER_IMAGE_DIGEST', ''),
                'configSource': 'saved',
            }
    except ImportError:
        pass  # config_store not available (e.g., in tests)
    
    # Default: disabled
    return {
        'url': env_url,  # May be partially set
        'token': env_token,
        'mode': env_mode or 'off',
        'timeoutSeconds': int(os.environ.get('REMOTE_WORKER_TIMEOUT_SECONDS', '7200')),
        'pollSeconds': int(os.environ.get('REMOTE_WORKER_POLL_SECONDS', '2')),
        'uploadMode': os.environ.get('REMOTE_WORKER_UPLOAD_MODE', 'pull'),
        'expectedGitCommit': os.environ.get('EXPECTED_WORKER_GIT_COMMIT', ''),
        'expectedImageDigest': os.environ.get('EXPECTED_WORKER_IMAGE_DIGEST', ''),
        'configSource': 'default',
    }


def is_remote_worker_available() -> bool:
    """Check if remote worker is configured and reachable."""
    config = get_worker_config()
    if not config['url'] or not config['token']:
        return False
    
    try:
        # Use proper headers to avoid Cloudflare 403
        headers = _get_default_headers(config['token'])
        response = requests.get(
            urljoin(config['url'], '/v1/ping'),  # Use /v1/ping instead of /health
            headers=headers,
            timeout=(5, 10)
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
            result['error'] = 'Remote worker not configured (missing REMOTE_WORKER_URL or WORKER_TOKEN)'
        _worker_status_cache[cache_key] = result
        _worker_status_cache_time[cache_key] = time.time()
        return result
    
    # Ping with auth token - short timeout to avoid blocking UI
    # Use proper headers to avoid Cloudflare 403
    start_time = time.time()
    headers = _get_default_headers(config['token'])
    
    try:
        response = requests.get(
            urljoin(config['url'], '/v1/ping'),
            headers=headers,
            timeout=(5, 10)  # (connect, read) timeouts
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
        # Set all required headers to avoid Cloudflare 403
        self.session.headers.update(_get_default_headers(token))
    
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make authenticated request to worker."""
        url = f"{self.url}{path}"
        # Set default timeouts (connect, read) - increased read timeout for stability
        kwargs.setdefault('timeout', (10, 60))
        
        try:
            response = self.session.request(method, url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            # Log error without token
            log_event('warning', 'remote_worker_request_failed',
                      method=method, path=path,
                      error=_format_request_error(e, getattr(e, 'response', None)))
            raise
    
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
    
    def get_job_status(self, worker_job_id: str) -> requests.Response:
        """Get job status from worker (response is inspected by caller)."""
        return self._request('GET', f'/v1/jobs/{worker_job_id}')
    
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
    consecutive_unreachable = 0
    MAX_CONSECUTIVE_UNREACHABLE = 6  # Fail after 6 consecutive unreachable checks
    
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
            consecutive_unreachable += 1
            backoff = calculate_backoff(capacity_check_attempt)
            
            # Log degraded state when threshold exceeded (but don't fail)
            if consecutive_unreachable == MAX_CONSECUTIVE_UNREACHABLE:
                log_event('warning', 'remote_worker_degraded', 
                          jobId=job_id, consecutiveFailures=consecutive_unreachable,
                          message='Worker unreachable - entering degraded state, will keep retrying')
            else:
                log_event('warning', 'remote_capacity_check_failed',
                          jobId=job_id, error=capacity['error'], backoffSeconds=backoff,
                          consecutiveFailures=consecutive_unreachable)
            
            # Mark as degraded after threshold, but keep retrying (don't fail)
            is_degraded = consecutive_unreachable >= MAX_CONSECUTIVE_UNREACHABLE
            
            update_manifest_callback(
                status='queued_remote',
                progress={
                    'stage': 'queued_remote', 
                    'currentFile': f'Worker unreachable, retrying in {int(backoff)}s...'
                },
                remoteStatus={
                    'degraded': is_degraded,
                    'reason': 'worker_unreachable' if is_degraded else None,
                    'consecutiveErrors': consecutive_unreachable
                },
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
        
        # Reset consecutive failure counter and degraded status on successful check
        if consecutive_unreachable >= MAX_CONSECUTIVE_UNREACHABLE:
            log_event('info', 'remote_worker_recovered', 
                      jobId=job_id, previousConsecutiveFailures=consecutive_unreachable,
                      message='Worker contact re-established after degraded state')
        consecutive_unreachable = 0
        
        # Clear degraded status
        update_manifest_callback(
            remoteStatus={
                'degraded': False,
                'reason': None,
                'consecutiveErrors': 0
            }
        )
        
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
    
    # Validate model is supported (no aliasing - controller should have validated)
    model = options.get('model', 'large-v3')
    try:
        validate_model_for_worker(model)
    except ValueError as e:
        log_event('error', 'remote_model_unsupported',
                  jobId=job_id, model=model, error=str(e))
        update_manifest_callback(
            status='failed',
            finishedAt=datetime.now(timezone.utc).isoformat(),
            error={'code': 'UNSUPPORTED_MODEL', 'message': str(e)}
        )
        return False
    
    # Create worker options (exact model, no mapping)
    worker_options = dict(options)
    worker_options['model'] = model
    
    # Optimize diarization chunking for GPU (larger chunks = less overhead)
    # GPU workers can handle larger chunks efficiently
    if worker_options.get('diarizationEnabled'):
        effective = worker_options.get('diarizationEffective', {})
        original_chunk = effective.get('chunkSeconds', 180)
        original_overlap = effective.get('overlapSeconds', 5)
        
        # GPU-optimized defaults: 300s chunks, 8s overlap (if not user-specified)
        # These values reduce per-chunk overhead while preserving accuracy
        if effective.get('derived', True):  # Only override if not user-specified
            gpu_chunk = 300  # 5 minutes per chunk
            gpu_overlap = 8  # 8 seconds overlap for accuracy
            
            effective['chunkSeconds'] = gpu_chunk
            effective['overlapSeconds'] = gpu_overlap
            effective['gpuOptimized'] = True
            worker_options['diarizationEffective'] = effective
            
            log_event('info', 'remote_diarization_gpu_optimized',
                      jobId=job_id,
                      originalChunk=original_chunk, gpuChunk=gpu_chunk,
                      originalOverlap=original_overlap, gpuOverlap=gpu_overlap)
    
    log_event('info', 'remote_dispatch_started',
              jobId=job_id, sessionHash=session_hash,
              workerUrl=config['url'][:50],
              model=model)
    
    try:
        # Create job on worker
        result = client.create_job(
            controller_job_id=job_id,
            controller_session_hash=session_hash,
            inputs=remote_inputs,
            options=worker_options,
            callback_url=callback_url,
            outputs_upload_url=outputs_upload_url
        )
        
        worker_job_id = result.get('workerJobId')
        if not worker_job_id:
            raise ValueError('Worker did not return workerJobId')
        
        # Update manifest with complete remote job metadata
        # This is critical for:
        # 1. Linking controller job to worker job
        # 2. Worker knowing where to push outputs
        # 3. Debugging and audit trail
        now = datetime.now(timezone.utc).isoformat()
        update_manifest_callback(
            worker={
                'workerJobId': worker_job_id,
                'url': config['url'],
                'createdAt': now,
                'lastSeenAt': now
            },
            remote={
                'workerJobId': worker_job_id,
                'workerUrl': config['url'],
                'uploadMode': config['uploadMode'],
                'outputsUploadUrl': outputs_upload_url,
                'completeUrl': callback_url,
                'startedAt': now,
                'controllerBaseUrl': controller_base_url
            },
            executionMode='remote'
        )
        
        # Log complete dispatch details for debugging
        # This is the single source of truth for what was sent to worker
        log_event('info', 'remote_job_created',
                  jobId=job_id, 
                  workerJobId=worker_job_id,
                  model=model,
                  uploadMode=config['uploadMode'],
                  outputsUploadUrl=outputs_upload_url,
                  completeUrl=callback_url,
                  diarizationEnabled=worker_options.get('diarizationEnabled', False),
                  gpuOptimizedChunking=worker_options.get('diarizationEffective', {}).get('gpuOptimized', False),
                  chunkSeconds=worker_options.get('diarizationEffective', {}).get('chunkSeconds'),
                  overlapSeconds=worker_options.get('diarizationEffective', {}).get('overlapSeconds'))
        
        # Poll for completion with exponential backoff on errors
        poll_interval = config['pollSeconds']
        timeout = config['timeoutSeconds']
        start_time = time.time()
        consecutive_errors = 0
        consecutive_404s = 0
        MAX_CONSECUTIVE_404S = 3  # Require 3 consecutive 404s before declaring job lost
        
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
                log_event('error', 'remote_job_timeout', jobId=job_id, workerJobId=worker_job_id, 
                          elapsedSeconds=int(elapsed), timeoutSeconds=timeout)
                # Note: Worker may still be running and uploading outputs.
                # Use 'complete_with_errors' to indicate partial results may exist.
                # The worker will continue uploading outputs even after this timeout.
                update_manifest_callback(
                    status='complete_with_errors',
                    finishedAt=datetime.now(timezone.utc).isoformat(),
                    error={'code': 'REMOTE_WORKER_TIMEOUT', 
                           'message': f'Controller timed out after {timeout}s. Worker may still be processing - check for partial results.'}
                )
                return False
            
            # Poll worker status
            try:
                response = client.get_job_status(worker_job_id)

                if response.status_code != 200:
                    body_preview = (response.text or '')[:300]
                    classified = _classify_poll_http_error(response.status_code)
                    now_iso = datetime.now(timezone.utc).isoformat()

                    if classified is not None:
                        # Special handling for 404 - require multiple consecutive 404s
                        # A single 404 could be a transient issue (proxy hiccup, etc)
                        if response.status_code == 404:
                            consecutive_404s += 1
                            
                            if consecutive_404s < MAX_CONSECUTIVE_404S:
                                # Not enough consecutive 404s yet - treat as transient
                                log_event('warning', 'remote_job_404_transient',
                                          jobId=job_id,
                                          workerJobId=worker_job_id,
                                          consecutive404s=consecutive_404s,
                                          maxRequired=MAX_CONSECUTIVE_404S,
                                          message='Got 404, waiting for more before declaring lost')
                                
                                # Update manifest to show we're having issues but not failed yet
                                update_manifest_callback(
                                    progress={
                                        'currentFile': f'Worker returned 404 ({consecutive_404s}/{MAX_CONSECUTIVE_404S}), retrying...',
                                    },
                                    remote={
                                        'workerJobId': worker_job_id,
                                        'lastError': {
                                            'code': 'TRANSIENT_404',
                                            'message': f'Worker returned 404 ({consecutive_404s}/{MAX_CONSECUTIVE_404S})',
                                            'at': now_iso,
                                        },
                                    }
                                )
                                
                                # Wait and retry
                                time.sleep(poll_interval * 2)  # Double wait on 404
                                continue
                            
                            # Multiple consecutive 404s - job is truly lost
                            log_event('error', 'remote_job_lost',
                                      jobId=job_id,
                                      workerJobId=worker_job_id,
                                      consecutive404s=consecutive_404s,
                                      message='Worker lost the job after multiple 404s')
                            
                            # Check if we should auto-retry
                            current_retry_count = 0
                            # Signal auto-retry by returning special value
                            # The caller will check manifest and re-dispatch
                            update_manifest_callback(
                                status='failed',
                                finishedAt=now_iso,
                                error={'code': classified['code'], 'message': classified['message']},
                                remote={
                                    'workerJobId': worker_job_id,
                                    'retryCount': current_retry_count,
                                    'lastError': {
                                        'code': classified['code'],
                                        'message': classified['message'],
                                        'statusCode': response.status_code,
                                        'at': now_iso,
                                    },
                                    'shouldAutoRetry': True,
                                }
                            )
                            return 'retry'  # Signal to caller to retry
                        
                        # Other fatal errors (auth, etc)
                        log_event('error', 'remote_poll_fatal_http',
                                  jobId=job_id,
                                  workerJobId=worker_job_id,
                                  statusCode=response.status_code,
                                  bodyPreview=body_preview)

                        update_manifest_callback(
                            status='failed',
                            finishedAt=now_iso,
                            error={'code': classified['code'], 'message': classified['message']},
                            remote={
                                'workerJobId': worker_job_id,
                                'lastError': {
                                    'code': classified['code'],
                                    'message': classified['message'],
                                    'statusCode': response.status_code,
                                    'bodyPreview': body_preview,
                                    'at': now_iso,
                                },
                            }
                        )
                        return False

                    # Non-fatal HTTP (5xx/other): treat like transient network error
                    response.raise_for_status()

                status = response.json()
                consecutive_errors = 0  # Reset on success
                consecutive_404s = 0  # Reset 404 counter on success
                
                # Update manifest with progress
                worker_progress = status.get('progress', {})
                progress_update = {
                    'stage': status.get('stage', 'unknown'),
                    'currentFileIndex': worker_progress.get('currentFileIndex', 0),
                    'totalFiles': worker_progress.get('totalFiles', len(inputs)),
                    'chunkIndex': worker_progress.get('chunkIndex', 0),
                    'totalChunks': worker_progress.get('totalChunks', 0),
                    'percent': worker_progress.get('percent', 0),
                    'remoteStatus': status.get('status')
                }
                current_file = worker_progress.get('currentFile')
                if current_file:
                    progress_update['currentFile'] = current_file

                update_manifest_callback(
                    progress=progress_update,
                    remoteStatus={
                        'degraded': False,
                        'reason': None,
                        'consecutiveErrors': 0,
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
                    # Worker reports complete - outputs should have been pushed
                    # We mark as complete but also record remote completion info
                    # The actual outputs verification happens via the callback endpoint
                    finished_at = datetime.now(timezone.utc).isoformat()
                    
                    # Get outputs info from worker status if available
                    worker_outputs = status.get('outputs', [])
                    
                    log_event('info', 'remote_job_complete', 
                              jobId=job_id, workerJobId=worker_job_id,
                              workerOutputCount=len(worker_outputs))
                    
                    update_manifest_callback(
                        status='complete',
                        finishedAt=finished_at,
                        remote={
                            'workerJobId': worker_job_id,
                            'completedAt': finished_at,
                            'workerReportedOutputs': len(worker_outputs)
                        }
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
                
                # Update manifest with error info and degraded status
                now_iso = datetime.now(timezone.utc).isoformat()
                is_degraded = consecutive_errors >= 3
                update_manifest_callback(
                    lastErrorCode='REMOTE_WORKER_UNREACHABLE',
                    lastErrorMessage=str(e)[:200],
                    lastErrorAt=now_iso,
                    remoteStatus={
                        'degraded': is_degraded,
                        'reason': 'poll_timeout' if is_degraded else None,
                        'consecutiveErrors': consecutive_errors,
                    },
                    remote={
                        'workerJobId': worker_job_id,
                        'lastError': {
                            'code': 'REMOTE_WORKER_UNREACHABLE',
                            'message': str(e)[:200],
                            'at': now_iso,
                        },
                    }
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
