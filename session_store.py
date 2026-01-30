"""
Session and path utilities for multi-user server isolation.

This module centralizes all session management, path resolution, and file
operations to ensure users cannot access each other's data in server mode.
"""

import os
import re
import json
import secrets
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any


# =============================================================================
# App Mode Configuration
# =============================================================================

def get_app_mode() -> str:
    """Get the application mode from environment. Default: 'server'."""
    return os.environ.get('APP_MODE', 'server').lower()


def is_server_mode() -> bool:
    """Check if running in server (multi-user) mode."""
    return get_app_mode() == 'server'


def is_local_mode() -> bool:
    """Check if running in local (single-user) mode."""
    return get_app_mode() == 'local'


def data_root() -> str:
    """Get the data root directory from environment. Default: '/data'."""
    return os.environ.get('DATA_ROOT', '/data')


# =============================================================================
# ID Generation
# =============================================================================

def new_id(nbytes: int = 16) -> str:
    """Generate a URL-safe random token."""
    return secrets.token_urlsafe(nbytes)


# =============================================================================
# Session Management
# =============================================================================

COOKIE_NAME = 'bt_session'


def get_or_create_session_id(request, response=None) -> str:
    """
    Get existing session ID from cookie or create a new one.
    
    Args:
        request: Flask request object
        response: Flask response object (optional, for setting cookie)
    
    Returns:
        Session ID string
    """
    session_id = request.cookies.get(COOKIE_NAME)
    
    if not session_id:
        session_id = new_id(32)
        if response is not None:
            _set_session_cookie(request, response, session_id)
    
    return session_id


def _set_session_cookie(request, response, session_id: str) -> None:
    """Set the session cookie on the response."""
    secure = (
        request.is_secure or 
        os.environ.get('COOKIE_SECURE', '0') == '1'
    )
    response.set_cookie(
        COOKIE_NAME,
        session_id,
        httponly=True,
        samesite='Lax',
        secure=secure,
        max_age=60 * 60 * 24 * 365  # 1 year
    )


def set_new_session_cookie(request, response, session_id: str) -> None:
    """Public method to set session cookie on response."""
    _set_session_cookie(request, response, session_id)


# =============================================================================
# Session Directory Structure
# =============================================================================

def get_sessions_dir() -> str:
    """Get the sessions root directory path."""
    return os.path.join(data_root(), 'sessions')


def session_dir(session_id: str) -> str:
    """Get the session directory path."""
    return os.path.join(data_root(), 'sessions', session_id)


def ensure_session_dirs(session_id: str) -> Dict[str, str]:
    """
    Create session directories if they don't exist.
    
    Returns:
        Dict with 'uploads' and 'jobs' paths
    """
    base = session_dir(session_id)
    uploads_dir = os.path.join(base, 'uploads')
    jobs_dir = os.path.join(base, 'jobs')
    
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)
    
    return {
        'base': base,
        'uploads': uploads_dir,
        'jobs': jobs_dir
    }


def session_meta_path(session_id: str) -> str:
    """Get path to session metadata file."""
    return os.path.join(session_dir(session_id), 'session.json')


def session_preferences_path(session_id: str) -> str:
    """Get path to session preferences file."""
    return os.path.join(session_dir(session_id), 'preferences.json')


def touch_session(session_id: str) -> None:
    """Create or update session metadata with timestamps."""
    ensure_session_dirs(session_id)
    meta_path = session_meta_path(session_id)
    
    now = datetime.now(timezone.utc).isoformat()
    
    if os.path.exists(meta_path):
        meta = read_json(meta_path) or {}
        meta['lastSeenAt'] = now
    else:
        # New session: generate CSRF token
        meta = {
            'createdAt': now,
            'lastSeenAt': now,
            'csrfToken': new_id(24)
        }
    
    atomic_write_json(meta_path, meta)


def get_csrf_token(session_id: str) -> Optional[str]:
    """Get CSRF token for a session, creating one if needed."""
    meta_path = session_meta_path(session_id)
    meta = read_json(meta_path)
    
    if not meta:
        return None
    
    if 'csrfToken' not in meta:
        # Generate and save CSRF token for existing session
        meta['csrfToken'] = new_id(24)
        atomic_write_json(meta_path, meta)
    
    return meta.get('csrfToken')


def validate_csrf_token(session_id: str, token: str) -> bool:
    """Validate CSRF token matches session."""
    if not token:
        return False
    expected = get_csrf_token(session_id)
    if not expected:
        return False
    return secrets.compare_digest(token, expected)


# =============================================================================
# Path Safety
# =============================================================================

def is_safe_path(base_dir: str, target_path: str) -> bool:
    """
    Check if target_path is safely within base_dir.
    Prevents path traversal attacks.
    """
    try:
        base = os.path.realpath(base_dir)
        target = os.path.realpath(target_path)
        return target.startswith(base + os.sep) or target == base
    except (ValueError, OSError):
        return False


# =============================================================================
# Job Directory Structure
# =============================================================================

def job_dir(session_id: str, job_id: str) -> str:
    """Get the job directory path."""
    return os.path.join(session_dir(session_id), 'jobs', job_id)


def job_outputs_dir(session_id: str, job_id: str) -> str:
    """Get the job outputs directory path."""
    return os.path.join(job_dir(session_id, job_id), 'outputs')


def job_manifest_path(session_id: str, job_id: str) -> str:
    """Get path to job manifest file."""
    return os.path.join(job_dir(session_id, job_id), 'job.json')


def ensure_job_dirs(session_id: str, job_id: str) -> Dict[str, str]:
    """
    Create job directories if they don't exist.
    
    Returns:
        Dict with 'base' and 'outputs' paths
    """
    base = job_dir(session_id, job_id)
    outputs = job_outputs_dir(session_id, job_id)
    
    os.makedirs(outputs, exist_ok=True)
    
    return {
        'base': base,
        'outputs': outputs
    }


REVIEWABLE_OUTPUT_TYPES = {
    'speaker-markdown', 'markdown', 'diarization-json', 'json', 'vtt', 'srt'
}


def list_jobs(session_id: str, limit: int = 20) -> list:
    """
    List jobs for a session, most recent first.
    
    Returns:
        List of job summary dicts with reviewable flag
    """
    jobs_path = os.path.join(session_dir(session_id), 'jobs')
    if not os.path.exists(jobs_path):
        return []
    
    jobs = []
    for job_id in os.listdir(jobs_path):
        manifest_path = job_manifest_path(session_id, job_id)
        manifest = read_json(manifest_path)
        if manifest:
            outputs = manifest.get('outputs', [])
            valid_outputs = [o for o in outputs if not o.get('error')]
            inputs = manifest.get('inputs', [])
            options = manifest.get('options', {})

            # Job is reviewable if it has any transcript-like output
            reviewable = any(
                o.get('type') in REVIEWABLE_OUTPUT_TYPES 
                for o in valid_outputs
            )

            primary_input = inputs[0] if inputs else {}
            primary_filename = (
                primary_input.get('originalFilename')
                or primary_input.get('filename')
                or primary_input.get('storedName')
                or ''
            )

            def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
                if not ts:
                    return None
                try:
                    # Handle trailing "Z" from ISO timestamps
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except ValueError:
                    return None

            started_at = _parse_iso(manifest.get('startedAt'))
            finished_at = _parse_iso(manifest.get('finishedAt'))
            elapsed_seconds: Optional[float] = None
            if started_at and finished_at:
                elapsed_seconds = max(0.0, (finished_at - started_at).total_seconds())
            
            jobs.append({
                'jobId': manifest.get('jobId', job_id),
                'createdAt': manifest.get('createdAt'),
                'startedAt': manifest.get('startedAt'),
                'finishedAt': manifest.get('finishedAt'),
                'status': manifest.get('status'),
                'inputCount': len(manifest.get('inputs', [])),
                'outputCount': len(valid_outputs),
                'reviewable': reviewable,
                'primaryFilename': primary_filename,
                'primaryDurationSec': primary_input.get('durationSec'),
                'elapsedSeconds': elapsed_seconds,
                'model': options.get('model'),
                'backend': options.get('backend'),
                'executionMode': manifest.get('executionMode', 'local'),
                'diarizationEnabled': bool(options.get('diarizationEnabled')),
                'finished': manifest.get('finished', False),
                'controllerTimedOut': manifest.get('controllerTimedOut', False),
                'controllerTimedOutAt': manifest.get('controllerTimedOutAt'),
            })
    
    # Sort by createdAt descending
    jobs.sort(key=lambda j: j.get('createdAt', ''), reverse=True)
    
    return jobs[:limit]


# =============================================================================
# File Operations
# =============================================================================

def atomic_write_json(path: str, obj: Any) -> None:
    """
    Write JSON atomically using temp file + rename.
    This prevents partial reads during concurrent access.
    """
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_json(path: str) -> Optional[Dict[str, Any]]:
    """Read JSON file, return None if not found or invalid."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# =============================================================================
# Filename Sanitization
# =============================================================================

# Pattern for allowed characters: alphanumeric, dot, dash, underscore
_SAFE_FILENAME_PATTERN = re.compile(r'[^a-zA-Z0-9._-]')


def sanitize_filename(name: str) -> str:
    """
    Sanitize filename to safe characters only.
    Keeps: alphanumeric, dot, dash, underscore
    Replaces others with underscore.
    """
    if not name:
        return 'unnamed'
    
    # Replace unsafe characters
    safe = _SAFE_FILENAME_PATTERN.sub('_', name)
    
    # Collapse multiple underscores
    while '__' in safe:
        safe = safe.replace('__', '_')
    
    # Remove leading/trailing underscores and dots
    safe = safe.strip('_.')
    
    # Ensure we have something
    if not safe:
        return 'unnamed'
    
    return safe


# =============================================================================
# Upload Helpers
# =============================================================================

def uploads_dir(session_id: str) -> str:
    """Get the uploads directory for a session."""
    return os.path.join(session_dir(session_id), 'uploads')


def find_upload_by_id(session_id: str, upload_id: str) -> Optional[str]:
    """
    Find an uploaded file by its ID prefix.
    
    Returns:
        Full path to the file, or None if not found
    """
    uploads = uploads_dir(session_id)
    if not os.path.exists(uploads):
        return None
    
    prefix = f"{upload_id}_"
    for filename in os.listdir(uploads):
        if filename.startswith(prefix):
            return os.path.join(uploads, filename)
    
    return None


def list_uploads(session_id: str) -> list:
    """List all uploads for a session."""
    uploads = uploads_dir(session_id)
    if not os.path.exists(uploads):
        return []
    
    result = []
    for filename in os.listdir(uploads):
        filepath = os.path.join(uploads, filename)
        if os.path.isfile(filepath):
            # Parse upload_id from filename (format: {upload_id}_{original_name})
            parts = filename.split('_', 1)
            upload_id = parts[0] if len(parts) > 1 else filename
            original_name = parts[1] if len(parts) > 1 else filename
            
            result.append({
                'id': upload_id,
                'filename': original_name,
                'storedName': filename,
                'path': filepath,
                'size': os.path.getsize(filepath)
            })
    
    return result


# =============================================================================
# Session Cleanup
# =============================================================================

def get_session_ttl_hours() -> int:
    """Get session TTL in hours from environment."""
    try:
        return int(os.environ.get('SESSION_TTL_HOURS', '24'))
    except ValueError:
        return 24


def get_job_ttl_days() -> int:
    """Get job TTL in days from environment."""
    try:
        return int(os.environ.get('JOB_TTL_DAYS', '7'))
    except ValueError:
        return 7


def get_job_stale_minutes() -> int:
    """Get job stale threshold in minutes from environment."""
    try:
        return int(os.environ.get('JOB_STALE_MINUTES', '30'))
    except ValueError:
        return 30


def get_max_job_runtime_minutes() -> int:
    """Get maximum job runtime in minutes from environment."""
    try:
        return int(os.environ.get('MAX_JOB_RUNTIME_MINUTES', '120'))
    except ValueError:
        return 120


def get_max_session_mb() -> int:
    """Get maximum session storage in MB from environment."""
    try:
        return int(os.environ.get('MAX_TOTAL_SESSION_MB', '2000'))
    except ValueError:
        return 2000


def is_job_stale(manifest: Dict[str, Any]) -> bool:
    """
    Check if a running job is stale (no updates for too long).
    
    Args:
        manifest: Job manifest dict
    
    Returns:
        True if job is running but hasn't updated recently
    """
    if manifest.get('status') != 'running':
        return False
    
    updated_at = manifest.get('updatedAt')
    if not updated_at:
        return True
    
    try:
        from datetime import timedelta
        last_update = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=get_job_stale_minutes())
        return last_update < stale_threshold
    except (ValueError, TypeError):
        return True


def mark_job_stale(session_id: str, job_id: str) -> None:
    """Mark a stale job as failed."""
    manifest_path = job_manifest_path(session_id, job_id)
    manifest = read_json(manifest_path)
    if not manifest:
        return
    
    now = datetime.now(timezone.utc).isoformat()
    manifest['status'] = 'failed'
    manifest['finishedAt'] = now
    manifest['updatedAt'] = now
    manifest['error'] = {
        'code': 'STALE_JOB',
        'message': 'Job stopped responding and was marked as failed'
    }
    atomic_write_json(manifest_path, manifest)


def get_session_disk_usage_mb(session_id: str) -> float:
    """Get total disk usage for a session in MB."""
    session_path = session_dir(session_id)
    if not os.path.exists(session_path):
        return 0.0
    
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(session_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_bytes += os.path.getsize(filepath)
            except OSError:
                pass
    
    return total_bytes / (1024 * 1024)


def cleanup_expired_sessions(exclude_session_id: Optional[str] = None) -> Dict[str, int]:
    """
    Remove expired sessions based on SESSION_TTL_HOURS.
    
    Args:
        exclude_session_id: Session ID to exclude from cleanup (e.g., current request's session)
    
    Returns:
        Dict with 'checked' and 'deleted' counts
    """
    import shutil
    from datetime import timedelta
    
    sessions_root = os.path.join(data_root(), 'sessions')
    if not os.path.exists(sessions_root):
        return {'checked': 0, 'deleted': 0}
    
    ttl_hours = get_session_ttl_hours()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
    
    checked = 0
    deleted = 0
    
    for session_id in os.listdir(sessions_root):
        # Never delete the current session
        if exclude_session_id and session_id == exclude_session_id:
            continue
        
        session_path = os.path.join(sessions_root, session_id)
        if not os.path.isdir(session_path):
            continue
        
        checked += 1
        meta_path = os.path.join(session_path, 'session.json')
        meta = read_json(meta_path)
        
        if meta and 'lastSeenAt' in meta:
            try:
                last_seen = datetime.fromisoformat(meta['lastSeenAt'].replace('Z', '+00:00'))
                if last_seen < cutoff:
                    shutil.rmtree(session_path, ignore_errors=True)
                    deleted += 1
            except (ValueError, TypeError):
                pass
    
    return {'checked': checked, 'deleted': deleted}


# =============================================================================
# Configuration Limits
# =============================================================================

def get_max_upload_mb() -> int:
    """Get maximum upload size in MB from environment."""
    try:
        return int(os.environ.get('MAX_UPLOAD_MB', '500'))
    except ValueError:
        return 500


def get_max_files_per_job() -> int:
    """Get maximum files per job from environment."""
    try:
        return int(os.environ.get('MAX_FILES_PER_JOB', '50'))
    except ValueError:
        return 50


# =============================================================================
# Session Sharing
# =============================================================================

def shared_sessions_path(session_id: str) -> str:
    """Get path to shared sessions file for a session."""
    return os.path.join(session_dir(session_id), 'shared_sessions.json')


def get_shared_sessions(session_id: str) -> list:
    """Get all share tokens for a session."""
    path = shared_sessions_path(session_id)
    data = read_json(path)
    return data.get('shares', []) if data else []


def add_share_token(session_id: str, token: str, mode: str, password_hash: str = None) -> dict:
    """
    Add a share token for a session.
    
    Args:
        session_id: The session to share
        token: The share token
        mode: 'read' or 'edit'
        password_hash: Optional bcrypt hash of password
    
    Returns:
        The created share entry
    """
    path = shared_sessions_path(session_id)
    data = read_json(path) or {'shares': []}
    
    entry = {
        'token': token,
        'mode': mode,
        'passwordHash': password_hash,
        'createdAt': datetime.now(timezone.utc).isoformat(),
        'createdBy': session_id
    }
    
    data['shares'].append(entry)
    atomic_write_json(path, data)
    
    return entry


def revoke_share_token(session_id: str, token: str) -> bool:
    """
    Revoke a share token.
    
    Returns:
        True if token was found and revoked, False otherwise
    """
    path = shared_sessions_path(session_id)
    data = read_json(path)
    if not data or 'shares' not in data:
        return False
    
    original_len = len(data['shares'])
    data['shares'] = [s for s in data['shares'] if s.get('token') != token]
    
    if len(data['shares']) < original_len:
        atomic_write_json(path, data)
        return True
    return False


def find_share_by_token(token: str) -> Optional[dict]:
    """
    Find a share entry by token across all sessions.
    
    Returns:
        Dict with 'sessionId', 'mode', 'passwordHash' if found, None otherwise
    """
    sessions_root = os.path.join(data_root(), 'sessions')
    if not os.path.exists(sessions_root):
        return None
    
    for session_id in os.listdir(sessions_root):
        session_path = os.path.join(sessions_root, session_id)
        if not os.path.isdir(session_path):
            continue
        
        shares_path = os.path.join(session_path, 'shared_sessions.json')
        if not os.path.exists(shares_path):
            continue
        
        data = read_json(shares_path)
        if not data or 'shares' not in data:
            continue
        
        for share in data['shares']:
            if share.get('token') == token:
                return {
                    'sessionId': session_id,
                    'mode': share.get('mode', 'read'),
                    'passwordHash': share.get('passwordHash'),
                    'createdAt': share.get('createdAt')
                }
    
    return None
