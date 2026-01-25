"""
Server-side configuration store for Bulk Transcribe.

Stores configuration in JSON files under DATA_ROOT/config/.
Used for settings that can be changed at runtime without restarting.

Security notes:
- Token values are stored but NEVER returned via API
- Config files should have restricted permissions (0600)
- Never log token values
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from logger import log_event


def _get_config_dir() -> Path:
    """Get the config directory path, creating it if needed."""
    from session_store import data_root
    config_dir = Path(data_root()) / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically (write to temp file, then rename)."""
    # Write to temp file in same directory (ensures same filesystem for rename)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        # Atomic rename
        os.replace(temp_path, path)
        # Restrict permissions (owner read/write only)
        os.chmod(path, 0o600)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log_event('warning', 'config_read_error', path=str(path), error=str(e))
        return None


# =============================================================================
# Remote Worker Config
# =============================================================================

_REMOTE_WORKER_CONFIG_FILE = 'remote_worker.json'


def get_remote_worker_config() -> Dict[str, Any]:
    """
    Get saved remote worker configuration.
    
    Returns:
        {
            'url': str,
            'mode': str ('off' | 'optional' | 'required'),
            'tokenSet': bool,  # True if token is stored (never return actual token)
            'updatedAt': str | None
        }
    
    Note: Token is stored but NEVER returned by this function.
    """
    config_path = _get_config_dir() / _REMOTE_WORKER_CONFIG_FILE
    data = _read_json(config_path)
    
    if not data:
        return {
            'url': '',
            'mode': 'off',
            'tokenSet': False,
            'updatedAt': None
        }
    
    return {
        'url': data.get('url', ''),
        'mode': data.get('mode', 'off'),
        'tokenSet': bool(data.get('token')),
        'updatedAt': data.get('updatedAt')
    }


def get_remote_worker_token() -> str:
    """
    Get the stored remote worker token.
    
    This is a separate function to make it clear this is sensitive data.
    Should only be called internally when making requests to the worker.
    
    Returns:
        The stored token, or empty string if not set.
    """
    config_path = _get_config_dir() / _REMOTE_WORKER_CONFIG_FILE
    data = _read_json(config_path)
    
    if not data:
        return ''
    
    return data.get('token', '')


def get_remote_worker_url() -> str:
    """Get the stored remote worker URL."""
    config_path = _get_config_dir() / _REMOTE_WORKER_CONFIG_FILE
    data = _read_json(config_path)
    
    if not data:
        return ''
    
    return data.get('url', '')


def get_remote_worker_mode() -> str:
    """Get the stored remote worker mode."""
    config_path = _get_config_dir() / _REMOTE_WORKER_CONFIG_FILE
    data = _read_json(config_path)
    
    if not data:
        return 'off'
    
    return data.get('mode', 'off')


def save_remote_worker_config(
    url: Optional[str] = None,
    token: Optional[str] = None,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save remote worker configuration.
    
    Only updates fields that are provided (not None).
    Token can be explicitly cleared by passing empty string.
    
    Args:
        url: Worker URL (e.g., https://xxx-8477.proxy.runpod.net)
        token: Worker token (pass None to keep existing, '' to clear)
        mode: 'off' | 'optional' | 'required'
    
    Returns:
        The saved config (without token, same as get_remote_worker_config)
    """
    config_path = _get_config_dir() / _REMOTE_WORKER_CONFIG_FILE
    
    # Load existing config
    existing = _read_json(config_path) or {}
    
    # Update only provided fields
    if url is not None:
        existing['url'] = url
    if token is not None:
        existing['token'] = token
    if mode is not None:
        if mode not in ('off', 'optional', 'required'):
            raise ValueError(f"Invalid mode: {mode}")
        existing['mode'] = mode
    
    existing['updatedAt'] = datetime.now(timezone.utc).isoformat()
    
    # Save atomically
    _atomic_write_json(config_path, existing)
    
    # Log (without token)
    log_event('info', 'remote_worker_config_saved',
              url=existing.get('url', ''),
              mode=existing.get('mode', 'off'),
              tokenSet=bool(existing.get('token')))
    
    # Return safe version (no token)
    return get_remote_worker_config()


def clear_remote_worker_config() -> None:
    """Clear all remote worker configuration."""
    config_path = _get_config_dir() / _REMOTE_WORKER_CONFIG_FILE
    if config_path.exists():
        config_path.unlink()
        log_event('info', 'remote_worker_config_cleared')
