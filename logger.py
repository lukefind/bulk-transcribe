"""
Structured logging helper for bulk-transcribe.

Provides consistent JSON logging format with key fields for observability.
"""

import hashlib
import json
import os
import time
import logging
import warnings
from typing import Dict, Any, Optional

# Configure JSON logger
logger = logging.getLogger('bulk-transcribe')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def hash_id(value: str, n: int = 8) -> str:
    """
    Generate a short hash of an identifier for logging.
    
    Args:
        value: The value to hash (e.g., session ID)
        n: Number of characters to return (default 8)
    
    Returns:
        First n characters of SHA256 hex digest
    """
    if not value:
        return ''
    return hashlib.sha256(value.encode()).hexdigest()[:n]


def configure_runtime_noise() -> None:
    """
    Suppress noisy third-party warnings in production logs.
    
    Call once at process startup. Controlled by SUPPRESS_THIRD_PARTY_WARNINGS env.
    """
    if os.environ.get('SUPPRESS_THIRD_PARTY_WARNINGS', '1') == '0':
        return
    
    # Suppress known noisy modules
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"speechbrain\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"whisper\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"pyannote\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"speechbrain\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\..*")


def _sanitize_fields(fields: dict) -> dict:
    """
    Sanitize log fields to remove sensitive data.
    
    - Replaces sessionId/session_id with sessionHash
    - Never logs HF_TOKEN or other secrets
    """
    sanitized = {}
    
    for key, value in fields.items():
        # Convert session IDs to hashes
        if key in ('sessionId', 'session_id'):
            if value:
                sanitized['sessionHash'] = hash_id(str(value), 8)
            continue
        
        # Skip any token fields
        if 'token' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
            continue
        
        sanitized[key] = value
    
    return sanitized


def log_event(level: str, event: str, **fields) -> None:
    """
    Log a structured event with consistent fields.
    
    Args:
        level: Log level ('info', 'warn', 'error', 'debug')
        event: Event name (e.g., 'job_started', 'diarization_finished')
        **fields: Additional fields to include in the log
        
    Note:
        sessionId/session_id fields are automatically converted to sessionHash
    """
    log_entry = {
        'timestamp': time.time(),
        'event': event,
        'level': level.lower()
    }
    
    # Sanitize and add provided fields
    sanitized = _sanitize_fields(fields)
    log_entry.update(sanitized)
    
    # Log as JSON
    log_line = json.dumps(log_entry, separators=(',', ':'))
    
    if level.lower() == 'error':
        logger.error(log_line)
    elif level.lower() == 'warn':
        logger.warning(log_line)
    elif level.lower() == 'debug':
        logger.debug(log_line)
    else:
        logger.info(log_line)

def log_duration(event: str, duration_ms: int, **fields) -> None:
    """
    Log an event with duration.
    
    Args:
        event: Event name
        duration_ms: Duration in milliseconds
        **fields: Additional fields
    """
    log_event('info', event, durationMs=duration_ms, **fields)

def with_timer(event: str, **fields):
    """
    Context manager to log duration of a block.
    
    Usage:
        with with_timer('diarization_run', jobId='123', file='audio.wav'):
            # do work
    """
    class Timer:
        def __init__(self, event, **fields):
            self.event = event
            self.fields = fields
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            log_event('info', f'{self.event}_started', **self.fields)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration_ms = int((time.time() - self.start_time) * 1000)
                if exc_type:
                    log_event('error', f'{self.event}_failed', 
                             durationMs=duration_ms, 
                             error=str(exc_val),
                             **self.fields)
                else:
                    log_event('info', f'{self.event}_finished', 
                             durationMs=duration_ms, 
                             **self.fields)
    
    return Timer(event, **fields)
