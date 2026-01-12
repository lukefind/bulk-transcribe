"""
Structured logging helper for bulk-transcribe.

Provides consistent JSON logging format with key fields for observability.
"""

import json
import time
import logging
from typing import Dict, Any, Optional

# Configure JSON logger
logger = logging.getLogger('bulk-transcribe')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def log_event(level: str, event: str, **fields) -> None:
    """
    Log a structured event with consistent fields.
    
    Args:
        level: Log level ('info', 'warn', 'error', 'debug')
        event: Event name (e.g., 'job_started', 'diarization_finished')
        **fields: Additional fields to include in the log
    """
    log_entry = {
        'timestamp': time.time(),
        'event': event,
        'level': level.lower()
    }
    
    # Add provided fields
    log_entry.update(fields)
    
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
