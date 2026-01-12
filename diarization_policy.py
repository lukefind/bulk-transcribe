"""
Diarization Policy Module - Single source of truth for diarization configuration.

This module computes effective diarization parameters (max duration, chunk size, overlap)
based on server limits, user preferences, and deterministic scaling rules.

All diarization policy decisions flow through compute_diarization_policy().
"""

import os
from typing import Optional


def get_server_policy_config() -> dict:
    """
    Load server-side diarization policy limits from environment variables.
    
    Returns:
        dict with server policy configuration
    """
    return {
        'serverMaxDurationSeconds': int(os.environ.get('DIARIZATION_SERVER_MAX_DURATION_SECONDS', '1800')),
        'defaultMaxDurationSeconds': int(os.environ.get('DIARIZATION_DEFAULT_MAX_DURATION_SECONDS', '180')),
        'minChunkSeconds': int(os.environ.get('DIARIZATION_POLICY_MIN_CHUNK_SECONDS', '60')),
        'maxChunkSeconds': int(os.environ.get('DIARIZATION_POLICY_MAX_CHUNK_SECONDS', '600')),
        'overlapRatio': float(os.environ.get('DIARIZATION_POLICY_DEFAULT_OVERLAP_RATIO', '0.03')),
        'minOverlapSeconds': int(os.environ.get('DIARIZATION_POLICY_MIN_OVERLAP_SECONDS', '2')),
        'maxOverlapSeconds': int(os.environ.get('DIARIZATION_POLICY_MAX_OVERLAP_SECONDS', '15')),
    }


def _derive_chunk_seconds(max_duration_seconds: int) -> int:
    """
    Deterministic scaling rule for chunk size based on max duration.
    
    This provides predictable behavior as max duration changes.
    
    Args:
        max_duration_seconds: The effective max diarization duration
        
    Returns:
        Recommended chunk size in seconds
    """
    if max_duration_seconds <= 180:
        return 150
    elif max_duration_seconds <= 600:
        return 180
    elif max_duration_seconds <= 1200:
        return 240
    else:
        return 300


def _derive_overlap_seconds(
    chunk_seconds: int,
    overlap_ratio: float,
    min_overlap: int,
    max_overlap: int
) -> int:
    """
    Compute overlap seconds from chunk size and ratio, with clamping.
    
    Args:
        chunk_seconds: The chunk size in seconds
        overlap_ratio: Ratio of overlap to chunk (e.g., 0.03 = 3%)
        min_overlap: Minimum overlap in seconds
        max_overlap: Maximum overlap in seconds
        
    Returns:
        Overlap in seconds, clamped to bounds
    """
    computed = round(chunk_seconds * overlap_ratio)
    return max(min_overlap, min(max_overlap, computed))


def compute_diarization_policy(
    *,
    diarization_enabled: bool,
    diarization_auto_split: bool,
    requested_max_duration_seconds: Optional[int] = None,
    requested_chunk_seconds: Optional[int] = None,
    requested_overlap_seconds: Optional[int] = None,
    server_max_duration_seconds: Optional[int] = None,
    default_max_duration_seconds: Optional[int] = None,
    min_chunk_seconds: Optional[int] = None,
    max_chunk_seconds: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
    min_overlap_seconds: Optional[int] = None,
    max_overlap_seconds: Optional[int] = None,
) -> Optional[dict]:
    """
    Compute effective diarization policy from user preferences and server limits.
    
    This is the single source of truth for all diarization configuration.
    The UI, job creation, and worker all use this function to ensure consistency.
    
    Args:
        diarization_enabled: Whether diarization is enabled for this job
        diarization_auto_split: Whether auto-split is enabled for long files
        requested_max_duration_seconds: User-requested max duration (optional)
        requested_chunk_seconds: User override for chunk size (optional)
        requested_overlap_seconds: User override for overlap (optional)
        server_max_duration_seconds: Server hard cap (from env, optional - will load from env if None)
        default_max_duration_seconds: Default max duration (from env, optional)
        min_chunk_seconds: Minimum chunk size (from env, optional)
        max_chunk_seconds: Maximum chunk size (from env, optional)
        overlap_ratio: Default overlap ratio (from env, optional)
        min_overlap_seconds: Minimum overlap (from env, optional)
        max_overlap_seconds: Maximum overlap (from env, optional)
        
    Returns:
        dict with effective policy, or None if diarization disabled:
        {
            "maxDurationSeconds": int,
            "autoSplit": bool,
            "chunkSeconds": int,
            "overlapSeconds": int,
            "derived": bool,  # true if chunk/overlap computed rather than user override
            "clamped": {
                "maxDurationClamped": bool,
                "maxDurationOriginal": int | None,
                "chunkSecondsClamped": bool,
                "chunkSecondsOriginal": int | None,
                "overlapSecondsClamped": bool,
                "overlapSecondsOriginal": int | None,
            }
        }
    """
    if not diarization_enabled:
        return None
    
    # Load server config for any missing parameters
    server_config = get_server_policy_config()
    
    if server_max_duration_seconds is None:
        server_max_duration_seconds = server_config['serverMaxDurationSeconds']
    if default_max_duration_seconds is None:
        default_max_duration_seconds = server_config['defaultMaxDurationSeconds']
    if min_chunk_seconds is None:
        min_chunk_seconds = server_config['minChunkSeconds']
    if max_chunk_seconds is None:
        max_chunk_seconds = server_config['maxChunkSeconds']
    if overlap_ratio is None:
        overlap_ratio = server_config['overlapRatio']
    if min_overlap_seconds is None:
        min_overlap_seconds = server_config['minOverlapSeconds']
    if max_overlap_seconds is None:
        max_overlap_seconds = server_config['maxOverlapSeconds']
    
    clamped = {
        'maxDurationClamped': False,
        'maxDurationOriginal': None,
        'chunkSecondsClamped': False,
        'chunkSecondsOriginal': None,
        'overlapSecondsClamped': False,
        'overlapSecondsOriginal': None,
    }
    
    # Compute effective max duration
    if requested_max_duration_seconds is not None:
        max_duration = requested_max_duration_seconds
    else:
        max_duration = default_max_duration_seconds
    
    # Clamp max duration to [30, server_max]
    original_max = max_duration
    max_duration = max(30, min(server_max_duration_seconds, max_duration))
    if max_duration != original_max:
        clamped['maxDurationClamped'] = True
        clamped['maxDurationOriginal'] = original_max
    
    # Determine if chunk/overlap are derived or user-specified
    derived = requested_chunk_seconds is None and requested_overlap_seconds is None
    
    # Compute chunk seconds
    if requested_chunk_seconds is not None:
        chunk_seconds = requested_chunk_seconds
    else:
        chunk_seconds = _derive_chunk_seconds(max_duration)
    
    # Clamp chunk seconds
    original_chunk = chunk_seconds
    chunk_seconds = max(min_chunk_seconds, min(max_chunk_seconds, chunk_seconds))
    if chunk_seconds != original_chunk:
        clamped['chunkSecondsClamped'] = True
        clamped['chunkSecondsOriginal'] = original_chunk
    
    # Compute overlap seconds
    if requested_overlap_seconds is not None:
        overlap_seconds = requested_overlap_seconds
    else:
        overlap_seconds = _derive_overlap_seconds(
            chunk_seconds, overlap_ratio, min_overlap_seconds, max_overlap_seconds
        )
    
    # Clamp overlap seconds
    original_overlap = overlap_seconds
    overlap_seconds = max(min_overlap_seconds, min(max_overlap_seconds, overlap_seconds))
    # Also ensure overlap < chunk
    if overlap_seconds >= chunk_seconds:
        overlap_seconds = max(min_overlap_seconds, chunk_seconds - 1)
    if overlap_seconds != original_overlap:
        clamped['overlapSecondsClamped'] = True
        clamped['overlapSecondsOriginal'] = original_overlap
    
    return {
        'maxDurationSeconds': max_duration,
        'autoSplit': diarization_auto_split,
        'chunkSeconds': chunk_seconds,
        'overlapSeconds': overlap_seconds,
        'derived': derived,
        'clamped': clamped,
    }


def estimate_chunk_count(
    file_duration_seconds: float,
    chunk_seconds: int,
    overlap_seconds: int
) -> int:
    """
    Estimate the number of chunks for a file given chunk/overlap settings.
    
    Args:
        file_duration_seconds: Duration of the audio file
        chunk_seconds: Chunk size in seconds
        overlap_seconds: Overlap in seconds
        
    Returns:
        Estimated number of chunks (minimum 1)
    """
    if file_duration_seconds <= 0:
        return 0
    if file_duration_seconds <= chunk_seconds:
        return 1
    
    effective_step = chunk_seconds - overlap_seconds
    if effective_step <= 0:
        effective_step = 1
    
    return max(1, int((file_duration_seconds - overlap_seconds) / effective_step) + 1)


def format_duration_human(seconds: int) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable string like "3m", "15m", "1h 30m"
    """
    if seconds < 60:
        return f"{seconds}s"
    
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"
