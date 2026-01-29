#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) module using Silero VAD.

This module provides pre-processing to filter out silence from audio
before sending to Whisper, reducing hallucinations on quiet sections.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Lazy imports to avoid loading torch at module import time
_vad_model = None
_vad_utils = None


def _load_vad_model():
    """Lazy-load Silero VAD model."""
    global _vad_model, _vad_utils
    
    if _vad_model is not None:
        return _vad_model, _vad_utils
    
    import torch
    
    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    _vad_model = model
    _vad_utils = utils
    
    return model, utils


def get_speech_timestamps(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = True
) -> List[Dict[str, float]]:
    """
    Detect speech segments in an audio file using Silero VAD.
    
    Args:
        audio_path: Path to audio file
        threshold: Speech probability threshold (0.0-1.0). Higher = stricter.
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence duration to split segments
        speech_pad_ms: Padding around speech segments
        return_seconds: If True, return timestamps in seconds; else samples
    
    Returns:
        List of dicts with 'start' and 'end' keys (in seconds or samples)
    """
    import torch
    import torchaudio
    
    model, utils = _load_vad_model()
    get_speech_ts = utils[0]
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed (Silero VAD expects 16kHz)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000
    
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Get speech timestamps
    speech_timestamps = get_speech_ts(
        wav.squeeze(),
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=return_seconds,
        sampling_rate=sr
    )
    
    return speech_timestamps


def filter_audio_by_vad(
    audio_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    speech_pad_ms: int = 100,
    max_gap_to_merge_ms: int = 300
) -> Tuple[str, List[Dict[str, float]], float]:
    """
    Filter an audio file to keep only speech segments.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for output file (auto-generated if None)
        threshold: VAD threshold (0.0-1.0)
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence to split segments
        speech_pad_ms: Padding around speech segments
        max_gap_to_merge_ms: Merge segments with gaps smaller than this
    
    Returns:
        Tuple of (output_path, speech_segments, total_speech_duration)
    """
    import torch
    import torchaudio
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_path,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True
    )
    
    if not speech_timestamps:
        # No speech detected - return original file
        return audio_path, [], 0.0
    
    # Merge close segments
    merged_segments = []
    max_gap = max_gap_to_merge_ms / 1000.0
    
    for seg in speech_timestamps:
        if merged_segments and (seg['start'] - merged_segments[-1]['end']) < max_gap:
            # Merge with previous
            merged_segments[-1]['end'] = seg['end']
        else:
            merged_segments.append({'start': seg['start'], 'end': seg['end']})
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Extract speech segments
    speech_chunks = []
    for seg in merged_segments:
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        speech_chunks.append(wav[:, start_sample:end_sample])
    
    # Concatenate with small silence gaps between segments
    silence_samples = int(0.1 * sr)  # 100ms silence between segments
    silence = torch.zeros(wav.shape[0], silence_samples)
    
    output_chunks = []
    for i, chunk in enumerate(speech_chunks):
        output_chunks.append(chunk)
        if i < len(speech_chunks) - 1:
            output_chunks.append(silence)
    
    output_wav = torch.cat(output_chunks, dim=1)
    
    # Calculate total speech duration
    total_speech = sum(seg['end'] - seg['start'] for seg in merged_segments)
    
    # Save output
    if output_path is None:
        suffix = Path(audio_path).suffix
        fd, output_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
    
    torchaudio.save(output_path, output_wav, sr)
    
    return output_path, merged_segments, total_speech


def create_timestamp_mapping(
    original_segments: List[Dict[str, float]],
    silence_gap: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Create a mapping from filtered audio timestamps back to original timestamps.
    
    Args:
        original_segments: Speech segments from VAD (in original audio time)
        silence_gap: Gap inserted between segments in filtered audio
    
    Returns:
        List of mapping entries with 'filtered_start', 'filtered_end', 
        'original_start', 'original_end'
    """
    mapping = []
    filtered_time = 0.0
    
    for seg in original_segments:
        duration = seg['end'] - seg['start']
        mapping.append({
            'filtered_start': filtered_time,
            'filtered_end': filtered_time + duration,
            'original_start': seg['start'],
            'original_end': seg['end']
        })
        filtered_time += duration + silence_gap
    
    return mapping


def remap_timestamps(
    segments: List[Dict[str, Any]],
    timestamp_mapping: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remap Whisper segment timestamps from filtered audio back to original audio.
    
    Args:
        segments: Whisper segments with 'start' and 'end' keys
        timestamp_mapping: Mapping from create_timestamp_mapping()
    
    Returns:
        Segments with remapped timestamps
    """
    if not timestamp_mapping:
        return segments
    
    remapped = []
    
    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        
        new_start = _remap_single_timestamp(seg_start, timestamp_mapping)
        new_end = _remap_single_timestamp(seg_end, timestamp_mapping)
        
        remapped_seg = seg.copy()
        remapped_seg['start'] = new_start
        remapped_seg['end'] = new_end
        
        # Also remap word timestamps if present
        if 'words' in remapped_seg:
            remapped_seg['words'] = remap_timestamps(remapped_seg['words'], timestamp_mapping)
        
        remapped.append(remapped_seg)
    
    return remapped


def _remap_single_timestamp(
    filtered_time: float,
    timestamp_mapping: List[Dict[str, Any]]
) -> float:
    """
    Remap a single timestamp from filtered audio time to original audio time.
    
    Handles edge cases:
    - Timestamp falls exactly within a mapped segment
    - Timestamp falls in a gap between segments (maps to nearest boundary)
    - Timestamp is before first segment or after last segment
    """
    if not timestamp_mapping:
        return filtered_time
    
    # Check if timestamp falls within any mapped segment
    for m in timestamp_mapping:
        if m['filtered_start'] <= filtered_time <= m['filtered_end']:
            offset = filtered_time - m['filtered_start']
            return m['original_start'] + offset
    
    # Timestamp is in a gap or outside all segments
    # Find the closest segment boundary and map to it
    
    # Before first segment
    if filtered_time < timestamp_mapping[0]['filtered_start']:
        # Map proportionally from start
        return timestamp_mapping[0]['original_start']
    
    # After last segment
    if filtered_time > timestamp_mapping[-1]['filtered_end']:
        # Map to end of last segment
        return timestamp_mapping[-1]['original_end']
    
    # In a gap between segments - find which gap and interpolate
    for i in range(len(timestamp_mapping) - 1):
        curr_end = timestamp_mapping[i]['filtered_end']
        next_start = timestamp_mapping[i + 1]['filtered_start']
        
        if curr_end < filtered_time < next_start:
            # Timestamp is in the gap between segment i and i+1
            # Map to the original time at the boundary
            # Use the end of the current original segment as the reference
            gap_duration = next_start - curr_end
            gap_offset = filtered_time - curr_end
            gap_ratio = gap_offset / gap_duration if gap_duration > 0 else 0
            
            # Interpolate between end of current original segment and start of next
            orig_curr_end = timestamp_mapping[i]['original_end']
            orig_next_start = timestamp_mapping[i + 1]['original_start']
            
            return orig_curr_end + gap_ratio * (orig_next_start - orig_curr_end)
    
    # Fallback (shouldn't reach here)
    return filtered_time
