"""
Speaker diarization module using pyannote.audio.

This module provides speaker diarization functionality without any Flask
or session dependencies. It is a pure computation module.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def check_diarization_access(hf_token: str) -> Dict[str, any]:
    """
    Verify HuggingFace access to pyannote models without downloading models.
    
    Args:
        hf_token: HuggingFace token
    
    Returns:
        Dict with access information:
        - ok: bool - True if access to both repos
        - missing: list[str] - List of repos that can't be accessed
        - message: str|None - Error message if any
    """
    from huggingface_hub import HfApi
    
    repos_to_check = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0"
    ]
    
    missing = []
    message = None
    
    try:
        api = HfApi(token=hf_token)
        
        for repo_id in repos_to_check:
            try:
                api.model_info(repo_id=repo_id)
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "403" in error_msg or "Cannot access gated repo" in error_msg:
                    missing.append(repo_id)
                else:
                    message = f"Unexpected error checking {repo_id}: {e}"
                    return {"ok": False, "missing": missing, "message": message}
    
    except Exception as e:
        message = f"Failed to initialize HfApi: {e}"
        return {"ok": False, "missing": missing, "message": message}
    
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "message": message
    }


def _convert_to_wav(
    audio_path: str, 
    target_sample_rate: int = 16000,
    temp_dir: Optional[str] = None,
    max_wav_mb: Optional[int] = None,
    job_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Convert audio to WAV format at 16kHz mono for pyannote compatibility.
    
    Args:
        audio_path: Path to the input audio file
        target_sample_rate: Target sample rate (default 16kHz)
        temp_dir: Directory for temp files (default: system temp)
        max_wav_mb: Maximum output WAV size in MB (default: from env or 500)
        job_id: Job ID for logging
        session_id: Session ID for logging
    
    Returns:
        Path to the converted WAV file (or original if already compatible)
    
    Raises:
        RuntimeError: If conversion fails or output too large
    """
    import subprocess
    import tempfile
    import time
    from logger import log_event
    
    # Get max WAV size from env if not provided
    if max_wav_mb is None:
        max_wav_mb = int(os.environ.get('DIARIZATION_MAX_WAV_MB', '500'))
    
    log_fields = {'file': os.path.basename(audio_path)}
    if job_id:
        log_fields['jobId'] = job_id
    if session_id:
        log_fields['sessionId'] = session_id
    
    # Check if already a compatible WAV file
    try:
        from audio_utils import get_audio_info, is_wav_pcm_compatible
        info = get_audio_info(audio_path)
        if is_wav_pcm_compatible(info):
            log_event('info', 'wav_convert_skipped', reason='already_compatible', **log_fields)
            return audio_path
    except Exception:
        pass  # If probe fails, try conversion anyway
    
    # Create temp WAV file in specified directory
    start_time = time.time()
    log_event('info', 'wav_convert_started', targetSampleRate=target_sample_rate, **log_fields)
    
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir)
    else:
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    
    try:
        # Use ffmpeg to convert to 16kHz mono WAV
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', str(target_sample_rate),
            '-ac', '1',  # mono
            '-c:a', 'pcm_s16le',  # 16-bit PCM
            temp_wav.name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")
        
        # Check output size
        output_size_bytes = os.path.getsize(temp_wav.name)
        output_size_mb = output_size_bytes / (1024 * 1024)
        
        duration_ms = int((time.time() - start_time) * 1000)
        log_event('info', 'wav_convert_finished', 
                 durationMs=duration_ms, 
                 outputSizeMB=round(output_size_mb, 2),
                 **log_fields)
        
        if output_size_mb > max_wav_mb:
            os.unlink(temp_wav.name)
            raise RuntimeError(
                f"Converted WAV too large: {output_size_mb:.1f}MB exceeds limit of {max_wav_mb}MB. "
                f"Use shorter audio or increase DIARIZATION_MAX_WAV_MB."
            )
        
        return temp_wav.name
        
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)
        raise RuntimeError("WAV conversion timed out after 5 minutes")
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)
        raise


def load_pipeline(
    device: str = 'cpu',
    job_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Load the pyannote diarization pipeline.
    
    Args:
        device: PyTorch device ('cpu', 'cuda', 'mps')
        job_id: Job ID for logging (optional)
        session_id: Session ID for logging (optional)
    
    Returns:
        Loaded pyannote Pipeline
    
    Raises:
        ImportError: If pyannote.audio is not installed
        RuntimeError: If pipeline loading fails
    """
    from logger import log_event, with_timer
    
    log_fields = {'stage': 'diarization'}
    if job_id:
        log_fields['jobId'] = job_id
    if session_id:
        log_fields['sessionId'] = session_id
    
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        log_event('error', 'diarization_import_failed', error=str(e), **log_fields)
        raise ImportError(
            "pyannote.audio is required for speaker diarization. "
            "Install with: pip install pyannote.audio"
        ) from e
    
    # Get HuggingFace token from environment
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        log_event('error', 'diarization_no_token', **log_fields)
        raise RuntimeError(
            "HuggingFace token required for pyannote. "
            "Set HF_TOKEN environment variable. "
            "Get token from https://huggingface.co/settings/tokens"
        )
    
    # Load the diarization pipeline with logging
    log_event('info', 'diarization_model_loading_started', model='pyannote/speaker-diarization-3.1', **log_fields)
    
    try:
        with with_timer('diarization_model_loading', **log_fields):
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
        if pipeline is None:
            log_event('error', 'diarization_model_none', **log_fields)
            raise RuntimeError(
                "Pipeline.from_pretrained returned None. "
                "This may be due to missing model access. "
                "Please ensure you have accepted the user agreement at: "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
    except Exception as e:
        error_msg = str(e)
        log_event('error', 'diarization_model_load_failed', error=error_msg, **log_fields)
        
        if "Cannot access gated repo" in error_msg:
            raise RuntimeError(
                "Model access required. Please visit the following URLs and accept the user agreements:\n"
                "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  - https://huggingface.co/pyannote/segmentation-3.0\n"
                "Then restart the container."
            ) from e
        else:
            raise RuntimeError(f"Failed to load diarization model: {e}") from e
    
    # Move to device if not CPU
    if device != 'cpu':
        import torch
        log_event('info', 'diarization_device_move', device=device, **log_fields)
        pipeline.to(torch.device(device))
    
    return pipeline


def run_diarization(
    audio_path: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    device: str = 'cpu',
    job_id: Optional[str] = None,
    session_id: Optional[str] = None,
    temp_dir: Optional[str] = None
) -> list[dict]:
    """
    Run speaker diarization on an audio file.
    
    Args:
        audio_path: Path to the audio file
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        num_speakers: Exact number of speakers if known (optional)
        device: PyTorch device ('cpu', 'cuda', 'mps')
        job_id: Job ID for logging (optional)
        session_id: Session ID for logging (optional)
        temp_dir: Directory for temp files (optional)
    
    Returns:
        List of speaker segments:
        [
            {"start": 2.1, "end": 6.3, "speaker": "SPEAKER_00"},
            {"start": 6.5, "end": 10.2, "speaker": "SPEAKER_01"},
            ...
        ]
    
    Raises:
        ImportError: If pyannote.audio is not installed
        RuntimeError: If diarization fails
    """
    from logger import log_event, with_timer
    
    # Convert audio to WAV format for pyannote compatibility
    converted_path = None
    try:
        converted_path = _convert_to_wav(
            audio_path, 
            temp_dir=temp_dir,
            job_id=job_id,
            session_id=session_id
        )
        if converted_path != audio_path:
            # Use converted file for diarization
            diarization_audio_path = converted_path
        else:
            diarization_audio_path = audio_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {e}")
    
    # Log diarization start with memory info
    log_fields = {
        'stage': 'diarization',
        'file': os.path.basename(audio_path)
    }
    if job_id:
        log_fields['jobId'] = job_id
    if session_id:
        log_fields['sessionId'] = session_id
    
    # Log memory usage at start
    try:
        from audio_utils import get_memory_usage_mb
        mem_mb = get_memory_usage_mb()
        if mem_mb is not None:
            log_fields['memoryMB'] = round(mem_mb, 1)
    except Exception:
        pass
    
    log_event('info', 'diarization_started', **log_fields)
    
    # Load pipeline
    pipeline = load_pipeline(device=device, job_id=job_id, session_id=session_id)
    
    # Build diarization parameters
    diarize_params = {}
    if num_speakers is not None:
        diarize_params['num_speakers'] = num_speakers
    else:
        if min_speakers is not None:
            diarize_params['min_speakers'] = min_speakers
        if max_speakers is not None:
            diarize_params['max_speakers'] = max_speakers
    
    log_event('info', 'diarization_params', params=diarize_params, **log_fields)
    
    # Run diarization with timing
    try:
        with with_timer('diarization_run', **log_fields):
            diarization = pipeline(diarization_audio_path, **diarize_params)
    except Exception as e:
        log_event('error', 'diarization_run_failed', error=str(e), **log_fields)
        raise RuntimeError(f"Diarization failed: {e}") from e
    finally:
        # Clean up converted temp file
        if converted_path and converted_path != audio_path and os.path.exists(converted_path):
            try:
                os.unlink(converted_path)
            except:
                pass
    
    # Convert to list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })
    
    # Log memory usage at end
    end_mem_mb = None
    try:
        from audio_utils import get_memory_usage_mb
        end_mem_mb = get_memory_usage_mb()
    except Exception:
        pass
    
    # Remove memoryMB from log_fields to avoid duplicate when adding end memory
    finish_log_fields = {k: v for k, v in log_fields.items() if k != 'memoryMB'}
    finish_log_fields['numSegments'] = len(segments)
    finish_log_fields['numSpeakers'] = len(set(s['speaker'] for s in segments))
    if end_mem_mb is not None:
        finish_log_fields['memoryMB'] = round(end_mem_mb, 1)
    
    log_event('info', 'diarization_finished', **finish_log_fields)
    
    return segments


def merge_transcript_with_speakers(
    transcript_segments: list[dict],
    speaker_segments: list[dict],
    merge_gap_threshold: float = 1.0,
    job_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> list[dict]:
    """
    Merge Whisper transcript segments with speaker diarization.
    
    Args:
        transcript_segments: Whisper segments with {start, end, text}
        speaker_segments: Diarization segments with {start, end, speaker}
        merge_gap_threshold: Max gap (seconds) to merge consecutive same-speaker segments
        job_id: Job ID for logging (optional)
        session_id: Session ID for logging (optional)
    
    Returns:
        List of merged segments:
        [
            {"start": 2.1, "end": 6.3, "speaker": "Speaker 1", "text": "..."},
            ...
        ]
    """
    from logger import log_event, with_timer
    
    # Log merge start
    log_fields = {
        'stage': 'merge',
        'numTranscriptSegments': len(transcript_segments),
        'numSpeakerSegments': len(speaker_segments)
    }
    if job_id:
        log_fields['jobId'] = job_id
    if session_id:
        log_fields['sessionId'] = session_id
    
    log_event('info', 'merge_started', **log_fields)
    if not transcript_segments:
        return []
    
    if not speaker_segments:
        # No diarization - return transcript as-is with unknown speaker
        return [
            {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "speaker": "Speaker",
                "text": seg.get("text", "").strip()
            }
            for seg in transcript_segments
        ]
    
    # Assign speaker to each transcript segment based on overlap
    assigned_segments = []
    for tseg in transcript_segments:
        t_start = tseg.get("start", 0)
        t_end = tseg.get("end", 0)
        text = tseg.get("text", "").strip()
        
        if not text:
            continue
        
        # Find speaker with maximum overlap
        best_speaker = None
        best_overlap = 0
        
        for sseg in speaker_segments:
            s_start = sseg["start"]
            s_end = sseg["end"]
            
            # Calculate overlap
            overlap_start = max(t_start, s_start)
            overlap_end = min(t_end, s_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg["speaker"]
        
        # Convert SPEAKER_XX to friendly name
        speaker_name = _friendly_speaker_name(best_speaker) if best_speaker else "Speaker"
        
        assigned_segments.append({
            "start": t_start,
            "end": t_end,
            "speaker": speaker_name,
            "text": text
        })
    
    # Merge consecutive segments from same speaker if gap is small
    merged = []
    for seg in assigned_segments:
        if not merged:
            merged.append(seg.copy())
            continue
        
        last = merged[-1]
        gap = seg["start"] - last["end"]
        
        if last["speaker"] == seg["speaker"] and gap <= merge_gap_threshold:
            # Merge: extend end time and append text
            last["end"] = seg["end"]
            last["text"] = last["text"] + " " + seg["text"]
        else:
            merged.append(seg.copy())
    
    log_event('info', 'merge_finished', 
             numMergedSegments=len(merged),
             **log_fields)
    
    return merged


def _friendly_speaker_name(speaker_id: str) -> str:
    """Convert SPEAKER_XX to Speaker N."""
    if not speaker_id:
        return "Speaker"
    
    # Extract number from SPEAKER_00, SPEAKER_01, etc.
    if speaker_id.startswith("SPEAKER_"):
        try:
            num = int(speaker_id.split("_")[1])
            return f"Speaker {num + 1}"
        except (IndexError, ValueError):
            pass
    
    return speaker_id


def merge_chunk_speaker_segments(
    all_chunk_segments: list[list[dict]],
    chunk_infos: list[dict],
    overlap_seconds: float = 5.0,
    max_speakers: int = 6
) -> list[dict]:
    """
    Merge speaker segments from multiple chunks into a single timeline.
    
    This function handles:
    1. Shifting chunk-local timestamps to global timeline
    2. Reconciling speaker identities across chunks using time continuity
    3. Merging overlapping segments from chunk boundaries
    
    Args:
        all_chunk_segments: List of speaker segment lists, one per chunk
        chunk_infos: List of chunk info dicts with startSec, endSec
        overlap_seconds: Overlap duration between chunks
        max_speakers: Maximum number of global speakers to track
    
    Returns:
        Merged list of speaker segments in global timeline
    """
    if not all_chunk_segments:
        return []
    
    # Build global speaker mapping using time continuity heuristic
    # Track which chunk-local speakers map to which global speakers
    global_speakers = []  # List of global speaker IDs
    speaker_mapping = {}  # (chunk_idx, local_speaker) -> global_speaker
    
    all_global_segments = []
    
    for chunk_idx, (chunk_segments, chunk_info) in enumerate(zip(all_chunk_segments, chunk_infos)):
        chunk_start = chunk_info.get('startSec', 0)
        
        for seg in chunk_segments:
            local_speaker = seg.get('speaker', 'SPEAKER_00')
            local_start = seg.get('start', 0)
            local_end = seg.get('end', 0)
            
            # Shift to global timeline
            global_start = chunk_start + local_start
            global_end = chunk_start + local_end
            
            # Map local speaker to global speaker
            mapping_key = (chunk_idx, local_speaker)
            
            if mapping_key not in speaker_mapping:
                # Try to find a matching global speaker from previous chunk
                # based on time continuity (speaker active near chunk boundary)
                matched_global = None
                
                if chunk_idx > 0:
                    prev_chunk_end = chunk_infos[chunk_idx - 1].get('endSec', 0)
                    boundary_window = overlap_seconds * 1.5
                    
                    # Look for segments near the boundary in previous chunk
                    for prev_seg in all_global_segments:
                        if prev_seg['end'] >= prev_chunk_end - boundary_window:
                            # Check if this segment's speaker is a candidate
                            if global_start <= prev_seg['end'] + boundary_window:
                                matched_global = prev_seg['speaker']
                                break
                
                if matched_global and len(global_speakers) < max_speakers:
                    speaker_mapping[mapping_key] = matched_global
                else:
                    # Create new global speaker
                    new_speaker = f"SPEAKER_{len(global_speakers):02d}"
                    if len(global_speakers) < max_speakers:
                        global_speakers.append(new_speaker)
                        speaker_mapping[mapping_key] = new_speaker
                    else:
                        # Reuse last speaker if at max
                        speaker_mapping[mapping_key] = global_speakers[-1]
            
            global_speaker = speaker_mapping[mapping_key]
            
            all_global_segments.append({
                'speaker': global_speaker,
                'start': round(global_start, 3),
                'end': round(global_end, 3)
            })
    
    # Sort by start time
    all_global_segments.sort(key=lambda x: x['start'])
    
    # Merge overlapping segments from same speaker
    merged = []
    for seg in all_global_segments:
        if not merged:
            merged.append(seg.copy())
            continue
        
        last = merged[-1]
        
        # Check for overlap or small gap with same speaker
        if last['speaker'] == seg['speaker'] and seg['start'] <= last['end'] + 0.5:
            # Extend the previous segment
            last['end'] = max(last['end'], seg['end'])
        else:
            merged.append(seg.copy())
    
    return merged


def run_chunked_diarization(
    audio_path: str,
    chunks: list,
    pipeline,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    job_id: Optional[str] = None,
    session_id: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> list[dict]:
    """
    Run diarization on audio chunks and merge results.
    
    Args:
        audio_path: Path to original audio file (for logging)
        chunks: List of ChunkInfo objects from split_audio_to_wav_chunks
        pipeline: Loaded pyannote Pipeline
        min_speakers: Minimum speakers hint
        max_speakers: Maximum speakers hint
        num_speakers: Fixed number of speakers
        job_id: Job ID for logging
        session_id: Session ID for logging
        progress_callback: Optional callback(chunk_idx, total_chunks, status)
    
    Returns:
        Merged speaker segments in global timeline
    """
    from logger import log_event
    
    log_fields = {'stage': 'chunked_diarization'}
    if job_id:
        log_fields['jobId'] = job_id
    if session_id:
        log_fields['sessionId'] = session_id
    
    all_chunk_segments = []
    chunk_infos = []
    total_chunks = len(chunks)
    
    log_event('info', 'chunked_diarization_started',
              totalChunks=total_chunks,
              **log_fields)
    
    # Build pipeline params
    params = {}
    if num_speakers is not None:
        params['num_speakers'] = num_speakers
    else:
        if min_speakers is not None:
            params['min_speakers'] = min_speakers
        if max_speakers is not None:
            params['max_speakers'] = max_speakers
    
    for chunk in chunks:
        chunk_idx = chunk.index
        chunk_path = chunk.path
        
        if progress_callback:
            progress_callback(chunk_idx, total_chunks, f'Diarizing chunk {chunk_idx + 1}/{total_chunks}')
        
        log_event('info', 'chunk_diarization_started',
                  chunkIndex=chunk_idx,
                  totalChunks=total_chunks,
                  chunkPath=os.path.basename(chunk_path),
                  **log_fields)
        
        try:
            # Run diarization on chunk
            if params:
                diarization = pipeline(chunk_path, **params)
            else:
                diarization = pipeline(chunk_path)
            
            # Extract segments
            chunk_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                chunk_segments.append({
                    'speaker': speaker,
                    'start': round(turn.start, 3),
                    'end': round(turn.end, 3)
                })
            
            all_chunk_segments.append(chunk_segments)
            chunk_infos.append(chunk.to_dict())
            
            log_event('info', 'chunk_diarization_finished',
                      chunkIndex=chunk_idx,
                      numSegments=len(chunk_segments),
                      **log_fields)
            
        except Exception as e:
            log_event('error', 'chunk_diarization_failed',
                      chunkIndex=chunk_idx,
                      error=str(e)[:200],
                      **log_fields)
            # Continue with empty segments for this chunk
            all_chunk_segments.append([])
            chunk_infos.append(chunk.to_dict())
    
    # Merge all chunk segments
    overlap_seconds = 5.0  # Default, could be passed in
    if len(chunk_infos) >= 2:
        # Estimate overlap from chunk boundaries
        overlap_seconds = max(0, chunk_infos[0].get('endSec', 0) - chunk_infos[1].get('startSec', 0))
    
    merged_segments = merge_chunk_speaker_segments(
        all_chunk_segments,
        chunk_infos,
        overlap_seconds=overlap_seconds,
        max_speakers=max_speakers or 6
    )
    
    log_event('info', 'chunked_diarization_finished',
              totalChunks=total_chunks,
              totalSegments=len(merged_segments),
              **log_fields)
    
    return merged_segments


def format_speaker_markdown(
    merged_segments: list[dict],
    filename: str,
    transcript_segments: Optional[list[dict]] = None
) -> str:
    """
    Format merged segments as reviewer-friendly markdown.
    
    Args:
        merged_segments: Output from merge_transcript_with_speakers
        filename: Original audio filename for header
        transcript_segments: Original transcript segments (for fallback if no diarization)
    
    Returns:
        Markdown string optimized for human review
    """
    lines = [f"## File: {filename}", ""]
    
    # Handle empty merged segments
    if not merged_segments:
        if transcript_segments and len(transcript_segments) > 0:
            # Diarization produced no speaker segments but transcript exists
            lines.append("> Diarization produced no speaker segments; transcript shown without speaker labels.")
            lines.append("")
            for seg in transcript_segments:
                timestamp = _format_timestamp(seg.get("start", 0))
                text = seg.get("text", "").strip()
                if text:
                    lines.append(f"[{timestamp}] {text}")
                    lines.append("")
        else:
            # No transcript and no diarization
            lines.append("> No speaker segments detected.")
            lines.append("")
            lines.append("This file produced no transcribable content or speaker segments.")
            lines.append("")
        return "\n".join(lines)
    
    for seg in merged_segments:
        timestamp = _format_timestamp(seg["start"])
        speaker = seg["speaker"]
        text = seg["text"]
        
        lines.append(f"[{timestamp}] {speaker}:")
        lines.append(text)
        lines.append("")
    
    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_rttm(
    speaker_segments: list[dict],
    filename: str
) -> str:
    """
    Format speaker segments as RTTM (Rich Transcription Time Marked).
    
    RTTM format:
    SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    
    Args:
        speaker_segments: Raw diarization segments
        filename: Audio filename (without extension)
    
    Returns:
        RTTM formatted string
    """
    lines = []
    file_id = os.path.splitext(filename)[0]
    
    for seg in speaker_segments:
        start = seg["start"]
        duration = round(seg["end"] - seg["start"], 3)
        speaker = seg["speaker"]
        
        line = f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        lines.append(line)
    
    return "\n".join(lines)


def format_diarization_json(
    speaker_segments: list[dict],
    merged_segments: list[dict],
    filename: str
) -> dict:
    """
    Create structured JSON output for diarization results.
    
    Args:
        speaker_segments: Raw diarization segments
        merged_segments: Merged transcript+speaker segments
        filename: Original audio filename
    
    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "source": filename,
        "speakers": _get_unique_speakers(speaker_segments),
        "rawDiarization": speaker_segments,
        "mergedTranscript": merged_segments
    }


def _get_unique_speakers(segments: list[dict]) -> list[str]:
    """Extract unique speaker IDs in order of first appearance."""
    seen = set()
    speakers = []
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker not in seen:
            seen.add(speaker)
            speakers.append(speaker)
    return speakers
