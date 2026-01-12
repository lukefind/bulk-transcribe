"""
Speaker diarization module using pyannote.audio.

This module provides speaker diarization functionality without any Flask
or session dependencies. It is a pure computation module.
"""

import os
from typing import Optional


def run_diarization(
    audio_path: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    device: str = 'cpu'
) -> list[dict]:
    """
    Run speaker diarization on an audio file.
    
    Args:
        audio_path: Path to the audio file
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        num_speakers: Exact number of speakers if known (optional)
        device: PyTorch device ('cpu', 'cuda', 'mps')
    
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
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for speaker diarization. "
            "Install with: pip install pyannote.audio"
        ) from e
    
    # Get HuggingFace token from environment
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise RuntimeError(
            "HuggingFace token required for pyannote. "
            "Set HF_TOKEN environment variable. "
            "Get token from https://huggingface.co/settings/tokens"
        )
    
    # Load the diarization pipeline
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        if pipeline is None:
            raise RuntimeError("Pipeline.from_pretrained returned None")
    except Exception as e:
        raise RuntimeError(f"Failed to load diarization model: {e}") from e
    
    # Move to device if not CPU
    if device != 'cpu':
        import torch
        pipeline.to(torch.device(device))
    
    # Build diarization parameters
    diarize_params = {}
    if num_speakers is not None:
        diarize_params['num_speakers'] = num_speakers
    else:
        if min_speakers is not None:
            diarize_params['min_speakers'] = min_speakers
        if max_speakers is not None:
            diarize_params['max_speakers'] = max_speakers
    
    # Run diarization
    try:
        diarization = pipeline(audio_path, **diarize_params)
    except Exception as e:
        raise RuntimeError(f"Diarization failed: {e}") from e
    
    # Convert to list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })
    
    return segments


def merge_transcript_with_speakers(
    transcript_segments: list[dict],
    speaker_segments: list[dict],
    merge_gap_threshold: float = 1.0
) -> list[dict]:
    """
    Merge Whisper transcript segments with speaker diarization.
    
    Args:
        transcript_segments: Whisper segments with {start, end, text}
        speaker_segments: Diarization segments with {start, end, speaker}
        merge_gap_threshold: Max gap (seconds) to merge consecutive same-speaker segments
    
    Returns:
        List of merged segments:
        [
            {"start": 2.1, "end": 6.3, "speaker": "Speaker 1", "text": "..."},
            ...
        ]
    """
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


def format_speaker_markdown(
    merged_segments: list[dict],
    filename: str
) -> str:
    """
    Format merged segments as reviewer-friendly markdown.
    
    Args:
        merged_segments: Output from merge_transcript_with_speakers
        filename: Original audio filename for header
    
    Returns:
        Markdown string optimized for human review
    """
    lines = [f"## File: {filename}", ""]
    
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
