"""
Transcription options and segment post-processing for bulk-transcribe.

This module provides:
- TranscribeOptions: Explicit decoding parameters for Whisper
- Segment post-processing that never invents timestamps
- Quality presets for different use cases
"""

from dataclasses import dataclass, replace
from typing import Optional, List, Dict, Any


@dataclass
class TranscribeOptions:
    """
    Explicit transcription parameters for openai-whisper.
    
    These parameters control decoding behavior to reduce hallucinations,
    improve segment quality, and handle silence/noise more gracefully.
    """
    
    # Language handling
    language: Optional[str] = None  # None = auto-detect
    
    # Decoding parameters
    temperature: float = 0.0  # 0 = greedy decoding (most deterministic)
    beam_size: Optional[int] = None
    best_of: Optional[int] = None
    
    # Silence/noise handling - these reduce hallucinations during silence
    no_speech_threshold: float = 0.6  # Segments with no_speech_prob > this are skipped
    logprob_threshold: float = -1.0  # Segments with avg logprob < this may be skipped
    compression_ratio_threshold: float = 2.4  # Segments with compression ratio > this may be skipped
    
    # Timestamp handling
    word_timestamps: bool = False  # Enable word-level timestamps
    
    # Segment post-processing
    merge_short_segments: bool = True  # Merge segments < min_segment_duration
    min_segment_duration: float = 0.5  # Minimum segment duration in seconds
    max_segment_duration: Optional[float] = None  # Max duration (only split if word_timestamps=True)
    
    # Condition on previous text (can help continuity but may cause repetition)
    condition_on_previous_text: bool = True
    
    # Initial prompt (can help with domain-specific vocabulary)
    initial_prompt: Optional[str] = None
    
    def to_whisper_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for whisper.transcribe()."""
        kwargs = {
            'language': self.language,
            'temperature': self.temperature,
            'no_speech_threshold': self.no_speech_threshold,
            'logprob_threshold': self.logprob_threshold,
            'compression_ratio_threshold': self.compression_ratio_threshold,
            'word_timestamps': self.word_timestamps,
            'condition_on_previous_text': self.condition_on_previous_text,
            'verbose': False,
        }

        if self.beam_size is not None:
            kwargs['beam_size'] = int(self.beam_size)
        if self.best_of is not None:
            kwargs['best_of'] = int(self.best_of)
        
        if self.initial_prompt:
            kwargs['initial_prompt'] = self.initial_prompt
        
        return kwargs


# Quality presets
QUALITY_PRESETS = {
    'balanced': TranscribeOptions(
        temperature=0.0,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=True,
    ),
    'conservative': TranscribeOptions(
        temperature=0.0,
        no_speech_threshold=0.5,  # More permissive (keeps more)
        logprob_threshold=-0.8,  # Stricter quality threshold
        compression_ratio_threshold=2.0,  # Stricter repetition detection
        condition_on_previous_text=False,  # Reduces repetition loops
    ),
    'aggressive': TranscribeOptions(
        temperature=0.0,
        no_speech_threshold=0.7,  # Stricter (skips more)
        logprob_threshold=-1.5,  # More lenient quality threshold
        compression_ratio_threshold=3.0,  # More lenient repetition threshold
        condition_on_previous_text=True,
    ),
}


def get_preset(name: str) -> TranscribeOptions:
    """Get a quality preset by name."""
    # Return a copy so callers can safely override fields without mutating the preset.
    return replace(QUALITY_PRESETS.get(name, QUALITY_PRESETS['balanced']))


def postprocess_segments(
    segments: List[Dict[str, Any]],
    merge_short: bool = True,
    min_duration: float = 0.5,
    max_duration: Optional[float] = None,
    word_timestamps_available: bool = False,
) -> List[Dict[str, Any]]:
    """
    Post-process Whisper segments to improve readability.
    
    Rules:
    1. Never invent timestamps - only use times from Whisper output
    2. Merge pathologically short segments (< min_duration) with adjacent segments
    3. Only split long segments if word_timestamps are available and accurate
    4. Preserve all original timing information
    
    Args:
        segments: List of Whisper segments with 'start', 'end', 'text', optional 'words'
        merge_short: Whether to merge segments shorter than min_duration
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration (only enforced if word timestamps exist)
        word_timestamps_available: Whether word-level timestamps are available
    
    Returns:
        List of processed segments
    """
    if not segments:
        return []
    
    result = []
    
    # Step 1: Filter out empty segments
    filtered = [s for s in segments if s.get('text', '').strip()]
    
    if not filtered:
        return []
    
    # Attach source IDs for auditability (preserve original Whisper segment IDs if present)
    filtered = [_attach_source_ids(s) for s in filtered]

    # Step 2: Merge short segments if enabled
    if merge_short:
        filtered = _merge_short_segments(filtered, min_duration)
    
    # Step 3: Split long segments only if word timestamps are available
    if max_duration and word_timestamps_available:
        filtered = _split_long_segments_by_words(filtered, max_duration)
    
    # Step 4: Assign stable IDs and clean up
    for seg in filtered:
        start = float(seg['start'])
        end = float(seg['end'])
        if end < start:
            continue

        index = len(result)
        out = {
            'id': f"seg_{index + 1:03d}",
            'index': index,
            'start': start,
            'end': end,
            'text': seg['text'].strip(),
        }

        source_ids = seg.get('_source_ids')
        if isinstance(source_ids, list) and source_ids:
            out['source_id'] = '+'.join(str(x) for x in source_ids)

        if seg.get('words'):
            out['words'] = seg.get('words')
        result.append(out)
    
    return result


def _merge_short_segments(
    segments: List[Dict[str, Any]],
    min_duration: float,
) -> List[Dict[str, Any]]:
    """
    Merge segments shorter than min_duration with adjacent segments.
    
    Strategy:
    - Prefer merging with the previous segment (more natural reading flow)
    - Only merge if the combined segment is reasonable
    - Never merge across long gaps (> 2 seconds)
    """
    if len(segments) <= 1:
        return segments

    result: List[Dict[str, Any]] = []

    max_merge_gap_seconds = 0.5

    for seg in segments:
        start = float(seg['start'])
        end = float(seg['end'])
        duration = end - start

        if duration >= min_duration or not result:
            result.append(seg)
            continue

        prev = result[-1]
        prev_end = float(prev['end'])
        gap = start - prev_end

        # Never merge across long gaps; keep the short segment standalone.
        if gap > max_merge_gap_seconds:
            result.append(seg)
            continue

        # Merge into previous segment; this does not invent any timestamps.
        merged = {
            'start': float(prev['start']),
            'end': max(prev_end, end),
            'text': (prev.get('text', '').strip() + ' ' + seg.get('text', '').strip()).strip(),
            'words': _merge_words(prev.get('words'), seg.get('words')),
            '_source_ids': _merge_source_ids(prev.get('_source_ids'), seg.get('_source_ids')),
        }

        result[-1] = merged

    return result


def _merge_words(words1: Optional[List], words2: Optional[List]) -> Optional[List]:
    """Merge word lists from two segments."""
    if words1 is None and words2 is None:
        return None
    if words1 is None:
        return words2
    if words2 is None:
        return words1
    return words1 + words2


def _split_long_segments_by_words(
    segments: List[Dict[str, Any]],
    max_duration: float,
) -> List[Dict[str, Any]]:
    """
    Split segments longer than max_duration using word boundaries.
    
    Only splits if word timestamps are available and accurate.
    Never invents timestamps.
    """
    result = []
    
    for seg in segments:
        seg_start = float(seg['start'])
        seg_end = float(seg['end'])
        duration = seg_end - seg_start
        words = seg.get('words')
        
        if duration <= max_duration or not words:
            # Segment is short enough or no word timestamps
            result.append(seg)
            continue

        if not _valid_word_timestamps(words, seg_start=seg_start, seg_end=seg_end):
            # Do not split if word timestamps are missing/invalid.
            result.append(seg)
            continue
        
        # Split at word boundaries
        chunks = _split_words_into_chunks(words, max_duration)

        split_segments: List[Dict[str, Any]] = []
        ok = True
        for chunk_words in chunks:
            if not chunk_words:
                continue

            chunk_start = chunk_words[0].get('start')
            chunk_end = chunk_words[-1].get('end')
            if not isinstance(chunk_start, (int, float)) or not isinstance(chunk_end, (int, float)):
                ok = False
                break
            chunk_start = float(chunk_start)
            chunk_end = float(chunk_end)
            if chunk_end < chunk_start:
                ok = False
                break

            chunk_text = ''.join(w.get('word', '') for w in chunk_words).strip()
            split_segments.append({
                'start': chunk_start,
                'end': chunk_end,
                'text': chunk_text,
                'words': chunk_words,
                '_source_ids': seg.get('_source_ids'),
            })

        if not ok or not split_segments:
            result.append(seg)
        else:
            result.extend(split_segments)
    
    return result


def _split_words_into_chunks(
    words: List[Dict[str, Any]],
    max_duration: float,
) -> List[List[Dict[str, Any]]]:
    """
    Split a list of words into chunks of max_duration.
    
    Tries to split at natural boundaries (after punctuation) when possible.
    """
    if not words:
        return []
    
    chunks = []
    current_chunk = []
    chunk_start = words[0].get('start', 0)
    
    for word in words:
        word_end = word.get('end', chunk_start)
        chunk_duration = word_end - chunk_start
        
        if chunk_duration > max_duration and current_chunk:
            # Chunk is full, start a new one
            chunks.append(current_chunk)
            current_chunk = [word]
            chunk_start = word.get('start', word_end)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def _valid_word_timestamps(words: List[Dict[str, Any]], seg_start: float, seg_end: float) -> bool:
    """Return True only if all words have numeric start/end and are monotonic."""
    if not isinstance(words, list) or not words:
        return False

    last_end: Optional[float] = None
    for w in words:
        if not isinstance(w, dict):
            return False
        ws = w.get('start')
        we = w.get('end')
        if not isinstance(ws, (int, float)) or not isinstance(we, (int, float)):
            return False
        ws = float(ws)
        we = float(we)
        if we < ws:
            return False
        if last_end is not None and ws < last_end - 1e-4:
            return False
        # Loose bounds: allow tiny numeric drift.
        if ws < seg_start - 0.5 or we > seg_end + 0.5:
            return False
        last_end = we

    return True


def _attach_source_ids(seg: Dict[str, Any]) -> Dict[str, Any]:
    """Attach internal _source_ids list from seg['id'] if present; do not modify input."""
    if not isinstance(seg, dict):
        return seg
    if '_source_ids' in seg:
        return seg

    source_id = seg.get('id')
    if source_id is None:
        return seg

    out = dict(seg)
    out['_source_ids'] = [str(source_id)]
    return out


def _merge_source_ids(left: Any, right: Any) -> List[str]:
    """Merge two internal _source_ids values into a single list of strings."""
    merged: List[str] = []
    if isinstance(left, list):
        merged.extend(str(x) for x in left if x is not None and str(x) != '')
    if isinstance(right, list):
        merged.extend(str(x) for x in right if x is not None and str(x) != '')
    return merged


# ============================================================================
# VAD Pre-Chunking (Future Enhancement)
# ============================================================================
# 
# VAD-based pre-chunking would split audio into speech segments before
# transcription, which can improve quality for:
# - Long recordings with significant silence
# - Recordings with background noise
# - Multi-speaker recordings
#
# Implementation plan (not implemented in this phase):
# 1. Use silero-vad (small, PyTorch-based, ~1MB model)
# 2. Split audio at silence boundaries with overlap/padding
# 3. Transcribe each chunk separately
# 4. Time-shift and merge results
#
# Requirements:
# - silero-vad pip package (or vendor the model)
# - Careful timestamp adjustment after chunking
# - UI toggle in Advanced section (off by default)
#
# This is deferred because:
# - Adds packaging complexity
# - Requires careful testing with various audio types
# - Current whisper segmentation is often sufficient

VAD_AVAILABLE = False

def check_vad_available() -> bool:
    """Check if VAD dependencies are available."""
    global VAD_AVAILABLE
    if VAD_AVAILABLE:
        return True
    try:
        import torch
        # silero-vad would be checked here
        # For now, always return False since we're not implementing VAD yet
        return False
    except ImportError:
        return False


def preprocess_with_vad(audio_path: str, padding: float = 0.5) -> List[Dict[str, Any]]:
    """
    Stub for VAD-based audio pre-chunking.
    
    When implemented, this would:
    1. Load audio
    2. Run VAD to detect speech segments
    3. Return list of chunks with start/end times and audio data
    
    Args:
        audio_path: Path to audio file
        padding: Seconds of padding before/after each speech segment
    
    Returns:
        List of dicts with 'start', 'end', 'audio' keys
        
    Currently returns empty list (VAD not implemented).
    """
    if not check_vad_available():
        return []
    
    # Future implementation would go here
    return []
