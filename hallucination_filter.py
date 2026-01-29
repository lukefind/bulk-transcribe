#!/usr/bin/env python3
"""
Hallucination detection and filtering for Whisper transcripts.

Whisper is known to hallucinate on:
- Silence or very quiet audio
- Background noise
- Non-speech audio

Common hallucination patterns:
- Repeated phrases ("Thank you. Thank you. Thank you.")
- Foreign language insertions in English audio
- Generic filler phrases on silence
- Very short segments with low confidence
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


# Phrases that are ALWAYS suspicious (rarely real speech in normal recordings)
ALWAYS_SUSPICIOUS_PHRASES = [
    r'^please subscribe\.?$',
    r'^subscribe\.?$',
    r'^like and subscribe\.?$',
    r'^see you next time\.?$',
    r'^music\.?$',
    r'^\[music\]$',
    r'^applause\.?$',
    r'^\[applause\]$',
    r'^silence\.?$',
    r'^\.+$',  # Just periods
    r'^\s*$',  # Empty/whitespace
]

# Phrases that are only suspicious if Whisper metrics are also bad
# (these are common in real speech, so need corroborating evidence)
SUSPICIOUS_WITH_BAD_METRICS = [
    r'^thank you\.?$',
    r'^thanks\.?$',
    r'^bye\.?$',
    r'^goodbye\.?$',
    r'^okay\.?$',
    r'^ok\.?$',
]

# Compile patterns for efficiency
_ALWAYS_SUSPICIOUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ALWAYS_SUSPICIOUS_PHRASES]
_SUSPICIOUS_WITH_METRICS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_WITH_BAD_METRICS]

# Characters that indicate foreign language hallucination in English audio
FOREIGN_CHARS = set('你好谢谢再见こんにちはありがとうさようなら안녕하세요감사합니다')

# Threshold for detecting repetition
REPETITION_THRESHOLD = 3  # Same phrase 3+ times in a row


def detect_hallucination_indicators(segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a segment for hallucination indicators.
    
    Args:
        segment: Whisper segment with 'text', 'start', 'end', and optionally
                 'no_speech_prob', 'avg_logprob', 'compression_ratio'
    
    Returns:
        Dict with hallucination indicators:
        - is_likely_hallucination: bool
        - confidence: float (0-1, higher = more likely hallucination)
        - reasons: list of reason strings
    """
    text = segment.get('text', '').strip()
    reasons = []
    confidence = 0.0
    
    # Check for empty or very short text
    if len(text) < 2:
        reasons.append('empty_or_very_short')
        confidence = max(confidence, 0.9)

    # Check Whisper's own confidence metrics first (needed for conditional phrase detection)
    no_speech_prob = segment.get('no_speech_prob', 0)
    avg_logprob = segment.get('avg_logprob', 0)
    compression_ratio = segment.get('compression_ratio', 1.0)
    duration = segment.get('end', 0) - segment.get('start', 0)

    # Determine if Whisper metrics are suspicious
    has_bad_metrics = (
        no_speech_prob > 0.5 or  # Whisper somewhat uncertain if speech
        avg_logprob < -1.2 or    # Low confidence
        compression_ratio > 2.0   # Repetitive
    )

    # Check for always-suspicious phrases (rarely real speech)
    for pattern in _ALWAYS_SUSPICIOUS_PATTERNS:
        if pattern.match(text):
            reasons.append('known_hallucination_phrase')
            confidence = max(confidence, 0.85)
            break

    # Check for phrases that need corroborating bad metrics
    # (common in real speech, only flag if Whisper is uncertain)
    if has_bad_metrics:
        for pattern in _SUSPICIOUS_WITH_METRICS_PATTERNS:
            if pattern.match(text):
                reasons.append('suspicious_phrase_with_bad_metrics')
                confidence = max(confidence, 0.7)
                break

    # Check for foreign characters in supposedly English audio
    if any(c in FOREIGN_CHARS for c in text):
        reasons.append('foreign_characters')
        confidence = max(confidence, 0.85)

    # Flag high no_speech_prob on its own
    if no_speech_prob > 0.8:
        reasons.append('high_no_speech_prob')
        confidence = max(confidence, 0.7 + (no_speech_prob - 0.8) * 0.5)

    # Flag very low logprob
    if avg_logprob < -1.5:
        reasons.append('low_avg_logprob')
        confidence = max(confidence, 0.6)

    # Flag high compression ratio
    if compression_ratio > 2.5:
        reasons.append('high_compression_ratio')
        confidence = max(confidence, 0.7)

    # Check for very short duration with text
    if duration < 0.3 and len(text) > 10:
        reasons.append('too_much_text_for_duration')
        confidence = max(confidence, 0.6)
    
    return {
        'is_likely_hallucination': confidence > 0.6,
        'confidence': confidence,
        'reasons': reasons
    }


def detect_repetition_hallucinations(
    segments: List[Dict[str, Any]],
    threshold: int = REPETITION_THRESHOLD
) -> List[int]:
    """
    Detect sequences of repeated segments (common hallucination pattern).
    
    Args:
        segments: List of Whisper segments
        threshold: Number of repetitions to consider hallucination
    
    Returns:
        List of segment indices that are likely repetition hallucinations
    """
    if len(segments) < threshold:
        return []
    
    hallucination_indices = []
    
    # Normalize texts for comparison
    texts = [s.get('text', '').strip().lower() for s in segments]
    
    i = 0
    while i < len(texts):
        # Count consecutive repetitions
        current_text = texts[i]
        if not current_text or len(current_text) < 3:
            i += 1
            continue
        
        repeat_count = 1
        j = i + 1
        while j < len(texts) and texts[j] == current_text:
            repeat_count += 1
            j += 1
        
        if repeat_count >= threshold:
            # Mark all but the first as hallucinations
            for k in range(i + 1, j):
                hallucination_indices.append(k)
        
        i = j if j > i + 1 else i + 1
    
    return hallucination_indices


def filter_hallucinations(
    segments: List[Dict[str, Any]],
    remove: bool = False,
    flag_only: bool = True,
    repetition_threshold: int = REPETITION_THRESHOLD,
    confidence_threshold: float = 0.6
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Filter or flag hallucinations in a transcript.

    Args:
        segments: List of Whisper segments
        remove: If True, remove likely hallucinations entirely
        flag_only: If True, add 'hallucination_warning' field but keep segments
        repetition_threshold: Threshold for repetition detection
        confidence_threshold: Minimum confidence to flag as hallucination (0.3-0.9)

    Returns:
        Tuple of (filtered_segments, stats)
    """
    if not segments:
        return segments, {'total': 0, 'flagged': 0, 'removed': 0}

    # Detect repetition hallucinations
    repetition_indices = set(detect_repetition_hallucinations(segments, repetition_threshold))

    filtered = []
    stats = {
        'total': len(segments),
        'flagged': 0,
        'removed': 0,
        'reasons': Counter()
    }

    for i, segment in enumerate(segments):
        indicators = detect_hallucination_indicators(segment)

        # Also check if this is a repetition hallucination
        if i in repetition_indices:
            indicators['confidence'] = max(indicators['confidence'], 0.85)
            if 'repetition' not in indicators['reasons']:
                indicators['reasons'].append('repetition')

        # Apply configurable confidence threshold
        is_hallucination = indicators['confidence'] > confidence_threshold

        if is_hallucination:
            stats['flagged'] += 1
            for reason in indicators['reasons']:
                stats['reasons'][reason] += 1
            
            if remove:
                stats['removed'] += 1
                continue
            
            if flag_only:
                segment = segment.copy()
                segment['hallucination_warning'] = {
                    'confidence': indicators['confidence'],
                    'reasons': indicators['reasons']
                }
        
        filtered.append(segment)
    
    # Convert Counter to dict for JSON serialization
    stats['reasons'] = dict(stats['reasons'])
    
    return filtered, stats


def get_hallucination_summary(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get a summary of hallucination indicators in a transcript.
    
    Args:
        segments: List of Whisper segments
    
    Returns:
        Summary dict with counts and examples
    """
    flagged_segments = []
    
    for i, segment in enumerate(segments):
        indicators = detect_hallucination_indicators(segment)
        if indicators['is_likely_hallucination']:
            flagged_segments.append({
                'index': i,
                'text': segment.get('text', '')[:100],
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'confidence': indicators['confidence'],
                'reasons': indicators['reasons']
            })
    
    # Also check for repetitions
    repetition_indices = detect_repetition_hallucinations(segments)
    
    return {
        'total_segments': len(segments),
        'flagged_count': len(flagged_segments),
        'repetition_count': len(repetition_indices),
        'flagged_examples': flagged_segments[:10],  # First 10 examples
        'has_issues': len(flagged_segments) > 0 or len(repetition_indices) > 0
    }
