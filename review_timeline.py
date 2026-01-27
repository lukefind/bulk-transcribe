"""
ReviewTimeline: Canonical model for transcript review with speaker assignment.

Parses transcript and diarization outputs into a unified timeline of chunks
that can be displayed, edited, and exported.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


# Speaker colors for UI display (12 distinct colors)
SPEAKER_COLORS = [
    '#3B82F6',  # blue
    '#EF4444',  # red
    '#10B981',  # green
    '#F59E0B',  # amber
    '#8B5CF6',  # violet
    '#EC4899',  # pink
    '#06B6D4',  # cyan
    '#F97316',  # orange
    '#6366F1',  # indigo
    '#14B8A6',  # teal
    '#A855F7',  # purple
    '#84CC16',  # lime
]


@dataclass
class Speaker:
    """A speaker detected in the transcript."""
    id: str
    label: str
    color: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    """A chunk of transcript text with timing and speaker info."""
    chunk_id: str
    start: float
    end: float
    speaker_id: Optional[str]
    text: str
    confidence: Optional[float] = None
    origin: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewTimeline:
    """Complete timeline for review, derived from transcript/diarization outputs."""
    version: int = 1
    source: Dict[str, str] = field(default_factory=dict)
    speakers: List[Speaker] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'source': self.source,
            'speakers': [s.to_dict() for s in self.speakers],
            'chunks': [c.to_dict() for c in self.chunks],
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def get_speaker(self, speaker_id: str) -> Optional[Speaker]:
        for s in self.speakers:
            if s.id == speaker_id:
                return s
        return None
    
    def add_speaker(self, speaker_id: str, label: Optional[str] = None) -> Speaker:
        """Add a speaker if not already present, return the speaker."""
        existing = self.get_speaker(speaker_id)
        if existing:
            return existing
        
        idx = len(self.speakers)
        color = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        if label is None:
            # Convert SPEAKER_00 to "Speaker 1"
            if speaker_id.startswith('SPEAKER_'):
                num = int(speaker_id.split('_')[1]) + 1
                label = f'Speaker {num}'
            else:
                label = speaker_id
        
        speaker = Speaker(id=speaker_id, label=label, color=color)
        self.speakers.append(speaker)
        return speaker


class TimelineParser:
    """Parses various transcript formats into a ReviewTimeline."""
    
    def __init__(self, job_id: str, input_id: str, filename: str):
        self.job_id = job_id
        self.input_id = input_id
        self.filename = filename
    
    def parse(self, 
              transcript_json: Optional[str] = None,
              diarization_json: Optional[str] = None,
              speaker_md: Optional[str] = None,
              transcript_md: Optional[str] = None) -> ReviewTimeline:
        """
        Parse available outputs into a ReviewTimeline.
        
        Priority:
        1. If diarization_json exists, use it for speaker segments
        2. Merge with transcript_json for text if available
        3. Fall back to speaker_md or transcript_md parsing
        """
        timeline = ReviewTimeline(
            source={
                'jobId': self.job_id,
                'inputId': self.input_id,
                'filename': self.filename,
            }
        )
        
        # Try diarization JSON first (has speaker info)
        if diarization_json:
            self._parse_diarization_json(timeline, diarization_json, transcript_json)
        elif transcript_json:
            self._parse_transcript_json(timeline, transcript_json)
        elif speaker_md:
            self._parse_speaker_md(timeline, speaker_md)
        elif transcript_md:
            self._parse_transcript_md(timeline, transcript_md)
        
        return timeline
    
    def _parse_diarization_json(self, timeline: ReviewTimeline, 
                                 diarization_json: str,
                                 transcript_json: Optional[str] = None):
        """Parse diarization JSON with speaker segments."""
        try:
            data = json.loads(diarization_json)
        except json.JSONDecodeError:
            return

        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict):
            segments = data.get('segments', [])
        else:
            segments = []
        
        # Also parse transcript JSON for better text if available
        transcript_segments = []
        if transcript_json:
            try:
                t_data = json.loads(transcript_json)
                if isinstance(t_data, list):
                    transcript_segments = t_data
                elif isinstance(t_data, dict):
                    transcript_segments = t_data.get('segments', [])
            except json.JSONDecodeError:
                pass
        
        # Build chunks from diarization segments
        for idx, seg in enumerate(segments):
            speaker_id = seg.get('speaker')
            if speaker_id:
                timeline.add_speaker(speaker_id)
            
            # Get text - prefer from diarization, fall back to transcript overlap
            text = seg.get('text', '')
            if not text and transcript_segments:
                text = self._get_overlapping_text(
                    seg.get('start', 0), 
                    seg.get('end', 0),
                    transcript_segments
                )
            
            chunk = Chunk(
                chunk_id=f't_{idx:06d}',
                start=seg.get('start', 0),
                end=seg.get('end', 0),
                speaker_id=speaker_id,
                text=text.strip(),
                origin={'diarization_segment_idx': idx}
            )
            timeline.chunks.append(chunk)
    
    def _parse_transcript_json(self, timeline: ReviewTimeline, transcript_json: str):
        """Parse Whisper transcript JSON (no speaker info)."""
        try:
            data = json.loads(transcript_json)
        except json.JSONDecodeError:
            return
        
        segments = data.get('segments', [])
        
        # Add default speaker for non-diarized transcripts
        default_speaker = timeline.add_speaker('SPEAKER_00', 'Speaker 1')
        
        # Group segments into reasonable chunks (aim for 1-3 sentences)
        chunks = self._group_segments_into_chunks(segments)
        
        for idx, chunk_data in enumerate(chunks):
            chunk = Chunk(
                chunk_id=f't_{idx:06d}',
                start=chunk_data['start'],
                end=chunk_data['end'],
                speaker_id=default_speaker.id,
                text=chunk_data['text'].strip(),
                origin={'transcript_segment_ids': chunk_data['segment_ids']}
            )
            timeline.chunks.append(chunk)
    
    def _parse_speaker_md(self, timeline: ReviewTimeline, speaker_md: str):
        """Parse speaker markdown format: [00:00:12] Speaker 1: text"""
        # Pattern: [MM:SS] or [HH:MM:SS] followed by optional speaker and text
        pattern = r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*(?:(SPEAKER_\d+|Speaker\s*\d+)\s*:\s*)?(.*)'
        
        for idx, match in enumerate(re.finditer(pattern, speaker_md, re.MULTILINE)):
            hours = int(match.group(1)) if match.group(3) else 0
            mins = int(match.group(1)) if not match.group(3) else int(match.group(2))
            secs = int(match.group(3)) if match.group(3) else int(match.group(2))
            
            start = hours * 3600 + mins * 60 + secs
            
            speaker_raw = match.group(4)
            text = match.group(5).strip()
            
            # Normalize speaker ID
            speaker_id = None
            if speaker_raw:
                if speaker_raw.startswith('Speaker'):
                    num = int(re.sub(r'\D', '', speaker_raw)) - 1
                    speaker_id = f'SPEAKER_{num:02d}'
                else:
                    speaker_id = speaker_raw
                timeline.add_speaker(speaker_id)
            
            chunk = Chunk(
                chunk_id=f't_{idx:06d}',
                start=start,
                end=start,  # End time not available in MD format
                speaker_id=speaker_id,
                text=text,
                origin={'source': 'speaker_md', 'line': idx}
            )
            timeline.chunks.append(chunk)
    
    def _parse_transcript_md(self, timeline: ReviewTimeline, transcript_md: str):
        """Parse transcript markdown format: [00:00:12] text"""
        pattern = r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*(.*)'
        
        default_speaker = timeline.add_speaker('SPEAKER_00', 'Speaker 1')
        
        for idx, match in enumerate(re.finditer(pattern, transcript_md, re.MULTILINE)):
            hours = int(match.group(1)) if match.group(3) else 0
            mins = int(match.group(1)) if not match.group(3) else int(match.group(2))
            secs = int(match.group(3)) if match.group(3) else int(match.group(2))
            
            start = hours * 3600 + mins * 60 + secs
            text = match.group(4).strip()
            
            chunk = Chunk(
                chunk_id=f't_{idx:06d}',
                start=start,
                end=start,
                speaker_id=default_speaker.id,
                text=text,
                origin={'source': 'transcript_md', 'line': idx}
            )
            timeline.chunks.append(chunk)
    
    def _get_overlapping_text(self, start: float, end: float, 
                               transcript_segments: List[dict]) -> str:
        """Get text from transcript segments that overlap with the time range."""
        texts = []
        for seg in transcript_segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Check for overlap
            if seg_start < end and seg_end > start:
                texts.append(seg.get('text', ''))
        
        return ' '.join(texts)
    
    def _group_segments_into_chunks(self, segments: List[dict], 
                                     max_duration: float = 30.0,
                                     max_chars: int = 500) -> List[dict]:
        """
        Group transcript segments into reasonable chunks.
        
        Heuristics:
        - Don't exceed max_duration seconds
        - Don't exceed max_chars characters
        - Try to break at sentence boundaries
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk = {
            'start': segments[0].get('start', 0),
            'end': segments[0].get('end', 0),
            'text': '',
            'segment_ids': []
        }
        
        for idx, seg in enumerate(segments):
            seg_text = seg.get('text', '').strip()
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Check if we should start a new chunk
            duration = seg_end - current_chunk['start']
            new_length = len(current_chunk['text']) + len(seg_text)
            
            should_break = (
                duration > max_duration or
                new_length > max_chars or
                (current_chunk['text'] and 
                 current_chunk['text'].rstrip()[-1] in '.!?' and
                 duration > 10)  # Break at sentence if > 10s
            )
            
            if should_break and current_chunk['text']:
                chunks.append(current_chunk)
                current_chunk = {
                    'start': seg_start,
                    'end': seg_end,
                    'text': seg_text,
                    'segment_ids': [idx]
                }
            else:
                current_chunk['end'] = seg_end
                current_chunk['text'] += ' ' + seg_text if current_chunk['text'] else seg_text
                current_chunk['segment_ids'].append(idx)
        
        # Don't forget the last chunk
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        return chunks


def apply_review_state(timeline: ReviewTimeline, review_state: dict) -> ReviewTimeline:
    """
    Apply saved review state (speaker relabels, chunk edits) to a timeline.
    
    review_state format:
    {
        "speakerLabelMap": {"SPEAKER_00": "Matt", "SPEAKER_01": "Host"},
        "chunkEdits": {"t_000001": {"speakerId": "SPEAKER_02"}, ...},
        "uiPrefs": {...}
    }
    """
    # Apply speaker label renames
    label_map = review_state.get('speakerLabelMap', {})
    for speaker in timeline.speakers:
        if speaker.id in label_map:
            speaker.label = label_map[speaker.id]
    
    # Apply chunk edits
    chunk_edits = review_state.get('chunkEdits', {})
    for chunk in timeline.chunks:
        if chunk.chunk_id in chunk_edits:
            edit = chunk_edits[chunk.chunk_id]
            if 'speakerId' in edit:
                chunk.speaker_id = edit['speakerId']
                # Ensure speaker exists
                if chunk.speaker_id:
                    timeline.add_speaker(chunk.speaker_id)
            if 'text' in edit:
                chunk.text = edit['text']
    
    return timeline
