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

    def reorder_speakers_by_first_appearance(self) -> None:
        """
        Reorder speakers to match first appearance in the timeline.
        This avoids confusing speaker lists like "Speaker 10" appearing first.
        Colors are reassigned deterministically based on the new order.
        """
        first_seen: Dict[str, int] = {}
        for idx, chunk in enumerate(self.chunks):
            if chunk.speaker_id and chunk.speaker_id not in first_seen:
                first_seen[chunk.speaker_id] = idx

        def speaker_sort_key(s: Speaker) -> tuple[int, str]:
            if s.id in first_seen:
                return (0, f"{first_seen[s.id]:09d}")
            # Fallback: sort by numeric speaker suffix if present
            if s.id.startswith('SPEAKER_'):
                try:
                    return (1, f"{int(s.id.split('_')[1]):09d}")
                except ValueError:
                    pass
            return (2, s.label.lower())

        self.speakers.sort(key=speaker_sort_key)

        # Reassign colors based on new order for deterministic UI display.
        for idx, speaker in enumerate(self.speakers):
            speaker.color = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]

    def dedupe_chunks_strict(self) -> int:
        """
        Strict deduplication pass to remove duplicate/subset/overlapping chunks.
        Returns the number of chunks removed.
        
        Handles:
        1. Exact text duplicates at similar times
        2. Subset text (short text contained in longer text) at overlapping times
        3. Near-empty chunks (< 3 chars normalized)
        
        Keeps the "better" chunk: longer text > longer duration > earlier start.
        """
        import re
        
        def normalize_text(text: str) -> str:
            """Lowercase, strip, collapse whitespace."""
            if not text:
                return ''
            return re.sub(r'\s+', ' ', text.lower().strip())
        
        def is_subset_text(short: str, long: str) -> bool:
            """Check if short text is a subset of long text."""
            short_norm = normalize_text(short)
            long_norm = normalize_text(long)
            if len(short_norm) < 8:
                return False
            return short_norm == long_norm or short_norm in long_norm
        
        def calc_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
            """Calculate overlap ratio relative to shorter chunk."""
            overlap_seconds = max(0.0, min(a_end, b_end) - max(a_start, b_start))
            duration_a = max(1e-6, a_end - a_start)
            duration_b = max(1e-6, b_end - b_start)
            min_duration = min(duration_a, duration_b)
            return overlap_seconds / min_duration if min_duration > 0 else 0
        
        def chunk_score(chunk) -> tuple:
            """Score for choosing better chunk: (text_len, duration, -start)."""
            norm_len = len(normalize_text(chunk.text))
            duration = max(0.0, chunk.end - chunk.start)
            return (norm_len, duration, -chunk.start)
        
        if len(self.chunks) <= 1:
            return 0
        
        # Sort by (start, end) for deterministic processing
        self.chunks.sort(key=lambda c: (c.start, c.end))
        
        # Rolling window parameters
        WINDOW_TIME_SECONDS = 30.0
        WINDOW_MAX_CHUNKS = 12
        
        kept = []
        removed = 0
        
        for chunk in self.chunks:
            cur_norm = normalize_text(chunk.text)
            
            # Drop near-empty chunks
            if len(cur_norm) < 3:
                removed += 1
                continue
            
            if not kept:
                kept.append(chunk)
                continue
            
            # Build rolling window of recent candidates
            # Look at last N chunks where candidate.start >= chunk.start - WINDOW_TIME_SECONDS
            window_start_time = chunk.start - WINDOW_TIME_SECONDS
            recent_candidates = []
            for i in range(len(kept) - 1, max(-1, len(kept) - 1 - WINDOW_MAX_CHUNKS), -1):
                candidate = kept[i]
                if candidate.start >= window_start_time:
                    recent_candidates.append((i, candidate))
                else:
                    break  # Since kept is sorted by start, earlier chunks won't qualify
            
            # Check against each candidate in window, find best duplicate match
            best_dup_idx = None
            best_dup_score = None
            best_dup_candidate = None
            
            cur_len = len(cur_norm)
            
            for idx, candidate in recent_candidates:
                cand_norm = normalize_text(candidate.text)
                cand_len = len(cand_norm)
                
                # Cross-speaker guard: don't collapse real overlaps when different speakers have long text
                different_speakers = (chunk.speaker_id and candidate.speaker_id and chunk.speaker_id != candidate.speaker_id)
                long_text = min(cur_len, cand_len) >= 40
                
                if different_speakers and long_text:
                    continue
                
                # Calculate timing metrics
                start_delta = abs(chunk.start - candidate.start)
                overlap_ratio = calc_overlap_ratio(candidate.start, candidate.end, chunk.start, chunk.end)
                
                # Check for subset relationship (either direction)
                subset_relation = is_subset_text(cur_norm, cand_norm) or is_subset_text(cand_norm, cur_norm)
                
                # Exact match is always a duplicate if timing overlaps
                exact_match = cur_norm == cand_norm
                
                # Determine if likely duplicate
                # For subset-based dedupe, require minimum text length to avoid false positives on short phrases
                likely_duplicate = False
                if exact_match and (start_delta <= 1.0 or overlap_ratio >= 0.5):
                    likely_duplicate = True
                elif subset_relation and min(cur_len, cand_len) >= 20 and (start_delta <= 1.0 or overlap_ratio >= 0.6):
                    likely_duplicate = True
                
                if likely_duplicate:
                    # Score this duplicate candidate: prefer same speaker, exact match, higher overlap, closer start, longer text
                    same_speaker = 1 if (chunk.speaker_id and candidate.speaker_id and chunk.speaker_id == candidate.speaker_id) else 0
                    dup_score = (
                        same_speaker,
                        1 if exact_match else 0,
                        overlap_ratio,
                        -start_delta,
                        len(cand_norm)
                    )
                    
                    if best_dup_score is None or dup_score > best_dup_score:
                        best_dup_idx = idx
                        best_dup_score = dup_score
                        best_dup_candidate = candidate
            
            # After scanning all candidates, handle best duplicate if found
            if best_dup_idx is not None:
                # Keep the better chunk between current and best candidate
                cand_score = chunk_score(best_dup_candidate)
                cur_score = chunk_score(chunk)
                
                if cur_score > cand_score:
                    kept[best_dup_idx] = chunk
                
                removed += 1
            else:
                kept.append(chunk)
        
        self.chunks = kept
        return removed

    def postprocess_chunks(self) -> dict:
        """
        Post-processing pass to clean up chunks for human review.
        
        1) Drop contained fragments (small chunks fully inside larger ones with subset text)
        2) Merge adjacent same-speaker chunks with small gaps
        
        Returns stats dict and stores on self.postprocess_stats.
        """
        import re
        
        def normalize_text(text: str) -> str:
            if not text:
                return ''
            return re.sub(r'\s+', ' ', text.lower().strip())
        
        def duration(c) -> float:
            return max(0.0, c.end - c.start)
        
        def overlap_seconds(a, b) -> float:
            return max(0.0, min(a.end, b.end) - max(a.start, b.start))
        
        def overlap_ratio(a, b) -> float:
            ov = overlap_seconds(a, b)
            min_dur = min(duration(a), duration(b))
            return ov / min_dur if min_dur > 1e-6 else 0.0
        
        def is_contained(inner, outer) -> bool:
            return inner.start >= outer.start - 0.15 and inner.end <= outer.end + 0.15
        
        def text_is_subset(short_text: str, long_text: str) -> bool:
            short_norm = normalize_text(short_text)
            long_norm = normalize_text(long_text)
            if len(short_norm) < 12:
                return False
            return short_norm == long_norm or short_norm in long_norm
        
        before_count = len(self.chunks)
        
        if before_count <= 1:
            self.postprocess_stats = {
                'before': before_count,
                'afterContainmentDrop': before_count,
                'afterMerge': before_count,
                'droppedContained': 0,
                'merged': 0
            }
            return self.postprocess_stats
        
        # Sort by (start, end) for deterministic processing
        self.chunks.sort(key=lambda c: (c.start, c.end))
        
        # ========== PASS 1: Drop contained fragments ==========
        WINDOW_TIME = 40.0
        WINDOW_MAX = 20
        
        kept_after_drop = []
        dropped_contained = 0
        
        for chunk in self.chunks:
            chunk_norm = normalize_text(chunk.text)
            
            # Skip near-empty
            if len(chunk_norm) < 3:
                dropped_contained += 1
                continue
            
            if not kept_after_drop:
                kept_after_drop.append(chunk)
                continue
            
            # Build window of recent candidates
            window_start_time = chunk.start - WINDOW_TIME
            candidates = []
            for i in range(len(kept_after_drop) - 1, max(-1, len(kept_after_drop) - 1 - WINDOW_MAX), -1):
                cand = kept_after_drop[i]
                if cand.start >= window_start_time:
                    candidates.append((i, cand))
                else:
                    break
            
            # Check if chunk should be dropped as contained fragment
            should_drop = False
            for idx, cand in candidates:
                # Only drop if candidate is at least as long as chunk
                if duration(cand) < duration(chunk):
                    continue
                
                # Check containment
                if not is_contained(chunk, cand):
                    continue
                
                # Check text subset
                if not text_is_subset(chunk.text, cand.text):
                    continue
                
                # Same speaker OR very high overlap
                same_speaker = chunk.speaker_id and cand.speaker_id and chunk.speaker_id == cand.speaker_id
                high_overlap = overlap_ratio(chunk, cand) >= 0.9
                
                if same_speaker or high_overlap:
                    should_drop = True
                    break
            
            if should_drop:
                dropped_contained += 1
            else:
                kept_after_drop.append(chunk)
        
        after_containment = len(kept_after_drop)
        
        # ========== PASS 2: Merge adjacent same-speaker chunks ==========
        if len(kept_after_drop) <= 1:
            self.chunks = kept_after_drop
            self.postprocess_stats = {
                'before': before_count,
                'afterContainmentDrop': after_containment,
                'afterMerge': after_containment,
                'droppedContained': dropped_contained,
                'merged': 0
            }
            return self.postprocess_stats
        
        # Sort again to ensure order
        kept_after_drop.sort(key=lambda c: (c.start, c.end))
        
        merged_chunks = [kept_after_drop[0]]
        merge_count = 0
        
        for i in range(1, len(kept_after_drop)):
            cur = merged_chunks[-1]
            next_chunk = kept_after_drop[i]
            
            # Check merge conditions
            same_speaker = cur.speaker_id and next_chunk.speaker_id and cur.speaker_id == next_chunk.speaker_id
            if not same_speaker:
                merged_chunks.append(next_chunk)
                continue
            
            gap = next_chunk.start - cur.end
            if gap > 1.25:
                merged_chunks.append(next_chunk)
                continue
            
            combined_duration = next_chunk.end - cur.start
            if combined_duration > 45.0:
                merged_chunks.append(next_chunk)
                continue
            
            cur_norm = normalize_text(cur.text)
            next_norm = normalize_text(next_chunk.text)
            combined_text_len = len(cur_norm) + len(next_norm) + 1
            if combined_text_len > 600:
                merged_chunks.append(next_chunk)
                continue
            
            # Check hard boundary (sentence end + gap)
            cur_text_stripped = cur.text.rstrip() if cur.text else ''
            ends_with_punct = cur_text_stripped and cur_text_stripped[-1] in '.?!'
            if ends_with_punct and gap > 0.4:
                merged_chunks.append(next_chunk)
                continue
            
            # Merge!
            cur.end = next_chunk.end
            cur.text = (cur.text.rstrip() + ' ' + next_chunk.text.lstrip()).strip()
            
            # Track merged chunk IDs in origin
            if not hasattr(cur, 'origin') or cur.origin is None:
                cur.origin = {}
            merged_ids = cur.origin.get('mergedChunkIds', [cur.chunk_id])
            if next_chunk.chunk_id not in merged_ids:
                merged_ids.append(next_chunk.chunk_id)
            cur.origin['mergedChunkIds'] = merged_ids
            
            merge_count += 1
        
        self.chunks = merged_chunks
        
        self.postprocess_stats = {
            'before': before_count,
            'afterContainmentDrop': after_containment,
            'afterMerge': len(merged_chunks),
            'droppedContained': dropped_contained,
            'merged': merge_count
        }
        return self.postprocess_stats


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
        
        # Priority: transcript as structure with diarization for speakers
        if transcript_json and diarization_json:
            self._parse_transcript_with_diarization(timeline, transcript_json, diarization_json)
        elif diarization_json:
            self._parse_diarization_json(timeline, diarization_json, transcript_json)
        elif transcript_json:
            self._parse_transcript_json(timeline, transcript_json)
        elif speaker_md:
            self._parse_speaker_md(timeline, speaker_md)
        elif transcript_md:
            self._parse_transcript_md(timeline, transcript_md)

        # Normalize speaker ordering for a more intuitive review experience.
        timeline.reorder_speakers_by_first_appearance()
        
        # Final strict dedupe pass to remove duplicate/subset/overlapping chunks
        before_dedupe = len(timeline.chunks)
        dedupe_removed = timeline.dedupe_chunks_strict()
        after_dedupe = len(timeline.chunks)
        
        # Post-processing pass: drop contained fragments, merge same-speaker chunks
        postprocess_stats = timeline.postprocess_chunks()
        
        timeline.dedupe_stats = {
            'before': before_dedupe,
            'afterDedupe': after_dedupe,
            'dedupeRemoved': dedupe_removed,
            'postprocess': postprocess_stats
        }
        
        return timeline
    
    def _parse_transcript_with_diarization(self, timeline: ReviewTimeline,
                                            transcript_json: str,
                                            diarization_json: str):
        """
        Parse transcript JSON as canonical structure, using diarization only for speaker assignment.
        This reduces chattery speaker flips from diarization by using transcript segments as structure.
        """
        # A) Parse transcript segments
        try:
            transcript_data = json.loads(transcript_json)
        except json.JSONDecodeError:
            return
        
        if isinstance(transcript_data, list):
            transcript_segments = transcript_data
        elif isinstance(transcript_data, dict):
            transcript_segments = transcript_data.get('segments', [])
        else:
            transcript_segments = []
        
        if not transcript_segments:
            return
        
        # B) Parse diarization segments
        try:
            diar_data = json.loads(diarization_json)
        except json.JSONDecodeError:
            diar_data = []
        
        if isinstance(diar_data, list):
            diar_segments_raw = diar_data
        elif isinstance(diar_data, dict):
            diar_segments_raw = diar_data.get('segments', [])
        else:
            diar_segments_raw = []
        
        # C) Normalize diarization segments
        diar_segments = []
        for ds in diar_segments_raw:
            speaker = ds.get('speaker')
            if not speaker:
                continue
            start = float(ds.get('start', 0) or 0)
            end = float(ds.get('end', 0) or 0)
            if end <= start:
                continue
            diar_segments.append({'start': start, 'end': end, 'speaker': speaker})
        
        # D) Assign speaker to each transcript segment by overlap
        labeled_segments = []
        for ts in transcript_segments:
            ts_start = float(ts.get('start', 0) or 0)
            ts_end = float(ts.get('end', ts_start) or ts_start)
            if ts_end <= ts_start:
                continue
            
            text = (ts.get('text') or '').strip()
            segment_id = ts.get('id')
            
            # Compute overlap per speaker
            speaker_overlap = {}
            for ds in diar_segments:
                overlap = max(0.0, min(ts_end, ds['end']) - max(ts_start, ds['start']))
                if overlap > 0:
                    speaker_overlap[ds['speaker']] = speaker_overlap.get(ds['speaker'], 0.0) + overlap
            
            # Choose speaker with max overlap
            if speaker_overlap:
                speaker_id = max(speaker_overlap, key=speaker_overlap.get)
            else:
                speaker_id = 'SPEAKER_00'
            
            timeline.add_speaker(speaker_id)
            
            labeled_segments.append({
                'start': ts_start,
                'end': ts_end,
                'speaker_id': speaker_id,
                'text': text,
                'segment_id': segment_id
            })
        
        # E) Speaker smoothing pass - reduce rapid speaker flips on short segments
        for i, seg in enumerate(labeled_segments):
            dur = seg['end'] - seg['start']
            
            # Rule 1: Short segment (<=1.25s) with same-speaker neighbors
            if dur <= 1.25 and i > 0 and i < len(labeled_segments) - 1:
                prev_seg = labeled_segments[i - 1]
                next_seg = labeled_segments[i + 1]
                prev_dur = prev_seg['end'] - prev_seg['start']
                next_dur = next_seg['end'] - next_seg['start']
                
                if (prev_seg['speaker_id'] == next_seg['speaker_id'] and
                    seg['speaker_id'] != prev_seg['speaker_id'] and
                    (prev_dur >= 2.0 or next_dur >= 2.0)):
                    seg['speaker_id'] = prev_seg['speaker_id']
                    continue
            
            # Rule 2: Very short segment (<=0.8s) differs from both neighbors who agree
            if dur <= 0.8 and i > 0 and i < len(labeled_segments) - 1:
                prev_seg = labeled_segments[i - 1]
                next_seg = labeled_segments[i + 1]
                
                if (prev_seg['speaker_id'] == next_seg['speaker_id'] and
                    seg['speaker_id'] != prev_seg['speaker_id']):
                    seg['speaker_id'] = prev_seg['speaker_id']
        
        # F) Build chunks from labeled transcript segments
        MAX_CHUNK_SECONDS = 45
        MAX_CHUNK_CHARS = 600
        MAX_GAP_SECONDS = 1.25
        
        if not labeled_segments:
            return
        
        chunks = []
        current = {
            'start': labeled_segments[0]['start'],
            'end': labeled_segments[0]['end'],
            'speaker_id': labeled_segments[0]['speaker_id'],
            'text': labeled_segments[0]['text'],
            'segment_ids': [labeled_segments[0]['segment_id']]
        }
        
        for i in range(1, len(labeled_segments)):
            seg = labeled_segments[i]
            gap = seg['start'] - current['end']
            combined_duration = seg['end'] - current['start']
            combined_text_len = len(current['text']) + len(seg['text']) + 1
            
            # Check hard boundary (sentence end + gap)
            cur_text_stripped = current['text'].rstrip()
            ends_with_punct = cur_text_stripped and cur_text_stripped[-1] in '.?!'
            hard_boundary = ends_with_punct and gap > 0.4
            
            # Merge conditions
            should_merge = (
                seg['speaker_id'] == current['speaker_id'] and
                gap <= MAX_GAP_SECONDS and
                combined_duration <= MAX_CHUNK_SECONDS and
                combined_text_len <= MAX_CHUNK_CHARS and
                not hard_boundary
            )
            
            if should_merge:
                current['end'] = seg['end']
                current['text'] = (current['text'].rstrip() + ' ' + seg['text'].lstrip()).strip()
                if seg['segment_id'] is not None:
                    current['segment_ids'].append(seg['segment_id'])
            else:
                chunks.append(current)
                current = {
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker_id': seg['speaker_id'],
                    'text': seg['text'],
                    'segment_ids': [seg['segment_id']] if seg['segment_id'] is not None else []
                }
        
        # Don't forget the last chunk
        chunks.append(current)
        
        # G) Create Chunk objects
        for idx, c in enumerate(chunks):
            chunk = Chunk(
                chunk_id=f't_{idx:06d}',
                start=c['start'],
                end=c['end'],
                speaker_id=c['speaker_id'],
                text=c['text'],
                origin={
                    'transcriptSegmentIds': c['segment_ids'],
                    'speakerAssignedBy': 'diarization_overlap',
                    'speakerSmoothing': True
                }
            )
            timeline.chunks.append(chunk)
    
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

        # Normalize and deduplicate near-identical overlapping segments.
        # Some diarization outputs include duplicate segments at the same
        # timestamps with different speaker assignments.
        def _segment_key(seg: dict) -> tuple[float, float]:
            return (float(seg.get('start', 0) or 0), float(seg.get('end', 0) or 0))

        segments = sorted(segments, key=_segment_key)
        deduped_segments = []
        epsilon = 0.25
        for seg in segments:
            if not deduped_segments:
                deduped_segments.append(seg)
                continue

            prev = deduped_segments[-1]
            prev_start = float(prev.get('start', 0) or 0)
            prev_end = float(prev.get('end', 0) or 0)
            cur_start = float(seg.get('start', 0) or 0)
            cur_end = float(seg.get('end', 0) or 0)

            near_same_time = abs(cur_start - prev_start) <= epsilon and abs(cur_end - prev_end) <= epsilon

            prev_text = (prev.get('text') or '').strip()
            cur_text = (seg.get('text') or '').strip()

            overlap = max(0.0, min(cur_end, prev_end) - max(cur_start, prev_start))
            prev_duration = max(0.0, prev_end - prev_start)
            cur_duration = max(0.0, cur_end - cur_start)
            min_duration = max(1e-6, min(prev_duration, cur_duration))
            overlap_ratio = overlap / min_duration
            same_text = bool(prev_text) and prev_text.lower() == cur_text.lower()

            likely_duplicate = near_same_time or (overlap_ratio >= 0.8 and same_text)

            if not likely_duplicate:
                deduped_segments.append(seg)
                continue

            # Prefer segments with text; otherwise prefer longer duration.
            prev_score = (1 if prev_text else 0, len(prev_text), prev_duration)
            cur_score = (1 if cur_text else 0, len(cur_text), cur_duration)
            if cur_score > prev_score:
                deduped_segments[-1] = seg

        segments = deduped_segments
        
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
            
            # Get text. When transcript segments are available, always derive
            # text from the transcript and use diarization only for structure.
            text = ''
            if transcript_segments:
                text = self._get_overlapping_text(
                    seg.get('start', 0), 
                    seg.get('end', 0),
                    transcript_segments
                )
            else:
                text = seg.get('text', '')
            
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

        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict):
            segments = data.get('segments', [])
        else:
            segments = []
        
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
