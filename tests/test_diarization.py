"""
Tests for diarization functionality.

Run with: pytest tests/test_diarization.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from diarization import format_speaker_markdown, format_diarization_json


class TestFormatSpeakerMarkdown:
    """Test speaker markdown generation, including edge cases."""
    
    def test_empty_segments_no_transcript(self):
        """Regression test: empty speaker segments with no transcript should not crash."""
        result = format_speaker_markdown([], "test.wav", transcript_segments=[])
        assert result is not None
        assert "test.wav" in result
        assert "No speaker segments" in result or "no speaker segments" in result.lower()
    
    def test_empty_segments_with_transcript(self):
        """Empty speaker segments but transcript present should show transcript."""
        transcript = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.0, "end": 4.0, "text": "This is a test"}
        ]
        result = format_speaker_markdown([], "test.wav", transcript_segments=transcript)
        assert result is not None
        assert "test.wav" in result
        assert "Hello world" in result
        assert "This is a test" in result
    
    def test_normal_segments(self):
        """Normal case with speaker segments."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello"},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 4.0, "text": "Hi there"}
        ]
        result = format_speaker_markdown(segments, "test.wav")
        assert result is not None
        assert "SPEAKER_00" in result or "Speaker 1" in result
        assert "Hello" in result
        assert "Hi there" in result
    
    def test_none_segments_treated_as_empty(self):
        """None segments should be handled gracefully."""
        result = format_speaker_markdown(None, "test.wav", transcript_segments=[])
        assert result is not None
        assert "test.wav" in result


class TestFormatDiarizationJson:
    """Test diarization JSON generation."""
    
    def test_empty_segments(self):
        """Empty segments should produce valid JSON structure."""
        result = format_diarization_json([], [], "test.wav")
        assert result is not None
        assert "source" in result
        assert result["source"] == "test.wav"
        assert "rawDiarization" in result
        assert "mergedTranscript" in result
        assert "speakers" in result
    
    def test_normal_segments(self):
        """Normal case with segments."""
        speaker_segs = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}
        ]
        merged_segs = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello"}
        ]
        result = format_diarization_json(speaker_segs, merged_segs, "test.wav")
        assert len(result["rawDiarization"]) == 1
        assert len(result["mergedTranscript"]) == 1
        assert result["speakers"] == ["SPEAKER_00"]
