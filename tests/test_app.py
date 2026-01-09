#!/usr/bin/env python3
"""
Basic tests for Bulk Transcribe app.
Run with: ./venv/bin/python -m pytest tests/ -v
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAudioFileDetection:
    """Test audio file detection functionality."""
    
    def test_find_audio_files_empty_folder(self):
        from app import find_audio_files
        with tempfile.TemporaryDirectory() as tmpdir:
            files = find_audio_files(tmpdir)
            assert files == []
    
    def test_find_audio_files_with_supported_formats(self):
        from app import find_audio_files, SUPPORTED_FORMATS
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy files with supported extensions
            for ext in ['.mp3', '.wav', '.m4a']:
                Path(tmpdir, f'test{ext}').touch()
            # Create unsupported file
            Path(tmpdir, 'test.txt').touch()
            
            files = find_audio_files(tmpdir)
            assert len(files) == 3
            assert all(f.suffix in SUPPORTED_FORMATS for f in files)
    
    def test_find_audio_files_nonexistent_folder(self):
        from app import find_audio_files
        files = find_audio_files('/nonexistent/path/12345')
        assert files == []


class TestModelManagement:
    """Test model availability checking."""
    
    def test_check_model_available_returns_bool(self):
        from app import check_model_available
        result = check_model_available('tiny')
        assert isinstance(result, bool)
    
    def test_get_model_filepath_returns_path(self):
        from app import get_model_filepath
        path = get_model_filepath('turbo')
        assert isinstance(path, str)
        assert 'whisper' in path.lower() or '.cache' in path


class TestTranscriptionStatus:
    """Test transcription status structure."""
    
    def test_status_has_required_fields(self):
        from app import transcription_status
        required_fields = [
            'running', 'cancelled', 'current_file', 'completed', 
            'total', 'results', 'error', 'active_jobs', 'active_workers'
        ]
        for field in required_fields:
            assert field in transcription_status, f"Missing field: {field}"
    
    def test_status_initial_state(self):
        from app import transcription_status
        assert transcription_status['running'] == False
        assert transcription_status['cancelled'] == False
        assert transcription_status['completed'] == 0


class TestPreferences:
    """Test preferences loading/saving."""
    
    def test_load_preferences_returns_dict(self):
        from app import load_preferences
        prefs = load_preferences()
        assert isinstance(prefs, dict)
    
    def test_preferences_has_defaults(self):
        from app import load_preferences
        prefs = load_preferences()
        assert 'include_segments' in prefs
        assert 'include_timestamps' in prefs


class TestMarkdownGeneration:
    """Test markdown output generation."""
    
    def test_format_timestamp(self):
        from app import format_timestamp
        # Format is MM:SS.ss or HH:MM:SS.ss
        assert format_timestamp(0) == '00:00.00'
        assert format_timestamp(61) == '01:01.00'
        assert format_timestamp(3661) == '01:01:01.00'
    
    def test_format_timestamp_with_decimals(self):
        from app import format_timestamp
        result = format_timestamp(61.5)
        assert result == '01:01.50'


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
