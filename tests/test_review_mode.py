"""
Tests for review mode endpoints: audio streaming, text preview, speaker labels.

Run with: pytest tests/test_review_mode.py -v
"""

import os
import sys
import tempfile
import shutil
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(scope='module')
def test_data_root():
    """Create a temporary data root for testing."""
    tmpdir = tempfile.mkdtemp(prefix='bulk_transcribe_review_test_')
    os.environ['DATA_ROOT'] = tmpdir
    os.environ['APP_MODE'] = 'server'
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def app_client(test_data_root):
    """Create Flask test client."""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def session_with_job(test_data_root, app_client):
    """Create a session with a completed job that has outputs."""
    import re
    import session_store
    
    # Get session ID
    response = app_client.get('/')
    set_cookie = response.headers.get('Set-Cookie', '')
    match = re.search(r'bt_session=([^;]+)', set_cookie)
    session_id = match.group(1) if match else None
    
    # Create job directories
    job_id = 'test_review_job'
    dirs = session_store.ensure_job_dirs(session_id, job_id)
    
    # Create a fake audio file in uploads
    uploads_dir = session_store.uploads_dir(session_id)
    os.makedirs(uploads_dir, exist_ok=True)
    audio_path = os.path.join(uploads_dir, 'upload123_test.wav')
    with open(audio_path, 'wb') as f:
        # Write minimal WAV header
        f.write(b'RIFF')
        f.write((36).to_bytes(4, 'little'))  # file size - 8
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))  # chunk size
        f.write((1).to_bytes(2, 'little'))   # audio format (PCM)
        f.write((1).to_bytes(2, 'little'))   # num channels
        f.write((44100).to_bytes(4, 'little'))  # sample rate
        f.write((88200).to_bytes(4, 'little'))  # byte rate
        f.write((2).to_bytes(2, 'little'))   # block align
        f.write((16).to_bytes(2, 'little'))  # bits per sample
        f.write(b'data')
        f.write((0).to_bytes(4, 'little'))   # data size
    
    # Create output files
    outputs_dir = dirs['outputs']
    
    # Create speaker markdown output
    speaker_md_path = os.path.join(outputs_dir, 'test.speaker.md')
    with open(speaker_md_path, 'w') as f:
        f.write('[00:00] Speaker 1: Hello world\n')
        f.write('[00:05] Speaker 2: Hi there\n')
    
    # Create transcript markdown output
    transcript_md_path = os.path.join(outputs_dir, 'test.transcript.md')
    with open(transcript_md_path, 'w') as f:
        f.write('[00:00] Hello world\n')
        f.write('[00:05] Hi there\n')
    
    # Create diarization JSON output
    diarization_json_path = os.path.join(outputs_dir, 'test.diarization.json')
    with open(diarization_json_path, 'w') as f:
        json.dump({
            'segments': [
                {'start': 0, 'end': 5, 'speaker': 'SPEAKER_00', 'text': 'Hello world'},
                {'start': 5, 'end': 10, 'speaker': 'SPEAKER_01', 'text': 'Hi there'}
            ]
        }, f)
    
    # Create a binary file (should not be previewable)
    binary_path = os.path.join(outputs_dir, 'test.bin')
    with open(binary_path, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    
    # Create manifest
    manifest = {
        'jobId': job_id,
        'sessionId': session_id,
        'status': 'complete',
        'backend': 'cpu',
        'options': {'model': 'tiny'},
        'inputs': [{
            'uploadId': 'upload123',
            'path': audio_path,
            'originalFilename': 'test.wav',
            'durationSec': 10.0
        }],
        'outputs': [
            {'id': 'out1', 'filename': 'test.speaker.md', 'type': 'speaker_markdown', 'path': speaker_md_path},
            {'id': 'out2', 'filename': 'test.transcript.md', 'type': 'transcript_markdown', 'path': transcript_md_path},
            {'id': 'out3', 'filename': 'test.diarization.json', 'type': 'diarization_json', 'path': diarization_json_path},
            {'id': 'out4', 'filename': 'test.bin', 'type': 'binary', 'path': binary_path}
        ]
    }
    session_store.atomic_write_json(
        session_store.job_manifest_path(session_id, job_id),
        manifest
    )
    
    return {
        'session_id': session_id,
        'job_id': job_id,
        'audio_path': audio_path,
        'manifest': manifest
    }


class TestAudioEndpoint:
    """Test audio streaming endpoint."""
    
    def test_audio_endpoint_requires_manifest_input(self, app_client, session_with_job):
        """Audio endpoint should only serve files listed in manifest inputs."""
        job_id = session_with_job['job_id']
        
        # Valid input ID should work
        response = app_client.get(f'/api/jobs/{job_id}/audio/upload123')
        assert response.status_code == 200
        assert response.headers.get('Accept-Ranges') == 'bytes'
        
        # Invalid input ID should fail
        response = app_client.get(f'/api/jobs/{job_id}/audio/nonexistent')
        assert response.status_code == 404
    
    def test_audio_endpoint_session_isolation(self, test_data_root, session_with_job):
        """Audio endpoint should not allow access from different session."""
        from app import app
        import session_store
        
        job_id = session_with_job['job_id']
        
        # Create a new client (different session)
        app.config['TESTING'] = True
        with app.test_client() as other_client:
            other_client.get('/')  # Establish session
            
            # Should not find the job (belongs to different session)
            response = other_client.get(f'/api/jobs/{job_id}/audio/upload123')
            assert response.status_code == 404
    
    def test_audio_endpoint_range_request(self, app_client, session_with_job):
        """Audio endpoint should support Range requests."""
        job_id = session_with_job['job_id']
        
        # Request with Range header
        response = app_client.get(
            f'/api/jobs/{job_id}/audio/upload123',
            headers={'Range': 'bytes=0-10'}
        )
        
        # Should return 206 Partial Content
        assert response.status_code == 206
        assert 'Content-Range' in response.headers
        assert response.headers.get('Accept-Ranges') == 'bytes'


class TestOutputTextEndpoint:
    """Test output text preview endpoint."""
    
    def test_output_text_endpoint_only_allows_text_types(self, app_client, session_with_job):
        """Text preview should only work for allowed file types."""
        job_id = session_with_job['job_id']
        
        # Markdown should work
        response = app_client.get(f'/api/jobs/{job_id}/outputs/out1/text')
        assert response.status_code == 200
        data = response.get_json()
        assert 'content' in data
        assert data['mime'] == 'text/markdown'
        
        # JSON should work
        response = app_client.get(f'/api/jobs/{job_id}/outputs/out3/text')
        assert response.status_code == 200
        data = response.get_json()
        assert data['mime'] == 'application/json'
        
        # Binary should fail
        response = app_client.get(f'/api/jobs/{job_id}/outputs/out4/text')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_output_text_returns_content(self, app_client, session_with_job):
        """Text preview should return file content."""
        job_id = session_with_job['job_id']
        
        response = app_client.get(f'/api/jobs/{job_id}/outputs/out1/text')
        assert response.status_code == 200
        data = response.get_json()
        
        assert 'Speaker 1' in data['content']
        assert 'Hello world' in data['content']
        assert data['truncated'] == False
    
    def test_output_text_session_isolation(self, test_data_root, session_with_job):
        """Text preview should not allow access from different session."""
        from app import app
        
        job_id = session_with_job['job_id']
        
        with app.test_client() as other_client:
            other_client.get('/')
            
            response = other_client.get(f'/api/jobs/{job_id}/outputs/out1/text')
            assert response.status_code == 404


class TestSpeakerLabelsEndpoint:
    """Test speaker labels endpoints."""
    
    def test_get_speakers_returns_detected_speakers(self, app_client, session_with_job):
        """GET /speakers should return detected speakers."""
        job_id = session_with_job['job_id']
        
        response = app_client.get(f'/api/jobs/{job_id}/speakers')
        assert response.status_code == 200
        data = response.get_json()
        
        assert 'speakers' in data
        assert 'labels' in data
        assert len(data['speakers']) >= 1
    
    def test_put_speaker_labels_persists_to_manifest(self, app_client, session_with_job):
        """PUT /speakers should persist labels to manifest."""
        import session_store
        
        job_id = session_with_job['job_id']
        session_id = session_with_job['session_id']
        
        # Get CSRF token
        session_resp = app_client.get('/api/session')
        csrf_token = session_resp.get_json().get('csrfToken', '')
        
        # Update labels
        response = app_client.put(
            f'/api/jobs/{job_id}/speakers',
            json={'labels': {'SPEAKER_00': 'Alice', 'SPEAKER_01': 'Bob'}},
            headers={'X-CSRF-Token': csrf_token}
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['labels']['SPEAKER_00'] == 'Alice'
        
        # Verify persisted to manifest
        manifest = session_store.read_json(
            session_store.job_manifest_path(session_id, job_id)
        )
        assert manifest['speakerLabels']['SPEAKER_00'] == 'Alice'
        assert manifest['speakerLabels']['SPEAKER_01'] == 'Bob'
    
    def test_put_speaker_labels_rejects_invalid_names(self, app_client, session_with_job):
        """PUT /speakers should reject invalid label values."""
        job_id = session_with_job['job_id']
        
        # Get CSRF token
        session_resp = app_client.get('/api/session')
        csrf_token = session_resp.get_json().get('csrfToken', '')
        
        # Too long label
        long_label = 'A' * 50
        response = app_client.put(
            f'/api/jobs/{job_id}/speakers',
            json={'labels': {'SPEAKER_00': long_label}},
            headers={'X-CSRF-Token': csrf_token}
        )
        assert response.status_code == 400
        assert 'exceeds' in response.get_json()['error']
        
        # Invalid characters
        response = app_client.put(
            f'/api/jobs/{job_id}/speakers',
            json={'labels': {'SPEAKER_00': '<script>alert(1)</script>'}},
            headers={'X-CSRF-Token': csrf_token}
        )
        assert response.status_code == 400
        assert 'invalid' in response.get_json()['error'].lower()
    
    def test_put_speaker_labels_rejects_invalid_speaker_id(self, app_client, session_with_job):
        """PUT /speakers should reject invalid speaker ID format."""
        job_id = session_with_job['job_id']
        
        session_resp = app_client.get('/api/session')
        csrf_token = session_resp.get_json().get('csrfToken', '')
        
        response = app_client.put(
            f'/api/jobs/{job_id}/speakers',
            json={'labels': {'invalid_id': 'Test'}},
            headers={'X-CSRF-Token': csrf_token}
        )
        assert response.status_code == 400
        assert 'Invalid speaker ID' in response.get_json()['error']
    
    def test_speaker_labels_session_isolation(self, test_data_root, session_with_job):
        """Speaker labels should not be accessible from different session."""
        from app import app
        
        job_id = session_with_job['job_id']
        
        with app.test_client() as other_client:
            other_client.get('/')
            
            response = other_client.get(f'/api/jobs/{job_id}/speakers')
            assert response.status_code == 404
