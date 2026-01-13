"""Tests for worker /v1/ping endpoint resilience."""

import pytest
import os
import sys

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


class TestWorkerPingResilience:
    """Test that /v1/ping is resilient to diarization import failures."""
    
    def test_ping_returns_200_without_hf_token(self):
        """Test ping returns 200 JSON when HF_TOKEN is not set."""
        # Clear any existing HF tokens
        old_hf = os.environ.pop('HF_TOKEN', None)
        old_hf2 = os.environ.pop('HUGGINGFACE_TOKEN', None)
        os.environ['WORKER_TOKEN'] = 'test-token'
        
        try:
            # Import after clearing env
            from worker.app import app
            
            with app.test_client() as client:
                response = client.get('/v1/ping', 
                                      headers={'Authorization': 'Bearer test-token'})
                
                assert response.status_code == 200
                data = response.get_json()
                assert data is not None
                assert 'status' in data
                assert data['status'] == 'ok'
                # diarization should be False without HF token
                assert data.get('diarization') == False
        finally:
            # Restore env
            if old_hf:
                os.environ['HF_TOKEN'] = old_hf
            if old_hf2:
                os.environ['HUGGINGFACE_TOKEN'] = old_hf2
    
    def test_ping_returns_json_structure(self):
        """Test ping returns expected JSON structure."""
        os.environ['WORKER_TOKEN'] = 'test-token'
        
        try:
            from worker.app import app
            
            with app.test_client() as client:
                response = client.get('/v1/ping',
                                      headers={'Authorization': 'Bearer test-token'})
                
                assert response.status_code == 200
                data = response.get_json()
                
                # Required fields
                assert 'status' in data
                assert 'version' in data
                assert 'gpu' in data
                assert 'diarization' in data
                
                # Capacity fields (from heavy batch ops)
                assert 'activeJobs' in data
                assert 'maxConcurrentJobs' in data
        finally:
            pass
    
    def test_ping_diarization_false_without_token(self):
        """Test that diarization is False when no HF token is set."""
        old_hf = os.environ.pop('HF_TOKEN', None)
        old_hf2 = os.environ.pop('HUGGINGFACE_TOKEN', None)
        os.environ['WORKER_TOKEN'] = 'test-token'
        
        try:
            # Force re-detection by clearing cache
            import worker.app as worker_app
            worker_app._cached_capabilities = None
            
            from worker.app import app
            
            with app.test_client() as client:
                response = client.get('/v1/ping',
                                      headers={'Authorization': 'Bearer test-token'})
                
                data = response.get_json()
                assert data['diarization'] == False
                # Should NOT have diarizationError when token is missing
                # (error only appears when import actually fails)
                assert 'diarizationError' not in data or data.get('diarizationError') is None
        finally:
            if old_hf:
                os.environ['HF_TOKEN'] = old_hf
            if old_hf2:
                os.environ['HUGGINGFACE_TOKEN'] = old_hf2
            # Reset cache
            import worker.app as worker_app
            worker_app._cached_capabilities = None


class TestWorkerDependencyPins:
    """Test that worker dependencies are properly pinned."""
    
    def test_requirements_has_pinned_versions(self):
        """Test that requirements.txt uses exact version pins."""
        req_path = os.path.join(_parent_dir, 'worker', 'requirements.txt')
        
        with open(req_path, 'r') as f:
            content = f.read()
        
        # Should not have >= for critical deps
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
        
        for line in lines:
            # Skip comments
            if line.startswith('#'):
                continue
            # Check that we use == not >= for version pins
            if '>=' in line:
                pytest.fail(f"Found loose version constraint: {line}")
    
    def test_pyannote_version_pinned(self):
        """Test that pyannote.audio is pinned to 3.1.1."""
        req_path = os.path.join(_parent_dir, 'worker', 'requirements.txt')
        
        with open(req_path, 'r') as f:
            content = f.read()
        
        assert 'pyannote.audio==3.1.1' in content
