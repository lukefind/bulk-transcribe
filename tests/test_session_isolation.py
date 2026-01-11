"""
Minimal tests for session isolation and security.

Run with: pytest tests/ -q
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(scope='module')
def test_data_root():
    """Create a temporary data root for testing."""
    tmpdir = tempfile.mkdtemp(prefix='bulk_transcribe_test_')
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


class TestSessionCookie:
    """Test session cookie issuance."""
    
    def test_session_cookie_issued_on_first_request(self, app_client):
        """Session cookie should be issued on first request."""
        response = app_client.get('/')
        assert response.status_code == 200
        
        # Check for bt_session cookie
        cookies = {c.name: c for c in app_client.cookie_jar}
        assert 'bt_session' in cookies
        
        cookie = cookies['bt_session']
        assert len(cookie.value) > 20  # Should be a long random token
    
    def test_session_cookie_httponly(self, app_client):
        """Session cookie should be HttpOnly."""
        response = app_client.get('/')
        
        # Check Set-Cookie header for HttpOnly
        set_cookie = response.headers.get('Set-Cookie', '')
        assert 'HttpOnly' in set_cookie or 'httponly' in set_cookie.lower()
    
    def test_session_cookie_samesite(self, app_client):
        """Session cookie should have SameSite=Lax."""
        response = app_client.get('/')
        
        set_cookie = response.headers.get('Set-Cookie', '')
        assert 'SameSite=Lax' in set_cookie or 'samesite=lax' in set_cookie.lower()


class TestServerModeGating:
    """Test that legacy endpoints are blocked in server mode."""
    
    def test_preview_blocked_in_server_mode(self, app_client):
        """Legacy /preview endpoint should return 404 in server mode."""
        response = app_client.post('/preview', json={'folder': '/tmp'})
        assert response.status_code == 404
    
    def test_browse_blocked_in_server_mode(self, app_client):
        """Legacy /browse endpoint should return 404 in server mode."""
        response = app_client.post('/browse', json={'prompt': 'test'})
        assert response.status_code == 404
    
    def test_legacy_download_blocked_in_server_mode(self, app_client):
        """Legacy /download endpoint should return 404 in server mode."""
        response = app_client.get('/download?folder=/tmp')
        assert response.status_code == 404


class TestCsrfProtection:
    """Test CSRF protection for POST endpoints."""
    
    def test_upload_requires_csrf_token(self, app_client):
        """POST /api/uploads should require CSRF token."""
        # First request to get session
        app_client.get('/')
        
        # Try upload without CSRF token
        response = app_client.post('/api/uploads', data={})
        assert response.status_code == 403
        
        data = response.get_json()
        assert 'CSRF' in data.get('error', '')
    
    def test_job_create_requires_csrf_token(self, app_client):
        """POST /api/jobs should require CSRF token."""
        app_client.get('/')
        
        response = app_client.post('/api/jobs', json={'uploadIds': []})
        assert response.status_code == 403
    
    def test_csrf_token_endpoint_returns_token(self, app_client):
        """GET /api/session should return CSRF token."""
        app_client.get('/')
        
        response = app_client.get('/api/session')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'csrfToken' in data
        assert len(data['csrfToken']) > 20


class TestSecurityHeaders:
    """Test security headers on API endpoints."""
    
    def test_api_endpoints_have_no_store(self, app_client):
        """API endpoints should have Cache-Control: no-store."""
        app_client.get('/')
        
        response = app_client.get('/api/mode')
        assert response.status_code == 200
        assert response.headers.get('Cache-Control') == 'no-store'
    
    def test_api_endpoints_have_nosniff(self, app_client):
        """API endpoints should have X-Content-Type-Options: nosniff."""
        app_client.get('/')
        
        response = app_client.get('/api/mode')
        assert response.headers.get('X-Content-Type-Options') == 'nosniff'


class TestSessionIsolation:
    """Test that sessions are isolated from each other."""
    
    def test_different_sessions_have_different_ids(self, test_data_root):
        """Different clients should get different session IDs."""
        from app import app
        app.config['TESTING'] = True
        
        with app.test_client() as client1:
            client1.get('/')
            cookies1 = {c.name: c.value for c in client1.cookie_jar}
            session1 = cookies1.get('bt_session')
        
        with app.test_client() as client2:
            client2.get('/')
            cookies2 = {c.name: c.value for c in client2.cookie_jar}
            session2 = cookies2.get('bt_session')
        
        assert session1 != session2
    
    def test_job_not_found_for_different_session(self, test_data_root):
        """Jobs should not be accessible from different sessions."""
        from app import app
        import session_store
        
        app.config['TESTING'] = True
        
        # Create a job in session1
        session1_id = session_store.new_id(32)
        job_id = session_store.new_id(12)
        session_store.ensure_job_dirs(session1_id, job_id)
        
        manifest = {
            'jobId': job_id,
            'status': 'complete',
            'inputs': [],
            'outputs': []
        }
        session_store.atomic_write_json(
            session_store.job_manifest_path(session1_id, job_id),
            manifest
        )
        
        # Try to access from a different session
        with app.test_client() as client:
            client.get('/')  # Get a new session
            
            response = client.get(f'/api/jobs/{job_id}')
            assert response.status_code == 404


class TestApiEndpoints:
    """Test API endpoint functionality."""
    
    def test_api_mode_returns_server(self, app_client):
        """GET /api/mode should return server mode."""
        response = app_client.get('/api/mode')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['mode'] == 'server'
        assert data['isServer'] is True
    
    def test_api_jobs_list_empty(self, app_client):
        """GET /api/jobs should return empty list for new session."""
        app_client.get('/')
        
        response = app_client.get('/api/jobs')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'jobs' in data
        assert data['jobs'] == []
    
    def test_api_uploads_list_empty(self, app_client):
        """GET /api/uploads should return empty list for new session."""
        app_client.get('/')
        
        response = app_client.get('/api/uploads')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'uploads' in data
        assert data['uploads'] == []
