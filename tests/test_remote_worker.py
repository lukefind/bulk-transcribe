"""
Tests for remote GPU worker integration.
"""

import os
import time
import pytest
import tempfile


class TestRemoteWorkerConfig:
    """Test remote worker configuration."""
    
    def test_get_worker_config_defaults(self):
        """Test default configuration values."""
        # Clear any existing env vars
        for key in ['REMOTE_WORKER_URL', 'REMOTE_WORKER_TOKEN', 'REMOTE_WORKER_MODE']:
            os.environ.pop(key, None)
        
        from remote_worker import get_worker_config
        config = get_worker_config()
        
        assert config['url'] == ''
        assert config['token'] == ''
        assert config['mode'] == 'off'
        assert config['timeoutSeconds'] == 7200
        assert config['pollSeconds'] == 2
        assert config['uploadMode'] == 'pull'
    
    def test_get_worker_config_from_env(self):
        """Test configuration from environment variables."""
        os.environ['REMOTE_WORKER_URL'] = 'https://test-worker.example.com'
        os.environ['REMOTE_WORKER_TOKEN'] = 'test-token-123'
        os.environ['REMOTE_WORKER_MODE'] = 'optional'
        
        try:
            from remote_worker import get_worker_config
            config = get_worker_config()
            
            assert config['url'] == 'https://test-worker.example.com'
            assert config['token'] == 'test-token-123'
            assert config['mode'] == 'optional'
        finally:
            os.environ.pop('REMOTE_WORKER_URL', None)
            os.environ.pop('REMOTE_WORKER_TOKEN', None)
            os.environ.pop('REMOTE_WORKER_MODE', None)


class TestSignedUrls:
    """Test signed URL generation and verification."""
    
    def test_generate_signed_url(self):
        """Test signed URL generation."""
        from remote_worker import generate_signed_url
        
        url = generate_signed_url(
            base_url='https://controller.example.com',
            job_id='job123',
            input_id='input456',
            secret='test-secret',
            expires_in_seconds=3600
        )
        
        assert 'https://controller.example.com/api/jobs/job123/inputs/input456' in url
        assert 'expires=' in url
        assert 'sig=' in url
    
    def test_verify_signed_url_valid(self):
        """Test valid signature verification."""
        from remote_worker import generate_signed_url, verify_signed_url
        
        # Generate URL
        url = generate_signed_url(
            base_url='https://controller.example.com',
            job_id='job123',
            input_id='input456',
            secret='test-secret',
            expires_in_seconds=3600
        )
        
        # Extract params
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        
        expires = params['expires'][0]
        sig = params['sig'][0]
        
        # Verify
        assert verify_signed_url('job123', 'input456', expires, sig, 'test-secret')
    
    def test_verify_signed_url_wrong_secret(self):
        """Test signature verification with wrong secret."""
        from remote_worker import generate_signed_url, verify_signed_url
        
        url = generate_signed_url(
            base_url='https://controller.example.com',
            job_id='job123',
            input_id='input456',
            secret='test-secret',
            expires_in_seconds=3600
        )
        
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        
        expires = params['expires'][0]
        sig = params['sig'][0]
        
        # Verify with wrong secret should fail
        assert not verify_signed_url('job123', 'input456', expires, sig, 'wrong-secret')
    
    def test_verify_signed_url_expired(self):
        """Test expired signature verification."""
        from remote_worker import verify_signed_url
        
        # Use an expired timestamp
        expired_time = str(int(time.time()) - 100)
        
        # Generate a valid signature for the expired time
        import hmac
        import hashlib
        payload = f"job123:input456:{expired_time}"
        sig = hmac.new(
            'test-secret'.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()[:32]
        
        # Should fail due to expiry
        assert not verify_signed_url('job123', 'input456', expired_time, sig, 'test-secret')


class TestRemoteWorkerStatus:
    """Test remote worker status checking."""
    
    def test_status_not_configured(self):
        """Test status when worker is not configured."""
        for key in ['REMOTE_WORKER_URL', 'REMOTE_WORKER_TOKEN', 'REMOTE_WORKER_MODE']:
            os.environ.pop(key, None)
        
        from remote_worker import get_remote_worker_status
        status = get_remote_worker_status()
        
        assert status['configured'] == False
        assert status['connected'] == False
        # No error when mode is off (default)
        assert status['mode'] == 'off'
    
    def test_status_configured_but_unreachable(self):
        """Test status when worker is configured but unreachable."""
        os.environ['REMOTE_WORKER_URL'] = 'http://localhost:99999'
        os.environ['REMOTE_WORKER_TOKEN'] = 'test-token'
        os.environ['REMOTE_WORKER_MODE'] = 'optional'
        
        try:
            import remote_worker
            # Clear cache to ensure fresh check
            remote_worker._worker_status_cache = None
            remote_worker._worker_status_cache_time = 0
            
            status = remote_worker.get_remote_worker_status(force_refresh=True)
            
            assert status['configured'] == True
            assert status['connected'] == False
            assert status['error'] is not None
        finally:
            os.environ.pop('REMOTE_WORKER_URL', None)
            os.environ.pop('REMOTE_WORKER_TOKEN', None)
            os.environ.pop('REMOTE_WORKER_MODE', None)


class TestShouldUseRemoteWorker:
    """Test remote worker selection logic."""
    
    def test_mode_off(self):
        """Test that mode=off always returns False."""
        os.environ['REMOTE_WORKER_URL'] = 'http://localhost:8477'
        os.environ['REMOTE_WORKER_TOKEN'] = 'test-token'
        os.environ['REMOTE_WORKER_MODE'] = 'off'
        
        try:
            from remote_worker import should_use_remote_worker
            
            assert should_use_remote_worker(user_requested=False) == False
            assert should_use_remote_worker(user_requested=True) == False
        finally:
            os.environ.pop('REMOTE_WORKER_URL', None)
            os.environ.pop('REMOTE_WORKER_TOKEN', None)
            os.environ.pop('REMOTE_WORKER_MODE', None)
    
    def test_mode_required(self):
        """Test that mode=required always returns True."""
        os.environ['REMOTE_WORKER_URL'] = 'http://localhost:8477'
        os.environ['REMOTE_WORKER_TOKEN'] = 'test-token'
        os.environ['REMOTE_WORKER_MODE'] = 'required'
        
        try:
            from remote_worker import should_use_remote_worker
            
            assert should_use_remote_worker(user_requested=False) == True
            assert should_use_remote_worker(user_requested=True) == True
        finally:
            os.environ.pop('REMOTE_WORKER_URL', None)
            os.environ.pop('REMOTE_WORKER_TOKEN', None)
            os.environ.pop('REMOTE_WORKER_MODE', None)
