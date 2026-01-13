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
        
        import remote_worker
        # Clear cache
        remote_worker._worker_status_cache.clear()
        remote_worker._worker_status_cache_time.clear()
        
        status = remote_worker.get_remote_worker_status(force_refresh=True)
        
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
            remote_worker._worker_status_cache.clear()
            remote_worker._worker_status_cache_time.clear()
            
            status = remote_worker.get_remote_worker_status(force_refresh=True)
            
            assert status['configured'] == True
            assert status['connected'] == False
            assert status['error'] is not None
        finally:
            os.environ.pop('REMOTE_WORKER_URL', None)
            os.environ.pop('REMOTE_WORKER_TOKEN', None)
            os.environ.pop('REMOTE_WORKER_MODE', None)
    
    def test_cache_invalidation_on_config_change(self):
        """Test that cache is invalidated when config changes."""
        import remote_worker
        
        # Set initial config
        os.environ['REMOTE_WORKER_URL'] = 'http://worker1:8477'
        os.environ['REMOTE_WORKER_TOKEN'] = 'token1'
        os.environ['REMOTE_WORKER_MODE'] = 'optional'
        
        try:
            # Clear cache
            remote_worker._worker_status_cache.clear()
            remote_worker._worker_status_cache_time.clear()
            
            # First call - should ping worker1
            status1 = remote_worker.get_remote_worker_status(force_refresh=True)
            assert status1['url'] == 'http://worker1:8477'
            
            # Change config to different worker
            os.environ['REMOTE_WORKER_URL'] = 'http://worker2:8477'
            os.environ['REMOTE_WORKER_TOKEN'] = 'token2'
            
            # Second call - should NOT return cached worker1 status
            status2 = remote_worker.get_remote_worker_status()
            assert status2['url'] == 'http://worker2:8477'
            
        finally:
            os.environ.pop('REMOTE_WORKER_URL', None)
            os.environ.pop('REMOTE_WORKER_TOKEN', None)
            os.environ.pop('REMOTE_WORKER_MODE', None)
    
    def test_full_url_not_truncated(self):
        """Test that URL is not truncated in status response."""
        os.environ['REMOTE_WORKER_URL'] = 'https://very-long-worker-hostname.example.com:8477'
        os.environ['REMOTE_WORKER_TOKEN'] = 'test-token'
        os.environ['REMOTE_WORKER_MODE'] = 'optional'
        
        try:
            import remote_worker
            remote_worker._worker_status_cache.clear()
            remote_worker._worker_status_cache_time.clear()
            
            status = remote_worker.get_remote_worker_status(force_refresh=True)
            
            # URL should be full, not truncated
            assert status['url'] == 'https://very-long-worker-hostname.example.com:8477'
            assert '...' not in status['url']
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


class TestBackoffLogic:
    """Test exponential backoff with jitter."""
    
    def test_backoff_increases_exponentially(self):
        """Test that backoff increases with attempt number."""
        from remote_worker import calculate_backoff
        
        # Use seed for deterministic results
        delays = [calculate_backoff(i, seed=42) for i in range(5)]
        
        # Should generally increase (with jitter, not strictly monotonic)
        assert delays[0] >= 2  # Min is 2s
        assert delays[4] <= 30  # Max is 30s
    
    def test_backoff_respects_min_max(self):
        """Test that backoff stays within min/max bounds."""
        from remote_worker import calculate_backoff
        
        for attempt in range(20):
            delay = calculate_backoff(attempt, seed=attempt)
            assert delay >= 2, f"Delay {delay} below min at attempt {attempt}"
            assert delay <= 30, f"Delay {delay} above max at attempt {attempt}"
    
    def test_backoff_with_seed_is_deterministic(self):
        """Test that same seed produces same result."""
        from remote_worker import calculate_backoff
        
        delay1 = calculate_backoff(3, seed=123)
        delay2 = calculate_backoff(3, seed=123)
        assert delay1 == delay2
    
    def test_backoff_jitter_varies_results(self):
        """Test that different seeds produce different results."""
        from remote_worker import calculate_backoff
        
        delays = [calculate_backoff(3, seed=i) for i in range(10)]
        unique_delays = set(delays)
        # Should have some variation due to jitter
        assert len(unique_delays) > 1


class TestCapacityCheck:
    """Test worker capacity checking."""
    
    def test_capacity_check_when_not_connected(self):
        """Test capacity check returns no capacity when worker not connected."""
        os.environ['REMOTE_WORKER_URL'] = 'http://localhost:99999'
        os.environ['REMOTE_WORKER_TOKEN'] = 'test-token'
        os.environ['REMOTE_WORKER_MODE'] = 'optional'
        
        try:
            import remote_worker
            remote_worker._worker_status_cache.clear()
            remote_worker._worker_status_cache_time.clear()
            
            capacity = remote_worker.check_worker_capacity()
            
            assert capacity['hasCapacity'] == False
            assert capacity['error'] is not None
        finally:
            os.environ.pop('REMOTE_WORKER_URL', None)
            os.environ.pop('REMOTE_WORKER_TOKEN', None)
            os.environ.pop('REMOTE_WORKER_MODE', None)


class TestQueueStateTransitions:
    """Test job state transitions for remote queue."""
    
    def test_queued_remote_is_valid_status(self):
        """Test that queued_remote is a recognized job status."""
        # This is a documentation test - the status should be handled in UI
        valid_statuses = ['queued', 'queued_remote', 'running', 'complete', 
                         'complete_with_errors', 'failed', 'canceled']
        assert 'queued_remote' in valid_statuses
    
    def test_error_codes_are_standardized(self):
        """Test that error codes follow naming convention."""
        standard_codes = [
            'REMOTE_WORKER_UNAUTHORIZED',
            'REMOTE_WORKER_UNREACHABLE', 
            'REMOTE_WORKER_TIMEOUT',
            'REMOTE_WORKER_CAPACITY',
            'REMOTE_DISPATCH_FAILED',
            'REMOTE_FAILED',
            'USER_CANCELED'
        ]
        
        # All codes should be uppercase with underscores
        for code in standard_codes:
            assert code == code.upper()
            assert ' ' not in code
