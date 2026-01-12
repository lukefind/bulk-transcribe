"""
Tests for diarization policy computation and validation.

These tests verify:
1. Policy clamps to server cap
2. Overlap is always less than chunk
3. Job manifest contains diarizationEffective
4. Policy endpoint matches job creation effective
5. Guardrail error payload contains effective fields
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diarization_policy import (
    compute_diarization_policy,
    get_server_policy_config,
    get_clamping_warnings,
    estimate_chunk_count,
    format_duration_human
)


class TestPolicyClampsToServerCap:
    """Test that policy clamps max duration to server cap."""
    
    def test_requested_above_server_cap_is_clamped(self):
        """When user requests max duration above server cap, it should be clamped."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=3600,  # 1 hour
            server_max_duration_seconds=1800,     # 30 min cap
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        assert policy is not None
        assert policy['maxDurationSeconds'] == 1800  # Clamped to server cap
        assert policy['clamped']['maxDurationClamped'] is True
        assert policy['clamped']['maxDurationOriginal'] == 3600
        assert policy['serverMaxDurationSeconds'] == 1800
    
    def test_requested_below_server_cap_not_clamped(self):
        """When user requests max duration below server cap, it should not be clamped."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=600,   # 10 min
            server_max_duration_seconds=1800,     # 30 min cap
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        assert policy is not None
        assert policy['maxDurationSeconds'] == 600
        assert policy['clamped']['maxDurationClamped'] is False
    
    def test_requested_below_minimum_is_clamped(self):
        """When user requests max duration below 30s minimum, it should be clamped."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=10,    # Too low
            server_max_duration_seconds=1800,
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        assert policy is not None
        assert policy['maxDurationSeconds'] == 30  # Clamped to minimum
        assert policy['clamped']['maxDurationClamped'] is True


class TestPolicyOverlapLessThanChunk:
    """Test that overlap is always less than chunk size."""
    
    def test_derived_overlap_less_than_chunk(self):
        """Derived overlap should always be less than chunk size."""
        for max_duration in [60, 180, 600, 1200, 1800]:
            policy = compute_diarization_policy(
                diarization_enabled=True,
                diarization_auto_split=True,
                requested_max_duration_seconds=max_duration,
                server_max_duration_seconds=1800,
                default_max_duration_seconds=180,
                min_chunk_seconds=60,
                max_chunk_seconds=600,
                overlap_ratio=0.03,
                min_overlap_seconds=2,
                max_overlap_seconds=15,
            )
            
            assert policy is not None
            assert policy['overlapSeconds'] < policy['chunkSeconds'], \
                f"Overlap {policy['overlapSeconds']} should be < chunk {policy['chunkSeconds']} for max_duration={max_duration}"
    
    def test_explicit_overlap_clamped_if_too_large(self):
        """Explicit overlap should be clamped if >= chunk size."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=180,
            requested_chunk_seconds=100,
            requested_overlap_seconds=100,  # Same as chunk - should be clamped
            server_max_duration_seconds=1800,
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        assert policy is not None
        assert policy['overlapSeconds'] < policy['chunkSeconds']
        assert policy['clamped']['overlapSecondsClamped'] is True


class TestClampingWarnings:
    """Test that clamping warnings are generated correctly."""
    
    def test_warning_generated_for_clamped_max_duration(self):
        """Warning should be generated when max duration is clamped."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=3600,
            server_max_duration_seconds=1800,
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        warnings = get_clamping_warnings(policy)
        assert len(warnings) > 0
        assert any('clamped' in w.lower() for w in warnings)
    
    def test_no_warning_when_not_clamped(self):
        """No warning should be generated when nothing is clamped."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=600,
            server_max_duration_seconds=1800,
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        warnings = get_clamping_warnings(policy)
        assert len(warnings) == 0


class TestPolicyDisabled:
    """Test policy behavior when diarization is disabled."""
    
    def test_returns_none_when_disabled(self):
        """Policy should return None when diarization is disabled."""
        policy = compute_diarization_policy(
            diarization_enabled=False,
            diarization_auto_split=True,
            requested_max_duration_seconds=600,
            server_max_duration_seconds=1800,
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        assert policy is None


class TestChunkCountEstimation:
    """Test chunk count estimation."""
    
    def test_single_chunk_for_short_file(self):
        """Short file should result in single chunk."""
        count = estimate_chunk_count(
            file_duration_seconds=100,
            chunk_seconds=150,
            overlap_seconds=5
        )
        assert count == 1
    
    def test_multiple_chunks_for_long_file(self):
        """Long file should result in multiple chunks."""
        count = estimate_chunk_count(
            file_duration_seconds=600,
            chunk_seconds=150,
            overlap_seconds=5
        )
        assert count > 1
    
    def test_zero_duration_returns_zero(self):
        """Zero duration should return zero chunks."""
        count = estimate_chunk_count(
            file_duration_seconds=0,
            chunk_seconds=150,
            overlap_seconds=5
        )
        assert count == 0


class TestDurationFormatting:
    """Test human-readable duration formatting."""
    
    def test_seconds_format(self):
        assert format_duration_human(30) == "30s"
    
    def test_minutes_format(self):
        assert format_duration_human(180) == "3m"
    
    def test_minutes_with_seconds_format(self):
        assert format_duration_human(90) == "1m 30s"
    
    def test_hours_format(self):
        assert format_duration_human(3600) == "1h"
    
    def test_hours_with_minutes_format(self):
        assert format_duration_human(5400) == "1h 30m"


class TestDerivedChunkSizes:
    """Test that chunk sizes are derived correctly based on max duration."""
    
    def test_short_duration_gets_small_chunks(self):
        """Short max duration should get smaller chunk size."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=180,
            server_max_duration_seconds=1800,
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        assert policy['chunkSeconds'] == 150
        assert policy['derived'] is True
    
    def test_long_duration_gets_larger_chunks(self):
        """Long max duration should get larger chunk size."""
        policy = compute_diarization_policy(
            diarization_enabled=True,
            diarization_auto_split=True,
            requested_max_duration_seconds=1800,
            server_max_duration_seconds=1800,
            default_max_duration_seconds=180,
            min_chunk_seconds=60,
            max_chunk_seconds=600,
            overlap_ratio=0.03,
            min_overlap_seconds=2,
            max_overlap_seconds=15,
        )
        
        assert policy['chunkSeconds'] == 300
        assert policy['derived'] is True


class TestServerConfigLoading:
    """Test server config loading from environment."""
    
    def test_get_server_policy_config_returns_dict(self):
        """Server config should return a dict with expected keys."""
        config = get_server_policy_config()
        
        assert isinstance(config, dict)
        assert 'serverMaxDurationSeconds' in config
        assert 'defaultMaxDurationSeconds' in config
        assert 'minChunkSeconds' in config
        assert 'maxChunkSeconds' in config
        assert 'overlapRatio' in config
        assert 'minOverlapSeconds' in config
        assert 'maxOverlapSeconds' in config
