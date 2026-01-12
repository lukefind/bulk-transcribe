"""Tests for review overhaul."""

import pytest
import json
import os
import tempfile
import shutil
from datetime import datetime, timezone

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


@pytest.fixture(autouse=True)
def use_temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    original_data_root = os.environ.get('DATA_ROOT')
    os.environ['DATA_ROOT'] = temp_dir
    os.environ['APP_MODE'] = 'server'
    yield temp_dir
    if original_data_root:
        os.environ['DATA_ROOT'] = original_data_root
    else:
        os.environ.pop('DATA_ROOT', None)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestReviewableOutputTypes:
    def test_reviewable_types_defined(self):
        import session_store
        assert hasattr(session_store, 'REVIEWABLE_OUTPUT_TYPES')
        types = session_store.REVIEWABLE_OUTPUT_TYPES
        assert 'json' in types

    def test_job_with_json_output_is_reviewable(self, use_temp_data_dir):
        import session_store
        session_id = 'test_session'
        job_id = 'test_job'
        session_dir = session_store.session_dir(session_id)
        jobs_dir = os.path.join(session_dir, 'jobs')
        job_dir = os.path.join(jobs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        manifest = {
            'jobId': job_id,
            'status': 'complete',
            'createdAt': datetime.now(timezone.utc).isoformat(),
            'inputs': [],
            'outputs': [{'id': 'out1', 'type': 'json', 'filename': 'test.json'}]
        }
        # Use job.json as expected by job_manifest_path
        with open(os.path.join(job_dir, 'job.json'), 'w') as f:
            json.dump(manifest, f)
        jobs = session_store.list_jobs(session_id)
        assert len(jobs) == 1
        assert jobs[0]['reviewable'] == True

    def test_job_without_outputs_not_reviewable(self, use_temp_data_dir):
        import session_store
        session_id = 'test_session'
        job_id = 'test_job'
        session_dir = session_store.session_dir(session_id)
        jobs_dir = os.path.join(session_dir, 'jobs')
        job_dir = os.path.join(jobs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        manifest = {
            'jobId': job_id,
            'status': 'complete',
            'createdAt': datetime.now(timezone.utc).isoformat(),
            'inputs': [],
            'outputs': []
        }
        with open(os.path.join(job_dir, 'job.json'), 'w') as f:
            json.dump(manifest, f)
        jobs = session_store.list_jobs(session_id)
        assert len(jobs) == 1
        assert jobs[0]['reviewable'] == False


class TestReviewTimelineParser:
    def test_parse_diarization_json(self):
        from review_timeline import TimelineParser
        parser = TimelineParser('job1', 'input1', 'test.wav')
        diarization = json.dumps({
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'Hello', 'speaker': 'SPEAKER_00'},
                {'start': 2.0, 'end': 4.0, 'text': 'World', 'speaker': 'SPEAKER_01'}
            ]
        })
        timeline = parser.parse(diarization_json=diarization)
        assert len(timeline.chunks) == 2
        assert len(timeline.speakers) == 2

    def test_apply_review_state(self):
        from review_timeline import TimelineParser, apply_review_state
        parser = TimelineParser('job1', 'input1', 'test.wav')
        diarization = json.dumps({
            'segments': [{'start': 0.0, 'end': 2.0, 'text': 'Hello', 'speaker': 'SPEAKER_00'}]
        })
        timeline = parser.parse(diarization_json=diarization)
        state = {
            'speakerLabelMap': {'SPEAKER_00': 'Alice'},
            'chunkEdits': {timeline.chunks[0].chunk_id: {'speakerId': 'SPEAKER_01'}}
        }
        timeline = apply_review_state(timeline, state)
        assert timeline.speakers[0].label == 'Alice'
        assert timeline.chunks[0].speaker_id == 'SPEAKER_01'

    def test_to_dict_structure(self):
        from review_timeline import TimelineParser
        parser = TimelineParser('job1', 'input1', 'test.wav')
        diarization = json.dumps({
            'segments': [{'start': 0.0, 'end': 2.0, 'text': 'Hello', 'speaker': 'SPEAKER_00'}]
        })
        timeline = parser.parse(diarization_json=diarization)
        data = timeline.to_dict()
        assert 'version' in data
        assert 'speakers' in data
        assert 'chunks' in data
        assert len(data['speakers']) > 0
        assert 'id' in data['speakers'][0]
        assert 'color' in data['speakers'][0]
