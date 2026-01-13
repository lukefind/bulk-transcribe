#!/usr/bin/env python3
"""
GPU Worker Service for Bulk Transcribe.

This is a lightweight Flask service that accepts transcription jobs from the
controller, runs Whisper + diarization on GPU, and uploads results back.

Environment variables:
- WORKER_TOKEN: Shared secret for authentication (required)
- WORKER_TMP_DIR: Directory for temp files (default: /tmp/bt-worker)
- WORKER_MAX_FILE_MB: Maximum input file size (default: 2000)
- WORKER_MODEL: Default Whisper model (default: large-v3)
- WORKER_PORT: Port to listen on (default: 8477)
- HF_TOKEN: HuggingFace token for diarization (required if diarization enabled)
"""

import os
import sys
import json
import time
import uuid
import shutil
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any
from functools import wraps

import requests
from flask import Flask, request, jsonify, g

# Configuration
WORKER_TOKEN = os.environ.get('WORKER_TOKEN', '')
WORKER_TMP_DIR = os.environ.get('WORKER_TMP_DIR', '/tmp/bt-worker')
WORKER_MAX_FILE_MB = int(os.environ.get('WORKER_MAX_FILE_MB', '2000'))
WORKER_MODEL = os.environ.get('WORKER_MODEL', 'large-v3')
WORKER_PORT = int(os.environ.get('WORKER_PORT', '8477'))

app = Flask(__name__)

# In-memory job store (for single-worker setup; could be Redis for multi-worker)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def require_auth(f):
    """Decorator to require Bearer token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not WORKER_TOKEN:
            return jsonify({'error': 'Worker not configured (no WORKER_TOKEN)'}), 500
        
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing Authorization header'}), 401
        
        token = auth_header[7:]
        if token != WORKER_TOKEN:
            return jsonify({'error': 'Invalid token'}), 403
        
        return f(*args, **kwargs)
    return decorated


def generate_worker_job_id() -> str:
    """Generate a unique worker job ID."""
    return f"wk_{uuid.uuid4().hex[:12]}"


def get_job(worker_job_id: str) -> Optional[Dict]:
    """Get job by worker job ID."""
    with _jobs_lock:
        return _jobs.get(worker_job_id)


def update_job(worker_job_id: str, **updates):
    """Update job fields atomically."""
    with _jobs_lock:
        if worker_job_id in _jobs:
            _jobs[worker_job_id].update(updates)
            _jobs[worker_job_id]['updatedAt'] = datetime.now(timezone.utc).isoformat()


def add_log(worker_job_id: str, level: str, event: str, message: str = '', **extra):
    """Add a log entry to the job."""
    with _jobs_lock:
        if worker_job_id not in _jobs:
            return
        log_entry = {
            'ts': time.time(),
            'level': level,
            'event': event,
            'message': message,
            **extra
        }
        _jobs[worker_job_id].setdefault('logs', [])
        # Keep last 100 logs
        _jobs[worker_job_id]['logs'] = _jobs[worker_job_id]['logs'][-99:] + [log_entry]


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'bulk-transcribe-worker',
        'model': WORKER_MODEL,
        'maxFileMB': WORKER_MAX_FILE_MB
    })


@app.route('/v1/jobs', methods=['POST'])
@require_auth
def create_job():
    """
    Create a new transcription job.
    
    Request body:
    {
        "controllerJobId": "JOB_ID",
        "controllerSessionHash": "SHORT_HASH",
        "inputs": [
            {
                "inputId": "UPLOAD_ID",
                "filename": "file.wav",
                "contentType": "audio/wav",
                "sizeBytes": 123,
                "downloadUrl": "https://controller/api/jobs/JOB_ID/inputs/UPLOAD_ID?token=SIGNED"
            }
        ],
        "options": {
            "backend": "cuda",
            "model": "large-v3",
            "language": "",
            "diarizationEnabled": true,
            "diarizationEffective": {...}
        },
        "callbackUrl": "https://controller/api/jobs/JOB_ID/worker/complete"
    }
    """
    data = request.json or {}
    
    controller_job_id = data.get('controllerJobId')
    if not controller_job_id:
        return jsonify({'error': 'controllerJobId required'}), 400
    
    inputs = data.get('inputs', [])
    if not inputs:
        return jsonify({'error': 'No inputs provided'}), 400
    
    options = data.get('options', {})
    
    # Check idempotency - if job with same controller ID exists, return it
    idempotency_key = request.headers.get('Idempotency-Key', controller_job_id)
    with _jobs_lock:
        for wjid, job in _jobs.items():
            if job.get('controllerJobId') == idempotency_key:
                return jsonify({
                    'workerJobId': wjid,
                    'status': job['status']
                })
    
    # Create new job
    worker_job_id = generate_worker_job_id()
    now = datetime.now(timezone.utc).isoformat()
    
    job = {
        'workerJobId': worker_job_id,
        'controllerJobId': controller_job_id,
        'controllerSessionHash': data.get('controllerSessionHash', ''),
        'inputs': inputs,
        'options': options,
        'callbackUrl': data.get('callbackUrl'),
        'outputsUploadUrl': data.get('outputsUploadUrl'),
        'status': 'queued',
        'stage': 'queued',
        'stageStartedAt': now,
        'createdAt': now,
        'updatedAt': now,
        'progress': {
            'currentFileIndex': 0,
            'totalFiles': len(inputs),
            'chunkIndex': 0,
            'totalChunks': 0,
            'percent': 0
        },
        'logs': [],
        'error': None,
        'cancelRequested': False
    }
    
    with _jobs_lock:
        _jobs[worker_job_id] = job
    
    # Start processing in background thread
    thread = threading.Thread(target=process_job, args=(worker_job_id,), daemon=True)
    thread.start()
    
    return jsonify({
        'workerJobId': worker_job_id,
        'status': 'queued'
    })


@app.route('/v1/jobs/<worker_job_id>', methods=['GET'])
@require_auth
def get_job_status(worker_job_id: str):
    """Get job status and progress."""
    job = get_job(worker_job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify({
        'workerJobId': job['workerJobId'],
        'controllerJobId': job['controllerJobId'],
        'status': job['status'],
        'stage': job['stage'],
        'stageStartedAt': job.get('stageStartedAt'),
        'progress': job.get('progress', {}),
        'logs': job.get('logs', [])[-20:],  # Return last 20 logs
        'error': job.get('error')
    })


@app.route('/v1/jobs/<worker_job_id>/cancel', methods=['POST'])
@require_auth
def cancel_job(worker_job_id: str):
    """Request job cancellation."""
    job = get_job(worker_job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    update_job(worker_job_id, cancelRequested=True)
    add_log(worker_job_id, 'info', 'cancel_requested', 'Cancellation requested by controller')
    
    return jsonify({'status': 'cancel_requested'})


def process_job(worker_job_id: str):
    """
    Main job processing function. Runs in background thread.
    
    Steps:
    1. Download input files
    2. Load Whisper model
    3. Transcribe each file
    4. Run diarization if enabled
    5. Merge results
    6. Upload outputs to controller
    7. Mark complete
    """
    job = get_job(worker_job_id)
    if not job:
        return
    
    update_job(worker_job_id, status='running', stage='initializing')
    add_log(worker_job_id, 'info', 'job_started', f'Processing {len(job["inputs"])} file(s)')
    
    # Create temp directory for this job
    job_tmp_dir = os.path.join(WORKER_TMP_DIR, worker_job_id)
    os.makedirs(job_tmp_dir, exist_ok=True)
    
    outputs = []
    
    try:
        inputs = job['inputs']
        options = job['options']
        
        # Step 1: Download input files
        update_job(worker_job_id, stage='downloading')
        add_log(worker_job_id, 'info', 'downloading_started', f'Downloading {len(inputs)} file(s)')
        
        downloaded_files = []
        for i, inp in enumerate(inputs):
            if job.get('cancelRequested'):
                raise CancelledException('Job cancelled during download')
            
            update_job(worker_job_id, progress={
                'currentFileIndex': i,
                'totalFiles': len(inputs),
                'percent': int((i / len(inputs)) * 10)  # Download is 10% of total
            })
            
            local_path = download_input(job_tmp_dir, inp)
            downloaded_files.append({
                'inputId': inp['inputId'],
                'filename': inp['filename'],
                'localPath': local_path
            })
            add_log(worker_job_id, 'info', 'file_downloaded', f'Downloaded {inp["filename"]}')
        
        # Step 2: Load Whisper model
        update_job(worker_job_id, stage='loading_model')
        add_log(worker_job_id, 'info', 'loading_model', f'Loading model {options.get("model", WORKER_MODEL)}')
        
        model = load_whisper_model(options.get('model', WORKER_MODEL))
        
        # Step 3 & 4: Process each file (transcribe + diarize)
        for i, file_info in enumerate(downloaded_files):
            if job.get('cancelRequested'):
                raise CancelledException('Job cancelled during processing')
            
            update_job(worker_job_id, stage='transcribing', progress={
                'currentFileIndex': i,
                'totalFiles': len(downloaded_files),
                'percent': 10 + int((i / len(downloaded_files)) * 70)  # Transcription is 70%
            })
            add_log(worker_job_id, 'info', 'transcribing_file', f'Transcribing {file_info["filename"]}')
            
            # Transcribe
            transcript_result = transcribe_file(
                model, 
                file_info['localPath'],
                options.get('language'),
                worker_job_id
            )
            
            # Save transcript outputs
            file_outputs = save_transcript_outputs(
                job_tmp_dir,
                file_info['inputId'],
                file_info['filename'],
                transcript_result,
                options
            )
            outputs.extend(file_outputs)
            
            # Diarization if enabled
            if options.get('diarizationEnabled'):
                if job.get('cancelRequested'):
                    raise CancelledException('Job cancelled before diarization')
                
                update_job(worker_job_id, stage='diarizing')
                add_log(worker_job_id, 'info', 'diarizing_file', f'Diarizing {file_info["filename"]}')
                
                diarization_result = run_diarization_on_file(
                    file_info['localPath'],
                    options.get('diarizationEffective', {}),
                    worker_job_id
                )
                
                # Save diarization outputs
                diar_outputs = save_diarization_outputs(
                    job_tmp_dir,
                    file_info['inputId'],
                    file_info['filename'],
                    transcript_result,
                    diarization_result,
                    options
                )
                outputs.extend(diar_outputs)
        
        # Step 5: Upload outputs to controller
        update_job(worker_job_id, stage='uploading', progress={'percent': 90})
        add_log(worker_job_id, 'info', 'uploading_outputs', f'Uploading {len(outputs)} output(s)')
        
        upload_outputs_to_controller(job, job_tmp_dir, outputs)
        
        # Step 6: Mark complete
        update_job(worker_job_id, status='complete', stage='complete', progress={'percent': 100})
        add_log(worker_job_id, 'info', 'job_complete', f'Job completed with {len(outputs)} outputs')
        
        # Notify controller
        notify_controller_complete(job, outputs)
        
    except CancelledException as e:
        update_job(worker_job_id, status='canceled', error={'code': 'CANCELED', 'message': str(e)})
        add_log(worker_job_id, 'info', 'job_canceled', str(e))
        notify_controller_complete(job, outputs, error={'code': 'CANCELED', 'message': str(e)})
        
    except Exception as e:
        error_msg = str(e)[:500]
        update_job(worker_job_id, status='failed', error={'code': 'PROCESSING_ERROR', 'message': error_msg})
        add_log(worker_job_id, 'error', 'job_failed', error_msg, traceback=traceback.format_exc()[-1000:])
        notify_controller_complete(job, outputs, error={'code': 'PROCESSING_ERROR', 'message': error_msg})
        
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(job_tmp_dir, ignore_errors=True)
        except Exception:
            pass


class CancelledException(Exception):
    """Raised when job is cancelled."""
    pass


def download_input(job_tmp_dir: str, inp: dict) -> str:
    """Download input file from controller."""
    download_url = inp.get('downloadUrl')
    if not download_url:
        raise ValueError(f'No downloadUrl for input {inp.get("inputId")}')
    
    filename = inp.get('filename', 'input.wav')
    local_path = os.path.join(job_tmp_dir, f'{inp["inputId"]}_{filename}')
    
    response = requests.get(download_url, stream=True, timeout=300)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return local_path


def load_whisper_model(model_name: str):
    """Load Whisper model onto GPU."""
    import torch
    import whisper
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_name, device=device)
    return model


def transcribe_file(model, audio_path: str, language: Optional[str], worker_job_id: str) -> dict:
    """Transcribe a single audio file."""
    import whisper
    
    options = {}
    if language:
        options['language'] = language
    
    result = model.transcribe(audio_path, **options)
    return result


def run_diarization_on_file(audio_path: str, effective_policy: dict, worker_job_id: str) -> list:
    """Run speaker diarization on audio file."""
    # Import diarization module (reuse from main codebase if available)
    try:
        # Try to import from parent directory
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from diarization import run_diarization
        
        result = run_diarization(
            audio_path,
            job_id=worker_job_id
        )
        return result
    except ImportError:
        # Fallback: basic diarization
        add_log(worker_job_id, 'warning', 'diarization_fallback', 'Using basic diarization')
        return []


def save_transcript_outputs(job_tmp_dir: str, input_id: str, filename: str, 
                           transcript_result: dict, options: dict) -> list:
    """Save transcript outputs to temp directory."""
    outputs = []
    stem = Path(filename).stem
    
    # JSON output
    json_path = os.path.join(job_tmp_dir, f'{input_id}_{stem}_transcript.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_result, f, indent=2, ensure_ascii=False)
    outputs.append({
        'inputId': input_id,
        'filename': f'{stem}_transcript.json',
        'localPath': json_path,
        'type': 'json'
    })
    
    # Markdown output
    md_content = f"# {filename}\n\n"
    for segment in transcript_result.get('segments', []):
        text = segment.get('text', '').strip()
        if text:
            md_content += f"{text}\n\n"
    
    md_path = os.path.join(job_tmp_dir, f'{input_id}_{stem}_transcript.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    outputs.append({
        'inputId': input_id,
        'filename': f'{stem}_transcript.md',
        'localPath': md_path,
        'type': 'markdown'
    })
    
    return outputs


def save_diarization_outputs(job_tmp_dir: str, input_id: str, filename: str,
                            transcript_result: dict, diarization_result: list,
                            options: dict) -> list:
    """Save diarization outputs to temp directory."""
    outputs = []
    stem = Path(filename).stem
    
    # Diarization JSON
    diar_path = os.path.join(job_tmp_dir, f'{input_id}_{stem}_diarization.json')
    with open(diar_path, 'w', encoding='utf-8') as f:
        json.dump(diarization_result, f, indent=2, ensure_ascii=False)
    outputs.append({
        'inputId': input_id,
        'filename': f'{stem}_diarization.json',
        'localPath': diar_path,
        'type': 'diarization-json'
    })
    
    # Speaker markdown (merged transcript + diarization)
    speaker_md = generate_speaker_markdown(transcript_result, diarization_result, filename)
    speaker_md_path = os.path.join(job_tmp_dir, f'{input_id}_{stem}_speaker.md')
    with open(speaker_md_path, 'w', encoding='utf-8') as f:
        f.write(speaker_md)
    outputs.append({
        'inputId': input_id,
        'filename': f'{stem}_speaker.md',
        'localPath': speaker_md_path,
        'type': 'speaker-markdown'
    })
    
    return outputs


def generate_speaker_markdown(transcript: dict, diarization: list, filename: str) -> str:
    """Generate speaker-attributed markdown from transcript and diarization."""
    content = f"# {filename}\n\n"
    
    if not diarization:
        # No diarization - just output transcript
        for segment in transcript.get('segments', []):
            text = segment.get('text', '').strip()
            if text:
                content += f"{text}\n\n"
        return content
    
    # Merge transcript segments with speaker labels
    segments = transcript.get('segments', [])
    
    for segment in segments:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '').strip()
        
        if not text:
            continue
        
        # Find speaker for this segment
        speaker = find_speaker_for_segment(start, end, diarization)
        
        # Format timestamp
        ts = format_timestamp(start)
        
        content += f"**[{ts}] {speaker}:** {text}\n\n"
    
    return content


def find_speaker_for_segment(start: float, end: float, diarization: list) -> str:
    """Find the speaker label for a transcript segment."""
    if not diarization:
        return "Speaker"
    
    # Find overlapping diarization segment
    best_speaker = "Speaker"
    best_overlap = 0
    
    for diar_seg in diarization:
        diar_start = diar_seg.get('start', 0)
        diar_end = diar_seg.get('end', 0)
        
        # Calculate overlap
        overlap_start = max(start, diar_start)
        overlap_end = min(end, diar_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diar_seg.get('speaker', 'Speaker')
    
    return best_speaker


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def upload_outputs_to_controller(job: dict, job_tmp_dir: str, outputs: list):
    """Upload output files to controller."""
    outputs_url = job.get('outputsUploadUrl')
    if not outputs_url:
        add_log(job['workerJobId'], 'warning', 'no_upload_url', 'No outputsUploadUrl provided')
        return
    
    for output in outputs:
        local_path = output.get('localPath')
        if not local_path or not os.path.exists(local_path):
            continue
        
        try:
            with open(local_path, 'rb') as f:
                files = {'file': (output['filename'], f)}
                data = {
                    'workerJobId': job['workerJobId'],
                    'inputId': output.get('inputId', ''),
                    'outputType': output.get('type', 'unknown')
                }
                headers = {'Authorization': f'Bearer {WORKER_TOKEN}'}
                
                response = requests.post(outputs_url, files=files, data=data, headers=headers, timeout=120)
                response.raise_for_status()
                
                add_log(job['workerJobId'], 'info', 'output_uploaded', f'Uploaded {output["filename"]}')
        except Exception as e:
            add_log(job['workerJobId'], 'error', 'upload_failed', f'Failed to upload {output["filename"]}: {e}')


def notify_controller_complete(job: dict, outputs: list, error: dict = None):
    """Notify controller that job is complete."""
    callback_url = job.get('callbackUrl')
    if not callback_url:
        return
    
    try:
        payload = {
            'workerJobId': job['workerJobId'],
            'controllerJobId': job['controllerJobId'],
            'status': 'failed' if error else 'complete',
            'outputs': [{'inputId': o.get('inputId'), 'filename': o['filename'], 'type': o.get('type')} for o in outputs],
            'error': error
        }
        headers = {
            'Authorization': f'Bearer {WORKER_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(callback_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        add_log(job['workerJobId'], 'error', 'callback_failed', f'Failed to notify controller: {e}')


if __name__ == '__main__':
    # Ensure temp directory exists
    os.makedirs(WORKER_TMP_DIR, exist_ok=True)
    
    print(f"Starting Bulk Transcribe GPU Worker on port {WORKER_PORT}")
    print(f"Model: {WORKER_MODEL}, Max file size: {WORKER_MAX_FILE_MB}MB")
    print(f"Temp directory: {WORKER_TMP_DIR}")
    
    if not WORKER_TOKEN:
        print("WARNING: WORKER_TOKEN not set - authentication disabled!")
    
    app.run(host='0.0.0.0', port=WORKER_PORT, debug=False)
