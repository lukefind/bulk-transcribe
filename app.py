#!/usr/bin/env python3
"""
Web UI for Audio Transcription Tool using OpenAI Whisper
"""

# Prevent PyTorch from spawning subprocesses (causes duplicate dock icons on macOS)
import os
import sys
from pathlib import Path

# Set up ffmpeg path if running in PyInstaller bundle
if getattr(sys, 'frozen', False):
    base_path = Path(sys._MEIPASS)
    # PyInstaller puts binaries in different locations depending on structure
    possible_bin_paths = [
        base_path / 'bin',
        base_path / 'Frameworks' / 'bin',
        Path(sys.executable).parent / 'bin',
        Path(sys.executable).parent.parent / 'Frameworks' / 'bin',
    ]
    for bin_path in possible_bin_paths:
        if bin_path.exists() and (bin_path / 'ffmpeg').exists():
            os.environ['PATH'] = str(bin_path) + os.pathsep + os.environ.get('PATH', '')
            break

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Prevent multiprocessing from creating new app instances
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import subprocess
import threading
import queue
from dataclasses import replace
from flask import Flask, render_template, request, jsonify, send_file, g
import zipfile
import io
import tempfile
import shutil
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import json
from datetime import datetime, timezone

from transcribe_options import TranscribeOptions, get_preset, postprocess_segments
import session_store

app = Flask(__name__)

# Apply ProxyFix for HTTPS detection behind reverse proxy (Caddy, nginx)
# This ensures request.is_secure works correctly
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configure max upload size from environment
app.config['MAX_CONTENT_LENGTH'] = session_store.get_max_upload_mb() * 1024 * 1024

SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mp4', '.mov', '.avi']
AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
MODEL_INFO = {
    'tiny': {'size': '~75 MB', 'ram': '~1 GB', 'speed': 'Fastest', 'accuracy': 'Basic'},
    'base': {'size': '~145 MB', 'ram': '~1 GB', 'speed': 'Fast', 'accuracy': 'Good'},
    'small': {'size': '~465 MB', 'ram': '~2 GB', 'speed': 'Medium', 'accuracy': 'Better'},
    'medium': {'size': '~1.5 GB', 'ram': '~5 GB', 'speed': 'Slow', 'accuracy': 'High'},
    'large': {'size': '~3 GB', 'ram': '~10 GB', 'speed': 'Slowest', 'accuracy': 'Best'},
    'turbo': {'size': '~800 MB', 'ram': '~6 GB', 'speed': 'Fast', 'accuracy': 'Good'},
}
COMMON_LANGUAGES = [
    ('', 'Auto-detect'),
    ('en', 'English'),
    ('es', 'Spanish'),
    ('fr', 'French'),
    ('de', 'German'),
    ('it', 'Italian'),
    ('pt', 'Portuguese'),
    ('ja', 'Japanese'),
    ('ko', 'Korean'),
    ('zh', 'Chinese'),
    ('ru', 'Russian'),
    ('ar', 'Arabic'),
    ('hi', 'Hindi'),
]

transcription_status = {
    'running': False,
    'cancelled': False,
    'cancel_after_file': False,  # Stop after current file(s) complete
    'current_file': '',
    'current_file_progress': '',
    'current_file_percent': 0,
    'current_file_start': None,  # Timestamp when current file started
    'current_file_elapsed': 0,  # Seconds elapsed on current file
    'last_transcribed_text': '',
    'completed': 0,
    'total': 0,
    'results': [],
    'error': None,
    'downloading_model': False,
    'active_jobs': 0,
    'active_workers': 0,
    'current_files': [],  # List of files currently being processed (for parallel mode)
    'start_time': None,  # Job start timestamp
    'elapsed_seconds': 0,  # Total elapsed time
}

# Thread-safe lock for updating transcription_status
import threading
status_lock = threading.Lock()

# Request counter for periodic cleanup
_request_counter = 0
_cleanup_interval = 50  # Run cleanup every N requests


# =============================================================================
# Session Middleware
# =============================================================================

@app.before_request
def before_request_session():
    """Ensure session ID exists for all requests."""
    global _request_counter
    
    # Get or generate session ID
    session_id = request.cookies.get(session_store.COOKIE_NAME)
    if not session_id:
        session_id = session_store.new_id(32)
        g._new_session_id = session_id
    
    g.session_id = session_id
    
    # Periodic cleanup (exclude current session to prevent mid-request deletion)
    _request_counter += 1
    if _request_counter >= _cleanup_interval:
        _request_counter = 0
        try:
            session_store.cleanup_expired_sessions(exclude_session_id=session_id)
        except Exception:
            pass  # Don't fail requests due to cleanup errors


@app.after_request
def after_request_session(response):
    """Set session cookie if new, touch session metadata, and add security headers."""
    # Set cookie for new sessions
    if hasattr(g, '_new_session_id'):
        session_store.set_new_session_cookie(request, response, g._new_session_id)
    
    # Touch session to update lastSeenAt (skip for static files and healthz)
    if hasattr(g, 'session_id') and not request.path.startswith('/static') and request.path != '/healthz':
        try:
            session_store.touch_session(g.session_id)
        except Exception:
            pass  # Don't fail requests due to session touch errors
    
    # Security headers for API endpoints and downloads
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Pragma'] = 'no-cache'
        response.headers['X-Content-Type-Options'] = 'nosniff'
    
    return response


# CSRF-protected endpoints in server mode
_CSRF_PROTECTED_ENDPOINTS = {
    'api_upload_files',
    'api_create_job',
    'api_cancel_job',
}


def _require_csrf():
    """Check CSRF token for protected endpoints in server mode."""
    if not session_store.is_server_mode():
        return None  # No CSRF in local mode
    
    if request.endpoint not in _CSRF_PROTECTED_ENDPOINTS:
        return None
    
    token = request.headers.get('X-CSRF-Token', '')
    session_id = getattr(g, 'session_id', None)
    
    if not session_id or not session_store.validate_csrf_token(session_id, token):
        return jsonify({'error': 'Invalid or missing CSRF token'}), 403
    
    return None


@app.before_request
def before_request_csrf():
    """Validate CSRF token for protected endpoints."""
    result = _require_csrf()
    if result:
        return result


def choose_folder_dialog(prompt="Select a folder"):
    """Open native macOS folder picker dialog."""
    script = f'POSIX path of (choose folder with prompt "{prompt}")'
    try:
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def find_audio_files(input_folder: str) -> list:
    """Find all supported audio files in the input folder."""
    audio_files = []
    input_path = Path(input_folder)
    
    if not input_path.exists():
        return []
    
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            audio_files.append(file_path)
    
    return sorted(audio_files)


def get_whisper_cache_dir():
    """Get the whisper model cache directory."""
    return os.path.expanduser('~/.cache/whisper')


def get_preferences_file():
    """Get the path to the user preferences file."""
    config_dir = os.path.expanduser('~/.config/bulk-transcribe')
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, 'preferences.json')


def load_preferences():
    """Load user preferences from file."""
    prefs_file = get_preferences_file()
    default_prefs = {
        'default_model': None,
        'default_language': '',
        'include_segments': True,
        'include_full': False,
        'include_timestamps': True,
        'word_timestamps': False,
    }
    try:
        if os.path.exists(prefs_file):
            with open(prefs_file, 'r') as f:
                saved = json.load(f)
                default_prefs.update(saved)
    except Exception:
        pass
    return default_prefs


def save_preferences(prefs):
    """Save user preferences to file."""
    prefs_file = get_preferences_file()
    try:
        with open(prefs_file, 'w') as f:
            json.dump(prefs, f, indent=2)
        return True
    except Exception:
        return False


# Mapping from model name to actual filename in cache
MODEL_FILENAMES = {
    'tiny': 'tiny.pt',
    'base': 'base.pt',
    'small': 'small.pt',
    'medium': 'medium.pt',
    'large': 'large-v3.pt',
    'turbo': 'large-v3-turbo.pt',
}


def get_model_filepath(model: str) -> str:
    """Get the full path to a model file."""
    cache_dir = get_whisper_cache_dir()
    filename = MODEL_FILENAMES.get(model, f'{model}.pt')
    return os.path.join(cache_dir, filename)


def check_model_available(model: str) -> bool:
    """Check if a whisper model is already downloaded."""
    model_file = get_model_filepath(model)
    return os.path.exists(model_file)


def get_model_file_size(model: str) -> str:
    """Get the actual file size of a downloaded model."""
    model_file = get_model_filepath(model)
    if os.path.exists(model_file):
        size_bytes = os.path.getsize(model_file)
        if size_bytes >= 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        else:
            return f"{size_bytes / (1024 * 1024):.0f} MB"
    return None


def get_all_models_status():
    """Get status of all available models."""
    models = []
    for model_name in AVAILABLE_MODELS:
        info = MODEL_INFO.get(model_name, {})
        is_downloaded = check_model_available(model_name)
        actual_size = get_model_file_size(model_name) if is_downloaded else None
        models.append({
            'name': model_name,
            'downloaded': is_downloaded,
            'size': actual_size or info.get('size', 'Unknown'),
            'ram': info.get('ram', 'Unknown'),
            'speed': info.get('speed', 'Unknown'),
            'accuracy': info.get('accuracy', 'Unknown'),
        })
    return models


def transcribe_file(audio_file: Path, model: str, language: str, output_folder: Path, 
                    word_timestamps: bool = False, status_callback=None,
                    transcribe_options: TranscribeOptions = None) -> dict:
    """Transcribe a single audio file using Whisper Python library with progress callback."""
    global transcription_status
    
    import time
    import threading
    
    start_time = time.time()
    
    transcription_status['current_file_progress'] = f'Loading model ({model})...'
    transcription_status['current_file_percent'] = 0
    # Do not clear last_transcribed_text here; keep the last snippet visible during processing
    # and update it when we have new text.
    
    try:
        import whisper
        
        whisper_model = None
        if status_callback is not None and isinstance(status_callback, dict):
            whisper_model = status_callback.get('whisper_model')

        if whisper_model is None:
            transcription_status['current_file_progress'] = f'Loading {model} model...'
            transcription_status['current_file_percent'] = 5
            # MPS has bugs with some audio files, use CPU for stability
            whisper_model = whisper.load_model(model)
        
        transcription_status['current_file_progress'] = 'Transcribing audio...'
        transcription_status['current_file_percent'] = 10
        
        # Check for cancellation before transcription
        if transcription_status.get('cancelled'):
            return {'error': 'Cancelled by user'}
        
        # Get audio duration for progress estimation
        import whisper.audio
        audio_array = whisper.audio.load_audio(str(audio_file))
        audio_duration = len(audio_array) / whisper.audio.SAMPLE_RATE
        
        # Background thread for progress updates based on segments
        stop_progress = threading.Event()
        segments_received = []
        
        def progress_update():
            while not stop_progress.is_set():
                if transcription_status.get('cancelled'):
                    return
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
                
                # Estimate progress based on audio duration
                # CPU on Apple Silicon with turbo: ~0.5-1x realtime
                # With parallel workers, each file is slower due to CPU contention
                if audio_duration > 0:
                    estimated_total_time = audio_duration * 1.5  # Assume 0.67x realtime on CPU
                    estimated_pct = min(95, int(10 + (elapsed / estimated_total_time) * 85))
                else:
                    # Fallback: assume 30 seconds per percent after initial 10%
                    estimated_pct = min(95, 10 + int(elapsed / 30))
                
                # Only update if we're the current/only file being processed
                # In parallel mode, don't fight over the progress bar
                with status_lock:
                    if len(transcription_status.get('current_files', [])) <= 1:
                        transcription_status['current_file_percent'] = estimated_pct
                        transcription_status['current_file_progress'] = f'Transcribing... ({elapsed_str} elapsed)'
                
                stop_progress.wait(1.0)  # Check less frequently
        
        progress_thread = threading.Thread(target=progress_update, daemon=True)
        progress_thread.start()
        
        # Build transcription options (clone per call to avoid cross-thread mutation)
        if transcribe_options is None:
            transcribe_options = get_preset('balanced')
        else:
            transcribe_options = replace(transcribe_options)

        # Override with function parameters
        transcribe_options.language = language if language else None
        transcribe_options.word_timestamps = word_timestamps

        # Transcribe with explicit parameters
        whisper_kwargs = transcribe_options.to_whisper_kwargs()
        result = whisper.transcribe(
            whisper_model,
            str(audio_file),
            **whisper_kwargs
        )
        
        # Post-process segments to improve quality
        if result.get('segments'):
            result['segments'] = postprocess_segments(
                result['segments'],
                merge_short=transcribe_options.merge_short_segments,
                min_duration=transcribe_options.min_segment_duration,
                max_duration=transcribe_options.max_segment_duration,
                word_timestamps_available=word_timestamps and any(
                    s.get('words') for s in result['segments']
                ),
            )
        
        stop_progress.set()
        
        # Check for cancellation after transcription
        if transcription_status.get('cancelled'):
            return {'error': 'Cancelled by user'}
        
        transcription_status['current_file_progress'] = 'Saving results...'
        transcription_status['current_file_percent'] = 98
        
        # Update last transcribed text with final segment (thread-safe)
        if result.get('segments') and len(result['segments']) > 0:
            last_segment = result['segments'][-1]
            snippet = last_segment.get('text', '').strip()[:150]
            print(f"[DEBUG] Setting snippet: '{snippet[:50]}...' from {len(result['segments'])} segments")
            if snippet:
                with status_lock:
                    transcription_status['last_transcribed_text'] = snippet
        else:
            print(f"[DEBUG] No segments in result: {result.keys() if isinstance(result, dict) else type(result)}")
        
        # Save JSON output
        json_output_path = output_folder / (audio_file.stem + '.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
        transcription_status['current_file_progress'] = f'Complete ({elapsed_str})'
        transcription_status['current_file_percent'] = 100
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Transcription error: {error_msg}")
        print(traceback.format_exc())
        if 'CUDA' in error_msg or 'cuda' in error_msg:
            error_msg = 'GPU/CUDA error - try restarting the app'
        elif 'out of memory' in error_msg.lower():
            error_msg = 'Out of memory - try a smaller model'
        elif 'No such file' in error_msg or 'FileNotFoundError' in error_msg:
            error_msg = f'Audio file not found: {error_msg[:100]}'
        return {'error': error_msg[:200]}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def create_markdown(audio_file: Path, transcription_data: dict, model: str, 
                   language: str, output_folder: Path, options: dict = None) -> Path:
    """Create a markdown file from transcription data."""
    if options is None:
        options = {}
    
    include_full = options.get('include_full', True)
    include_segments = options.get('include_segments', True)
    include_timestamps = options.get('include_timestamps', True)
    include_word_timestamps = options.get('word_timestamps', False)
    
    output_name = audio_file.stem + '_transcription.md'
    output_path = output_folder / output_name
    
    content = f"# Transcription: {audio_file.name}\n\n"
    content += f"**Source File:** `{audio_file.name}`\n"
    content += f"**Model Used:** `{model}`\n"
    
    if language:
        content += f"**Language:** `{language}`\n"
    
    if 'language' in transcription_data:
        content += f"**Detected Language:** `{transcription_data['language']}`\n"
    
    # Add segment count for reference
    segment_count = len(transcription_data.get('segments', []))
    if segment_count > 0:
        content += f"**Segments:** `{segment_count}`\n"
    
    content += f"\n---\n\n"
    
    if include_full and 'text' in transcription_data:
        content += "## Full Transcription\n\n"
        content += transcription_data['text'].strip() + "\n\n"
    
    if include_segments and 'segments' in transcription_data:
        content += "## Segmented Transcription\n\n"
        for i, segment in enumerate(transcription_data['segments'], 1):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if include_timestamps:
                content += f"**[{format_timestamp(start_time)} - {format_timestamp(end_time)}]**\n\n"
            
            content += f"{text}\n\n"
            
            # Include word-level timestamps if available and requested
            if include_word_timestamps and segment.get('words'):
                words = segment['words']
                word_line = ' '.join(
                    f"`{w.get('word', '')}@{w.get('start', 0):.2f}`" 
                    for w in words if w.get('word')
                )
                if word_line:
                    content += f"<details><summary>Word timestamps</summary>\n\n{word_line}\n\n</details>\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path


def run_transcription(input_folder: str, output_folder: str, model: str, language: str, options: dict = None):
    """Background task to run transcription with optional parallel workers."""
    global transcription_status
    
    if options is None:
        options = {}
    
    processing_mode = options.get('processing_mode', 'metal')
    word_timestamps = options.get('word_timestamps', False)
    
    # Compute effective mode: Metal + word_timestamps forces CPU (MPS doesn't support DTW float64)
    effective_mode = processing_mode
    if processing_mode == 'metal' and word_timestamps:
        effective_mode = 'cpu'
    
    # Only force single worker when actually using Metal GPU; allow multiple for CPU
    if effective_mode == 'metal':
        num_workers = 1
    else:
        num_workers = max(1, min(2, options.get('workers', 1)))  # Max 2 workers for CPU mode
    
    import time as time_module
    transcription_status['running'] = True
    transcription_status['active_jobs'] = 1
    transcription_status['active_workers'] = 0
    transcription_status['cancelled'] = False
    transcription_status['cancel_after_file'] = False
    transcription_status['start_time'] = time_module.time()
    transcription_status['elapsed_seconds'] = 0
    transcription_status['last_transcribed_text'] = ''
    transcription_status['completed'] = 0
    transcription_status['results'] = []
    transcription_status['error'] = None
    transcription_status['current_files'] = []
    
    try:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = find_audio_files(input_folder)
        transcription_status['total'] = len(audio_files)
        
        if not audio_files:
            transcription_status['error'] = 'No audio files found in the input folder'
            transcription_status['running'] = False
            return
        
        overwrite_existing = options.get('overwrite_existing', False)
        
        # Build TranscribeOptions from advanced settings
        quality_preset = options.get('quality_preset', 'balanced')
        transcribe_opts = get_preset(quality_preset)
        
        # Override with specific settings from UI
        if options.get('no_speech_threshold') is not None:
            transcribe_opts.no_speech_threshold = float(options['no_speech_threshold'])
        if word_timestamps and options.get('max_segment_duration') is not None:
            transcribe_opts.max_segment_duration = float(options['max_segment_duration'])
        else:
            transcribe_opts.max_segment_duration = None
        
        transcribe_opts.merge_short_segments = True
        transcribe_opts.min_segment_duration = 0.5
        
        # Filter out files that already have output (if not overwriting)
        files_to_process = []
        for audio_file in audio_files:
            expected_output = output_path / f"{audio_file.stem}_transcription.md"
            if expected_output.exists() and not overwrite_existing:
                with status_lock:
                    transcription_status['results'].append({
                        'file': audio_file.name,
                        'status': 'skipped',
                        'message': 'Output file already exists',
                        'elapsed_seconds': 0.0,
                        'audio_seconds': None,
                        'speed_x': None,
                        'segments_count': None,
                    })
                    transcription_status['completed'] += 1
            else:
                files_to_process.append(audio_file)
        
        if not files_to_process:
            # All files were skipped
            return
        
        # Worker function for parallel processing
        def process_file(audio_file, worker_model):
            """Process a single file with the given model instance."""
            if transcription_status.get('cancelled'):
                return None
            
            # Track this file as being processed
            import time
            file_start_time = time.time()
            with status_lock:
                transcription_status['current_files'].append(audio_file.name)
                transcription_status['active_workers'] = len(transcription_status['current_files'])
                if len(transcription_status['current_files']) == 1:
                    transcription_status['current_file'] = audio_file.name
                    transcription_status['current_file_start'] = file_start_time
                else:
                    transcription_status['current_file'] = f"{len(transcription_status['current_files'])} files"
            
            try:
                result = transcribe_file(
                    audio_file,
                    model,
                    language,
                    output_path,
                    word_timestamps,
                    status_callback={'whisper_model': worker_model},
                    transcribe_options=transcribe_opts,
                )
            except Exception as e:
                result = {'error': str(e)[:200]}
            
            file_elapsed_seconds = max(0.0, time.time() - file_start_time)
            audio_seconds = None
            segments_count = None
            speed_x = None
            
            try:
                segments = result.get('segments') if isinstance(result, dict) else None
                if segments:
                    segments_count = len(segments)
                    last_end = segments[-1].get('end')
                    if isinstance(last_end, (int, float)):
                        audio_seconds = float(last_end)
            except Exception:
                pass
            
            if audio_seconds and file_elapsed_seconds > 0:
                speed_x = audio_seconds / file_elapsed_seconds
            
            # Record result
            with status_lock:
                if 'error' in result:
                    transcription_status['results'].append({
                        'file': audio_file.name,
                        'status': 'error',
                        'message': result['error'],
                        'elapsed_seconds': file_elapsed_seconds,
                        'audio_seconds': audio_seconds,
                        'speed_x': speed_x,
                        'segments_count': segments_count,
                    })
                else:
                    md_path = create_markdown(audio_file, result, model, language, output_path, options)
                    transcription_status['results'].append({
                        'file': audio_file.name,
                        'status': 'success',
                        'output': md_path.name,
                        'elapsed_seconds': file_elapsed_seconds,
                        'audio_seconds': audio_seconds,
                        'speed_x': speed_x,
                        'segments_count': segments_count,
                    })
                
                transcription_status['completed'] += 1
                
                # Remove from current files list
                if audio_file.name in transcription_status['current_files']:
                    transcription_status['current_files'].remove(audio_file.name)
                transcription_status['active_workers'] = len(transcription_status['current_files'])
                if transcription_status['current_files']:
                    if len(transcription_status['current_files']) == 1:
                        transcription_status['current_file'] = transcription_status['current_files'][0]
                    else:
                        transcription_status['current_file'] = f"{len(transcription_status['current_files'])} files"
                else:
                    transcription_status['current_file'] = ''
                    transcription_status['current_file_start'] = None
            
            return result
        
        if processing_mode in ('metal', 'cuda') or num_workers == 1:
            # Single-threaded mode with GPU (Metal on macOS, CUDA on Linux)
            transcription_status['current_file_progress'] = f'Loading {model} model...'
            transcription_status['current_file_percent'] = 1
            try:
                import whisper
                import torch
                
                # Determine device based on processing_mode and availability
                # Note: MPS doesn't support float64 required by word timestamp alignment (DTW),
                # so we must fall back to CPU when word timestamps are enabled.
                device = "cpu"
                if processing_mode == 'cuda' and torch.cuda.is_available():
                    device = "cuda"
                    transcription_status['current_file_progress'] = f'Loading {model} model (CUDA GPU)...'
                elif processing_mode == 'metal' and torch.backends.mps.is_available() and not word_timestamps:
                    device = "mps"
                    transcription_status['current_file_progress'] = f'Loading {model} model (Metal GPU)...'
                else:
                    if word_timestamps and processing_mode == 'metal':
                        transcription_status['current_file_progress'] = f'Loading {model} model (CPU - required for word timestamps)...'
                    else:
                        transcription_status['current_file_progress'] = f'Loading {model} model (CPU)...'
                whisper_model = whisper.load_model(model, device=device)
            except Exception as e:
                transcription_status['error'] = f'Failed to load model: {str(e)[:200]}'
                return
            
            for audio_file in files_to_process:
                if transcription_status['cancelled']:
                    transcription_status['error'] = 'Cancelled by user'
                    break
                if transcription_status['cancel_after_file']:
                    transcription_status['error'] = 'Stopped after completing file'
                    break
                process_file(audio_file, whisper_model)
        else:
            # Parallel mode: each worker loads its own model
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import whisper
            
            transcription_status['current_file_progress'] = f'Loading {num_workers} model instances...'
            transcription_status['current_file_percent'] = 1
            
            # Pre-load models for each worker
            worker_models = []
            try:
                for i in range(num_workers):
                    if transcription_status['cancelled']:
                        break
                    transcription_status['current_file_progress'] = f'Loading model {i+1}/{num_workers}...'
                    # Use CPU for parallel workers - MPS has issues with multiple models
                    worker_models.append(whisper.load_model(model, device='cpu'))
            except Exception as e:
                transcription_status['error'] = f'Failed to load models: {str(e)[:200]}'
                return
            
            if transcription_status['cancelled']:
                transcription_status['error'] = 'Cancelled by user'
                return
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                model_idx = 0
                
                for audio_file in files_to_process:
                    if transcription_status['cancelled']:
                        break
                    if transcription_status['cancel_after_file']:
                        break
                    # Round-robin assign models to workers
                    worker_model = worker_models[model_idx % len(worker_models)]
                    model_idx += 1
                    future = executor.submit(process_file, audio_file, worker_model)
                    futures[future] = audio_file
                
                # Wait for all to complete (or be cancelled)
                for future in as_completed(futures):
                    if transcription_status['cancelled']:
                        # Don't wait for remaining futures
                        break
                    try:
                        future.result()
                    except Exception:
                        pass
                    # Check cancel_after_file after each file completes
                    if transcription_status['cancel_after_file']:
                        break
            
            if transcription_status['cancelled']:
                transcription_status['error'] = 'Cancelled by user'
            elif transcription_status['cancel_after_file']:
                transcription_status['error'] = 'Stopped after completing file(s)'
        
    except Exception as e:
        transcription_status['error'] = str(e)
    finally:
        transcription_status['running'] = False
        transcription_status['active_jobs'] = 0
        transcription_status['active_workers'] = 0
        transcription_status['current_file'] = ''
        transcription_status['current_files'] = []


@app.route('/')
def index():
    return render_template('index.html', 
                         models=AVAILABLE_MODELS, 
                         languages=COMMON_LANGUAGES,
                         app_mode=session_store.get_app_mode(),
                         is_server_mode=session_store.is_server_mode())


@app.route('/start', methods=['POST'])
def start_transcription():
    """Start transcription with folder paths. Blocked in server mode."""
    global transcription_status
    
    if session_store.is_server_mode():
        return jsonify({'error': 'Folder-based transcription not available in server mode. Use /api/jobs.'}), 400
    
    if transcription_status['running']:
        return jsonify({'error': 'Transcription already in progress'}), 400
    
    data = request.json
    input_folder = data.get('input_folder', '').strip()
    output_folder = data.get('output_folder', '').strip()
    model = data.get('model', 'base')
    language = data.get('language', '')
    
    options = {
        'include_segments': data.get('include_segments', True),
        'include_full': data.get('include_full', True),
        'include_timestamps': data.get('include_timestamps', True),
        'word_timestamps': data.get('word_timestamps', False),
        'overwrite_existing': data.get('overwrite_existing', False),
        'processing_mode': data.get('processing_mode', 'metal'),
        'workers': data.get('workers', 1),
        'quality_preset': data.get('quality_preset', 'balanced'),
        'no_speech_threshold': data.get('no_speech_threshold'),
        'max_segment_duration': data.get('max_segment_duration'),
    }
    
    if not input_folder:
        return jsonify({'error': 'Input folder is required'}), 400
    if not output_folder:
        return jsonify({'error': 'Output folder is required'}), 400
    if not Path(input_folder).exists():
        return jsonify({'error': 'Input folder does not exist'}), 400
    
    # Start transcription in background thread
    thread = threading.Thread(
        target=run_transcription,
        args=(input_folder, output_folder, model, language, options)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/status')
def get_status():
    import time as time_module
    # Update elapsed time if running
    if transcription_status['running'] and transcription_status.get('start_time'):
        transcription_status['elapsed_seconds'] = time_module.time() - transcription_status['start_time']
    # Update current file elapsed time
    if transcription_status.get('current_file_start'):
        transcription_status['current_file_elapsed'] = time_module.time() - transcription_status['current_file_start']
    else:
        transcription_status['current_file_elapsed'] = 0
    return jsonify(transcription_status)


@app.route('/cancel', methods=['POST'])
def cancel_transcription():
    """Cancel the current transcription job.
    
    Modes:
    - 'force': Stop immediately (current file may be incomplete)
    - 'after_file': Complete current file(s), then stop
    """
    global transcription_status
    
    if not transcription_status['running']:
        return jsonify({'error': 'No transcription in progress'}), 400
    
    data = request.get_json() or {}
    mode = data.get('mode', 'force')
    
    if mode == 'after_file':
        transcription_status['cancel_after_file'] = True
        return jsonify({'status': 'cancelling_after_file'})
    else:
        # Force stop - set cancelled and immediately mark as not running
        # The background thread will clean up when it checks the flag
        transcription_status['cancelled'] = True
        transcription_status['running'] = False
        transcription_status['error'] = 'Force stopped by user'
        transcription_status['current_file'] = ''
        transcription_status['current_file_progress'] = ''
        transcription_status['current_file_percent'] = 0
        return jsonify({'status': 'stopped'})


@app.route('/healthz')
def healthz():
    """Health check endpoint for Docker/orchestration."""
    return jsonify({'ok': True})


@app.route('/preview')
def preview_files():
    """Preview files in a folder. Blocked in server mode."""
    if session_store.is_server_mode():
        return jsonify({'error': 'Folder preview not available in server mode. Use /api/uploads.'}), 404
    
    input_folder = request.args.get('folder', '')
    if not input_folder or not Path(input_folder).exists():
        return jsonify({'files': [], 'error': 'Folder not found'})
    
    files = find_audio_files(input_folder)
    return jsonify({
        'files': [f.name for f in files],
        'count': len(files)
    })


@app.route('/browse', methods=['POST'])
def browse_folder():
    """Open native folder picker. Blocked in server mode."""
    if session_store.is_server_mode():
        return jsonify({'error': 'Folder browser not available in server mode.'}), 404
    
    data = request.json or {}
    prompt = data.get('prompt', 'Select a folder')
    folder = choose_folder_dialog(prompt)
    if folder:
        return jsonify({'folder': folder})
    return jsonify({'folder': None, 'cancelled': True})


def get_recommended_model():
    """Get the recommended model to use (user default or best downloaded)."""
    prefs = load_preferences()
    
    # If user has set a default and it's downloaded, use it
    if prefs.get('default_model'):
        if check_model_available(prefs['default_model']):
            return prefs['default_model']
    
    # Otherwise, prefer turbo if downloaded (best speed/quality balance)
    if check_model_available('turbo'):
        return 'turbo'
    
    # Fallback to any downloaded model
    priority_order = ['turbo', 'base', 'small', 'medium', 'large', 'tiny']
    for model in priority_order:
        if check_model_available(model):
            return model
    
    # Fallback to turbo if nothing downloaded (will download on first use)
    return 'turbo'


@app.route('/models')
def list_models():
    """Get list of all models with their status."""
    prefs = load_preferences()
    recommended = get_recommended_model()
    return jsonify({
        'models': get_all_models_status(),
        'recommended': recommended,
        'user_default': prefs.get('default_model')
    })


@app.route('/preferences')
def get_preferences():
    """Get user preferences."""
    return jsonify(load_preferences())


@app.route('/preferences', methods=['POST'])
def set_preferences():
    """Save user preferences."""
    data = request.json or {}
    prefs = load_preferences()
    
    # Update only provided fields
    for key in ['default_model', 'default_language', 'include_segments', 
                'include_full', 'include_timestamps', 'word_timestamps']:
        if key in data:
            prefs[key] = data[key]
    
    if save_preferences(prefs):
        return jsonify({'status': 'saved', 'preferences': prefs})
    return jsonify({'error': 'Failed to save preferences'}), 500


@app.route('/models/set-default', methods=['POST'])
def set_default_model():
    """Set the default model."""
    data = request.json or {}
    model_name = data.get('model', '')
    
    if model_name and model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    prefs = load_preferences()
    prefs['default_model'] = model_name if model_name else None
    
    if save_preferences(prefs):
        return jsonify({'status': 'saved', 'default_model': prefs['default_model']})
    return jsonify({'error': 'Failed to save preference'}), 500


model_download_status = {
    'downloading': False,
    'model': '',
    'progress': ''
}


@app.route('/models/download', methods=['POST'])
def download_model():
    """Download a whisper model."""
    global model_download_status
    
    if model_download_status['downloading']:
        return jsonify({'error': 'A download is already in progress'}), 400
    
    data = request.json or {}
    model_name = data.get('model', '')
    
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    if check_model_available(model_name):
        return jsonify({'error': f'Model {model_name} is already downloaded'}), 400
    
    def do_download():
        global model_download_status
        model_download_status['downloading'] = True
        model_download_status['model'] = model_name
        model_download_status['progress'] = 'Initializing download...'
        
        model_sizes = {
            'tiny': '75 MB',
            'base': '145 MB', 
            'small': '465 MB',
            'medium': '1.5 GB',
            'large': '3 GB',
            'turbo': '800 MB'
        }
        size_str = model_sizes.get(model_name, 'Unknown size')
        
        try:
            import time
            start_time = time.time()
            
            model_download_status['progress'] = f'Connecting to download server...'
            time.sleep(0.5)
            
            model_download_status['progress'] = f'Downloading {model_name} model ({size_str})... This may take several minutes.'
            
            # Monitor the cache directory for file size changes
            model_file = get_model_filepath(model_name)
            
            def update_progress():
                while model_download_status['downloading']:
                    elapsed = time.time() - start_time
                    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
                    
                    if os.path.exists(model_file):
                        current_size = os.path.getsize(model_file)
                        if current_size >= 1024 * 1024 * 1024:
                            size_downloaded = f"{current_size / (1024 * 1024 * 1024):.2f} GB"
                        else:
                            size_downloaded = f"{current_size / (1024 * 1024):.0f} MB"
                        model_download_status['progress'] = f'Downloading {model_name}: {size_downloaded} of {size_str} ({elapsed_str} elapsed)'
                    else:
                        model_download_status['progress'] = f'Downloading {model_name} model ({size_str})... {elapsed_str} elapsed'
                    time.sleep(1)
            
            # Start progress monitor in separate thread
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
            
            # Do the actual download
            import whisper
            whisper.load_model(model_name)
            
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
            model_download_status['progress'] = f'Complete! Downloaded {model_name} in {elapsed_str}'
            
        except Exception as e:
            model_download_status['progress'] = f'Error: {str(e)[:100]}'
        finally:
            model_download_status['downloading'] = False
    
    thread = threading.Thread(target=do_download, daemon=True)
    thread.start()
    
    return jsonify({'status': 'downloading', 'model': model_name})


@app.route('/models/download/status')
def download_status():
    """Get the current model download status."""
    return jsonify(model_download_status)


@app.route('/models/delete', methods=['POST'])
def delete_model():
    """Delete a downloaded whisper model."""
    data = request.json or {}
    model_name = data.get('model', '')
    
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    model_file = get_model_filepath(model_name)
    
    if not os.path.exists(model_file):
        return jsonify({'error': f'Model {model_name} is not downloaded'}), 400
    
    try:
        os.remove(model_file)
        return jsonify({'status': 'deleted', 'model': model_name})
    except Exception as e:
        return jsonify({'error': f'Failed to delete: {str(e)}'}), 500


@app.route('/upload', methods=['POST'])
def upload_files_legacy():
    """Upload files to a folder path. Blocked in server mode."""
    if session_store.is_server_mode():
        return jsonify({'error': 'Folder-based upload not available in server mode. Use /api/uploads.'}), 400
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Get target folder from form data or use default
    target_folder = request.form.get('folder', os.environ.get('INPUT_DIR', '/data/input'))
    target_path = Path(target_folder)
    
    # Create folder if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    uploaded = []
    skipped = []
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
            
        filename = secure_filename(file.filename)
        if not filename:
            continue
            
        # Check if it's a supported audio format
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            skipped.append({'name': filename, 'reason': f'Unsupported format: {ext}'})
            continue
        
        try:
            filepath = target_path / filename
            file.save(str(filepath))
            uploaded.append(filename)
        except Exception as e:
            errors.append({'name': filename, 'error': str(e)[:100]})
    
    return jsonify({
        'uploaded': uploaded,
        'skipped': skipped,
        'errors': errors,
        'folder': str(target_path),
        'total_uploaded': len(uploaded)
    })


@app.route('/download', methods=['GET'])
def download_transcriptions_legacy():
    """Download from folder path. Blocked in server mode."""
    if session_store.is_server_mode():
        return jsonify({'error': 'Folder-based download not available in server mode. Use /api/jobs/<job_id>/download.'}), 404
    
    output_folder = request.args.get('folder', os.environ.get('OUTPUT_DIR', '/data/output'))
    output_path = Path(output_folder)
    
    if not output_path.exists():
        return jsonify({'error': 'Output folder does not exist'}), 404
    
    # Find all transcription files (markdown and json)
    files_to_zip = []
    for ext in ['*.md', '*.json', '*.txt', '*.srt']:
        files_to_zip.extend(output_path.glob(ext))
    
    if not files_to_zip:
        return jsonify({'error': 'No transcription files found'}), 404
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filepath in files_to_zip:
            zf.write(filepath, filepath.name)
    
    zip_buffer.seek(0)
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'transcriptions_{timestamp}.zip'
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )


@app.route('/download/list', methods=['GET'])
def list_transcriptions_legacy():
    """List files in folder. Blocked in server mode."""
    if session_store.is_server_mode():
        return jsonify({'error': 'Folder-based listing not available in server mode. Use /api/jobs/<job_id>/outputs.'}), 404
    
    output_folder = request.args.get('folder', os.environ.get('OUTPUT_DIR', '/data/output'))
    output_path = Path(output_folder)
    
    if not output_path.exists():
        return jsonify({'files': [], 'folder': str(output_path), 'exists': False})
    
    files = []
    for ext in ['*.md', '*.json', '*.txt', '*.srt']:
        for filepath in output_path.glob(ext):
            stat = filepath.stat()
            files.append({
                'name': filepath.name,
                'size': stat.st_size,
                'modified': stat.st_mtime
            })
    
    # Sort by modification time, newest first
    files.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify({
        'files': files,
        'folder': str(output_path),
        'exists': True,
        'count': len(files)
    })


# =============================================================================
# Session-Isolated API Endpoints (Server Mode)
# =============================================================================

# In-memory job runners keyed by (session_id, job_id) for cancellation
_active_jobs = {}
_active_jobs_lock = threading.Lock()


@app.route('/api/uploads', methods=['POST'])
def api_upload_files():
    """Upload audio files to session-isolated storage."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Enforce file count limit
    max_files = session_store.get_max_files_per_job()
    if len(files) > max_files:
        return jsonify({'error': f'Too many files. Maximum {max_files} per upload.'}), 400
    
    session_id = g.session_id
    session_store.ensure_session_dirs(session_id)
    uploads_path = Path(session_store.uploads_dir(session_id))
    
    uploaded = []
    skipped = []
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
        
        original_name = file.filename
        sanitized = session_store.sanitize_filename(original_name)
        ext = Path(sanitized).suffix.lower()
        
        # Check supported format
        if ext not in SUPPORTED_FORMATS:
            skipped.append({'name': original_name, 'reason': f'Unsupported format: {ext}'})
            continue
        
        try:
            upload_id = session_store.new_id(8)
            stored_name = f"{upload_id}_{sanitized}"
            filepath = uploads_path / stored_name
            file.save(str(filepath))
            
            uploaded.append({
                'id': upload_id,
                'filename': original_name,
                'storedName': stored_name,
                'size': filepath.stat().st_size
            })
        except Exception as e:
            errors.append({'name': original_name, 'error': str(e)[:100]})
    
    return jsonify({
        'uploads': uploaded,
        'skipped': skipped,
        'errors': errors
    })


@app.route('/api/uploads', methods=['GET'])
def api_list_uploads():
    """List uploaded files for current session."""
    session_id = g.session_id
    uploads = session_store.list_uploads(session_id)
    return jsonify({'uploads': uploads})


@app.route('/api/jobs', methods=['POST'])
def api_create_job():
    """Create and start a transcription job."""
    session_id = g.session_id
    
    # Check for existing running job in this session
    with _active_jobs_lock:
        for key, job_info in _active_jobs.items():
            if key[0] == session_id and job_info.get('running'):
                return jsonify({'error': 'A job is already running in this session'}), 409
    
    data = request.json or {}
    upload_ids = data.get('uploadIds', [])
    options = data.get('options', {})
    
    if not upload_ids:
        return jsonify({'error': 'No upload IDs provided'}), 400
    
    # Resolve upload IDs to file paths
    inputs = []
    for upload_id in upload_ids:
        filepath = session_store.find_upload_by_id(session_id, upload_id)
        if not filepath:
            return jsonify({'error': f'Upload not found: {upload_id}'}), 404
        
        # Extract original filename from stored name
        stored_name = os.path.basename(filepath)
        parts = stored_name.split('_', 1)
        original_name = parts[1] if len(parts) > 1 else stored_name
        
        inputs.append({
            'uploadId': upload_id,
            'path': filepath,
            'originalFilename': original_name
        })
    
    # Create job
    job_id = session_store.new_id(12)
    dirs = session_store.ensure_job_dirs(session_id, job_id)
    
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        'jobId': job_id,
        'sessionId': session_id,
        'createdAt': now,
        'updatedAt': now,
        'status': 'queued',
        'options': {
            'model': options.get('model', 'base'),
            'language': options.get('language', ''),
            'includeSegments': options.get('includeSegments', True),
            'includeFull': options.get('includeFull', True),
            'includeTimestamps': options.get('includeTimestamps', True),
            'wordTimestamps': options.get('wordTimestamps', False),
            'qualityPreset': options.get('qualityPreset', 'balanced'),
        },
        'inputs': inputs,
        'outputs': [],
        'progress': {
            'total': len(inputs),
            'completed': 0,
            'currentFile': '',
            'percent': 0
        },
        'error': None
    }
    
    session_store.atomic_write_json(session_store.job_manifest_path(session_id, job_id), manifest)
    
    # Start job in background thread
    def run_job():
        _run_session_job(session_id, job_id, inputs, manifest['options'], dirs['outputs'])
    
    with _active_jobs_lock:
        _active_jobs[(session_id, job_id)] = {'running': True, 'cancel_requested': False}
    
    job_thread = threading.Thread(target=run_job, daemon=True)
    job_thread.start()
    
    return jsonify({'jobId': job_id})


def _run_session_job(session_id: str, job_id: str, inputs: list, options: dict, outputs_dir: str):
    """Run transcription job and update manifest with progress."""
    manifest_path = session_store.job_manifest_path(session_id, job_id)
    
    def update_manifest(**updates):
        manifest = session_store.read_json(manifest_path) or {}
        manifest['updatedAt'] = datetime.now(timezone.utc).isoformat()
        for key, value in updates.items():
            if key == 'progress':
                manifest.setdefault('progress', {}).update(value)
            else:
                manifest[key] = value
        session_store.atomic_write_json(manifest_path, manifest)
        return manifest
    
    def is_cancelled():
        with _active_jobs_lock:
            job_info = _active_jobs.get((session_id, job_id), {})
            return job_info.get('cancel_requested', False)
    
    try:
        update_manifest(status='running')
        
        # Load whisper model
        import whisper
        model_name = options.get('model', 'base')
        language = options.get('language', '') or None
        
        update_manifest(progress={'currentFile': f'Loading {model_name} model...'})
        
        # Determine device
        import torch
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't support word timestamps (float64 for DTW)
            if not options.get('wordTimestamps', False):
                device = 'mps'
        
        whisper_model = whisper.load_model(model_name, device=device)
        
        outputs = []
        total = len(inputs)
        
        for idx, input_info in enumerate(inputs):
            if is_cancelled():
                update_manifest(status='cancelled')
                return
            
            filepath = input_info['path']
            original_name = input_info['originalFilename']
            base_name = Path(original_name).stem
            
            update_manifest(progress={
                'currentFile': original_name,
                'completed': idx,
                'percent': int((idx / total) * 100)
            })
            
            try:
                # Transcribe
                result = whisper.transcribe(
                    whisper_model,
                    filepath,
                    language=language,
                    word_timestamps=options.get('wordTimestamps', False)
                )
                
                # Generate output files
                output_id = session_store.new_id(8)
                
                # Markdown output
                md_filename = f"{base_name}.md"
                md_path = os.path.join(outputs_dir, md_filename)
                
                content = f"# {original_name}\n\n"
                if options.get('includeFull', True):
                    content += f"## Full Transcript\n\n{result['text'].strip()}\n\n"
                
                if options.get('includeSegments', True) and result.get('segments'):
                    content += "## Segments\n\n"
                    for seg in result['segments']:
                        if options.get('includeTimestamps', True):
                            start = seg.get('start', 0)
                            end = seg.get('end', 0)
                            content += f"**[{start:.2f} - {end:.2f}]** "
                        content += f"{seg.get('text', '').strip()}\n\n"
                
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                outputs.append({
                    'id': output_id,
                    'inputFilename': original_name,
                    'filename': md_filename,
                    'path': md_path,
                    'type': 'markdown',
                    'size': os.path.getsize(md_path)
                })
                
                # JSON output
                json_filename = f"{base_name}.json"
                json_path = os.path.join(outputs_dir, json_filename)
                
                json_output = {
                    'source': original_name,
                    'text': result['text'],
                    'language': result.get('language', ''),
                    'segments': result.get('segments', [])
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)
                
                outputs.append({
                    'id': session_store.new_id(8),
                    'inputFilename': original_name,
                    'filename': json_filename,
                    'path': json_path,
                    'type': 'json',
                    'size': os.path.getsize(json_path)
                })
                
            except Exception as e:
                # Record error but continue with other files
                outputs.append({
                    'id': session_store.new_id(8),
                    'inputFilename': original_name,
                    'error': str(e)[:200]
                })
        
        update_manifest(
            status='complete',
            outputs=outputs,
            progress={'completed': total, 'percent': 100, 'currentFile': ''}
        )
        
    except Exception as e:
        update_manifest(status='failed', error=str(e)[:500])
    
    finally:
        with _active_jobs_lock:
            if (session_id, job_id) in _active_jobs:
                _active_jobs[(session_id, job_id)]['running'] = False


@app.route('/api/jobs/<job_id>', methods=['GET'])
def api_get_job(job_id):
    """Get job status and details."""
    session_id = g.session_id
    manifest_path = session_store.job_manifest_path(session_id, job_id)
    manifest = session_store.read_json(manifest_path)
    
    if not manifest:
        return jsonify({'error': 'Job not found'}), 404
    
    # Don't expose internal paths
    safe_manifest = {
        'jobId': manifest.get('jobId'),
        'createdAt': manifest.get('createdAt'),
        'updatedAt': manifest.get('updatedAt'),
        'status': manifest.get('status'),
        'options': manifest.get('options'),
        'inputs': [{'uploadId': i.get('uploadId'), 'filename': i.get('originalFilename')} for i in manifest.get('inputs', [])],
        'outputs': [{'id': o.get('id'), 'filename': o.get('filename'), 'type': o.get('type'), 'size': o.get('size'), 'error': o.get('error')} for o in manifest.get('outputs', [])],
        'progress': manifest.get('progress'),
        'error': manifest.get('error')
    }
    
    return jsonify(safe_manifest)


@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def api_cancel_job(job_id):
    """Cancel a running job."""
    session_id = g.session_id
    
    with _active_jobs_lock:
        job_info = _active_jobs.get((session_id, job_id))
        if job_info and job_info.get('running'):
            job_info['cancel_requested'] = True
            return jsonify({'status': 'cancelling'})
    
    # Check if job exists
    manifest_path = session_store.job_manifest_path(session_id, job_id)
    manifest = session_store.read_json(manifest_path)
    
    if not manifest:
        return jsonify({'error': 'Job not found'}), 404
    
    if manifest.get('status') in ('complete', 'failed', 'cancelled'):
        return jsonify({'status': manifest.get('status'), 'message': 'Job already finished'})
    
    return jsonify({'status': 'not_running'})


@app.route('/api/jobs/<job_id>/outputs', methods=['GET'])
def api_list_job_outputs(job_id):
    """List outputs for a job."""
    session_id = g.session_id
    manifest_path = session_store.job_manifest_path(session_id, job_id)
    manifest = session_store.read_json(manifest_path)
    
    if not manifest:
        return jsonify({'error': 'Job not found'}), 404
    
    outputs = [
        {'id': o.get('id'), 'filename': o.get('filename'), 'type': o.get('type'), 'size': o.get('size')}
        for o in manifest.get('outputs', [])
        if not o.get('error')
    ]
    
    return jsonify({'outputs': outputs})


@app.route('/api/jobs/<job_id>/download', methods=['GET'])
def api_download_job(job_id):
    """Download all job outputs as a zip file."""
    session_id = g.session_id
    manifest_path = session_store.job_manifest_path(session_id, job_id)
    manifest = session_store.read_json(manifest_path)
    
    if not manifest:
        return jsonify({'error': 'Job not found'}), 404
    
    outputs = manifest.get('outputs', [])
    files_to_zip = [o for o in outputs if o.get('path') and not o.get('error')]
    
    if not files_to_zip:
        return jsonify({'error': 'No output files available'}), 404
    
    # Verify all files exist and are within job directory
    job_outputs_path = session_store.job_outputs_dir(session_id, job_id)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for output in files_to_zip:
            filepath = output.get('path')
            if filepath and os.path.exists(filepath) and session_store.is_safe_path(job_outputs_path, filepath):
                zf.write(filepath, output.get('filename'))
    
    zip_buffer.seek(0)
    
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    zip_filename = f'transcriptions_{job_id[:8]}_{timestamp}.zip'
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )


@app.route('/api/jobs/<job_id>/outputs/<output_id>', methods=['GET'])
def api_download_output(job_id, output_id):
    """Download a single output file."""
    session_id = g.session_id
    manifest_path = session_store.job_manifest_path(session_id, job_id)
    manifest = session_store.read_json(manifest_path)
    
    if not manifest:
        return jsonify({'error': 'Job not found'}), 404
    
    # Find output by ID
    output = None
    for o in manifest.get('outputs', []):
        if o.get('id') == output_id:
            output = o
            break
    
    if not output or output.get('error'):
        return jsonify({'error': 'Output not found'}), 404
    
    filepath = output.get('path')
    job_outputs_path = session_store.job_outputs_dir(session_id, job_id)
    
    # Security: verify file is within job outputs directory
    if not filepath or not os.path.exists(filepath) or not session_store.is_safe_path(job_outputs_path, filepath):
        return jsonify({'error': 'Output file not accessible'}), 404
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=output.get('filename')
    )


@app.route('/api/session', methods=['GET'])
def api_get_session():
    """Get session info including CSRF token (server mode only)."""
    if not session_store.is_server_mode():
        return jsonify({'error': 'Not available in local mode'}), 404
    
    session_id = g.session_id
    csrf_token = session_store.get_csrf_token(session_id)
    
    response = jsonify({
        'csrfToken': csrf_token
    })
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/api/jobs', methods=['GET'])
def api_list_jobs():
    """List jobs for current session."""
    session_id = g.session_id
    jobs = session_store.list_jobs(session_id, limit=20)
    return jsonify({'jobs': jobs})


@app.route('/api/mode', methods=['GET'])
def api_get_mode():
    """Get current application mode."""
    return jsonify({
        'mode': session_store.get_app_mode(),
        'isServer': session_store.is_server_mode(),
        'isLocal': session_store.is_local_mode()
    })


def run_with_webview():
    """Run the app with PyWebView for a native window experience."""
    import socket
    import threading
    
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    
    # Start Flask in a background thread
    def run_flask():
        app.run(debug=False, host='127.0.0.1', port=port, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Give Flask a moment to start
    import time
    time.sleep(0.5)
    
    # Create native window with PyWebView
    import webview
    
    # Get the app icon path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(script_dir, 'AppIcon.icns')
    if not os.path.exists(icon_path):
        icon_path = None
    
    window = webview.create_window(
        'Bulk Transcribe',
        f'http://localhost:{port}',
        width=1100,
        height=850,
        min_size=(800, 600)
    )
    webview.start(icon=icon_path)


def run_with_browser():
    """Run the app with browser (fallback mode)."""
    import socket
    import webbrowser
    import threading
    
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    print("Starting Bulk Transcribe...")
    print(f"Open http://localhost:{port} in your browser")
    
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f'http://localhost:{port}')
    
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, host='127.0.0.1', port=port)


# Required for PyInstaller on macOS - must be at module level
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    import sys
    
    # Check for --browser flag to force browser mode
    if '--browser' in sys.argv:
        run_with_browser()
    else:
        try:
            run_with_webview()
        except ImportError:
            print("PyWebView not available, falling back to browser mode")
            run_with_browser()
