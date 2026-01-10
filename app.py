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
from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

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


def transcribe_file(audio_file: Path, model: str, language: str, output_folder: Path, word_timestamps: bool = False, status_callback=None) -> dict:
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
        
        # Transcribe - whisper doesn't have a callback, but we can use the result segments
        result = whisper.transcribe(
            whisper_model,
            str(audio_file),
            language=language if language else None,
            word_timestamps=word_timestamps,
            verbose=False
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
    
    output_name = audio_file.stem + '_transcription.md'
    output_path = output_folder / output_name
    
    content = f"# Transcription: {audio_file.name}\n\n"
    content += f"**Source File:** `{audio_file.name}`\n"
    content += f"**Model Used:** `{model}`\n"
    
    if language:
        content += f"**Language:** `{language}`\n"
    
    if 'language' in transcription_data:
        content += f"**Detected Language:** `{transcription_data['language']}`\n"
    
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
                content += f"### Segment {i} ({format_timestamp(start_time)} - {format_timestamp(end_time)})\n\n"
            else:
                content += f"### Segment {i}\n\n"
            content += f"{text}\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path


def run_transcription(input_folder: str, output_folder: str, model: str, language: str, options: dict = None):
    """Background task to run transcription with optional parallel workers."""
    global transcription_status
    
    if options is None:
        options = {}
    
    processing_mode = options.get('processing_mode', 'metal')
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
        
        word_timestamps = options.get('word_timestamps', False)
        overwrite_existing = options.get('overwrite_existing', False)
        
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
                device = "cpu"
                if processing_mode == 'cuda' and torch.cuda.is_available():
                    device = "cuda"
                    transcription_status['current_file_progress'] = f'Loading {model} model (CUDA GPU)...'
                elif processing_mode == 'metal' and torch.backends.mps.is_available():
                    device = "mps"
                    transcription_status['current_file_progress'] = f'Loading {model} model (Metal GPU)...'
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
                         languages=COMMON_LANGUAGES)


@app.route('/start', methods=['POST'])
def start_transcription():
    global transcription_status
    
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
    """Open native folder picker and return selected path."""
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
