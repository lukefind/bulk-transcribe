"""
Audio utility functions for metadata extraction and validation.
"""
import json
import os
import subprocess
from typing import Optional


def get_memory_usage_mb() -> Optional[float]:
    """
    Get current process memory usage in MB.
    Works in Docker containers using cgroup v2 or v1.
    
    Returns:
        Memory usage in MB, or None if unavailable
    """
    # Try cgroup v2 first
    cgroup_v2_path = '/sys/fs/cgroup/memory.current'
    if os.path.exists(cgroup_v2_path):
        try:
            with open(cgroup_v2_path, 'r') as f:
                bytes_used = int(f.read().strip())
                return bytes_used / (1024 * 1024)
        except Exception:
            pass
    
    # Try cgroup v1
    cgroup_v1_path = '/sys/fs/cgroup/memory/memory.usage_in_bytes'
    if os.path.exists(cgroup_v1_path):
        try:
            with open(cgroup_v1_path, 'r') as f:
                bytes_used = int(f.read().strip())
                return bytes_used / (1024 * 1024)
        except Exception:
            pass
    
    # Fallback to /proc/self/status
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # VmRSS is in kB
                    kb = int(line.split()[1])
                    return kb / 1024
    except Exception:
        pass
    
    return None


class AudioProbeError(Exception):
    """Raised when audio probing fails."""
    pass


def get_audio_info(path: str) -> dict:
    """
    Get audio file metadata using ffprobe.
    
    Args:
        path: Path to the audio file
    
    Returns:
        Dictionary with:
        - durationSec: float (duration in seconds)
        - codec: str (audio codec name)
        - sampleRate: int (sample rate in Hz)
        - channels: int (number of audio channels)
        - formatName: str (container format)
    
    Raises:
        AudioProbeError: If ffprobe fails or metadata unavailable
    """
    if not os.path.exists(path):
        raise AudioProbeError(f"File not found: {path}")
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            '-select_streams', 'a:0',  # First audio stream only
            path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise AudioProbeError(
                f"ffprobe failed with code {result.returncode}: {result.stderr[:200]}"
            )
        
        data = json.loads(result.stdout)
        
        # Extract format info
        format_info = data.get('format', {})
        duration_str = format_info.get('duration')
        format_name = format_info.get('format_name', 'unknown')
        
        if duration_str is None:
            raise AudioProbeError("Could not determine audio duration")
        
        duration_sec = float(duration_str)
        
        # Extract stream info (first audio stream)
        streams = data.get('streams', [])
        audio_stream = None
        for stream in streams:
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break
        
        if audio_stream is None:
            raise AudioProbeError("No audio stream found in file")
        
        codec = audio_stream.get('codec_name', 'unknown')
        sample_rate = int(audio_stream.get('sample_rate', 0))
        channels = int(audio_stream.get('channels', 0))
        
        return {
            'durationSec': round(duration_sec, 3),
            'codec': codec,
            'sampleRate': sample_rate,
            'channels': channels,
            'formatName': format_name
        }
        
    except subprocess.TimeoutExpired:
        raise AudioProbeError("ffprobe timed out")
    except json.JSONDecodeError as e:
        raise AudioProbeError(f"Failed to parse ffprobe output: {e}")
    except ValueError as e:
        raise AudioProbeError(f"Invalid metadata value: {e}")
    except Exception as e:
        if isinstance(e, AudioProbeError):
            raise
        raise AudioProbeError(f"Unexpected error probing audio: {e}")


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string (e.g., '3m 45s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {secs}s"
    
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs}s"


def is_wav_pcm_compatible(info: dict) -> bool:
    """
    Check if audio is already WAV PCM format suitable for pyannote.
    
    Args:
        info: Audio info dict from get_audio_info()
    
    Returns:
        True if file is WAV PCM mono/stereo at acceptable sample rate
    """
    if info.get('codec') not in ('pcm_s16le', 'pcm_s24le', 'pcm_f32le'):
        return False
    
    if info.get('formatName') not in ('wav',):
        return False
    
    # Accept common sample rates
    sample_rate = info.get('sampleRate', 0)
    if sample_rate not in (16000, 22050, 44100, 48000):
        return False
    
    # Accept mono or stereo
    channels = info.get('channels', 0)
    if channels not in (1, 2):
        return False
    
    return True


class ChunkInfo:
    """Information about an audio chunk."""
    def __init__(self, index: int, start_sec: float, end_sec: float, path: str):
        self.index = index
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.path = path
    
    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'startSec': self.start_sec,
            'endSec': self.end_sec,
            'path': self.path
        }


def split_audio_to_wav_chunks(
    input_path: str,
    out_dir: str,
    chunk_seconds: int = 150,
    overlap_seconds: int = 5,
    sample_rate: int = 16000
) -> list[ChunkInfo]:
    """
    Split audio file into overlapping WAV chunks for diarization.
    
    Args:
        input_path: Path to input audio file
        out_dir: Directory to write chunk files
        chunk_seconds: Duration of each chunk in seconds
        overlap_seconds: Overlap between consecutive chunks
        sample_rate: Output sample rate (default 16kHz for pyannote)
    
    Returns:
        List of ChunkInfo objects describing each chunk
    
    Raises:
        AudioProbeError: If audio probing or splitting fails
    """
    # Get audio duration
    info = get_audio_info(input_path)
    total_duration = info['durationSec']
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    chunks = []
    chunk_index = 0
    start_sec = 0.0
    
    # Calculate step size (chunk minus overlap)
    step_sec = chunk_seconds - overlap_seconds
    
    while start_sec < total_duration:
        # Calculate end time for this chunk
        end_sec = min(start_sec + chunk_seconds, total_duration)
        duration = end_sec - start_sec
        
        # Skip very short final chunks (less than 5 seconds)
        if duration < 5 and chunk_index > 0:
            break
        
        # Generate chunk filename
        chunk_filename = f"chunk_{chunk_index:03d}.wav"
        chunk_path = os.path.join(out_dir, chunk_filename)
        
        # Extract chunk using ffmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-ss', str(start_sec),
            '-i', input_path,
            '-t', str(duration),
            '-ac', '1',  # Mono
            '-ar', str(sample_rate),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            chunk_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                raise AudioProbeError(
                    f"ffmpeg chunk extraction failed: {result.stderr[:200]}"
                )
        except subprocess.TimeoutExpired:
            raise AudioProbeError(f"ffmpeg timed out extracting chunk {chunk_index}")
        
        chunks.append(ChunkInfo(
            index=chunk_index,
            start_sec=start_sec,
            end_sec=end_sec,
            path=chunk_path
        ))
        
        chunk_index += 1
        start_sec += step_sec
        
        # Safety limit: max 100 chunks
        if chunk_index >= 100:
            break
    
    return chunks


def cleanup_chunks(chunk_dir: str) -> None:
    """
    Remove chunk files and directory.
    
    Args:
        chunk_dir: Directory containing chunk files
    """
    import shutil
    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir, ignore_errors=True)
