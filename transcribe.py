#!/usr/bin/env python3
"""
Audio Transcription Tool using OpenAI Whisper
Transcribes all audio files in a folder and outputs markdown files.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import json


def setup_ffmpeg_path():
    """Set up PATH to include bundled ffmpeg if running in PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)
        ffmpeg_path = base_path / 'bin' / 'ffmpeg'
        if ffmpeg_path.exists():
            # Add the bin directory to PATH
            os.environ['PATH'] = str(base_path / 'bin') + os.pathsep + os.environ.get('PATH', '')
            return str(ffmpeg_path)
    return None


class AudioTranscriber:
    """Main class for handling audio transcription with Whisper."""
    
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mp4', '.mov', '.avi']
    
    def __init__(self, input_folder: str, output_folder: str, model: str = "base", 
                 language: Optional[str] = None, verbose: bool = False):
        # Set up ffmpeg path if bundled
        setup_ffmpeg_path()
        
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.model = model
        self.language = language
        self.verbose = verbose
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def find_audio_files(self) -> List[Path]:
        """Find all supported audio files in the input folder."""
        audio_files = []
        
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_folder}")
        
        for file_path in self.input_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                audio_files.append(file_path)
        
        return sorted(audio_files)
    
    def transcribe_file(self, audio_file: Path) -> dict:
        """Transcribe a single audio file using Whisper."""
        cmd = [
            'whisper',
            str(audio_file),
            '--model', self.model,
            '--output_format', 'json'
        ]
        
        if self.language:
            cmd.extend(['--language', self.language])
        
        if self.verbose:
            cmd.append('--verbose')
        
        try:
            if self.verbose:
                print(f"Transcribing: {audio_file}")
                print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the JSON output
            transcription_data = json.loads(result.stdout)
            return transcription_data
            
        except subprocess.CalledProcessError as e:
            print(f"Error transcribing {audio_file}: {e}")
            print(f"Stderr: {e.stderr}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output for {audio_file}: {e}")
            return None
    
    def create_markdown_file(self, audio_file: Path, transcription_data: dict) -> Path:
        """Create a markdown file from transcription data."""
        # Generate output filename
        output_name = audio_file.stem + '_transcription.md'
        output_path = self.output_folder / output_name
        
        # Create markdown content
        content = f"# Transcription: {audio_file.name}\n\n"
        content += f"**Source File:** `{audio_file.name}`\n"
        content += f"**Model Used:** `{self.model}`\n"
        
        if self.language:
            content += f"**Language:** `{self.language}`\n"
        
        if 'language' in transcription_data:
            content += f"**Detected Language:** `{transcription_data['language']}`\n"
        
        content += f"\n---\n\n"
        
        # Add the full transcription
        if 'text' in transcription_data:
            content += "## Full Transcription\n\n"
            content += transcription_data['text'].strip() + "\n\n"
        
        # Add segments if available
        if 'segments' in transcription_data:
            content += "## Segmented Transcription\n\n"
            for i, segment in enumerate(transcription_data['segments'], 1):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                content += f"### Segment {i} ({start_time:.2f}s - {end_time:.2f}s)\n\n"
                content += f"{text}\n\n"
        
        # Write the markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_path
    
    def transcribe_all(self) -> List[Path]:
        """Transcribe all audio files in the input folder."""
        audio_files = self.find_audio_files()
        
        if not audio_files:
            print("No supported audio files found in the input folder.")
            return []
        
        print(f"Found {len(audio_files)} audio files to transcribe.")
        
        output_files = []
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            
            # Transcribe the file
            transcription_data = self.transcribe_file(audio_file)
            
            if transcription_data:
                # Create markdown file
                output_path = self.create_markdown_file(audio_file, transcription_data)
                output_files.append(output_path)
                print(f"✓ Created: {output_path.name}")
            else:
                print(f"✗ Failed to transcribe: {audio_file.name}")
        
        return output_files


def get_available_models():
    """Get list of available Whisper models."""
    return ['tiny', 'base', 'small', 'medium', 'large', 'turbo']


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/audio/files /path/to/output
  %(prog)s /path/to/audio/files /path/to/output --model small --language en
  %(prog)s /path/to/audio/files /path/to/output --model large --verbose
        """
    )
    
    parser.add_argument('input_folder', help='Folder containing audio files')
    parser.add_argument('output_folder', help='Folder to save transcription markdown files')
    parser.add_argument('--model', choices=get_available_models(), default='base',
                       help='Whisper model to use (default: base)')
    parser.add_argument('--language', help='Language code (e.g., en, es, fr, de)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available Whisper models:")
        for model in get_available_models():
            print(f"  - {model}")
        return
    
    try:
        # Create transcriber instance
        transcriber = AudioTranscriber(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            model=args.model,
            language=args.language,
            verbose=args.verbose
        )
        
        # Run transcription
        print(f"Starting transcription with model: {args.model}")
        if args.language:
            print(f"Language: {args.language}")
        
        output_files = transcriber.transcribe_all()
        
        # Summary
        print(f"\n{'='*50}")
        print(f"Transcription complete!")
        print(f"Processed {len(output_files)} files successfully.")
        print(f"Output files saved to: {args.output_folder}")
        
        if output_files:
            print("\nCreated files:")
            for file_path in output_files:
                print(f"  - {file_path.name}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
