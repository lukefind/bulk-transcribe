#!/usr/bin/env python3
"""
Interactive Transcription Tool using OpenAI Whisper
Uses native macOS folder picker dialogs
"""

import subprocess
import json
import sys
from pathlib import Path


SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mp4', '.mov', '.avi']
MODELS = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
LANGUAGES = [
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
]


def choose_folder(prompt="Select a folder"):
    """Open native macOS folder picker dialog."""
    script = f'POSIX path of (choose folder with prompt "{prompt}")'
    try:
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def find_audio_files(folder):
    """Find all supported audio files in a folder."""
    audio_files = []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        return []
    
    for file_path in folder_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            audio_files.append(file_path)
    
    return sorted(audio_files)


def transcribe_file(audio_file, model, language, output_folder):
    """Transcribe a single audio file using Whisper CLI."""
    cmd = [
        'whisper',
        str(audio_file),
        '--model', model,
        '--output_format', 'json',
        '--output_dir', str(output_folder)
    ]
    
    if language:
        cmd.extend(['--language', language])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            return {'error': result.stderr or 'Transcription failed'}
        
        json_output_path = output_folder / (audio_file.stem + '.json')
        if json_output_path.exists():
            with open(json_output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {'error': 'JSON output file not found'}
            
    except subprocess.TimeoutExpired:
        return {'error': 'Transcription timed out'}
    except Exception as e:
        return {'error': str(e)}


def create_markdown(audio_file, transcription_data, model, language, output_folder):
    """Create a markdown file from transcription data."""
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
    
    if 'text' in transcription_data:
        content += "## Full Transcription\n\n"
        content += transcription_data['text'].strip() + "\n\n"
    
    if 'segments' in transcription_data:
        content += "## Segmented Transcription\n\n"
        for i, segment in enumerate(transcription_data['segments'], 1):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            content += f"### Segment {i} ({start_time:.2f}s - {end_time:.2f}s)\n\n"
            content += f"{text}\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path


def select_option(options, prompt, default_index=0):
    """Display options and let user select one."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = " [default]" if i == default_index else ""
        print(f"  {i + 1}. {opt}{marker}")
    
    while True:
        choice = input(f"Enter choice (1-{len(options)}) or press Enter for default: ").strip()
        if choice == '':
            return options[default_index]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def main():
    print("=" * 50)
    print("  Whisper Transcription Tool")
    print("=" * 50)
    
    # Select input folder
    print("\n[1/4] Select INPUT folder containing audio files...")
    input_folder = choose_folder("Select folder containing audio files")
    
    if not input_folder:
        print("No input folder selected. Exiting.")
        sys.exit(1)
    
    print(f"Input: {input_folder}")
    
    # Preview files
    audio_files = find_audio_files(input_folder)
    if not audio_files:
        print("No audio files found in the selected folder.")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files[:5]:
        print(f"  - {f.name}")
    if len(audio_files) > 5:
        print(f"  ... and {len(audio_files) - 5} more")
    
    # Select output folder
    print("\n[2/4] Select OUTPUT folder for transcriptions...")
    output_folder = choose_folder("Select folder to save transcriptions")
    
    if not output_folder:
        print("No output folder selected. Exiting.")
        sys.exit(1)
    
    print(f"Output: {output_folder}")
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select model
    model = select_option(MODELS, "[3/4] Select Whisper model:", default_index=1)
    print(f"Model: {model}")
    
    # Select language
    lang_names = [name for code, name in LANGUAGES]
    lang_name = select_option(lang_names, "[4/4] Select language:", default_index=0)
    language = ''
    for code, name in LANGUAGES:
        if name == lang_name:
            language = code
            break
    print(f"Language: {lang_name}")
    
    # Confirm
    print("\n" + "=" * 50)
    print("Ready to transcribe:")
    print(f"  Input:    {input_folder}")
    print(f"  Output:   {output_folder}")
    print(f"  Model:    {model}")
    print(f"  Language: {lang_name}")
    print(f"  Files:    {len(audio_files)}")
    print("=" * 50)
    
    confirm = input("\nProceed? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Cancelled.")
        sys.exit(0)
    
    # Run transcription
    print("\nStarting transcription...\n")
    
    success_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {audio_file.name}")
        
        result = transcribe_file(audio_file, model, language, output_path)
        
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            md_path = create_markdown(audio_file, result, model, language, output_path)
            print(f"  -> {md_path.name}")
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Complete! {success_count}/{len(audio_files)} files transcribed.")
    print(f"Output saved to: {output_folder}")
    print("=" * 50)


if __name__ == "__main__":
    main()
