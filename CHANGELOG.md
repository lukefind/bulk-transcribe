# Changelog

All notable changes to Bulk Transcribe will be documented in this file.

## [1.1.0] - 2026-01-09

### Added
- **Two cancel options**: "Cancel After File" (yellow) completes current file then stops; "Force Stop" (red) stops immediately
- **Current file elapsed time**: Shows how long the current file has been processing
- **ETA improvements**: Only shows after first file completes, formatted as hours/minutes for clarity

### Changed
- Metal GPU mode is now the default and recommended processing mode
- Improved progress tracking with per-file timing and speed metrics (e.g., "5.8x realtime")

### Fixed
- Force Stop now immediately updates the UI instead of appearing stuck
- Cancel buttons properly reset after job completion

## [1.0.0] - 2026-01-08

### Added
- Initial release
- Native macOS app with Metal GPU acceleration
- Batch transcription of audio files to markdown
- Model management (download, delete, switch models)
- Skip existing files to resume interrupted jobs
- Multiple output formats (segmented, full text, timestamps)
- Real-time progress tracking
- Persistent settings across sessions
