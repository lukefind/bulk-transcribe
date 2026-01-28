# Review Workflow

The Review workspace is designed for fast human correction. Machine output is a first-pass annotation layer, not ground truth.

---

## Core Actions

### Rename speakers
- Click any speaker label in the sidebar
- Type a new name and press Enter
- Labels + colors persist per session

### Change speaker colors
- Click the color dot in the sidebar
- Pick a palette color
- Colors persist and are used in exports

### Edit text
- Select a chunk and press **E**
- **Enter** saves and exits
- **Shift+Enter** inserts a newline
- **Esc** cancels

### Split interjections (mid‑segment speaker change)
While editing a chunk:
- **Cmd/Ctrl + S** → split at cursor
- **Cmd/Ctrl + Shift + S** → split selection (3‑way)

This lets you insert short interjections as their own chunk and assign a new speaker.

### Assign speaker
- Hover a chunk to reveal speaker buttons
- Press keys **1–9** to assign quickly

### Speaker hotkey numbers

In **Edit Speakers**, assign hotkey numbers (1–9) per file:
- Numbers affect shortcuts **in this file only**
- Reassigning a number clears it from the previous speaker
- Use **Copy from previous** to carry assignments to the next file
- Use **Reset** to clear all assigned numbers

### Merged/Raw View Toggle

The review toolbar includes a **Merged** checkbox:

- **Merged (default)**: Combines adjacent same-speaker chunks for readability.
- **Raw**: Shows diarization-aligned chunks with more speaker switches.

Use Raw view when debugging speaker assignment or when you need to see the original segmentation.

### Info Modal

Click the **ⓘ** button in the toolbar for a quick reference on view modes, Fast Switching, exports, and keyboard shortcuts.

---

## Exports

From Review:

- **Project (.btproj)** — full archive for backup/transfer
- **Markdown (.md)** — timestamped speaker transcript
- **Timeline (.json)** — structured review data
- **Word (.docx)** — timestamped transcript with speaker colors
- **PDF (.pdf)** — timestamped transcript with speaker colors

---

## Tips

- Use **Fixed speakers** for known counts (e.g., 3 for 1 adult + 2 kids)
- Use **Fast Switching** only when needed — it is slower
- Keep chunk durations short if diarization misses rapid turn‑taking

---

## Export / Import Sessions

### Export Session

Export all jobs in your current session as a single zip archive:

1. Click **Export Session** in the Recent Jobs header
2. The zip contains:
   - `session.json` — session metadata
   - `session_summary.json` — stats (job count, files, review state count)
   - `jobs/<job_id>/` — per-job folders with:
     - `job.json` — full job manifest
     - `review_state.json` — speaker labels, colors, merges, edits
    - `timeline.json` — parsed timeline with edits applied
    - `outputs/` — all output files (JSON, Markdown, etc.)

Exports run in the background with a progress modal and download link.

### Import Session

Restore a previously exported session:

1. Click **Import Session** in the Recent Jobs header
2. Select a session export zip
3. Jobs are restored with all editor state (speaker labels, colors, edits)
4. If a job ID already exists, a new ID is generated automatically

## Session Sharing

Share a live server session without exporting:
- Click **Share Session**
- Choose **Read-only** or **Full edit**
- Copy the token (and optional password) to another user

Join a shared session:
- Click **Join Session** and paste the token
- Read-only sessions show a banner and disable edits

**Use cases:**
- Backup/restore work across machines
- Share review progress with collaborators
- Archive completed sessions
