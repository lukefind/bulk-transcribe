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
