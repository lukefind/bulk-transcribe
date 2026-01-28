# Transcript & Review Schema

This document describes the primary output files and review artifacts.

---

## Machine Transcript (transcript.json)

Generated directly by Whisper. Fields vary slightly by model, but typically include:

```json
{
  "text": "...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "...",
      "words": [
        {"word": "...", "start": 0.1, "end": 0.4}
      ]
    }
  ]
}
```

Notes:
- **Do not edit** this file after generation.
- Word timestamps are included only if enabled.

---

## Diarization Output (diarization.json)

Raw diarization output. Structure depends on the diarization pipeline, but includes time ranges and speaker ids.

---

## Speaker Markdown (speaker.md)

A human-readable, timestamped speaker transcript:

```
[00:23] Speaker 1: text...
[00:28] Speaker 2: text...
```

---

## Review State (review_state.json)

Stores edits made in the Review UI. Review state is **scoped per input file** to avoid cross‑contamination in batch jobs.

```json
{
  "perInput": {
    "<inputId>": {
      "speakerLabelMap": {
        "SPEAKER_00": "Alice"
      },
      "speakerColorMap": {
        "SPEAKER_00": "#3B82F6"
      },
      "speakerNumberMap": {
        "SPEAKER_00": 1
      },
      "chunkEdits": {
        "t_000123": {
          "text": "corrected text",
          "speakerId": "SPEAKER_01"
        }
      }
    }
  },
  "uiPrefs": {}
}
```

Notes:
- `perInput` keys are upload IDs from the job manifest.
- Chunk IDs may include split suffixes (e.g., `t_000123_a`).
- This file is currently mutable. Future versions may migrate to an append‑only edits log.

---

## Review Timeline (timeline.json)

Structured representation of the transcript used in Review UI and exports:

```json
{
  "speakers": [
    {"id": "SPEAKER_00", "label": "Speaker 1", "color": "#3B82F6"}
  ],
  "chunks": [
    {
      "chunk_id": "t_000000",
      "start": 0.0,
      "end": 3.2,
      "text": "...",
      "speaker_id": "SPEAKER_00",
      "origin": {"transcript_segment_ids": [0]}
    }
  ]
}
```

Notes:
- Speaker labels/colors reflect **review edits**.
- Chunks may be split/merged by review actions.
- Segments may carry `hallucination_warning` (flag-only; text is preserved).
