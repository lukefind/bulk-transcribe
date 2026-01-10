from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List


# Ensure repo root is importable when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from transcribe_options import TranscribeOptions, postprocess_segments


def _assert_contiguous_ids(segments: List[Dict[str, Any]]) -> None:
    ids = [s.get('id') for s in segments]
    assert all(isinstance(x, str) for x in ids), f"Non-string ids: {ids}"
    assert all(re.match(r"^seg_\d{3}$", x) for x in ids), f"Unexpected id format: {ids}"

    indexes = [s.get('index') for s in segments]
    assert indexes == list(range(len(segments))), f"Indexes not contiguous: {indexes}"


def main() -> None:
    # Case 1: No word timestamps -> must not split even if max_duration is set.
    whisper_result_no_words = {
        "segments": [
            {"id": 0, "start": 0.0, "end": 12.0, "text": "hello world"},
            {"id": 1, "start": 12.0, "end": 12.2, "text": "uh"},
        ]
    }

    processed_no_words = postprocess_segments(
        whisper_result_no_words["segments"],
        merge_short=True,
        min_duration=0.5,
        max_duration=5.0,
        word_timestamps_available=False,
    )

    assert all(s["end"] >= s["start"] for s in processed_no_words)
    _assert_contiguous_ids(processed_no_words)

    # source_id should be present when input had id(s)
    assert all("source_id" in s for s in processed_no_words), "Expected source_id for segments with input ids"

    # Confirm no splitting occurred (should be 1 merged segment or 2 segments, but never > input count here)
    assert len(processed_no_words) <= len(whisper_result_no_words["segments"])

    # Case 2: Word timestamps present and valid -> splitting may occur.
    words = [
        {"word": "Hello", "start": 0.0, "end": 1.0},
        {"word": " ", "start": 1.0, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 2.0},
        {"word": "!", "start": 2.0, "end": 2.1},
        {"word": " More", "start": 2.1, "end": 4.0},
        {"word": " words", "start": 4.0, "end": 6.0},
    ]

    whisper_result_with_words = {
        "segments": [
            {"id": 7, "start": 0.0, "end": 6.0, "text": "Hello world! More words", "words": words}
        ]
    }

    processed_with_words = postprocess_segments(
        whisper_result_with_words["segments"],
        merge_short=True,
        min_duration=0.5,
        max_duration=3.0,
        word_timestamps_available=True,
    )

    assert all(s["end"] >= s["start"] for s in processed_with_words)
    _assert_contiguous_ids(processed_with_words)

    assert all(s.get("source_id") == "7" for s in processed_with_words), "Expected source_id to preserve input id as string"

    # With max_duration=3.0 and a 6s segment, we expect splitting into multiple segments.
    assert len(processed_with_words) > 1, "Expected split when word timestamps are available"

    # Case 3: Word timestamps flag true but words missing -> must not split.
    whisper_result_flag_true_but_missing_words = {
        "segments": [
            {"id": "abc", "start": 0.0, "end": 12.0, "text": "no words here", "words": None}
        ]
    }

    processed_missing_words = postprocess_segments(
        whisper_result_flag_true_but_missing_words["segments"],
        merge_short=True,
        min_duration=0.5,
        max_duration=5.0,
        word_timestamps_available=True,
    )

    assert all(s["end"] >= s["start"] for s in processed_missing_words)
    _assert_contiguous_ids(processed_missing_words)
    assert len(processed_missing_words) == 1, "Should not split when words are missing"

    assert processed_missing_words[0].get("source_id") == "abc"

    # Case 4: merge should combine source ids
    whisper_result_merge = {
        "segments": [
            {"id": 10, "start": 0.0, "end": 1.0, "text": "hi"},
            {"id": 11, "start": 1.0, "end": 1.2, "text": "there"},
        ]
    }

    processed_merge = postprocess_segments(
        whisper_result_merge["segments"],
        merge_short=True,
        min_duration=0.5,
        max_duration=None,
        word_timestamps_available=False,
    )
    assert all(s["end"] >= s["start"] for s in processed_merge)
    _assert_contiguous_ids(processed_merge)
    assert processed_merge[0].get("source_id") == "10+11", f"Unexpected merged source_id: {processed_merge[0].get('source_id')}"

    # Sanity: TranscribeOptions kwargs build.
    opts = TranscribeOptions(language=None)
    kwargs = opts.to_whisper_kwargs()
    assert "language" in kwargs and "temperature" in kwargs

    print("OK")


if __name__ == "__main__":
    main()
