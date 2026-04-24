"""Tests for src/video_names — VLM-backed Speaker → name mapping.

Qwen 2.5-VL, ffmpeg, and OpenCV are all faked here so the tests stay fast
and don't require model weights. The actual pieces being exercised are:

  - timestamp picking from diarization (Step 1)
  - active-tile bbox heuristic (Step 2, via synthetic HSV frames)
  - VLM answer parsing + majority vote (Steps 3 & 4)
  - end-to-end pipeline wiring with mocked frame extraction + VLM
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import video_names
from src.schemas import Transcript, TranscriptSegment
from src.video_names import (
    VideoSpeakerEvidence,
    detect_active_tile,
    find_active_tile_bbox,
    identify_speakers_from_video,
    majority_vote,
    parse_vlm_answer,
    pick_sample_timestamps,
)


def _cv2_safely_importable() -> bool:
    """Probe ``import cv2`` in a subprocess so we skip tests cleanly on
    broken OpenCV installs (some Apple Silicon venvs segfault during the
    cv2 import — that crash can't be caught with try/except in-process).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import cv2"],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


_CV2_AVAILABLE = _cv2_safely_importable()
_requires_cv2 = pytest.mark.skipif(
    not _CV2_AVAILABLE,
    reason="cv2 not importable in this environment",
)


# ─── Helpers ───────────────────────────────────────────────────────────────
def _seg(
    speaker: str, start: float, end: float, text: str = "hello"
) -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker, start_time=start, end_time=end, text=text
    )


def _transcript(segments: list[TranscriptSegment]) -> Transcript:
    return Transcript(
        source_path="x.mp4",
        source_type="video",
        duration_seconds=segments[-1].end_time if segments else 0.0,
        segments=segments,
    )


# ─── Step 1: pick_sample_timestamps ────────────────────────────────────────
class TestPickSampleTimestamps:
    def test_picks_middle_of_longest_turns(self) -> None:
        # Speaker 1 has three turns; longest is (20, 40) → midpoint 30.
        segs = [
            _seg("Speaker 1", 0.0, 5.0),     # 5s
            _seg("Speaker 2", 5.0, 10.0),
            _seg("Speaker 1", 20.0, 40.0),   # 20s ← longest
            _seg("Speaker 1", 50.0, 60.0),   # 10s
        ]
        picked = pick_sample_timestamps(segs, n_per_speaker=1, min_duration=3.0)
        assert picked["Speaker 1"] == [30.0]

    def test_takes_multiple_samples_per_speaker(self) -> None:
        segs = [
            _seg("Speaker 1", 0.0, 5.0),     # 5s
            _seg("Speaker 1", 10.0, 30.0),   # 20s
            _seg("Speaker 1", 40.0, 48.0),   # 8s
        ]
        picked = pick_sample_timestamps(segs, n_per_speaker=3, min_duration=3.0)
        # All three kept, sorted by duration desc: 20s (mid 20.0), 8s (44.0), 5s (2.5)
        assert picked["Speaker 1"] == [20.0, 44.0, 2.5]

    def test_drops_turns_shorter_than_min_duration(self) -> None:
        segs = [
            _seg("Speaker 1", 0.0, 1.0),   # 1s, too short
            _seg("Speaker 1", 2.0, 2.5),   # 0.5s, too short
        ]
        assert pick_sample_timestamps(segs, 3, min_duration=3.0) == {}

    def test_ignores_unknown_and_blank_speakers(self) -> None:
        segs = [
            _seg("", 0.0, 10.0),
            _seg("Speaker ?", 10.0, 20.0),
            _seg("Speaker 1", 20.0, 30.0),
        ]
        picked = pick_sample_timestamps(segs, 3, min_duration=3.0)
        assert list(picked.keys()) == ["Speaker 1"]

    def test_caps_samples_at_n_per_speaker(self) -> None:
        segs = [_seg("Speaker 1", i * 10.0, i * 10.0 + 8.0) for i in range(10)]
        picked = pick_sample_timestamps(segs, n_per_speaker=2, min_duration=3.0)
        assert len(picked["Speaker 1"]) == 2


# ─── Step 2: active-tile bbox detection (synthetic OpenCV frame) ───────────
def _synthetic_frame_with_border(
    color_bgr: tuple[int, int, int],
    w: int = 1280,
    h: int = 720,
    box: tuple[int, int, int, int] = (300, 200, 400, 300),
    border_width: int = 8,
):
    """Return a BGR numpy frame with a thick coloured border around ``box``."""
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    frame = np.full((h, w, 3), 30, dtype=np.uint8)  # dark grey background
    x, y, bw, bh = box
    cv2.rectangle(
        frame, (x, y), (x + bw, y + bh),
        color=color_bgr,
        thickness=border_width,
    )
    return frame


def _synthetic_frame_with_green_border(
    w: int = 1280,
    h: int = 720,
    box: tuple[int, int, int, int] = (300, 200, 400, 300),
    border_width: int = 8,
):
    """Return a BGR numpy frame with a Zoom-style green border."""
    return _synthetic_frame_with_border(
        (90, 240, 100), w=w, h=h, box=box, border_width=border_width
    )


@_requires_cv2
class TestFindActiveTileBbox:
    def test_detects_green_border(self) -> None:
        frame = _synthetic_frame_with_green_border(box=(300, 200, 400, 300))
        bbox = find_active_tile_bbox(frame)
        assert bbox is not None
        x, y, w, h = bbox
        # Allow a few px slack for morphology expansion.
        assert abs(x - 300) <= 10 and abs(y - 200) <= 10
        assert abs(w - 400) <= 20 and abs(h - 300) <= 20

    def test_detects_meet_blue_border(self) -> None:
        # #1a73e8 in BGR is (232, 115, 26) — Meet's active-speaker blue.
        frame = _synthetic_frame_with_border(
            (232, 115, 26), box=(300, 200, 400, 300)
        )
        bbox, reason = detect_active_tile(frame)
        assert bbox is not None, f"meet-blue border not detected: {reason}"
        assert "meet-blue" in reason

    def test_detects_teams_purple_border(self) -> None:
        # #5b5fc7 in BGR is (199, 95, 91) — Teams Fluent accent.
        frame = _synthetic_frame_with_border(
            (199, 95, 91), box=(300, 200, 400, 300)
        )
        bbox, reason = detect_active_tile(frame)
        assert bbox is not None, f"teams-purple border not detected: {reason}"
        assert "teams-purple" in reason

    def test_returns_none_on_frame_with_no_highlight(self) -> None:
        import numpy as np
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        assert find_active_tile_bbox(frame) is None

    def test_reason_explains_missing_highlight(self) -> None:
        # Same as above but also check the diagnostic reason — this is
        # what shows up in the [video-names] log line, and the user
        # relies on it to tell "no border in this video" from
        # "border rejected as too small/large".
        import numpy as np
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        bbox, reason = detect_active_tile(frame)
        assert bbox is None
        assert "no-palette-signal" in reason

    def test_returns_none_on_tiny_highlight(self) -> None:
        # A 40x30 green square is below the 10%-of-frame sanity threshold.
        frame = _synthetic_frame_with_green_border(box=(10, 10, 40, 30))
        bbox, reason = detect_active_tile(frame)
        assert bbox is None
        assert "bbox-too-small" in reason


class TestFindActiveTileBboxNoCv2:
    def test_returns_none_when_cv2_missing(self) -> None:
        # Simulate OpenCV not being installed — must not crash.
        with patch.dict("sys.modules", {"cv2": None}):
            assert find_active_tile_bbox(object()) is None
            bbox, reason = detect_active_tile(object())
            assert bbox is None
            assert reason == "cv2-unavailable"


# ─── Step 3: VLM answer parsing ────────────────────────────────────────────
class TestParseVlmAnswer:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Alice", "Alice"),
            ("alice", "Alice"),
            ("  Alice  ", "Alice"),
            ("Alice.", "Alice"),
            ("'Alice'", "Alice"),
            ("Alice Smith", "Alice"),              # multi-word → first name
            ("Alice (You)", "Alice"),              # (You) suffix stripped
            ("Alice [Host]", "Alice"),
            ("The name is Alice", "Alice"),        # model chatter lead-in
            ("Name: Alice", "Alice"),
        ],
    )
    def test_parses_clean_names(self, raw: str, expected: str) -> None:
        assert parse_vlm_answer(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "unknown",
            "Unknown",
            "UNKNOWN",
            "n/a",
            "None",
            "no",
            "nothing",
        ],
    )
    def test_rejects_unknown_answers(self, raw: str) -> None:
        assert parse_vlm_answer(raw) is None

    def test_rejects_pure_chrome(self) -> None:
        # Nothing-but-UI-chrome should never survive.
        assert parse_vlm_answer("Host") is None
        assert parse_vlm_answer("You") is None
        assert parse_vlm_answer("Participant") is None

    def test_handles_host_dash_name_prefix(self) -> None:
        assert parse_vlm_answer("Host — Alice") == "Alice"
        assert parse_vlm_answer("Cohost - Bob") == "Bob"

    def test_rejects_digits_and_symbols(self) -> None:
        assert parse_vlm_answer("1234") is None
        assert parse_vlm_answer("!!!") is None


# ─── Step 4: majority vote ─────────────────────────────────────────────────
class TestMajorityVote:
    def test_unanimous_wins(self) -> None:
        assert majority_vote(["Alice", "Alice", "Alice"]) == "Alice"

    def test_two_of_three_wins(self) -> None:
        assert majority_vote(["Alice", "Alice", "Bob"]) == "Alice"

    def test_tie_drops_to_none(self) -> None:
        assert majority_vote(["Alice", "Bob"]) is None

    def test_single_real_answer_wins_when_others_are_none(self) -> None:
        # One confident answer with no contradicting evidence is still
        # accepted — this is the common case when 2 of 3 frames returned
        # "unknown" (e.g. the active-tile crop failed on those frames).
        assert majority_vote(["Alice", None, None]) == "Alice"

    def test_all_none_returns_none(self) -> None:
        assert majority_vote([None, None, None]) is None

    def test_ignores_none_answers(self) -> None:
        # Two Alices out of three REAL answers → majority.
        assert majority_vote(["Alice", None, "Alice", "Bob"]) == "Alice"

    def test_case_insensitive_aggregation(self) -> None:
        # 'alice' and 'Alice' are the same vote; returned in the most
        # common casing (Alice).
        assert majority_vote(["alice", "Alice", "Bob"]) == "Alice"


# ─── End-to-end identify_speakers_from_video ───────────────────────────────
class TestIdentifySpeakersFromVideo:
    def _patched_env(self, vlm_answers_by_speaker: dict[str, list[str | None]]):
        """Patch out VLM loading, frame extraction, and image preparation.

        ``vlm_answers_by_speaker`` maps each Speaker N to the sequence of
        answers the (mocked) VLM should return as the pipeline walks
        through their sampled frames.
        """
        # Fake processor + model are just sentinels; _ask_vlm is patched.
        fake_model = MagicMock(name="qwen-vl-model")
        fake_processor = MagicMock(name="qwen-vl-processor")

        # Build the answer stream in the order identify_speakers_from_video
        # will consume it. Python's dict iteration preserves insertion
        # order, so we walk the mapping in the same order as the production
        # code walks the picked-samples dict.
        answer_stream: list[str | None] = []
        for _spk, answers in vlm_answers_by_speaker.items():
            answer_stream.extend(answers)

        ask_vlm = MagicMock(side_effect=answer_stream)

        patches = [
            patch.object(video_names, "VIDEO_VLM_ENABLED", True),
            patch.object(video_names, "VIDEO_VLM_SAMPLES_PER_SPEAKER", 3),
            patch.object(video_names, "VIDEO_VLM_MIN_SEGMENT_DURATION", 1.0),
            patch.object(
                video_names, "_get_vlm", return_value=(fake_model, fake_processor)
            ),
            patch.object(
                video_names, "_extract_frame_at",
                side_effect=lambda v, t, out: out,
            ),
            patch.object(
                video_names, "_prepare_crop",
                # (image, crop_note) — truthy image is all _ask_vlm needs.
                side_effect=lambda p: (object(), "cropped-tile (test)"),
            ),
            patch.object(video_names, "_ask_vlm", new=ask_vlm),
        ]
        return patches, ask_vlm

    def _run(self, patches, tr: Transcript) -> VideoSpeakerEvidence:
        for p in patches:
            p.start()
        try:
            return identify_speakers_from_video(tr.source_path, tr)
        finally:
            for p in reversed(patches):
                p.stop()

    def test_disabled_returns_empty_evidence(self) -> None:
        tr = _transcript([_seg("Speaker 1", 0.0, 20.0)])
        with patch.object(video_names, "VIDEO_VLM_ENABLED", False):
            ev = identify_speakers_from_video(tr.source_path, tr)
        assert ev.mapping == {} and ev.roster == []

    def test_no_diarized_speakers_short_circuits(self) -> None:
        tr = _transcript([_seg("", 0.0, 20.0)])
        patches = [
            patch.object(video_names, "VIDEO_VLM_ENABLED", True),
            patch.object(
                video_names, "_get_vlm", return_value=(MagicMock(), MagicMock())
            ),
        ]
        ev = self._run(patches, tr)
        assert ev.mapping == {} and ev.roster == []

    def test_majority_vote_produces_mapping(self) -> None:
        tr = _transcript([
            _seg("Speaker 1", 0.0, 15.0),
            _seg("Speaker 1", 20.0, 30.0),
            _seg("Speaker 1", 40.0, 50.0),
            _seg("Speaker 2", 60.0, 75.0),
            _seg("Speaker 2", 80.0, 90.0),
            _seg("Speaker 2", 100.0, 110.0),
        ])
        patches, ask_vlm = self._patched_env({
            "Speaker 1": ["Alice", "Alice", "Alice"],
            "Speaker 2": ["Bob", "Bob", None],
        })
        ev = self._run(patches, tr)
        assert ev.mapping == {"Speaker 1": "Alice", "Speaker 2": "Bob"}
        assert set(ev.roster) == {"Alice", "Bob"}
        # One VLM call per sampled frame.
        assert ask_vlm.call_count == 6

    def test_disagreeing_samples_drop_from_mapping(self) -> None:
        # All three samples disagree → no confident winner for Speaker 1.
        tr = _transcript([
            _seg("Speaker 1", 0.0, 15.0),
            _seg("Speaker 1", 20.0, 35.0),
            _seg("Speaker 1", 40.0, 55.0),
        ])
        patches, _ = self._patched_env({
            "Speaker 1": ["Alice", "Bob", "Carol"],
        })
        ev = self._run(patches, tr)
        # Speaker 1 is unresolved, but all names still land in the roster
        # as a vocabulary hint for the transcript-LLM fallback.
        assert ev.mapping == {}
        assert set(ev.roster) == {"Alice", "Bob", "Carol"}

    def test_stronger_claim_wins_when_same_name_contested(self) -> None:
        # Both speakers majority-vote to "Alice", but Speaker 1 did so
        # unanimously (3/3) while Speaker 2's claim is weaker (2/2 real
        # votes, with a None sample). The stronger claim wins; Speaker 2
        # has no fallback vote once Alice is banned, so they stay unmapped.
        tr = _transcript([
            _seg("Speaker 1", 0.0, 15.0),
            _seg("Speaker 1", 20.0, 30.0),
            _seg("Speaker 1", 40.0, 50.0),
            _seg("Speaker 2", 60.0, 75.0),
            _seg("Speaker 2", 80.0, 90.0),
            _seg("Speaker 2", 100.0, 110.0),
        ])
        patches, _ = self._patched_env({
            "Speaker 1": ["Alice", "Alice", "Alice"],
            "Speaker 2": ["Alice", "Alice", None],
        })
        ev = self._run(patches, tr)
        assert ev.mapping == {"Speaker 1": "Alice"}
        assert ev.roster == ["Alice"]

    def test_tied_claim_drops_both(self) -> None:
        # Equal-strength claims on the same name — no basis to pick a
        # winner. Both speakers stay unmapped, but the name still lands
        # in the roster as a vocabulary hint.
        tr = _transcript([
            _seg("Speaker 1", 0.0, 15.0),
            _seg("Speaker 1", 20.0, 30.0),
            _seg("Speaker 1", 40.0, 50.0),
            _seg("Speaker 2", 60.0, 75.0),
            _seg("Speaker 2", 80.0, 90.0),
            _seg("Speaker 2", 100.0, 110.0),
        ])
        patches, _ = self._patched_env({
            "Speaker 1": ["Alice", "Alice", "Alice"],
            "Speaker 2": ["Alice", "Alice", "Alice"],
        })
        ev = self._run(patches, tr)
        assert ev.mapping == {}
        assert ev.roster == ["Alice"]

    def test_weaker_claimant_falls_back_to_runner_up(self) -> None:
        # Regression for the Meet-recording failure mode: the VLM reads
        # a pinned/featured tile for most frames, but one sample catches
        # a different name. Speaker 2 votes Varun unanimously; Speaker 1
        # votes Varun 2/3 with one Himanshi read. Old behaviour dropped
        # both on conflict. New behaviour: Speaker 2 → Varun (stronger),
        # Speaker 1 falls back to Himanshi (the remaining real vote once
        # Varun is banned).
        tr = _transcript([
            _seg("Speaker 1", 0.0, 15.0),
            _seg("Speaker 1", 20.0, 30.0),
            _seg("Speaker 1", 40.0, 50.0),
            _seg("Speaker 2", 60.0, 75.0),
            _seg("Speaker 2", 80.0, 90.0),
            _seg("Speaker 2", 100.0, 110.0),
        ])
        patches, _ = self._patched_env({
            "Speaker 1": ["Varun", "Varun", "Himanshi"],
            "Speaker 2": ["Varun", "Varun", "Varun"],
        })
        ev = self._run(patches, tr)
        assert ev.mapping == {
            "Speaker 1": "Himanshi",
            "Speaker 2": "Varun",
        }
        assert set(ev.roster) == {"Varun", "Himanshi"}

    def test_all_unknown_answers_yield_empty_evidence(self) -> None:
        tr = _transcript([
            _seg("Speaker 1", 0.0, 15.0),
            _seg("Speaker 1", 20.0, 35.0),
            _seg("Speaker 1", 40.0, 55.0),
        ])
        patches, _ = self._patched_env({
            "Speaker 1": [None, None, None],
        })
        ev = self._run(patches, tr)
        assert ev.mapping == {} and ev.roster == []

    def test_failed_frame_extraction_doesnt_crash(self) -> None:
        # _extract_frame_at returning None on every call must leave us
        # with no answers — pipeline should just produce empty evidence.
        tr = _transcript([
            _seg("Speaker 1", 0.0, 15.0),
            _seg("Speaker 1", 20.0, 35.0),
            _seg("Speaker 1", 40.0, 55.0),
        ])
        patches = [
            patch.object(video_names, "VIDEO_VLM_ENABLED", True),
            patch.object(video_names, "VIDEO_VLM_SAMPLES_PER_SPEAKER", 3),
            patch.object(video_names, "VIDEO_VLM_MIN_SEGMENT_DURATION", 1.0),
            patch.object(
                video_names, "_get_vlm", return_value=(MagicMock(), MagicMock())
            ),
            patch.object(video_names, "_extract_frame_at", return_value=None),
            patch.object(video_names, "_ask_vlm", return_value="Alice"),
        ]
        ev = self._run(patches, tr)
        assert ev.mapping == {} and ev.roster == []
