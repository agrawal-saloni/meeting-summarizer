"""Tests for src/video_names — OCR-backed on-screen name roster.

We never actually run EasyOCR here; the reader is faked so the tests
stay fast and don't need model weights. What we're really exercising
is the name-token filter and the aggregation-by-frequency contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src import video_names
from src.video_names import candidate_tokens, extract_video_names


# ─── _candidate_tokens ─────────────────────────────────────────────────────
class TestCandidateTokens:
    def test_extracts_first_name_from_tile_caption(self) -> None:
        # Typical Zoom tile caption.
        assert candidate_tokens("Alice Smith") == ["Alice", "Smith"]

    def test_strips_trailing_punctuation(self) -> None:
        assert candidate_tokens("Alice.") == ["Alice"]
        assert candidate_tokens("(Bob)") == ["Bob"]

    def test_dedupes_within_single_caption(self) -> None:
        assert candidate_tokens("Alice Alice Smith") == ["Alice", "Smith"]

    def test_rejects_ui_chrome_and_stopwords(self) -> None:
        # The conferencing-UI stopword set should drop toolbar/overlay
        # labels that OCR picks up on every frame.
        assert candidate_tokens("Mute Unmute Participants Share Screen") == []
        assert candidate_tokens("OK") == []

    def test_keeps_real_names_next_to_ui_chrome(self) -> None:
        # A real name that happens to sit next to a button label still wins.
        assert candidate_tokens("Mute Alice Smith") == ["Alice", "Smith"]

    def test_rejects_all_caps_acronyms(self) -> None:
        # Stop-wordy tokens like "QA" / "PM" must not leak into the roster.
        assert candidate_tokens("QA PM") == []

    def test_rejects_numbers_and_mixed(self) -> None:
        assert candidate_tokens("Room1 123") == []


# ─── extract_video_names ───────────────────────────────────────────────────
class _FakeReader:
    """Stand-in for an easyocr.Reader, one page of mocked text per frame."""

    def __init__(self, per_frame_texts: list[list[str]]) -> None:
        self._per_frame = per_frame_texts
        self._idx = 0

    def readtext(self, *_args, **_kwargs):  # noqa: D401 - mock
        texts = self._per_frame[self._idx]
        self._idx += 1
        return texts


def _fake_frames(tmp_path: Path, n: int) -> list[Path]:
    """Create ``n`` empty PNG stand-ins and return their paths."""
    paths: list[Path] = []
    for i in range(n):
        p = tmp_path / f"frame_{i:05d}.png"
        p.write_bytes(b"")
        paths.append(p)
    return paths


class TestExtractVideoNames:
    def test_returns_empty_when_ocr_disabled(self, tmp_path: Path) -> None:
        with patch.object(video_names, "VIDEO_OCR_ENABLED", False):
            assert extract_video_names(tmp_path / "meeting.mp4") == []

    def test_returns_empty_when_reader_unavailable(self, tmp_path: Path) -> None:
        with patch.object(video_names, "VIDEO_OCR_ENABLED", True), \
             patch.object(video_names, "_get_reader", return_value=None):
            assert extract_video_names(tmp_path / "meeting.mp4") == []

    def test_orders_names_by_frequency(self, tmp_path: Path) -> None:
        # Alice appears on every frame (pinned tile); Bob only on some.
        per_frame = [
            ["Alice Smith"],
            ["Alice Smith", "Bob Jones"],
            ["Alice Smith"],
        ]
        reader = _FakeReader(per_frame)
        frames = _fake_frames(tmp_path, n=len(per_frame))

        with patch.object(video_names, "VIDEO_OCR_ENABLED", True), \
             patch.object(video_names, "_get_reader", return_value=reader), \
             patch.object(video_names, "_extract_frames", return_value=frames):
            names = extract_video_names(tmp_path / "meeting.mp4")

        # Frequency: Alice=3, Smith=3, Bob=1, Jones=1. Alice/Smith tie →
        # Counter.most_common preserves insertion order, so Alice (seen
        # first) comes before Smith. Bob and Jones follow.
        assert names[:2] == ["Alice", "Smith"]
        assert set(names) == {"Alice", "Smith", "Bob", "Jones"}
        # Bob/Jones must come after the higher-frequency names.
        assert names.index("Bob") > names.index("Alice")
        assert names.index("Jones") > names.index("Smith")

    def test_swallows_per_frame_ocr_errors(self, tmp_path: Path) -> None:
        reader = MagicMock()
        # First frame blows up, second frame returns a usable caption.
        reader.readtext.side_effect = [RuntimeError("bad frame"), ["Carol"]]
        frames = _fake_frames(tmp_path, n=2)

        with patch.object(video_names, "VIDEO_OCR_ENABLED", True), \
             patch.object(video_names, "_get_reader", return_value=reader), \
             patch.object(video_names, "_extract_frames", return_value=frames):
            names = extract_video_names(tmp_path / "meeting.mp4")

        assert names == ["Carol"]

    def test_returns_empty_when_no_frames_sampled(self, tmp_path: Path) -> None:
        reader = MagicMock()
        with patch.object(video_names, "VIDEO_OCR_ENABLED", True), \
             patch.object(video_names, "_get_reader", return_value=reader), \
             patch.object(video_names, "_extract_frames", return_value=[]):
            assert extract_video_names(tmp_path / "meeting.mp4") == []
        reader.readtext.assert_not_called()
