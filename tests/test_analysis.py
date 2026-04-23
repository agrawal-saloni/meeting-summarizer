"""Tests for the fused map + reduce flow in src/analysis."""

from __future__ import annotations

from unittest.mock import patch

from src.analysis import analyze, extract_action_items, summarize
from src.schemas import Transcript, TranscriptSegment


def _seg(speaker: str, text: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker, start_time=start, end_time=start + 1.0, text=text
    )


def _transcript(segments: list[TranscriptSegment]) -> Transcript:
    return Transcript(
        source_path="x.txt",
        source_type="transcript",
        duration_seconds=segments[-1].end_time if segments else 0.0,
        segments=segments,
    )


# ─── Fixtures for the mocked LLM ───────────────────────────────────────────
_MAP_RETURN = {
    "chunk_summary": "Alice and Bob agreed to ship on Friday.",
    "action_items": [
        {
            "owner": "Bob",
            "task": "Send the proposal",
            "due_date": None,
            "confidence": 0.9,
            "source_quote": "Bob: I'll send it out Wednesday.",
        }
    ],
}

_REDUCE_RETURN = {
    "overview": "The team planned a Friday ship.",
    "key_decisions": ["Ship on Friday"],
    "discussion_points": ["Ship timing"],
    "open_questions": [],
}


def _fake_complete(**kwargs):
    # json_mode is always True for our calls; key on presence of the
    # reduce prompt text to distinguish map vs reduce.
    system = kwargs.get("system", "")
    if "chunk-level summaries" in system or "chunk summaries" in system:
        return _REDUCE_RETURN
    return _MAP_RETURN


# ─── Token-budget behavior: map is called exactly once per chunk ───────────
class TestAnalyzeFusion:
    def test_analyze_calls_map_once_per_chunk(self) -> None:
        t = _transcript([
            _seg("Speaker 1", "Hello."),
            _seg("Speaker 2", "Hi."),
        ])
        with patch(
            "src.analysis.complete", side_effect=_fake_complete
        ) as mock_complete:
            summary, items = analyze(t)

        # 1 chunk → 1 map call + 1 reduce call = 2 total.
        assert mock_complete.call_count == 2
        assert summary.overview == "The team planned a Friday ship."
        assert len(items) == 1
        assert items[0].owner == "Bob"

    def test_map_uses_map_model_and_reduce_uses_reduce_model(self) -> None:
        t = _transcript([_seg("Speaker 1", "Hello.")])
        models_seen: list[str] = []

        def capture(**kwargs):
            models_seen.append(kwargs.get("model") or "")
            return _fake_complete(**kwargs)

        with patch("src.analysis.complete", side_effect=capture):
            analyze(t)

        # First call is map, second is reduce.
        assert models_seen[0] == "llama-3.1-8b-instant"  # MAP_MODEL default
        assert models_seen[1] == "llama-3.3-70b-versatile"  # REDUCE_MODEL default

    def test_summarize_and_extract_share_output_when_called_via_analyze(
        self,
    ) -> None:
        # This is the key fusion guarantee: when report/realtime call
        # analyze(), they get both outputs from a single pass of map calls.
        t = _transcript([_seg("Speaker 1", "Hello.")])
        with patch(
            "src.analysis.complete", side_effect=_fake_complete
        ) as mock_complete:
            summary, items = analyze(t)

        assert mock_complete.call_count == 2  # not 3, not 4
        assert summary.overview
        assert items


# ─── Backward-compat wrappers ──────────────────────────────────────────────
class TestBackwardCompat:
    def test_summarize_still_returns_meeting_summary(self) -> None:
        t = _transcript([_seg("Speaker 1", "Hello.")])
        with patch("src.analysis.complete", side_effect=_fake_complete):
            out = summarize(t)
        assert out.overview == "The team planned a Friday ship."

    def test_extract_action_items_still_returns_list(self) -> None:
        t = _transcript([_seg("Speaker 1", "Hello.")])
        with patch("src.analysis.complete", side_effect=_fake_complete):
            items = extract_action_items(t)
        assert len(items) == 1
        assert items[0].task == "Send the proposal"


# ─── Edge cases ────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_empty_transcript_skips_all_llm_calls(self) -> None:
        t = _transcript([])
        # Pydantic requires at least one segment? Let's just use a safe path:
        # analyze handles no-chunks. We verify via chunk_segments returning [].
        with patch("src.analysis.complete") as mock_complete:
            summary, items = analyze(t)
        mock_complete.assert_not_called()
        assert summary.overview == ""
        assert items == []

    def test_tolerates_malformed_action_item_from_llm(self) -> None:
        t = _transcript([_seg("Speaker 1", "Hello.")])
        bad_map = {
            "chunk_summary": "Something happened.",
            "action_items": [
                {"owner": "Bob"},  # missing required fields
                {
                    "owner": "Alice",
                    "task": "Do the thing",
                    "due_date": None,
                    "confidence": 0.8,
                    "source_quote": "Alice: I'll do the thing.",
                },
            ],
        }

        def fake(**kwargs):
            if "chunk-level summaries" in kwargs.get("system", "") or (
                "chunk summaries" in kwargs.get("system", "")
            ):
                return _REDUCE_RETURN
            return bad_map

        with patch("src.analysis.complete", side_effect=fake):
            _, items = analyze(t)

        # Malformed item dropped, valid item kept.
        assert len(items) == 1
        assert items[0].owner == "Alice"
