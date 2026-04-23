"""Tests for the regex pre-pass + unambiguous shortcut in src/speaker_names."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.schemas import Transcript, TranscriptSegment
from src.speaker_names import (
    apply_speaker_names,
    collect_evidence,
    extract_self_intros,
    extract_vocatives,
    format_bundle,
    infer_speaker_names,
    resolve_unambiguous,
)


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


# ─── extract_self_intros ───────────────────────────────────────────────────
class TestSelfIntros:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Hi everyone, I'm Alice from engineering.", ["Alice"]),
            ("I am Bob and I'll be running point.", ["Bob"]),
            ("My name is Carol, nice to meet you.", ["Carol"]),
            ("My name's Dave.", ["Dave"]),
            ("This is Eve, joining late.", ["Eve"]),
            ("Frank here — sorry I'm late.", ["Frank"]),
            ("Grace speaking.", ["Grace"]),
            ("I’m Heidi from product.", ["Heidi"]),  # smart apostrophe
        ],
    )
    def test_matches_intros(self, text: str, expected: list[str]) -> None:
        assert extract_self_intros(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "I'm fine, thanks.",
            "I am here.",
            "This is great news.",
            "My name is on the list.",  # no proper name follows
            "Yeah okay sure.",
            "Hello everyone.",
        ],
    )
    def test_rejects_non_names(self, text: str) -> None:
        assert extract_self_intros(text) == []

    def test_dedupes_within_one_segment(self) -> None:
        text = "I'm Alice. As I said, I'm Alice from engineering."
        assert extract_self_intros(text) == ["Alice"]


# ─── extract_vocatives ─────────────────────────────────────────────────────
class TestVocatives:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Thanks, Alice.", ["Alice"]),
            ("Thank you Bob for the update.", ["Bob"]),
            ("Carol, can you take that one?", ["Carol"]),
            ("So what do you think, Dave?", ["Dave"]),
        ],
    )
    def test_matches_vocatives(self, text: str, expected: list[str]) -> None:
        assert extract_vocatives(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Thanks for the help.",
            "Yeah sure.",
            "We should look at this.",
            "Okay so the plan is to ship.",  # "Okay" is a stopword
        ],
    )
    def test_rejects_non_vocatives(self, text: str) -> None:
        assert extract_vocatives(text) == []


# ─── collect_evidence ──────────────────────────────────────────────────────
class TestCollectEvidence:
    def test_self_intro_attributed_to_own_speaker(self) -> None:
        ev = collect_evidence([
            _seg("Speaker 1", "Hi, I'm Alice."),
            _seg("Speaker 2", "Cool."),
        ])
        assert "Alice" in ev["Speaker 1"].self_intros
        assert "Alice" not in ev["Speaker 2"].self_intros

    def test_vocative_attributed_to_adjacent_speaker(self) -> None:
        ev = collect_evidence([
            _seg("Speaker 1", "Let's begin."),
            _seg("Speaker 2", "Thanks, Alice."),  # addresses Speaker 1
            _seg("Speaker 3", "Sure."),
        ])
        # Speaker 2's vocative attributes "Alice" to its adjacent speakers
        # (Speaker 1 and Speaker 3), not to Speaker 2 itself.
        assert "Alice" in ev["Speaker 1"].addressed_as
        assert "Alice" in ev["Speaker 3"].addressed_as
        assert "Alice" not in ev["Speaker 2"].addressed_as

    def test_skips_blank_speakers(self) -> None:
        ev = collect_evidence([_seg("", "I'm Alice.")])
        assert ev == {}

    def test_caps_snippets_per_name(self) -> None:
        # 5 self-introductions of the same name → snippet bucket capped.
        segs = [_seg("Speaker 1", f"I'm Alice (turn {i}).") for i in range(5)]
        ev = collect_evidence(segs)
        assert len(ev["Speaker 1"].self_intros["Alice"]) <= 2


# ─── resolve_unambiguous ───────────────────────────────────────────────────
class TestResolveUnambiguous:
    def test_clean_self_intros_resolve_directly(self) -> None:
        ev = collect_evidence([
            _seg("Speaker 1", "Hi, I'm Alice."),
            _seg("Speaker 2", "And I'm Bob."),
        ])
        mapping = resolve_unambiguous(["Speaker 1", "Speaker 2"], ev)
        assert mapping == {"Speaker 1": "Alice", "Speaker 2": "Bob"}

    def test_partial_evidence_returns_partial_mapping(self) -> None:
        # Speaker 2 never self-introduces → omitted from mapping, no LLM needed.
        ev = collect_evidence([
            _seg("Speaker 1", "Hi, I'm Alice."),
            _seg("Speaker 2", "Sounds good."),
        ])
        mapping = resolve_unambiguous(["Speaker 1", "Speaker 2"], ev)
        assert mapping == {"Speaker 1": "Alice"}

    def test_competing_self_intros_defer_to_llm(self) -> None:
        # Same speaker says two different names → ambiguous.
        ev = collect_evidence([
            _seg("Speaker 1", "I'm Alice."),
            _seg("Speaker 1", "Actually, I'm Alex, sorry."),
        ])
        assert resolve_unambiguous(["Speaker 1"], ev) is None

    def test_name_collision_defers_to_llm(self) -> None:
        # Two speakers both "introduce themselves" as Alice → defer.
        ev = collect_evidence([
            _seg("Speaker 1", "I'm Alice."),
            _seg("Speaker 2", "I'm Alice too, weird."),
        ])
        assert resolve_unambiguous(["Speaker 1", "Speaker 2"], ev) is None

    def test_no_evidence_returns_empty_mapping(self) -> None:
        ev = collect_evidence([
            _seg("Speaker 1", "Yes."),
            _seg("Speaker 2", "No."),
        ])
        assert resolve_unambiguous(["Speaker 1", "Speaker 2"], ev) == {}


# ─── infer_speaker_names: end-to-end shortcut behavior ─────────────────────
class TestInferSpeakerNames:
    def test_unambiguous_skips_llm(self) -> None:
        t = _transcript([
            _seg("Speaker 1", "Hi, I'm Alice."),
            _seg("Speaker 2", "Hey Alice, I'm Bob."),
        ])
        with patch("src.speaker_names.complete") as mock_complete:
            mapping = infer_speaker_names(t)
        mock_complete.assert_not_called()
        assert mapping == {"Speaker 1": "Alice", "Speaker 2": "Bob"}

    def test_ambiguous_calls_llm_with_bundle(self) -> None:
        # Speaker 1 has two competing self-intros → must defer to LLM.
        t = _transcript([
            _seg("Speaker 1", "I'm Alice."),
            _seg("Speaker 1", "Sorry I meant I'm Alex."),
            _seg("Speaker 2", "Got it."),
        ])
        with patch(
            "src.speaker_names.complete",
            return_value={"Speaker 1": "Alex", "Speaker 2": None},
        ) as mock_complete:
            mapping = infer_speaker_names(t)

        assert mock_complete.called
        kwargs = mock_complete.call_args.kwargs
        # Bundle is sent (not the full transcript).
        assert "Self-introductions" in kwargs["user"]
        assert kwargs["json_mode"] is True
        # Routed to the small/cheap model.
        assert kwargs["model"] == "llama-3.1-8b-instant"
        # LLM-returned nulls are filtered out.
        assert mapping == {"Speaker 1": "Alex"}

    def test_empty_transcript_returns_empty(self) -> None:
        t = _transcript([_seg("", "untracked utterance")])
        with patch("src.speaker_names.complete") as mock_complete:
            assert infer_speaker_names(t) == {}
        mock_complete.assert_not_called()

    def test_no_self_intros_but_vocatives_defers_to_llm(self) -> None:
        # No speaker self-introduces, but "Thanks, Alice" / "Bob, can you..."
        # vocatives exist. The regex alone can't safely assign these, so we
        # should call the LLM rather than silently returning {}.
        t = _transcript([
            _seg("Speaker 1", "Let's get started."),
            _seg("Speaker 2", "Thanks, Alice."),
            _seg("Speaker 1", "Sure."),
            _seg("Speaker 2", "Bob, can you own that?"),
            _seg("Speaker 3", "Yep, I'll take it."),
        ])
        with patch(
            "src.speaker_names.complete",
            return_value={"Speaker 1": "Alice", "Speaker 3": "Bob"},
        ) as mock_complete:
            mapping = infer_speaker_names(t)

        mock_complete.assert_called_once()
        kwargs = mock_complete.call_args.kwargs
        assert kwargs["model"] == "llama-3.1-8b-instant"
        assert "Addressed by others" in kwargs["user"]
        assert mapping == {"Speaker 1": "Alice", "Speaker 3": "Bob"}

    def test_truly_no_evidence_still_skips_llm(self) -> None:
        # No self-intros AND no vocatives → nothing for the LLM to work
        # with, so don't bother calling it.
        t = _transcript([
            _seg("Speaker 1", "Yeah."),
            _seg("Speaker 2", "Okay."),
            _seg("Speaker 1", "Sounds good."),
        ])
        with patch("src.speaker_names.complete") as mock_complete:
            mapping = infer_speaker_names(t)
        mock_complete.assert_not_called()
        assert mapping == {}


# ─── apply_speaker_names ───────────────────────────────────────────────────
class TestApplyMapping:
    def test_rewrites_only_mapped_labels(self) -> None:
        t = _transcript([
            _seg("Speaker 1", "Hi."),
            _seg("Speaker 2", "Hey."),
        ])
        out = apply_speaker_names(t, {"Speaker 1": "Alice"})
        assert [s.speaker for s in out.segments] == ["Alice", "Speaker 2"]

    def test_empty_mapping_returns_input(self) -> None:
        t = _transcript([_seg("Speaker 1", "Hi.")])
        assert apply_speaker_names(t, {}) is t


# ─── Bundle formatting ─────────────────────────────────────────────────────
def test_format_bundle_lists_every_label() -> None:
    ev = collect_evidence([
        _seg("Speaker 1", "I'm Alice."),
        _seg("Speaker 2", "Yes."),
    ])
    bundle = format_bundle(["Speaker 1", "Speaker 2"], ev)
    assert "=== Speaker 1 ===" in bundle
    assert "=== Speaker 2 ===" in bundle
    assert "Alice" in bundle
    assert "(none found)" in bundle  # Speaker 2 has no evidence
