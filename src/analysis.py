"""Core analysis — chunking, summarization, and action item extraction.

All LLM-facing logic lives here. Input is a Transcript, outputs are a
MeetingSummary and/or a list of ActionItems.

Token-budget design
-------------------
Summarization and action-item extraction both walk the same chunks. Rather
than paying for the transcript content twice, we use a FUSED MAP step: one
prompt per chunk produces both a chunk-level summary and a list of action
items in a single JSON response. The REDUCE step then merges the chunk
summaries into the final structured MeetingSummary, while action items are
dedup'd in Python.

Models:
  - MAP  → MAP_MODEL     (cheap/fast, default 8B)
  - REDUCE → REDUCE_MODEL (synthesis-heavy, default primary LLM_MODEL)

Single-call entry point is ``analyze()``. The older ``summarize()`` and
``extract_action_items()`` functions are kept as thin wrappers for
backward compatibility; callers that need both outputs (report builders,
real-time streaming) should call ``analyze`` directly to avoid re-running
the map step twice.
"""

from __future__ import annotations

import tiktoken

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TOKENS,
    MAP_MODEL,
    PROMPTS_DIR,
    REDUCE_MODEL,
)
from src.llm_client import complete
from src.schemas import ActionItem, MeetingSummary, Transcript, TranscriptSegment

_encoding = tiktoken.get_encoding("cl100k_base")


# ─── Chunking ──────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    """Render segments for the LLM. Speaker labels are already normalized
    upstream by `_normalize_speaker_labels` in input_processing — empty
    speaker means no diarization, so we just emit the text."""
    return "\n".join(
        s.text if not s.speaker else f"{s.speaker}: {s.text}" for s in segments
    )


def chunk_segments(
    segments: list[TranscriptSegment],
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[list[TranscriptSegment]]:
    """Greedy chunking. Never splits a segment; preserves overlap between chunks."""
    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(f"{seg.speaker}: {seg.text}")
        if current_tokens + seg_tokens > max_tokens and current:
            chunks.append(current)
            tail, tail_tokens = [], 0
            for s in reversed(current):
                t = count_tokens(f"{s.speaker}: {s.text}")
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_tokens += t
            current = tail.copy()
            current_tokens = tail_tokens
        current.append(seg)
        current_tokens += seg_tokens

    if current:
        chunks.append(current)
    return chunks


# ─── Prompt loading ────────────────────────────────────────────────────────
def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


# ─── Fused map step ────────────────────────────────────────────────────────
def _run_fused_map(
    chunks: list[list[TranscriptSegment]], prompt_version: str
) -> list[dict]:
    """Run the combined summary + action-items prompt once per chunk.

    Returns a list of ``{"chunk_summary": str, "action_items": [...]}`` dicts,
    one per chunk, in order. Missing keys are normalized so downstream code
    can rely on the schema without extra guards.
    """
    prompt = _load_prompt(f"map_{prompt_version}.txt")
    results: list[dict] = []
    for chunk in chunks:
        raw = complete(
            system=prompt,
            user=segments_to_text(chunk),
            json_mode=True,
            temperature=0.0,
            model=MAP_MODEL or None,
        )
        if not isinstance(raw, dict):
            raw = {}
        results.append({
            "chunk_summary": raw.get("chunk_summary", "") or "",
            "action_items": raw.get("action_items", []) or [],
        })
    return results


# ─── Public entry points ───────────────────────────────────────────────────
def analyze(
    transcript: Transcript, prompt_version: str = "v1"
) -> tuple[MeetingSummary, list[ActionItem]]:
    """Single-pass summarization + action-item extraction.

    Runs the fused map prompt once over the transcript chunks, then does
    one reduce call to build the final MeetingSummary. Action items are
    collected directly from the map output and dedup'd in Python.
    """
    chunks = chunk_segments(transcript.segments)
    if not chunks:
        return _empty_summary(), []

    map_results = _run_fused_map(chunks, prompt_version)

    summary = _reduce_summaries(
        [r["chunk_summary"] for r in map_results if r["chunk_summary"]],
        prompt_version,
    )

    items: list[ActionItem] = []
    for r in map_results:
        for raw in r["action_items"]:
            try:
                items.append(ActionItem(**raw))
            except Exception:  # noqa: BLE001 — tolerate partial LLM output
                continue

    return summary, _dedupe(items)


def summarize(
    transcript: Transcript, prompt_version: str = "v1"
) -> MeetingSummary:
    """Return just the MeetingSummary. Thin wrapper around ``analyze``.

    Callers that also need action items should call ``analyze`` directly to
    avoid running the map step twice.
    """
    return analyze(transcript, prompt_version=prompt_version)[0]


def extract_action_items(
    transcript: Transcript, prompt_version: str = "v1"
) -> list[ActionItem]:
    """Return just the action items. Thin wrapper around ``analyze``.

    Callers that also need the summary should call ``analyze`` directly to
    avoid running the map step twice.
    """
    return analyze(transcript, prompt_version=prompt_version)[1]


# ─── Reduce step ───────────────────────────────────────────────────────────
def _reduce_summaries(
    chunk_summaries: list[str], prompt_version: str
) -> MeetingSummary:
    """Fold chunk summaries into the final structured MeetingSummary.

    Uses REDUCE_MODEL (the primary/70B model by default) because synthesis
    across chunks benefits from the stronger model even though the inputs
    (chunk summaries) are small.
    """
    if not chunk_summaries:
        return _empty_summary()

    reduce_prompt = _load_prompt(f"summarize_reduce_{prompt_version}.txt")
    final = complete(
        system=reduce_prompt,
        user="\n\n---\n\n".join(chunk_summaries),
        json_mode=True,
        temperature=0.0,
        model=REDUCE_MODEL or None,
    )
    return MeetingSummary(**final)


def _empty_summary() -> MeetingSummary:
    return MeetingSummary(
        overview="",
        key_decisions=[],
        discussion_points=[],
        open_questions=[],
    )


def _dedupe(items: list[ActionItem]) -> list[ActionItem]:
    """Drop near-duplicate items (same owner + leading 60 chars of task)."""
    seen: list[ActionItem] = []
    for item in items:
        key = (item.owner.lower().strip(), item.task.lower().strip()[:60])
        if any(
            (s.owner.lower().strip(), s.task.lower().strip()[:60]) == key
            for s in seen
        ):
            continue
        seen.append(item)
    return seen
