"""Core analysis — chunking, summarization, and action item extraction.

All LLM-facing logic lives here. Input is a Transcript, outputs are a
MeetingSummary and/or a list of ActionItems.
"""

from __future__ import annotations

import tiktoken

from config import CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, PROMPTS_DIR
from src.llm_client import complete
from src.schemas import ActionItem, MeetingSummary, Transcript, TranscriptSegment

_encoding = tiktoken.get_encoding("cl100k_base")


# ─── Chunking ──────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(f"{s.speaker}: {s.text}" for s in segments)


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


# ─── Summarization ─────────────────────────────────────────────────────────
def summarize(
    transcript: Transcript, prompt_version: str = "v1"
) -> MeetingSummary:
    """Map-reduce summarization: summarize each chunk, then merge into JSON."""
    chunks = chunk_segments(transcript.segments)
    map_prompt = _load_prompt(f"summarize_{prompt_version}.txt")
    reduce_prompt = _load_prompt(f"summarize_reduce_{prompt_version}.txt")

    chunk_summaries = [
        complete(system=map_prompt, user=segments_to_text(chunk), json_mode=False)
        for chunk in chunks
    ]
    final = complete(
        system=reduce_prompt,
        user="\n\n---\n\n".join(chunk_summaries),
        json_mode=True,
    )
    return MeetingSummary(**final)


# ─── Action Item Extraction ────────────────────────────────────────────────
def extract_action_items(
    transcript: Transcript, prompt_version: str = "v1"
) -> list[ActionItem]:
    """Extract action items chunk-wise, then dedupe across chunks."""
    chunks = chunk_segments(transcript.segments)
    prompt = _load_prompt(f"extract_{prompt_version}.txt")

    all_items: list[ActionItem] = []
    for chunk in chunks:
        result = complete(
            system=prompt, user=segments_to_text(chunk), json_mode=True
        )
        for item in result.get("action_items", []):
            all_items.append(ActionItem(**item))
    return _dedupe(all_items)


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
