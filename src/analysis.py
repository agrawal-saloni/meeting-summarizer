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


# ─── Summarization ─────────────────────────────────────────────────────────
def summarize(
    transcript: Transcript, prompt_version: str = "v1"
) -> MeetingSummary:
    """Map-reduce summarization: summarize each chunk, then merge into JSON."""
    chunks = chunk_segments(transcript.segments)
    map_prompt = _load_prompt(f"summarize_{prompt_version}.txt")
    reduce_prompt = _load_prompt(f"summarize_reduce_{prompt_version}.txt")

    chunk_summaries = [
        complete(
            system=map_prompt,
            user=segments_to_text(chunk),
            json_mode=False,
            temperature=0.0,
        )
        for chunk in chunks
    ]
    final = complete(
        system=reduce_prompt,
        user="\n\n---\n\n".join(chunk_summaries),
        json_mode=True,
        temperature=0.0,
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
            system=prompt,
            user=segments_to_text(chunk),
            json_mode=True,
            temperature=0.0,
        )
        for item in result.get("action_items", []):
            all_items.append(ActionItem(**item))
    return _dedupe(all_items)


# ─── Speaker name inference ────────────────────────────────────────────────
_NAME_INFERENCE_PROMPT = """\
You are analyzing a meeting transcript where speakers are labeled
"Speaker 1", "Speaker 2", … by an automatic diarization system.

Your job: determine the real first name of each speaker WHEN possible, by
looking for evidence in the transcript. Strong evidence includes:
  - Self-introductions ("Hi, I'm Alice.", "This is Bob.", "Alice here.")
  - Other people addressing them by name ("Bob, can you draft that?",
    "Thanks, Carol.", "I agree with Alice.")
  - Sign-offs that reveal identity

Rules:
- Only assign a name when the evidence is unambiguous (ideally 2+ references).
- If you are not confident, return null for that speaker — DO NOT GUESS.
- Use just the first name (no titles, no surnames) unless the surname is
  the only consistent reference.
- The same name must NOT be assigned to two different speakers.

Respond with a single valid JSON object whose keys are the speaker labels
present in the transcript, mapping each to a name string or null:
{"Speaker 1": "Alice", "Speaker 2": null, ...}
"""


def infer_speaker_names(transcript: Transcript) -> dict[str, str]:
    """Ask the LLM to map "Speaker N" labels to real names where possible.

    Returns a partial mapping: only speakers the LLM is confident about are
    included. Speakers without identified names are omitted (so callers can
    leave their labels untouched).
    """
    speaker_labels = sorted({s.speaker for s in transcript.segments if s.speaker})
    if len(speaker_labels) < 1:
        return {}

    text = segments_to_text(transcript.segments)
    user = f"Speakers in this transcript: {', '.join(speaker_labels)}\n\n{text}"
    raw = complete(
        system=_NAME_INFERENCE_PROMPT,
        user=user,
        json_mode=True,
        temperature=0.0,
    )

    mapping: dict[str, str] = {}
    seen_names: set[str] = set()
    for label in speaker_labels:
        name = raw.get(label)
        if not isinstance(name, str):
            continue
        clean = name.strip()
        if not clean or clean.lower() in seen_names:
            continue
        mapping[label] = clean
        seen_names.add(clean.lower())
    return mapping


def apply_speaker_names(
    transcript: Transcript, mapping: dict[str, str]
) -> Transcript:
    """Return a copy of ``transcript`` with speaker labels rewritten in-place."""
    if not mapping:
        return transcript
    new_segments = [
        seg.model_copy(update={"speaker": mapping.get(seg.speaker, seg.speaker)})
        for seg in transcript.segments
    ]
    return transcript.model_copy(update={"segments": new_segments})


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
