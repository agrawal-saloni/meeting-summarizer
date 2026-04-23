"""Speaker → real-name resolution.

Strategy (cheap → expensive):

  1. Regex pre-pass over the transcript collects two kinds of evidence per
     "Speaker N":
       - self-introductions in their OWN turns ("I'm Alice", "this is Bob")
       - vocatives in ADJACENT turns from other speakers ("Thanks, Carol",
         "Alice, can you draft that?")

  2. If every speaker is either covered by a single, conflict-free
     self-introduction or has no evidence at all, we return the mapping
     directly with zero LLM tokens.

  3. Otherwise we send a tiny per-speaker evidence bundle (~hundreds of
     tokens, not the whole transcript) to a small/cheap model — by default
     the 8B model, configured via ``SPEAKER_NAME_MODEL`` — and let it
     disambiguate. The 70B model never sees this task.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from config import SPEAKER_NAME_MODEL
from src.llm_client import complete
from src.schemas import Transcript, TranscriptSegment

# ─── Name token + stop-word filtering ──────────────────────────────────────
# A "name candidate" is a single Capitalized word, 2–15 letters. We
# deliberately avoid multi-word names here — they're rare in self-intros
# and any false positive blows up the evidence bundle.
_NAME = r"[A-Z][a-zA-Z'\-]{1,14}"

# Words that look like names (capitalized at sentence start) but almost
# never are. Extend conservatively — over-pruning means a real name is
# silently dropped.
_STOPWORDS: frozenset[str] = frozenset(
    w.lower()
    for w in [
        # filler / discourse
        "Hi", "Hello", "Hey", "Yeah", "Yes", "No", "Nope", "Okay", "OK",
        "Sure", "Right", "Well", "So", "Um", "Uh", "Uhm", "Hmm", "Oh",
        "Ah", "Mm", "Mhm",
        "Thanks", "Thank", "Sorry", "Please", "Welcome", "Great", "Good",
        "Cool", "Nice", "Awesome", "Perfect", "Alright",
        "Just", "Like", "Look", "Listen", "Wait", "Actually", "Anyway",
        "Maybe", "Probably", "Really", "Basically", "Honestly", "Obviously",
        # pronouns / articles / connectors that can start a sentence
        "I", "We", "You", "They", "He", "She", "It",
        "My", "Your", "Our", "Their", "His", "Her",
        "And", "But", "Or", "So", "If", "When", "While", "Because",
        "The", "A", "An", "This", "That", "These", "Those",
        # time / calendar
        "Today", "Tomorrow", "Yesterday", "Tonight", "Now", "Then",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        # titles (we want the name *after* these, not the title itself)
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sir", "Madam",
        # common project/meeting nouns that often appear capitalized
        "Q", "QA", "PM", "CEO", "CTO", "VP",
    ]
)


def _is_name_token(tok: str) -> bool:
    if not tok or len(tok) < 2 or len(tok) > 15:
        return False
    if not tok[0].isupper():
        return False
    if tok.lower() in _STOPWORDS:
        return False
    return tok.replace("'", "").replace("-", "").isalpha()


# ─── Patterns ──────────────────────────────────────────────────────────────
# Self-introductions: matched against a speaker's OWN segment text.
_SELF_INTRO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\b(?:I['’]?m|I am)\s+({_NAME})\b"),
    re.compile(rf"\bmy name(?:'s| is)\s+({_NAME})\b", re.IGNORECASE),
    re.compile(rf"\bthis is\s+({_NAME})\b", re.IGNORECASE),
    re.compile(rf"^\s*({_NAME})\s+(?:here|speaking)\b"),
)

# Vocatives: matched against OTHER speakers' segments. The captured name
# is attributed to an adjacent speaker (the one being addressed).
_VOCATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Thanks, Alice." / "Thank you Bob"
    re.compile(rf"\b(?:thanks|thank you),?\s+({_NAME})\b", re.IGNORECASE),
    # Start-of-turn vocative: "Alice, can you …"
    re.compile(rf"^\s*({_NAME})\s*,\s+"),
    # End-of-turn vocative: "… what do you think, Bob?"
    re.compile(rf",\s*({_NAME})\s*[.!?]?\s*$"),
)

# How many example snippets to keep per (speaker, name) pair. Caps the
# evidence bundle size when one name is mentioned dozens of times.
_MAX_SNIPPETS_PER_NAME = 2
# Maximum chars per snippet kept in the bundle.
_MAX_SNIPPET_LEN = 160


# ─── Evidence collection ───────────────────────────────────────────────────
@dataclass
class _Evidence:
    """Per-speaker evidence collected from the transcript."""

    # name → list of own-turn snippets where the speaker introduced themselves
    self_intros: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # name → list of (by_speaker, snippet) where another speaker addressed them
    addressed_as: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_SNIPPET_LEN else text[: _MAX_SNIPPET_LEN - 1] + "…"


def _extract_self_intros(text: str) -> list[str]:
    """Return distinct name candidates a speaker uses to introduce themselves."""
    found: list[str] = []
    for pat in _SELF_INTRO_PATTERNS:
        for m in pat.finditer(text):
            name = m.group(1)
            if _is_name_token(name) and name not in found:
                found.append(name)
    return found


def _extract_vocatives(text: str) -> list[str]:
    """Return distinct name candidates that appear in vocative position."""
    found: list[str] = []
    for pat in _VOCATIVE_PATTERNS:
        for m in pat.finditer(text):
            name = m.group(1)
            if _is_name_token(name) and name not in found:
                found.append(name)
    return found


def _collect_evidence(
    segments: list[TranscriptSegment],
) -> dict[str, _Evidence]:
    """Walk the transcript once, building per-speaker evidence."""
    evidence: dict[str, _Evidence] = defaultdict(_Evidence)

    for i, seg in enumerate(segments):
        spk = seg.speaker
        if not spk:
            continue
        text = seg.text

        # Self-introductions belong to this speaker.
        for name in _extract_self_intros(text):
            bucket = evidence[spk].self_intros[name]
            if len(bucket) < _MAX_SNIPPETS_PER_NAME:
                bucket.append(_snippet(text))

        # Vocatives address an adjacent speaker (prev or next), not this one.
        # We attribute the candidate to every distinct adjacent speaker; the
        # LLM (or unambiguous shortcut) sorts out which actually owns the name.
        vocatives = _extract_vocatives(text)
        if not vocatives:
            continue

        adjacent: set[str] = set()
        for j in (i - 1, i + 1):
            if 0 <= j < len(segments):
                other = segments[j].speaker
                if other and other != spk:
                    adjacent.add(other)

        for name in vocatives:
            for target in adjacent:
                bucket = evidence[target].addressed_as[name]
                if len(bucket) < _MAX_SNIPPETS_PER_NAME:
                    bucket.append((spk, _snippet(text)))

    return evidence


# ─── Unambiguous shortcut ──────────────────────────────────────────────────
def _resolve_unambiguous(
    speaker_labels: list[str],
    evidence: dict[str, _Evidence],
) -> dict[str, str] | None:
    """Try to resolve names without an LLM call.

    Returns the mapping (possibly empty / partial) only when the evidence is
    fully unambiguous: every speaker either has exactly one self-introduced
    name or no self-introduction at all, AND no name is claimed by more than
    one speaker. Returns None when any ambiguity remains and the LLM should
    decide.
    """
    proposed: dict[str, str] = {}
    for spk in speaker_labels:
        intros = evidence.get(spk, _Evidence()).self_intros
        if not intros:
            continue
        # More than one distinct self-intro name → ambiguous, defer to LLM.
        if len(intros) > 1:
            return None
        proposed[spk] = next(iter(intros))

    # Name collision across speakers → defer to LLM.
    if len(set(proposed.values())) != len(proposed):
        return None

    return proposed


# ─── Evidence bundle for the LLM ───────────────────────────────────────────
def _format_bundle(
    speaker_labels: list[str],
    evidence: dict[str, _Evidence],
) -> str:
    """Render a compact evidence bundle the LLM can read at a glance."""
    lines: list[str] = [
        "Speaker labels in this meeting: " + ", ".join(speaker_labels),
        "",
    ]
    for spk in speaker_labels:
        ev = evidence.get(spk, _Evidence())
        lines.append(f"=== {spk} ===")

        if ev.self_intros:
            lines.append("Self-introductions (own turns):")
            for name, snippets in ev.self_intros.items():
                for snip in snippets:
                    lines.append(f"  - candidate={name!r} :: \"{snip}\"")
        else:
            lines.append("Self-introductions: (none found)")

        if ev.addressed_as:
            lines.append("Addressed by others (adjacent turns):")
            for name, refs in ev.addressed_as.items():
                for by_spk, snip in refs:
                    lines.append(
                        f"  - candidate={name!r} (from {by_spk}) :: \"{snip}\""
                    )
        else:
            lines.append("Addressed by others: (none found)")
        lines.append("")
    return "\n".join(lines).rstrip()


_LLM_PROMPT = """\
You are resolving "Speaker 1", "Speaker 2", … labels to real first names.

You will be given a compact evidence bundle (NOT the full transcript). For
each speaker, the bundle lists:
  - candidate names extracted from that speaker's OWN self-introductions
    (strongest evidence)
  - candidate names that other speakers used while addressing them in
    adjacent turns (weaker evidence)

Rules:
- Prefer self-introductions over being-addressed evidence.
- Only assign a name when the evidence is unambiguous. When in doubt
  (multiple competing candidates with no clear winner, single weak
  vocative, or a candidate also claimed strongly by another speaker),
  return null for that speaker — DO NOT GUESS.
- Use first names only.
- The same name MUST NOT be assigned to two different speakers.

Respond with a single valid JSON object whose keys are EXACTLY the
speaker labels listed in the bundle, mapping each to a name string or
null. Example: {"Speaker 1": "Alice", "Speaker 2": null}
"""


# ─── Public API ────────────────────────────────────────────────────────────
def infer_speaker_names(transcript: Transcript) -> dict[str, str]:
    """Map "Speaker N" labels to real names where evidence supports it.

    Cheap path: regex pre-pass + unambiguous shortcut → no LLM call.
    Expensive path: small evidence bundle → small/cheap model only.

    Returns a partial mapping (only confident speakers); callers should
    leave unlisted labels untouched.
    """
    speaker_labels = sorted({s.speaker for s in transcript.segments if s.speaker})
    if not speaker_labels:
        return {}

    evidence = _collect_evidence(transcript.segments)

    shortcut = _resolve_unambiguous(speaker_labels, evidence)
    has_vocative_evidence = any(ev.addressed_as for ev in evidence.values())

    # Take the shortcut only when it actually resolved someone, OR when
    # there's no evidence of any kind (including vocatives) for the LLM
    # to work with. An empty shortcut with vocative evidence present is
    # "insufficient regex evidence", not "resolved" — defer to the LLM.
    if shortcut is not None and (shortcut or not has_vocative_evidence):
        if shortcut:
            print(
                f"[speaker-names] regex resolved {len(shortcut)}/"
                f"{len(speaker_labels)} speakers via self-intros, skipping LLM"
            )
        else:
            print(
                f"[speaker-names] no evidence found for "
                f"{len(speaker_labels)} speakers, skipping LLM"
            )
        return shortcut

    bundle = _format_bundle(speaker_labels, evidence)
    reason = "ambiguous self-intros" if shortcut is None else "vocatives only"
    print(
        f"[speaker-names] regex {reason} → calling {SPEAKER_NAME_MODEL} "
        f"with {len(bundle)}-byte bundle ({len(speaker_labels)} speakers)"
    )
    raw = complete(
        system=_LLM_PROMPT,
        user=bundle,
        json_mode=True,
        temperature=0.0,
        model=SPEAKER_NAME_MODEL or None,
    )
    return _validate_llm_mapping(speaker_labels, raw)


def _validate_llm_mapping(
    speaker_labels: list[str], raw: object
) -> dict[str, str]:
    """Filter the LLM response: drop nulls, enforce one-name-per-speaker."""
    if not isinstance(raw, dict):
        return {}
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
    """Return a copy of ``transcript`` with speaker labels rewritten."""
    if not mapping:
        return transcript
    new_segments = [
        seg.model_copy(update={"speaker": mapping.get(seg.speaker, seg.speaker)})
        for seg in transcript.segments
    ]
    return transcript.model_copy(update={"segments": new_segments})


# ─── Helpers exposed for tests ─────────────────────────────────────────────
# Stable handles for the regex helpers so tests can exercise them directly
# without depending on internal collection logic.
extract_self_intros = _extract_self_intros
extract_vocatives = _extract_vocatives
collect_evidence = _collect_evidence
resolve_unambiguous = _resolve_unambiguous
format_bundle = _format_bundle
