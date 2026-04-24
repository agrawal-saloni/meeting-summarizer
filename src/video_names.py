"""Video → Speaker-to-name mapping via a Vision-Language Model.

Pipeline (four cheap steps):

  1. **Pick clean timestamps from diarization, not the video.** For each
     ``Speaker N`` we take 2–3 timestamps at the *middle* of their longest
     turns. Middle-of-turn means the active-speaker UI cue is fully
     committed — no mid-transition frames.

  2. **Narrow the crop before the VLM sees anything.** Zoom draws a
     green/yellow border around the active tile, Meet enlarges it, Teams
     adds a subtle outline. Cheap OpenCV heuristics (HSV mask + contour
     bbox) find the active tile in ~a few ms per frame, and we crop to
     just that tile plus its label strip. Smaller input = faster + more
     accurate VLM.

  3. **Ask the VLM a grounded question, not for OCR.** Instead of "read
     this text," we ask: *"What is the participant's name labeled on this
     tile? Respond with just the name, or 'unknown' if not visible."*
     VLMs handle this naturally — they shrug off "(You)" suffixes,
     "Host — Alice" prefixes, multi-word names, and partial occlusion.

  4. **Aggregate per speaker.** For each ``Speaker N`` we majority-vote
     across the 2–3 sampled frames. All three agree → high confidence,
     keep. Disagreement or all-unknown → drop, and let the transcript
     LLM path resolve it instead.

The VLM (Qwen 2.5-VL by default) and ``transformers`` are optional at
runtime: if either is missing or the model fails to load,
``identify_speakers_from_video`` returns empty evidence and the rest of
the pipeline keeps working.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import ffmpeg

from config import (
    VIDEO_VLM_ENABLED,
    VIDEO_VLM_MAX_NEW_TOKENS,
    VIDEO_VLM_MIN_SEGMENT_DURATION,
    VIDEO_VLM_MODEL,
    VIDEO_VLM_SAMPLES_PER_SPEAKER,
    VIDEO_VLM_TILE_PADDING,
)
from src.schemas import Transcript, TranscriptSegment

log = logging.getLogger(__name__)


# ─── Result type ───────────────────────────────────────────────────────────
@dataclass
class VideoSpeakerEvidence:
    """Output of the video-side name-resolution pass.

    ``mapping`` is a confident ``{Speaker N: name}`` dict — the VLM agreed
    across multiple frames for that speaker. Callers should apply it to
    the transcript as-is.

    ``roster`` is the union of every name the VLM returned at least once
    (majority-vote winner or not). It's weaker evidence than ``mapping``
    but useful as a *vocabulary hint* for the transcript-side LLM
    resolver: "these are the real names of people actually on this call."
    """

    mapping: dict[str, str] = field(default_factory=dict)
    roster: list[str] = field(default_factory=list)


EMPTY_EVIDENCE = VideoSpeakerEvidence()


# ─── Qwen 2.5-VL (lazy singleton) ──────────────────────────────────────────
_vlm_state: dict[str, object] = {}
_vlm_unavailable: bool = False


def _get_vlm() -> tuple[object, object] | None:
    """Return ``(model, processor)`` or ``None`` if the VLM is unavailable.

    Missing ``transformers`` / a failed model download / OOM on first
    load are all swallowed once and remembered so a broken install
    doesn't crash every meeting.
    """
    global _vlm_unavailable
    if _vlm_state:
        return _vlm_state["model"], _vlm_state["processor"]  # type: ignore[return-value]
    if _vlm_unavailable:
        return None
        

    try:
        import torch
        from transformers import (  # type: ignore
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError as e:
        print(
            f"[video-names] transformers / qwen-vl unavailable ({e}); "
            "skipping VLM-based name extraction. Install with "
            "`pip install transformers accelerate qwen-vl-utils`."
        )
        _vlm_unavailable = True
        return None

    t0 = time.time()
    print(f"[video-names] loading Qwen2.5-VL model={VIDEO_VLM_MODEL!r}…")

    # Primary path: let `accelerate` shard/offload the model automatically.
    # On installs without `accelerate` this raises an ImportError-ish
    # runtime error — we catch it and fall back to a plain single-device
    # load + manual `.to(device)`, which works on any torch install.
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VIDEO_VLM_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
    except (ImportError, ValueError, RuntimeError) as e:
        msg = str(e).lower()
        if "accelerate" in msg:
            print(
                "[video-names] `accelerate` not available — retrying VLM "
                "load without device_map=auto. Install `accelerate` for "
                "faster / GPU-sharded loading."
            )
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    VIDEO_VLM_MODEL,
                    torch_dtype="auto",
                )
                model = model.to(_resolve_torch_device(torch))
            except Exception as e2:  # noqa: BLE001
                print(
                    f"[video-names] Qwen2.5-VL load failed ({e2}); "
                    "disabling VLM name extraction for this run."
                )
                _vlm_unavailable = True
                return None
        else:
            print(
                f"[video-names] Qwen2.5-VL load failed ({e}); disabling VLM "
                "name extraction for this run."
            )
            _vlm_unavailable = True
            return None
    except Exception as e:  # noqa: BLE001
        print(
            f"[video-names] Qwen2.5-VL load failed ({e}); disabling VLM "
            "name extraction for this run."
        )
        _vlm_unavailable = True
        return None

    try:
        processor = AutoProcessor.from_pretrained(VIDEO_VLM_MODEL)
    except Exception as e:  # noqa: BLE001
        print(
            f"[video-names] Qwen2.5-VL processor load failed ({e}); "
            "disabling VLM name extraction for this run."
        )
        _vlm_unavailable = True
        return None

    _vlm_state["model"] = model
    _vlm_state["processor"] = processor
    print(f"[video-names] VLM ready ({time.time() - t0:.1f}s)")
    return model, processor


def _resolve_torch_device(torch_module):
    """Pick the best torch device available (CUDA → MPS → CPU)."""
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


# ─── Step 1: pick clean timestamps from diarization ────────────────────────
def _pick_sample_timestamps(
    segments: list[TranscriptSegment],
    n_per_speaker: int,
    min_duration: float,
) -> dict[str, list[float]]:
    """Return ``{speaker: [t1, t2, …]}`` picked from the *middle* of long turns.

    We treat each merged transcript segment as a "clean turn" (it already
    comes from diarization + same-speaker gluing). Sorting by duration
    and taking the midpoint of the top N guarantees we sample moments
    where only one person is speaking and the active-speaker UI cue
    (green border, enlarged tile) is fully committed.
    """
    by_speaker: dict[str, list[TranscriptSegment]] = defaultdict(list)
    for seg in segments:
        spk = seg.speaker
        if not spk or spk.strip().lower() in {"speaker ?", "unknown"}:
            continue
        if (seg.end_time - seg.start_time) < min_duration:
            continue
        by_speaker[spk].append(seg)

    out: dict[str, list[float]] = {}
    for spk, segs in by_speaker.items():
        segs.sort(key=lambda s: s.end_time - s.start_time, reverse=True)
        picked = segs[:n_per_speaker]
        out[spk] = [(s.start_time + s.end_time) / 2.0 for s in picked]
    return out


# ─── Frame extraction ──────────────────────────────────────────────────────
def _extract_frame_at(video_path: Path, timestamp: float, out_path: Path) -> Path | None:
    """Decode a single frame at ``timestamp`` seconds into ``out_path``.

    Uses ffmpeg's ``-ss`` seek (fast + reasonably accurate on keyframes;
    we don't need sub-frame precision for tile captions).
    """
    try:
        (
            ffmpeg
            .input(str(video_path), ss=max(0.0, timestamp))
            .output(str(out_path), vframes=1, format="image2")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        stderr = (e.stderr or b"").decode(errors="ignore")
        log.warning(
            "ffmpeg single-frame extraction failed at t=%.2fs: %s",
            timestamp, stderr[:200].strip(),
        )
        return None
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


# ─── Step 2: OpenCV active-tile heuristic ──────────────────────────────────
# Each palette is ``(name, lo_hsv, hi_hsv)`` in OpenCV conventions
# (H=0-179, S=0-255, V=0-255). We keep ranges conservative — we'd rather
# miss a dim border and fall back to full-frame than crop to some random
# same-coloured slide element. The palette *name* is purely for logging:
# a line like "full-frame (no-palette-signal)" vs "cropped-tile (meet-blue,
# ratio=0.42)" makes it obvious when a recording uses a border the
# heuristic doesn't cover.
_ACTIVE_TILE_PALETTES: tuple[
    tuple[str, tuple[int, int, int], tuple[int, int, int]], ...
] = (
    # Zoom's active-speaker vivid green border (~#00e676 family).
    ("zoom-green",   (40, 120, 120), (80, 255, 255)),
    # Zoom spotlight / pin yellow highlight (~#ffd400).
    ("zoom-yellow",  (20, 120, 150), (35, 255, 255)),
    # Google Meet active-speaker blue border (~#1a73e8 / #4285f4 family).
    # Requires medium-high saturation so we don't match pastel slide
    # backgrounds or the grey-blue chrome around the gallery.
    ("meet-blue",    (100, 150, 120), (118, 255, 255)),
    # Teams active-tile purple/indigo Fluent accent (~#5b5fc7 / #6264a7).
    ("teams-purple", (115, 80, 120), (130, 255, 255)),
)

# Active-tile bbox sanity limits: must cover a reasonable fraction of
# the frame (not a tiny icon, not the whole screen).
_TILE_MIN_SIZE_RATIO = 0.10
_TILE_MAX_SIZE_RATIO = 0.95

# A palette "has signal" when at least this fraction of the frame matched
# its HSV range. Anything below is treated as background noise and
# excluded from the combined mask, so the diagnostic string can honestly
# say WHICH palette(s) contributed to the detection.
_PALETTE_MIN_PIXEL_RATIO = 0.0005


def _detect_active_tile(
    frame,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Return ``(bbox, reason)`` for the active-speaker tile.

    ``bbox`` is ``None`` when no confident tile was found. ``reason`` is a
    short, human-readable string suitable for per-frame logging — it
    names the palette(s) that fired and the resulting area ratio on
    success, or explains why detection failed (no cv2, empty frame, no
    palette signal, bbox-too-small, bbox-too-large, etc.).
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None, "cv2-unavailable"

    if frame is None or getattr(frame, "size", 0) == 0:
        return None, "empty-frame"

    h_img, w_img = frame.shape[:2]
    frame_area = float(w_img * h_img)
    if frame_area <= 0:
        return None, "empty-frame"

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Run each palette individually so the log line can attribute the
    # eventual bbox to a specific UI ("meet-blue" vs "teams-purple").
    signals: list[tuple[str, float]] = []
    matched_palettes: list[str] = []
    combined_mask = None
    for name, lo, hi in _ACTIVE_TILE_PALETTES:
        m = cv2.inRange(
            hsv,
            np.array(lo, dtype=np.uint8),
            np.array(hi, dtype=np.uint8),
        )
        fraction = float((m > 0).sum()) / frame_area
        signals.append((name, fraction))
        if fraction < _PALETTE_MIN_PIXEL_RATIO:
            continue
        matched_palettes.append(name)
        combined_mask = m if combined_mask is None else cv2.bitwise_or(
            combined_mask, m
        )

    if combined_mask is None:
        top_name, top_fraction = max(
            signals, key=lambda s: s[1], default=("", 0.0)
        )
        return None, (
            f"no-palette-signal (top={top_name or 'none'} "
            f"{top_fraction * 100:.3f}%)"
        )

    # Active-tile borders are thin strokes. Close small gaps so the
    # whole rectangle registers as one contour rather than four edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, f"no-contours (palettes={'+'.join(matched_palettes)})"

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    ratio = (w * h) / frame_area
    palette_note = "+".join(matched_palettes)

    if ratio < _TILE_MIN_SIZE_RATIO:
        return None, (
            f"bbox-too-small (palette={palette_note}, ratio={ratio:.3f})"
        )
    if ratio > _TILE_MAX_SIZE_RATIO:
        return None, (
            f"bbox-too-large (palette={palette_note}, ratio={ratio:.3f})"
        )

    return (x, y, w, h), f"palette={palette_note}, ratio={ratio:.3f}"


def _find_active_tile_bbox(frame) -> tuple[int, int, int, int] | None:
    """Thin wrapper around :func:`_detect_active_tile` that returns just
    the bbox (or ``None``). Kept for backward compatibility with older
    callers and tests; prefer ``_detect_active_tile`` when you also want
    the diagnostic reason string for logging.
    """
    bbox, _reason = _detect_active_tile(frame)
    return bbox


def _crop_to_tile(frame, bbox: tuple[int, int, int, int], padding: float):
    """Crop to the tile bbox plus a padding margin (to include the label strip)."""
    x, y, w, h = bbox
    h_img, w_img = frame.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w_img, x + w + pad_x)
    y1 = min(h_img, y + h + pad_y)
    return frame[y0:y1, x0:x1]


def _prepare_crop(frame_path: Path) -> tuple[object | None, str]:
    """Load a frame → ``(PIL image, crop_note)``.

    ``crop_note`` is a short log-friendly string describing whether we
    cropped to a detected active-speaker tile ("cropped-tile (…)") or
    fell back to the full frame, and — in either case — the underlying
    :func:`_detect_active_tile` diagnostic reason. This makes failure
    modes visible in the run log: you can tell at a glance whether the
    heuristic never found any highlighted border, found one but rejected
    it as too small, etc.

    Returns ``(None, reason)`` on load errors so the caller can still
    log *why* the frame was skipped.
    """
    try:
        import cv2  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return None, "cv2-unavailable"

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None, "frame-decode-failed"

    bbox, detect_reason = _detect_active_tile(frame)
    if bbox:
        crop = _crop_to_tile(frame, bbox, VIDEO_VLM_TILE_PADDING)
        crop_note = f"cropped-tile ({detect_reason})"
    else:
        crop = frame
        crop_note = f"full-frame ({detect_reason})"

    # OpenCV → PIL (BGR → RGB).
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), crop_note


# ─── Step 3: ask the VLM a grounded question ───────────────────────────────
_VLM_PROMPT = (
    "What is the participant's name labeled on this video-call tile? "
    "Ignore any 'Host', '(You)', 'Cohost', or 'Presenter' suffixes or "
    "prefixes and return just the person's name. Respond with ONLY the "
    "name (first name is fine) or the word 'unknown' if no name label "
    "is visible. No punctuation, no quotes, no extra words."
)

# Post-process the VLM's free-form answer into a clean first name, or
# ``None`` when the answer is "unknown" / gibberish / chrome.
_CHROME_TOKENS = {
    "you", "host", "cohost", "co-host", "presenter", "meeting", "recording",
    "participant", "participants", "speaker", "guest", "organizer",
}
_UNKNOWN_TOKENS = {"unknown", "unclear", "none", "n/a", "na", "no", "nothing"}


def _parse_vlm_answer(raw: str) -> str | None:
    """Normalize a VLM answer to a first-name string or ``None`` (no match).

    We deliberately keep this conservative: when in doubt, return None
    and let majority voting / the transcript LLM fill the gap.
    """
    if not raw:
        return None
    # Strip whitespace, common quotes, trailing punctuation, and any
    # lead-in like "The name is …" the model may emit despite the prompt.
    text = raw.strip().strip("\"'`.,;:!? ")
    text = re.sub(
        r"^(the\s+(?:name|person|participant)"
        r"(?:\s+shown)?(?:\s+(?:is|appears\s+to\s+be))\s+"
        r"|name[:\- ]+|answer[:\- ]+)",
        "", text, flags=re.IGNORECASE,
    )
    text = text.strip("\"'`.,;:!? ")

    if not text:
        return None
    if text.lower() in _UNKNOWN_TOKENS:
        return None

    # Drop bracketed annotations ("(You)", "[Host]") and dash prefixes
    # ("Host — Alice" → "Alice").
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^]]*\]", "", text)
    text = re.sub(r"^[^—\-–]+[—\-–]\s*", "", text) if any(
        token in text.lower().split()[:2] for token in ("host", "cohost", "co-host")
    ) else text

    tokens = [t for t in re.split(r"\s+", text) if t]
    tokens = [t for t in tokens if t.lower() not in _CHROME_TOKENS]
    if not tokens:
        return None

    # Prefer the *first* plausibly-name-shaped token; this handles
    # "Alice Smith" → "Alice", "Alice (You)" after the bracket strip,
    # and ignores the model occasionally echoing part of the prompt.
    for tok in tokens:
        cleaned = tok.strip("\"'`.,;:!?")
        if not cleaned:
            continue
        if not cleaned[0].isalpha():
            continue
        if not (2 <= len(cleaned) <= 20):
            continue
        if not cleaned.replace("'", "").replace("-", "").isalpha():
            continue
        return cleaned[0].upper() + cleaned[1:]
    return None


def _ask_vlm(model, processor, image) -> str | None:
    """Ask Qwen2.5-VL for the name on this tile. Returns a cleaned name or ``None``."""
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except ImportError:
        # qwen-vl-utils is the official helper; without it we'd have to
        # reimplement image preprocessing. Treat it as a hard dep.
        log.warning("qwen-vl-utils not installed — cannot run VLM.")
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _VLM_PROMPT},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(  # type: ignore[operator]
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)  # type: ignore[attr-defined]

        generated = model.generate(  # type: ignore[attr-defined]
            **inputs, max_new_tokens=VIDEO_VLM_MAX_NEW_TOKENS
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        raw = processor.batch_decode(  # type: ignore[attr-defined]
            trimmed, skip_special_tokens=True
        )[0]
    except Exception as e:  # noqa: BLE001
        log.warning("VLM inference failed: %s", e)
        return None

    return _parse_vlm_answer(raw)


# ─── Step 4: majority-vote across samples ──────────────────────────────────
def _majority_vote(answers: list[str | None]) -> str | None:
    """Return the confident winning name, or ``None`` if none qualifies.

    Rules:
      - Ignore ``None`` / "unknown" answers from vote counting.
      - Need at least 1 real vote, and the winner must strictly outrank
        every other REAL answer (case-insensitive). This keeps
        ``["Varun", None, None]`` → ``"Varun"`` (a confident single
        read with no contradicting evidence) while still rejecting
        ``["Varun", "Himanshi", None]`` as a 1-1 tie.
    """
    real = [a for a in answers if a]
    if not real:
        return None

    # Case-insensitive vote aggregation, preserving the best-cased original.
    buckets: dict[str, list[str]] = defaultdict(list)
    for a in real:
        buckets[a.lower()].append(a)

    counts = Counter({k: len(v) for k, v in buckets.items()})
    (top_key, top_n), *rest = counts.most_common()
    if rest and rest[0][1] == top_n:
        # Tie — no confident winner.
        return None

    # Return the best-cased variant of the winning name. On ties we
    # prefer a Title-Case spelling (first char uppercase, rest lowercase)
    # over ALL-LOWERCASE or ALL-CAPS, since Title Case is how real tile
    # labels are almost always rendered.
    variants = Counter(buckets[top_key])

    def _case_rank(v: str) -> int:
        # Lower rank = more preferred. Title Case first, then mixed, then
        # all-lower, then all-upper (rare on tile labels).
        if v[:1].isupper() and v[1:].islower():
            return 0
        if v.islower():
            return 2
        if v.isupper():
            return 3
        return 1

    return sorted(
        variants.items(),
        key=lambda kv: (-kv[1], _case_rank(kv[0]), kv[0]),
    )[0][0]


# ─── Public API ────────────────────────────────────────────────────────────
def identify_speakers_from_video(
    video_path: str | Path,
    transcript: Transcript,
) -> VideoSpeakerEvidence:
    """Map ``Speaker N`` labels to real names using diarization + a VLM.

    Returns :data:`EMPTY_EVIDENCE` (never raises) when the feature is
    disabled, the VLM is unavailable, or the video has no diarized
    speakers. Callers should merge the returned mapping into the
    transcript *before* falling back to the transcript-based LLM
    resolver.
    """
    if not VIDEO_VLM_ENABLED:
        return EMPTY_EVIDENCE

    vlm = _get_vlm()
    if vlm is None:
        return EMPTY_EVIDENCE
    model, processor = vlm

    samples = _pick_sample_timestamps(
        transcript.segments,
        n_per_speaker=VIDEO_VLM_SAMPLES_PER_SPEAKER,
        min_duration=VIDEO_VLM_MIN_SEGMENT_DURATION,
    )
    if not samples:
        print("[video-names] no diarized speakers with long-enough turns — "
              "nothing to sample.")
        return EMPTY_EVIDENCE

    video_path = Path(video_path)
    t0 = time.time()
    n_total = sum(len(ts) for ts in samples.values())
    print(
        f"[video-names] sampling {n_total} frames across "
        f"{len(samples)} speakers (up to {VIDEO_VLM_SAMPLES_PER_SPEAKER}/speaker)"
    )

    per_speaker_answers: dict[str, list[str | None]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="meeting-vlm-") as tmp:
        tmp_dir = Path(tmp)
        idx = 0
        for spk, timestamps in samples.items():
            for ts in timestamps:
                frame_path = tmp_dir / f"frame_{idx:05d}.png"
                idx += 1
                extracted = _extract_frame_at(video_path, ts, frame_path)
                if extracted is None:
                    print(f"[video-names]  {spk} @ {ts:.1f}s → (frame extract failed)")
                    continue
                image, crop_note = _prepare_crop(extracted)
                if image is None:
                    print(
                        f"[video-names]  {spk} @ {ts:.1f}s → "
                        f"(frame {crop_note})"
                    )
                    continue
                answer = _ask_vlm(model, processor, image)
                per_speaker_answers[spk].append(answer)
                print(
                    f"[video-names]  {spk} @ {ts:.1f}s [{crop_note}] "
                    f"→ {answer or 'unknown'!r}"
                )

    return _finalize_evidence(
        per_speaker_answers, wall_seconds=time.time() - t0
    )


def _resolve_assignments(
    per_speaker_answers: dict[str, list[str | None]],
) -> dict[str, str]:
    """Assign one name per speaker, resolving conflicts by vote strength.

    Plain majority vote collapses badly when a single name dominates
    several speakers' samples — a common failure mode when the VLM can't
    find the active-speaker cue and just reads whichever tile is pinned
    / biggest / closest. In that scenario most speakers "majority-vote"
    to the pinned person, the old drop-on-any-conflict rule wipes the
    mapping, and we lose even the minority signal that ``Speaker N``
    occasionally got a *different* name read.

    Instead we do iterative Hungarian-lite assignment:

      1. For each un-assigned speaker, compute a majority winner from
         the votes that haven't been banned yet.
      2. Group proposals by name. If one speaker uniquely claims a name,
         assign it. If several claim the same name, the one with the
         highest raw vote count wins and the name is banned from
         everyone else's pool; losers re-vote on the next iteration
         (possibly landing on a runner-up). Ties ban the name from all
         claimants — conservative, no confident winner.
      3. Stop when no un-assigned speaker can propose a name.

    Example (Varun pinned in every frame except one, where Himanshi
    briefly took the tile)::

        Speaker 1: [Varun, Varun, Himanshi]   # majority Varun, 2/3
        Speaker 2: [Varun, Varun, Varun]      # majority Varun, 3/3

    Iteration 1: both claim Varun, Speaker 2's 3/3 beats 2/3 → Speaker 2
    → Varun; Varun banned for Speaker 1. Iteration 2: Speaker 1 re-votes
    on [Himanshi] → Himanshi. Final: ``{1: "Himanshi", 2: "Varun"}``.

    If both speakers were *equally* confident on the same name we still
    drop both (no basis to pick one over the other).
    """
    remaining: dict[str, list[str | None]] = {
        spk: list(answers) for spk, answers in per_speaker_answers.items()
    }
    mapping: dict[str, str] = {}
    banned: set[str] = set()

    # Hard cap on iterations as a paranoid guard against a logic bug
    # that would otherwise spin forever. One iteration per speaker is
    # more than enough in practice.
    max_iterations = max(1, 2 * len(remaining))

    for _ in range(max_iterations):
        # Step 1: propose a winner per un-assigned speaker using only
        # votes whose names haven't been banned yet.
        proposals: dict[str, tuple[str, int]] = {}
        for spk, answers in remaining.items():
            if spk in mapping:
                continue
            usable = [a for a in answers if a and a.lower() not in banned]
            winner = _majority_vote(usable)
            if winner is None:
                continue
            votes = sum(1 for a in usable if a.lower() == winner.lower())
            proposals[spk] = (winner, votes)

        if not proposals:
            break

        # Step 2: group proposals by (lowercased) name.
        by_name: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for spk, (name, votes) in proposals.items():
            by_name[name.lower()].append((spk, name, votes))

        changed = False
        for name_key, claimants in by_name.items():
            if len(claimants) == 1:
                spk, name, _ = claimants[0]
                mapping[spk] = name
                banned.add(name_key)
                changed = True
                continue

            # Conflict — sort by votes desc; uniquely-strongest wins.
            claimants.sort(key=lambda c: -c[2])
            top_votes = claimants[0][2]
            runner_up_votes = claimants[1][2]
            if top_votes > runner_up_votes:
                spk, name, _ = claimants[0]
                mapping[spk] = name
                banned.add(name_key)
                changed = True
            else:
                # Tied conflict — no basis to prefer one speaker over
                # the other. Ban the name from all tied claimants and
                # let them fall back to runner-ups on the next pass.
                banned.add(name_key)
                changed = True

        if not changed:
            break

    return mapping


def _finalize_evidence(
    per_speaker_answers: dict[str, list[str | None]],
    wall_seconds: float,
) -> VideoSpeakerEvidence:
    """Aggregate per-frame VLM answers into a mapping + roster."""
    roster_counts: Counter[str] = Counter()
    for answers in per_speaker_answers.values():
        for a in answers:
            if a:
                roster_counts[a] += 1

    mapping = _resolve_assignments(per_speaker_answers)

    # Roster ordering: name that showed up most often first, stable on ties.
    roster = [n for n, _ in roster_counts.most_common()]
    preview_mapping = ", ".join(f"{k}→{v}" for k, v in mapping.items()) or "(none)"
    preview_roster = ", ".join(roster[:10]) if roster else "(none)"
    print(
        f"[video-names] done ({wall_seconds:.1f}s) — "
        f"mapping: {preview_mapping}; roster: {preview_roster}"
    )
    return VideoSpeakerEvidence(mapping=mapping, roster=roster)


# ─── Helpers exposed for tests and sibling modules ─────────────────────────
pick_sample_timestamps = _pick_sample_timestamps
find_active_tile_bbox = _find_active_tile_bbox
detect_active_tile = _detect_active_tile
crop_to_tile = _crop_to_tile
parse_vlm_answer = _parse_vlm_answer
majority_vote = _majority_vote
resolve_assignments = _resolve_assignments
finalize_evidence = _finalize_evidence
