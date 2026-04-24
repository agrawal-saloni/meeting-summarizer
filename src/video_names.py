"""Video → on-screen participant names (OCR roster).

Most conferencing tools (Zoom, Meet, Teams, Webex) render each
participant's name as a text overlay on their tile. We sample frames
from the recording, OCR them, and return a deduplicated list of
name-shaped tokens seen on screen.

This module intentionally does NOT try to map a name to a specific
speaker — it only produces the *roster* of real names that appeared
somewhere in the video. See ``src/speaker_names.py`` for how that
roster is consumed by the final ``Speaker N → name`` resolver.

OCR is optional. If ``easyocr`` is not installed, ``extract_video_names``
returns an empty list after logging a one-line hint, so the rest of the
pipeline keeps working.
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import ffmpeg

from config import (
    VIDEO_OCR_ENABLED,
    VIDEO_OCR_LANGS,
    VIDEO_OCR_MAX_FRAMES,
    VIDEO_OCR_SAMPLE_FPS,
)
from src.speaker_names import is_name_token

# ─── EasyOCR reader (lazy singleton) ───────────────────────────────────────
# EasyOCR downloads ~100MB of model weights on first use and takes a few
# seconds to initialize, so we cache the Reader instance per process.
_reader: object | None = None
_reader_unavailable: bool = False


def _get_reader() -> object | None:
    """Return a cached EasyOCR ``Reader`` or ``None`` if unavailable.

    We swallow ImportError / init failures once and remember the outcome;
    a missing OCR backend shouldn't take down the whole meeting pipeline.
    """
    global _reader, _reader_unavailable
    if _reader is not None:
        return _reader
    if _reader_unavailable:
        return None
    try:
        import easyocr  # type: ignore
    except ImportError:
        print(
            "[video-names] easyocr not installed — skipping on-screen name "
            "extraction. Install with `pip install easyocr` to enable."
        )
        _reader_unavailable = True
        return None

    t0 = time.time()
    langs = list(VIDEO_OCR_LANGS) or ["en"]
    print(f"[video-names] loading EasyOCR reader (langs={langs})…")
    try:
        _reader = easyocr.Reader(langs, gpu=False, verbose=False)
    except Exception as e:  # noqa: BLE001
        # Most commonly the first-run weight download fails behind a proxy
        # (WRONG_VERSION_NUMBER / cert errors) or offline. EasyOCR reuses
        # any weights already sitting in ~/.EasyOCR/model/, so we can sidestep
        # its CDN entirely — point the user at that workaround.
        msg = str(e).lower()
        looks_networky = any(
            s in msg for s in ("ssl", "urlopen", "urllib", "connection",
                               "timed out", "unreachable", "download")
        )
        print(f"[video-names] EasyOCR init failed ({e}); disabling OCR.")
        if looks_networky:
            print(
                "[video-names] ↳ looks like a model-download problem. "
                "Pre-fetch the weights from GitHub and drop them in "
                "~/.EasyOCR/model/ to bypass EasyOCR's CDN:\n"
                "  mkdir -p ~/.EasyOCR/model && cd ~/.EasyOCR/model && \\\n"
                "    curl -fLO https://github.com/JaidedAI/EasyOCR/"
                "releases/download/pre-v1.1.6/craft_mlt_25k.zip && \\\n"
                "    curl -fLO https://github.com/JaidedAI/EasyOCR/"
                "releases/download/v1.3/english_g2.zip && \\\n"
                "    unzip -o craft_mlt_25k.zip && unzip -o english_g2.zip"
            )
        _reader_unavailable = True
        return None
    print(f"[video-names] reader ready ({time.time() - t0:.1f}s)")
    return _reader


# ─── Frame sampling ────────────────────────────────────────────────────────
def _extract_frames(
    video_path: Path,
    out_dir: Path,
    sample_fps: float,
    max_frames: int,
) -> list[Path]:
    """Write ``max_frames`` PNGs sampled at ``sample_fps`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.png")
    (
        ffmpeg
        .input(str(video_path))
        .output(pattern, vf=f"fps={sample_fps}", vframes=max_frames)
        .overwrite_output()
        .run(quiet=True)
    )
    return sorted(out_dir.glob("frame_*.png"))


# ─── Name filtering ────────────────────────────────────────────────────────
# Punctuation we strip before testing a token — OCR routinely bolts dots
# and commas onto the end of a caption ("Alice Smith.").
_STRIP_CHARS = ".,;:!?()[]{}\"'"


def _candidate_tokens(text: str) -> list[str]:
    """Split an OCR'd string into distinct name-shaped tokens."""
    found: list[str] = []
    for raw in text.split():
        tok = raw.strip(_STRIP_CHARS)
        if is_name_token(tok) and tok not in found:
            found.append(tok)
    return found


# ─── Public API ────────────────────────────────────────────────────────────
def extract_video_names(video_path: str | Path) -> list[str]:
    """Return on-screen participant names, ordered by frequency.

    Returns an empty list (never raises) when OCR is disabled, the OCR
    backend is unavailable, or ffmpeg can't read the file. Callers can
    treat an empty list as "no roster evidence" and proceed.
    """
    if not VIDEO_OCR_ENABLED:
        return []

    reader = _get_reader()
    if reader is None:
        return []

    video_path = Path(video_path)
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="meeting-ocr-") as tmp:
        tmp_dir = Path(tmp)
        try:
            frames = _extract_frames(
                video_path,
                tmp_dir,
                sample_fps=VIDEO_OCR_SAMPLE_FPS,
                max_frames=VIDEO_OCR_MAX_FRAMES,
            )
        except ffmpeg.Error as e:
            stderr = (e.stderr or b"").decode(errors="ignore")
            print(
                f"[video-names] ffmpeg frame extraction failed: "
                f"{stderr[:200].strip()}"
            )
            return []

        if not frames:
            print("[video-names] no frames sampled — nothing to OCR.")
            return []

        print(
            f"[video-names] OCRing {len(frames)} frames "
            f"(fps={VIDEO_OCR_SAMPLE_FPS}, cap={VIDEO_OCR_MAX_FRAMES})…"
        )

        counts: Counter[str] = Counter()
        for frame in frames:
            # detail=0 → return a flat list of strings, skipping bbox/confidence
            # (we don't need them since we aggregate across many frames).
            try:
                texts = reader.readtext(  # type: ignore[attr-defined]
                    str(frame), detail=0, paragraph=False
                )
            except Exception as e:  # noqa: BLE001
                # A single bad frame shouldn't abort the whole pass.
                print(f"[video-names]  …skipping {frame.name}: {e}")
                continue
            for txt in texts:
                for tok in _candidate_tokens(txt):
                    counts[tok] += 1

    names = [name for name, _ in counts.most_common()]
    preview = ", ".join(names[:10]) if names else "(none)"
    print(
        f"[video-names] done ({time.time() - t0:.1f}s) — "
        f"{len(names)} candidate(s): {preview}"
    )
    return names


# ─── Helpers exposed for tests ─────────────────────────────────────────────
candidate_tokens = _candidate_tokens
