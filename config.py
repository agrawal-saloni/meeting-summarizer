"""Central configuration — loads environment variables and defines constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
GOLD_DIR: Path = DATA_DIR / "gold"
OUTPUT_DIR: Path = DATA_DIR / "outputs"
PROMPTS_DIR: Path = ROOT_DIR / "prompts"

# ─── API keys ──────────────────────────────────────────────────────────────
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")  # pyannote requires this

# ─── Model defaults ────────────────────────────────────────────────────────
ASR_MODEL: str = os.getenv("ASR_MODEL", "distil-large-v3")
ASR_DEVICE: str = os.getenv("ASR_DEVICE", "auto")  # auto-detect GPU
ASR_COMPUTE_TYPE: str = os.getenv("ASR_COMPUTE_TYPE", "int8")
ASR_BEAM_SIZE: int = int(os.getenv("ASR_BEAM_SIZE", "1"))
ASR_VAD_FILTER: bool = os.getenv("ASR_VAD_FILTER", "true").lower() in (
    "1", "true", "yes", "on"
)

# ─── Segment merging ───────────────────────────────────────────────────────
# After ASR+diarization, glue together consecutive segments from the same
# speaker so a single utterance shows as one row instead of 4 fragments.
MERGE_MAX_GAP_SECONDS: float = float(os.getenv("MERGE_MAX_GAP_SECONDS", "1.5"))
MERGE_MAX_DURATION_SECONDS: float = float(
    os.getenv("MERGE_MAX_DURATION_SECONDS", "45")
)
MERGE_MAX_CHARS: int = int(os.getenv("MERGE_MAX_CHARS", "600"))

DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"

# ─── On-screen name extraction (video only) ────────────────────────────────
# When the input is a video, we can OCR sampled frames to harvest real
# participant names from tile captions / name overlays. The harvested
# roster is fed to the speaker-name resolver as an extra hint; it does
# NOT on its own map a name to a specific speaker.
VIDEO_OCR_ENABLED: bool = os.getenv("VIDEO_OCR_ENABLED", "true").lower() in (
    "1", "true", "yes", "on"
)
# Frames-per-second to sample. 0.1 → one frame every 10s.
VIDEO_OCR_SAMPLE_FPS: float = float(os.getenv("VIDEO_OCR_SAMPLE_FPS", "0.1"))
# Hard cap on number of frames sent through OCR (runtime bound). With the
# default 0.1 fps, 120 frames covers the first 20 minutes — usually more
# than enough since tile captions persist throughout the meeting.
VIDEO_OCR_MAX_FRAMES: int = int(os.getenv("VIDEO_OCR_MAX_FRAMES", "120"))
# OCR language(s). Comma-separated EasyOCR codes (e.g. "en" or "en,es").
VIDEO_OCR_LANGS: tuple[str, ...] = tuple(
    s.strip() for s in os.getenv("VIDEO_OCR_LANGS", "en").split(",") if s.strip()
)

LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_FALLBACK_MODEL: str | None = (
    os.getenv("LLM_FALLBACK_MODEL") or "llama-3.1-8b-instant"
)
# Speaker-name detection is a small extraction task — route it to the cheaper
# 8B model by default so it doesn't eat the 70B token-per-day budget. Override
# with SPEAKER_NAME_MODEL=<slug>, or set to "" / the same value as LLM_MODEL
# to keep it on the primary model.
SPEAKER_NAME_MODEL: str = (
    os.getenv("SPEAKER_NAME_MODEL") or "llama-3.1-8b-instant"
)
# Map/reduce split for summarize + extract_action_items: the per-chunk MAP
# step is straightforward extraction/condensation and runs well on the 8B
# model, while the REDUCE step benefits from 70B's synthesis quality.
# Keeping MAP off the 70B saves the majority of the daily TPD budget.
MAP_MODEL: str = os.getenv("MAP_MODEL") or "llama-3.1-8b-instant"
REDUCE_MODEL: str = os.getenv("REDUCE_MODEL") or LLM_MODEL
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0"))

# ─── Chunking ──────────────────────────────────────────────────────────────
CHUNK_TOKENS: int = 3000
CHUNK_OVERLAP_TOKENS: int = 200

# ─── Real-time simulation ──────────────────────────────────────────────────
REALTIME_WINDOW_SECONDS: int = 60
