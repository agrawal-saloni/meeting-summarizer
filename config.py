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

# ─── On-screen name extraction (video only, VLM-based) ─────────────────────
# When the input is a video, we can identify speakers directly from their
# participant tiles using a Vision-Language Model (Qwen 2.5-VL). The
# pipeline: pick clean mid-turn timestamps from diarization → crop to
# the active-speaker tile using cheap OpenCV heuristics → ask the VLM
# for the name on the tile → majority-vote across 2-3 samples per
# speaker. Output is a direct Speaker-N → name mapping PLUS a roster
# hint for the transcript-LLM fallback resolver.
VIDEO_VLM_ENABLED: bool = os.getenv("VIDEO_VLM_ENABLED", "true").lower() in (
    "1", "true", "yes", "on"
)
# HF model id of the Qwen 2.5-VL variant to use. The 3B instruct model
# is the best speed/accuracy trade-off for tile-name reading; bump to 7B
# for harder meetings (low-res recordings, stylized fonts) if you have
# the VRAM.
VIDEO_VLM_MODEL: str = os.getenv(
    "VIDEO_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
)
# How many frames to sample per diarized speaker. 3 is the sweet spot:
# enough to majority-vote confidently without quadrupling inference cost.
VIDEO_VLM_SAMPLES_PER_SPEAKER: int = int(
    os.getenv("VIDEO_VLM_SAMPLES_PER_SPEAKER", "3")
)
# A turn must last at least this many seconds before we'll sample its
# middle — shorter turns are prone to UI transitions (active-speaker
# border hasn't committed yet) and make the crop unreliable. 1.5s is
# the sweet spot: long enough that the UI has settled, short enough
# that quick interjectors ("yeah", "right, that makes sense") still
# get at least one frame sampled instead of being skipped entirely.
VIDEO_VLM_MIN_SEGMENT_DURATION: float = float(
    os.getenv("VIDEO_VLM_MIN_SEGMENT_DURATION", "1.5")
)
# Expand the detected active-tile bbox by this fraction on each side
# before cropping, so the label strip (usually just below/beside the
# video feed) stays inside the crop.
VIDEO_VLM_TILE_PADDING: float = float(
    os.getenv("VIDEO_VLM_TILE_PADDING", "0.08")
)
# Cap on VLM generation length — we only need the model to emit a name
# or the word "unknown". Short answers save tokens and reduce the chance
# of hallucinated follow-up sentences.
VIDEO_VLM_MAX_NEW_TOKENS: int = int(
    os.getenv("VIDEO_VLM_MAX_NEW_TOKENS", "16")
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
