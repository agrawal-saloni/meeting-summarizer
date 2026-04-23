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

DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"

LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_FALLBACK_MODEL: str | None = (
    os.getenv("LLM_FALLBACK_MODEL") or "llama-3.1-8b-instant"
)
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0"))

# ─── Chunking ──────────────────────────────────────────────────────────────
CHUNK_TOKENS: int = 3000
CHUNK_OVERLAP_TOKENS: int = 200

# ─── Real-time simulation ──────────────────────────────────────────────────
REALTIME_WINDOW_SECONDS: int = 60
