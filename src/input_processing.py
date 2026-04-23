"""Input processing — video/audio/transcript → normalized Transcript.

This module owns every step before the LLM sees a meeting:
  - detect input type
  - extract audio from video via ffmpeg
  - transcribe audio with faster-whisper  (GPU-accelerated when available)
  - overlay pyannote speaker diarization  (GPU-accelerated when available)
  - parse pre-existing transcripts (txt/srt/vtt)
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ffmpeg
import pysrt
import torch
import torchaudio
import webvtt
from faster_whisper import WhisperModel
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from pyannote.audio import Pipeline

from config import (
    ASR_BEAM_SIZE,
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_MODEL,
    ASR_VAD_FILTER,
    DIARIZATION_MODEL,
    HF_TOKEN,
    MERGE_MAX_CHARS,
    MERGE_MAX_DURATION_SECONDS,
    MERGE_MAX_GAP_SECONDS,
    PROCESSED_DIR,
)
from src.schemas import Transcript, TranscriptSegment

log = logging.getLogger(__name__)

# ─── File-type detection ───────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
TRANSCRIPT_EXTS = {".txt", ".srt", ".vtt"}

# Matches "Alice: Hello" or "[Alice] Hello"
_SPEAKER_LINE = re.compile(
    r"^\s*(?:\[(?P<spk1>[^\]]+)\]|(?P<spk2>[^:]+):)\s*(?P<text>.+)$"
)

# Lazy-loaded model singletons
_whisper_model: WhisperModel | None = None
_diarize_pipeline: Pipeline | None = None


# ─── GPU / device helpers ──────────────────────────────────────────────────
def _resolve_device() -> tuple[str, str]:
    """Decide on (device, compute_type) for Whisper.

    - If ASR_DEVICE is 'cuda' and CUDA is available → ('cuda', 'float16')
    - If ASR_DEVICE is 'cuda' but no GPU → fallback to CPU with a warning
    - If ASR_DEVICE is 'auto' → use cuda when available, else cpu
    - Otherwise honor the configured device and compute type
    """
    cuda_ok = torch.cuda.is_available()
    requested = (ASR_DEVICE or "auto").lower()

    if requested in ("cuda", "gpu"):
        if cuda_ok:
            return "cuda", ASR_COMPUTE_TYPE or "float16"
        print(f"[asr] ⚠️  ASR_DEVICE={requested} requested but no CUDA GPU found "
              f"— falling back to CPU")
        return "cpu", "int8"

    if requested == "auto":
        if cuda_ok:
            return "cuda", ASR_COMPUTE_TYPE or "float16"
        return "cpu", ASR_COMPUTE_TYPE or "int8"

    # Explicit CPU (or anything else)
    return requested, ASR_COMPUTE_TYPE or "int8"


def _resolve_torch_device() -> torch.device:
    """Pick the best PyTorch device for pyannote (CUDA → MPS → CPU).

    Note: faster-whisper uses CTranslate2, which does NOT support Apple MPS,
    so Whisper still runs on CPU on Macs. Pyannote, being plain PyTorch,
    can use MPS for a large speedup on Apple Silicon.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Public API ────────────────────────────────────────────────────────────
def detect_input_type(path: Path) -> str:
    """Return 'video' | 'audio' | 'transcript' based on file extension."""
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TRANSCRIPT_EXTS:
        return "transcript"
    raise ValueError(f"Unsupported file type: {ext}")


def load_meeting(path: str | Path, diarize: bool = True) -> Transcript:
    """Main entry point. Loads any supported input and returns a Transcript.

    Args:
        path: Path to video, audio, or transcript file.
        diarize: Run pyannote speaker diarization on audio/video inputs.
    """
    path = Path(path)
    input_type = detect_input_type(path)

    if input_type == "video":
        audio_path = _extract_audio_from_video(path)
        return _transcribe_audio(audio_path, diarize=diarize,
                                 source_path=str(path), source_type="video")
    if input_type == "audio":
        return _transcribe_audio(path, diarize=diarize,
                                 source_path=str(path), source_type="audio")
    return _parse_transcript_file(path)


# ─── Video → Audio ─────────────────────────────────────────────────────────
def _extract_audio_from_video(video_path: Path, sample_rate: int = 16000) -> Path:
    """Extract audio as 16kHz mono WAV using ffmpeg."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{video_path.stem}.wav"
    t0 = time.time()
    print(f"[ffmpeg] extracting audio: {video_path.name} → {out_path.name}")
    (
        ffmpeg
        .input(str(video_path))
        .output(str(out_path), ac=1, ar=sample_rate, vn=None,
                acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )
    print(f"[ffmpeg] done ({time.time()-t0:.1f}s)  "
          f"size={out_path.stat().st_size/1e6:.1f} MB")
    return out_path


# ─── ASR + Diarization ─────────────────────────────────────────────────────
def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        device, compute_type = _resolve_device()
        print(f"[whisper] loading model={ASR_MODEL!r}  device={device}  "
              f"compute_type={compute_type}")
        t0 = time.time()
        _whisper_model = WhisperModel(
            ASR_MODEL, device=device, compute_type=compute_type
        )
        print(f"[whisper] model ready ({time.time()-t0:.1f}s)")
    return _whisper_model


class DiarizationAccessError(RuntimeError):
    """Raised when the HF token can't download the pyannote model."""


_ACCESS_HELP = (
    f"Cannot download {DIARIZATION_MODEL!r} from Hugging Face.\n"
    "This model is gated. To fix:\n"
    "  1. Accept the user agreement at:\n"
    f"       https://huggingface.co/{DIARIZATION_MODEL}\n"
    "       https://huggingface.co/pyannote/segmentation-3.0\n"
    "     (must be logged in as the same user as your HF_TOKEN)\n"
    "  2. Use a token with read access to gated repos:\n"
    "     - classic 'Read' token (easiest), OR\n"
    "     - fine-grained token with 'Read access to contents of all "
    "public gated repos you can access' enabled.\n"
    "  3. Put it in .env as HF_TOKEN=... and restart.\n"
    "Alternatively, disable diarization in the UI to skip this step."
)


def _get_diarization_pipeline() -> Pipeline:
    global _diarize_pipeline
    if _diarize_pipeline is None:
        if not HF_TOKEN:
            raise DiarizationAccessError(
                "HF_TOKEN not set — required for pyannote diarization.\n"
                + _ACCESS_HELP
            )
        print(f"[pyannote] loading pipeline={DIARIZATION_MODEL!r}")
        t0 = time.time()
        try:
            _diarize_pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL, use_auth_token=HF_TOKEN
            )
        except (HfHubHTTPError, LocalEntryNotFoundError) as e:
            raise DiarizationAccessError(_ACCESS_HELP) from e

        if _diarize_pipeline is None:
            # pyannote returns None when the token is valid but lacks access
            raise DiarizationAccessError(_ACCESS_HELP)

        # Pyannote defaults to CPU. Move to the best available accelerator
        # (CUDA on Linux/Windows, MPS/Metal on Apple Silicon).
        device = _resolve_torch_device()
        try:
            _diarize_pipeline.to(device)
            print(f"[pyannote] pipeline moved to {device.type.upper()} "
                  f"({time.time()-t0:.1f}s)")
        except (RuntimeError, NotImplementedError) as e:
            # Some pyannote ops occasionally lack MPS kernels — fall back gracefully.
            print(f"[pyannote] ⚠️  could not use {device.type.upper()} ({e}); "
                  f"falling back to CPU")
            _diarize_pipeline.to(torch.device("cpu"))
    return _diarize_pipeline


def _run_asr(
    audio_path: Path,
) -> tuple[list[TranscriptSegment], "object", float]:
    """Run faster-whisper end-to-end and return (segments, info, asr_wall_seconds)."""
    model = _get_whisper()
    t0 = time.time()
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=ASR_BEAM_SIZE,
        vad_filter=ASR_VAD_FILTER,
    )
    print(f"[asr] language={info.language} (prob={info.language_probability:.2f})  "
          f"duration={info.duration:.1f}s  beam={ASR_BEAM_SIZE}  "
          f"vad={ASR_VAD_FILTER}  (detect {time.time()-t0:.1f}s)")

    segments: list[TranscriptSegment] = []
    t_iter = time.time()
    last_tick = 0.0
    for i, seg in enumerate(segments_iter, start=1):
        segments.append(
            TranscriptSegment(
                speaker="Speaker ?",
                start_time=seg.start,
                end_time=seg.end,
                text=seg.text.strip(),
            )
        )
        if i % 10 == 0 or seg.end - last_tick >= 30:
            rtf = (time.time() - t_iter) / max(seg.end, 0.01)
            print(f"[asr]  …seg {i:4d}  audio {seg.end:6.1f}s  "
                  f"wall {time.time()-t_iter:5.1f}s  rtf={rtf:.2f}x")
            last_tick = seg.end

    t_asr = time.time() - t_iter
    speed = info.duration / max(t_asr, 0.01)
    print(f"[asr] ✅ {len(segments)} segments, {info.duration:.1f}s audio in "
          f"{t_asr:.1f}s wall ({speed:.1f}x realtime)")
    return segments, info, t_asr


def _run_diarization(audio_path: Path) -> list[tuple[float, float, str]]:
    """Run pyannote diarization and return a list of (start, end, speaker) turns."""
    t0 = time.time()
    print(f"[diarize] running pyannote on {audio_path.name}")
    # See `_attach_speakers` history: pass an in-memory waveform to avoid
    # codec-related sample-count mismatches in pyannote's chunked reader.
    waveform, sample_rate = _load_mono_waveform(audio_path)
    diarization = _get_diarization_pipeline()(
        {"waveform": waveform, "sample_rate": sample_rate}
    )
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    n_spk = len({spk for _, _, spk in turns})
    print(f"[diarize] ✅ {n_spk} speakers, {len(turns)} turns "
          f"({time.time()-t0:.1f}s)")
    return turns


def _transcribe_audio(
    audio_path: Path,
    diarize: bool,
    source_path: str,
    source_type: str,
) -> Transcript:
    """Whisper ASR + optional pyannote diarization, run concurrently when both apply."""
    t_start = time.time()
    file_mb = audio_path.stat().st_size / 1e6
    print(f"[asr] file={audio_path.name}  size={file_mb:.1f} MB")

    # Pre-warm models in the foreground so concurrent threads don't race on
    # the lazy-init singletons (and so model-load logs aren't interleaved).
    _get_whisper()
    if diarize:
        _get_diarization_pipeline()

    if diarize:
        # Whisper runs on CPU (CTranslate2) and pyannote on MPS/CUDA when
        # available, so the two largely use disjoint hardware. Even on pure
        # CPU, both libs release the GIL during inference, so threading wins.
        print("[pipeline] running ASR + diarization concurrently")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="meet") as ex:
            asr_future = ex.submit(_run_asr, audio_path)
            diar_future = ex.submit(_run_diarization, audio_path)
            segments, info, t_asr = asr_future.result()
            turns = diar_future.result()
        segments = _merge_speakers(segments, turns)
    else:
        segments, info, t_asr = _run_asr(audio_path)
        print("[diarize] skipped (diarize=False)")

    # Normalize speaker labels for downstream consumers (UI, LLM, dedupe).
    # Pyannote ids → "Speaker 1/2/…"; unknown labels (no diarization) →
    # blank, so the UI doesn't display a meaningless "Speaker ?".
    segments = _normalize_speaker_labels(segments)

    # Glue together consecutive same-speaker segments so one statement
    # shows as one row, not 4 fragments. Capped by gap, duration, and chars
    # to avoid runaway merges (e.g. one speaker monologuing for 10 minutes).
    pre_merge = len(segments)
    segments = _merge_consecutive_segments(
        segments,
        max_gap=MERGE_MAX_GAP_SECONDS,
        max_duration=MERGE_MAX_DURATION_SECONDS,
        max_chars=MERGE_MAX_CHARS,
    )
    print(f"[merge] {pre_merge} → {len(segments)} segments "
          f"(gap≤{MERGE_MAX_GAP_SECONDS}s, dur≤{MERGE_MAX_DURATION_SECONDS}s, "
          f"chars≤{MERGE_MAX_CHARS})")

    total = time.time() - t_start
    speed = info.duration / max(t_asr, 0.01)
    print(f"[pipeline] total {total:.1f}s for {info.duration:.1f}s audio")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Transcript(
        source_path=source_path,
        source_type=source_type,  # type: ignore[arg-type]
        duration_seconds=info.duration,
        segments=segments,
        metadata={
            "language": info.language,
            "language_confidence": round(info.language_probability, 3),
            "asr_model": ASR_MODEL,
            "asr_wall_seconds": round(t_asr, 2),
            "total_wall_seconds": round(total, 2),
            "realtime_factor": round(speed, 2),
        },
    )


def _merge_speakers(
    segments: list[TranscriptSegment],
    turns: list[tuple[float, float, str]],
) -> list[TranscriptSegment]:
    """Assign each ASR segment the diarization speaker with maximum overlap."""
    return [
        seg.model_copy(update={
            "speaker": _best_speaker(seg.start_time, seg.end_time, turns)
        })
        for seg in segments
    ]


# Labels we treat as "no real speaker info" (case-insensitive).
_UNKNOWN_SPEAKER_LABELS = {"speaker ?", "unknown", ""}


def _normalize_speaker_labels(
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Rewrite raw speaker ids into clean, user-facing labels.

    - If diarization didn't run (every label is unknown / "Speaker ?"),
      blank the label so the UI just shows the text without a meaningless
      prefix and the LLM doesn't see a wall of identical tokens.
    - Otherwise renumber pyannote's "SPEAKER_00", "SPEAKER_01", … to
      "Speaker 1", "Speaker 2", … in order of first appearance. Keeps any
      genuinely unknown segments as "Speaker ?".
    """
    raw_labels = [s.speaker for s in segments]
    has_real = any(
        lbl.strip().lower() not in _UNKNOWN_SPEAKER_LABELS for lbl in raw_labels
    )
    if not has_real:
        return [s.model_copy(update={"speaker": ""}) for s in segments]

    rename: dict[str, str] = {}
    next_idx = 1
    for lbl in raw_labels:
        if lbl in rename:
            continue
        if lbl.strip().lower() in _UNKNOWN_SPEAKER_LABELS:
            rename[lbl] = "Speaker ?"
        else:
            rename[lbl] = f"Speaker {next_idx}"
            next_idx += 1
    return [s.model_copy(update={"speaker": rename[s.speaker]}) for s in segments]


def _merge_consecutive_segments(
    segments: list[TranscriptSegment],
    max_gap: float,
    max_duration: float,
    max_chars: int,
) -> list[TranscriptSegment]:
    """Coalesce adjacent segments from the same speaker into one statement.

    A new segment is appended to the previous one when ALL of the following hold:
      - same speaker label
      - gap between previous end and current start ≤ ``max_gap``
      - merged duration would not exceed ``max_duration``
      - merged text would not exceed ``max_chars``

    Otherwise, the current segment starts a new group. This produces clean
    paragraph-level rows for the UI while still breaking long monologues.
    """
    if not segments:
        return segments

    out: list[TranscriptSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        same_speaker = prev.speaker == seg.speaker
        small_gap = (seg.start_time - prev.end_time) <= max_gap
        within_duration = (seg.end_time - prev.start_time) <= max_duration
        within_chars = (len(prev.text) + 1 + len(seg.text)) <= max_chars

        if same_speaker and small_gap and within_duration and within_chars:
            out[-1] = prev.model_copy(update={
                "end_time": seg.end_time,
                "text": f"{prev.text.rstrip()} {seg.text.lstrip()}".strip(),
            })
        else:
            out.append(seg)
    return out


def _load_mono_waveform(
    audio_path: Path, target_sr: int = 16000
) -> tuple[torch.Tensor, int]:
    """Load audio as a (1, N) float32 mono tensor at ``target_sr``."""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_sr
        )
        sample_rate = target_sr
    return waveform, sample_rate


def _best_speaker(
    start: float, end: float, turns: list[tuple[float, float, str]]
) -> str:
    best_overlap, best_speaker = 0.0, "Speaker ?"
    for t_start, t_end, spk in turns:
        overlap = max(0.0, min(end, t_end) - max(start, t_start))
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, spk
    return best_speaker


# ─── Transcript parsing ────────────────────────────────────────────────────
def _parse_transcript_file(path: Path) -> Transcript:
    ext = path.suffix.lower()
    if ext == ".srt":
        segments = _parse_srt(path)
    elif ext == ".vtt":
        segments = _parse_vtt(path)
    else:
        segments = _parse_txt(path)

    duration = segments[-1].end_time if segments else 0.0
    return Transcript(
        source_path=str(path),
        source_type="transcript",
        duration_seconds=duration,
        segments=segments,
    )


def _parse_srt(path: Path) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    for sub in pysrt.open(str(path)):
        speaker, text = _split_speaker(sub.text)
        out.append(TranscriptSegment(
            speaker=speaker,
            start_time=sub.start.ordinal / 1000,
            end_time=sub.end.ordinal / 1000,
            text=text,
        ))
    return out


def _parse_vtt(path: Path) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    for caption in webvtt.read(str(path)):
        speaker, text = _split_speaker(caption.text)
        out.append(TranscriptSegment(
            speaker=speaker,
            start_time=_vtt_ts(caption.start),
            end_time=_vtt_ts(caption.end),
            text=text,
        ))
    return out


def _parse_txt(path: Path) -> list[TranscriptSegment]:
    """Plain text, one utterance per line, optional 'Speaker: text' prefix."""
    out: list[TranscriptSegment] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        speaker, text = _split_speaker(line)
        out.append(TranscriptSegment(
            speaker=speaker,
            start_time=float(i),
            end_time=float(i) + 1.0,
            text=text,
        ))
    return out


def _split_speaker(line: str) -> tuple[str, str]:
    m = _SPEAKER_LINE.match(line.strip())
    if not m:
        return "Speaker ?", line.strip()
    speaker = m.group("spk1") or m.group("spk2")
    return speaker.strip(), m.group("text").strip()


def _vtt_ts(ts: str) -> float:
    """'HH:MM:SS.mmm' -> seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)
