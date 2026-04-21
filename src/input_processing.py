"""Input processing — video/audio/transcript → normalized Transcript.

This module owns every step before the LLM sees a meeting:
  - detect input type
  - extract audio from video via ffmpeg
  - transcribe audio with faster-whisper
  - overlay pyannote speaker diarization
  - parse pre-existing transcripts (txt/srt/vtt)
"""

from __future__ import annotations

import re
from pathlib import Path

import ffmpeg
import pysrt
import webvtt
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config import (
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_MODEL,
    DIARIZATION_MODEL,
    HF_TOKEN,
    PROCESSED_DIR,
)
from src.schemas import Transcript, TranscriptSegment

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
    (
        ffmpeg
        .input(str(video_path))
        .output(str(out_path), ac=1, ar=sample_rate, vn=None,
                acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path


# ─── ASR + Diarization ─────────────────────────────────────────────────────
def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(ASR_MODEL, device=ASR_DEVICE,
                                      compute_type=ASR_COMPUTE_TYPE)
    return _whisper_model


def _get_diarization_pipeline() -> Pipeline:
    global _diarize_pipeline
    if _diarize_pipeline is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set — required for pyannote.")
        _diarize_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, use_auth_token=HF_TOKEN
        )
    return _diarize_pipeline


def _transcribe_audio(
    audio_path: Path,
    diarize: bool,
    source_path: str,
    source_type: str,
) -> Transcript:
    """Whisper ASR, with optional pyannote speaker overlay."""
    model = _get_whisper()
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5)

    segments = [
        TranscriptSegment(
            speaker="Speaker ?",
            start_time=seg.start,
            end_time=seg.end,
            text=seg.text.strip(),
        )
        for seg in segments_iter
    ]

    if diarize:
        segments = _attach_speakers(audio_path, segments)

    return Transcript(
        source_path=source_path,
        source_type=source_type,  # type: ignore[arg-type]
        duration_seconds=info.duration,
        segments=segments,
        metadata={"language": info.language, "asr_model": ASR_MODEL},
    )


def _attach_speakers(
    audio_path: Path,
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Run diarization, assign each segment the speaker with max overlap."""
    diarization = _get_diarization_pipeline()(str(audio_path))
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return [
        seg.model_copy(update={
            "speaker": _best_speaker(seg.start_time, seg.end_time, turns)
        })
        for seg in segments
    ]


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
