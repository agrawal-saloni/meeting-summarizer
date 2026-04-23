"""Simulated real-time processing — replay a transcript in time-based windows.

Yields partial MeetingReports as the meeting progresses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

from config import LLM_MODEL, REALTIME_WINDOW_SECONDS
from src.analysis import analyze
from src.schemas import MeetingReport, Transcript


def stream_report(
    transcript: Transcript,
    window_seconds: int = REALTIME_WINDOW_SECONDS,
    prompt_version: str = "v1",
) -> Iterator[MeetingReport]:
    """Yield a MeetingReport after each window of the meeting is 'heard'."""
    elapsed = 0.0
    end = transcript.duration_seconds

    while elapsed < end:
        elapsed += window_seconds
        partial_segments = [
            s for s in transcript.segments if s.end_time <= elapsed
        ]
        if not partial_segments:
            continue
        partial = transcript.model_copy(update={"segments": partial_segments})

        summary, action_items = analyze(partial, prompt_version=prompt_version)
        yield MeetingReport(
            transcript=partial,
            summary=summary,
            action_items=action_items,
            generated_at=datetime.utcnow().isoformat(),
            prompt_version=prompt_version,
            llm_model=LLM_MODEL,
        )
