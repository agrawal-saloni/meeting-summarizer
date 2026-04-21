"""Shared data models — the single source of truth for the transcript schema
and LLM output contracts. Every module in src/ consumes or produces these."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single utterance in the normalized transcript."""

    speaker: str = Field(..., description="Speaker label, e.g. 'Speaker A' or 'Alice'")
    start_time: float = Field(..., description="Start time in seconds from meeting start")
    end_time: float = Field(..., description="End time in seconds from meeting start")
    text: str = Field(..., description="Transcribed utterance text")


class Transcript(BaseModel):
    """Normalized representation of a meeting — produced by input_processing."""

    source_path: str
    source_type: Literal["video", "audio", "transcript"]
    duration_seconds: float
    segments: list[TranscriptSegment]
    metadata: dict = Field(default_factory=dict)


class ActionItem(BaseModel):
    """Structured action item extracted from the transcript."""

    owner: str = Field(..., description="Person responsible, or 'unassigned'")
    task: str = Field(..., description="Concise description of the task")
    due_date: str | None = Field(None, description="ISO date or free text if ambiguous")
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_quote: str = Field(..., description="Verbatim line from transcript")


class MeetingSummary(BaseModel):
    """Structured summary output from the LLM."""

    key_decisions: list[str]
    discussion_points: list[str]
    open_questions: list[str]
    overview: str = Field(..., description="2-3 sentence high-level summary")


class MeetingReport(BaseModel):
    """Final report: everything we want to render in the UI and export."""

    transcript: Transcript
    summary: MeetingSummary
    action_items: list[ActionItem]
    generated_at: str  # ISO timestamp
    prompt_version: str
    llm_model: str
