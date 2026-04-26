"""Streamlit UI — upload a meeting and view summary + action items."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import pandas as pd
import streamlit as st
from groq import APIStatusError, RateLimitError

from src.input_processing import (
    DiarizationAccessError,
    StreamEvent,
    stream_load_meeting,
)
from src.report import build_report, render_markdown, save_report
from src.schemas import Transcript, TranscriptSegment
from src.speaker_names import apply_speaker_names, infer_speaker_names
from src.video_names import identify_speakers_from_video


def _render_live_segments(
    container, segments: list[TranscriptSegment], *, speakers_ready: bool
) -> None:
    """Re-render the live-transcript placeholder.

    While diarization is still running, every segment carries the placeholder
    label ``"Speaker ?"``. We hide it so the captions read cleanly; the
    moment diarization back-fills real ids, ``speakers_ready`` flips to
    True and the labels appear in front of every line.
    """
    with container.container():
        if not segments:
            st.caption("Listening…")
            return
        st.caption(
            f"{len(segments)} segments transcribed"
            + ("" if speakers_ready else "  ·  speakers will appear once diarization finishes")
        )
        for seg in segments:
            label = seg.speaker.strip()
            show_label = (
                speakers_ready
                and label
                and label.lower() not in {"speaker ?", "unknown"}
            )
            prefix = f"**{label}** " if show_label else ""
            st.write(f"{prefix}`{seg.start_time:.1f}s`: {seg.text}")


def _consume_stream(
    events: Iterator[StreamEvent], placeholder
) -> Transcript:
    """Drain a ``stream_load_meeting`` iterator, updating the live placeholder.

    Returns the final ``Transcript`` from the terminal ``"done"`` event.
    """
    segments: list[TranscriptSegment] = []
    speakers_ready = False
    final: Transcript | None = None

    for event in events:
        if event.type == "segment" and event.segment is not None:
            segments.append(event.segment)
        elif event.type == "speakers":
            segments = list(event.segments)
            speakers_ready = True
        elif event.type == "done" and event.transcript is not None:
            final = event.transcript
            break
        _render_live_segments(
            placeholder, segments, speakers_ready=speakers_ready
        )

    if final is None:
        raise RuntimeError("Stream ended without a 'done' event")
    return final


def main() -> None:
    st.set_page_config(page_title="Meeting Summarizer", layout="wide")
    st.title("Meeting Summarizer & Action Item Extractor")

    with st.sidebar:
        st.header("Settings")
        prompt_version = st.selectbox("Prompt version", ["v1", "v2"], index=0)
        diarize = st.checkbox("Run speaker diarization", value=True)
        st.caption("Requires HF_TOKEN for pyannote.")
        detect_names = st.checkbox(
            "Try to detect speaker names",
            value=True,
            help=(
                "After diarization, ask the LLM to map Speaker 1/2/… to real "
                "names by scanning the transcript for self-introductions and "
                "direct address. Only confident matches are applied."
            ),
        )

    uploaded = st.file_uploader(
        "Upload meeting (video / audio / transcript)",
        type=["mp4", "mkv", "webm", "mov", "mp3", "wav", "m4a", "flac", "ogg",
              "txt", "srt", "vtt"],
    )
    if not uploaded:
        st.info("Waiting for a file…")
        return

    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    # Live-ASR section: Whisper segments stream into this placeholder as
    # they're produced. Diarization runs in parallel; speaker labels
    # back-fill across all rendered rows the moment its turns are ready.
    st.subheader("Live transcript")
    live_status = st.status(
        "Transcribing… (diarization runs in parallel)" if diarize
        else "Transcribing…",
        expanded=True,
    )
    with live_status:
        live_placeholder = st.empty()
        try:
            transcript = _consume_stream(
                stream_load_meeting(tmp_path, diarize=diarize),
                live_placeholder,
            )
        except DiarizationAccessError as e:
            st.error("Speaker diarization unavailable")
            st.code(str(e))
            st.info("Retrying without diarization…")
            live_placeholder = st.empty()
            transcript = _consume_stream(
                stream_load_meeting(tmp_path, diarize=False),
                live_placeholder,
            )
    live_status.update(
        label=f"Transcription complete · {len(transcript.segments)} segments",
        state="complete",
        expanded=False,
    )

    if detect_names and any(s.speaker for s in transcript.segments):
        roster: list[str] = []
        # Stage 1 (video only): VLM reads the active-speaker tile for each
        # diarized speaker across a few mid-turn frames. When it's
        # confident (majority vote), we apply the mapping immediately —
        # this is strictly stronger evidence than anything we can squeeze
        # out of the transcript alone.
        if transcript.source_type == "video":
            with st.spinner("Reading names from video (VLM)…"):
                try:
                    evidence = identify_speakers_from_video(
                        transcript.source_path, transcript
                    )
                except Exception as e:  # noqa: BLE001
                    evidence = None
                    st.warning(f"On-screen name extraction failed: {e}")
            if evidence is not None:
                roster = evidence.roster
                if evidence.mapping:
                    transcript = apply_speaker_names(transcript, evidence.mapping)
                    st.caption(
                        "From video: "
                        + ", ".join(
                            f"{k} → **{v}**" for k, v in evidence.mapping.items()
                        )
                    )
                if roster:
                    st.caption("Names visible on video: " + ", ".join(roster))

        # Stage 2: transcript-side resolver fills in any remaining
        # "Speaker N" labels using self-intros / vocatives (+ the VLM
        # roster as a vocabulary hint). Skip cleanly when the VLM
        # already named everyone.
        unresolved = any(
            s.speaker.startswith("Speaker ") for s in transcript.segments
        )
        if unresolved:
            with st.spinner("Detecting speaker names…"):
                try:
                    mapping = infer_speaker_names(transcript, roster=roster)
                except Exception as e:  # noqa: BLE001
                    mapping = {}
                    st.warning(f"Speaker-name detection failed: {e}")
            if mapping:
                transcript = apply_speaker_names(transcript, mapping)
                st.caption(
                    "From transcript: "
                    + ", ".join(f"{k} → **{v}**" for k, v in mapping.items())
                )

    report = None
    report_error: str | None = None
    with st.spinner("Summarizing + extracting action items…"):
        try:
            report = build_report(transcript, prompt_version=prompt_version)
        except RateLimitError as e:
            report_error = (
                "Groq rate limit hit. Wait a minute and retry, or set "
                "`LLM_FALLBACK_MODEL` in `.env` to switch to a smaller model."
                f"\n\nDetails: {e}"
            )
        except APIStatusError as e:
            status = getattr(e, "status_code", "?")
            report_error = (
                f"Groq API error (HTTP {status}): {e}. "
                "Check your `GROQ_API_KEY` and selected `LLM_MODEL`."
            )
        except Exception as e:  # noqa: BLE001
            report_error = f"Report generation failed: {type(e).__name__}: {e}"

    tab_summary, tab_actions, tab_transcript = st.tabs(
        ["Summary", "Action Items", "Transcript"]
    )

    with tab_summary:
        if report is not None:
            st.markdown(render_markdown(report))
        else:
            st.error(report_error or "Report generation failed.")
            st.info("Transcript is still available in the **Transcript** tab.")

    with tab_actions:
        if report is not None and report.action_items:
            df = pd.DataFrame([a.model_dump() for a in report.action_items])
            st.dataframe(df, use_container_width=True)
        elif report is not None:
            st.write("No action items detected.")
        else:
            st.error(report_error or "Report generation failed.")

    with tab_transcript:
        for seg in transcript.segments:
            prefix = f"**{seg.speaker}** " if seg.speaker else ""
            st.write(f"{prefix}`{seg.start_time:.1f}s`: {seg.text}")

    if report is not None:
        paths = save_report(report, stem=tmp_path.stem)
        st.download_button("Download .md", paths["md"].read_bytes(),
                           file_name=paths["md"].name)
        st.download_button("Download .docx", paths["docx"].read_bytes(),
                           file_name=paths["docx"].name)


if __name__ == "__main__":
    main()
