"""Streamlit UI — upload a meeting and view summary + action items."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import APIStatusError, RateLimitError

from src.analysis import apply_speaker_names, infer_speaker_names
from src.input_processing import DiarizationAccessError, load_meeting
from src.report import build_report, render_markdown, save_report


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

    with st.spinner("Transcribing…"):
        try:
            transcript = load_meeting(tmp_path, diarize=diarize)
        except DiarizationAccessError as e:
            st.error("Speaker diarization unavailable")
            st.code(str(e))
            st.info("Retrying without diarization…")
            transcript = load_meeting(tmp_path, diarize=False)

    if detect_names and any(s.speaker for s in transcript.segments):
        with st.spinner("Detecting speaker names…"):
            try:
                mapping = infer_speaker_names(transcript)
            except Exception as e:  # noqa: BLE001
                mapping = {}
                st.warning(f"Speaker-name detection failed: {e}")
        if mapping:
            transcript = apply_speaker_names(transcript, mapping)
            st.caption(
                "Identified: "
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
