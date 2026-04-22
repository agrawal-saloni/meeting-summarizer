"""Streamlit UI — upload a meeting and view summary + action items."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

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
    with st.spinner("Summarizing + extracting action items…"):
        report = build_report(transcript, prompt_version=prompt_version)

    tab_summary, tab_actions, tab_transcript = st.tabs(
        ["Summary", "Action Items", "Transcript"]
    )

    with tab_summary:
        st.markdown(render_markdown(report))

    with tab_actions:
        if report.action_items:
            df = pd.DataFrame([a.model_dump() for a in report.action_items])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No action items detected.")

    with tab_transcript:
        for seg in transcript.segments:
            st.write(f"**{seg.speaker}** `{seg.start_time:.1f}s`: {seg.text}")

    # Save and offer download
    paths = save_report(report, stem=tmp_path.stem)
    st.download_button("Download .md", paths["md"].read_bytes(),
                       file_name=paths["md"].name)
    st.download_button("Download .docx", paths["docx"].read_bytes(),
                       file_name=paths["docx"].name)


if __name__ == "__main__":
    main()
