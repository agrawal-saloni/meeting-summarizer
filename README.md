# Meeting Summarizer & Action Item Extractor

End-to-end pipeline that turns meeting recordings (video / audio / transcript)
into structured summaries with extracted action items.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys
```

You also need `ffmpeg` on the system PATH for video audio extraction:

```bash
# macOS
brew install ffmpeg
# Ubuntu
sudo apt-get install ffmpeg
```

## Run

```bash
# Batch mode via Streamlit UI
streamlit run app.py

# Evaluate on a benchmark
python -m evaluation.evaluate --benchmark ami --prompt v1
```

## Repository layout

```
meeting-summarizer/
├── app.py                          # Streamlit entrypoint
├── config.py                       # Env, paths, model defaults
├── requirements.txt
├── src/
│   ├── schemas.py                  # Pydantic models (single source of truth)
│   ├── input_processing.py         # video/audio/transcript → Transcript
│   ├── llm_client.py               # OpenAI + Anthropic unified wrapper
│   ├── analysis.py                 # chunking + summarize + extract_action_items
│   ├── report.py                   # build_report + markdown/docx rendering
│   └── realtime.py                 # Simulated streaming mode
├── prompts/                        # Versioned prompt templates
├── evaluation/
│   ├── metrics.py                  # ROUGE + BERTScore + matching + P/R/F1
│   └── evaluate.py                 # Benchmark runner CLI
├── data/                           # raw / processed / gold / outputs
├── notebooks/                      # Prompt experimentation
└── tests/
```

## Data flow

```
  file ──► input_processing.load_meeting() ──► Transcript
                                                   │
                                                   ▼
                                       analysis.summarize() +
                                 analysis.extract_action_items()
                                                   │
                                                   ▼
                                       report.build_report()
                                                   │
                                                   ▼
                                             MeetingReport
                                                   │
                                                   ▼
                                    Streamlit UI / .md / .docx
```
