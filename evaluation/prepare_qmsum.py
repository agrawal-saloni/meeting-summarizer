"""Download a small QMSum subset and convert it into our benchmark layout.

QMSum (Zhong et al., 2021) is a public meeting-summarization benchmark
covering Academic (AMI/ICSI) and Committee (Welsh Parliament) meetings.
Each meeting JSON contains:
  - ``meeting_transcripts``: list of ``{speaker, content}`` turns
  - ``general_query_list``: a meeting-level "Summarize the meeting" Q/A pair
  - ``specific_query_list``: query-focused summaries (unused here)

We materialize each selected meeting as:
  data/raw/qmsum/<id>.txt        (one "Speaker: text" line per turn)
  data/gold/qmsum/<id>.json      ({"summary": <general answer>,
                                   "action_items": []})

Action items are not annotated in QMSum, so the gold list stays empty;
``evaluation.evaluate`` skips action-item metrics when gold has no items.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from config import GOLD_DIR, RAW_DIR

# Small, diverse default subset from QMSum's test split:
#   - ES2004a: AMI scenario meeting (Product design, short intro session)
#   - TS3004a: AMI scenario meeting (different series, similar length)
#   - Bmr014:  ICSI research meeting (longer, dense technical dialogue)
#   - education_4: Welsh Parliament committee (policy discussion)
DEFAULT_MEETINGS = [
    "ES2004a",
    "TS3004a",
    "Bmr014",
    "education_4",
]

_RAW_URL = (
    "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/"
    "data/ALL/test/{name}.json"
)


def _download(name: str) -> dict:
    url = _RAW_URL.format(name=name)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _to_transcript_text(meeting: dict) -> str:
    """Render QMSum turns as 'Speaker: utterance' lines, one per turn."""
    lines: list[str] = []
    for turn in meeting.get("meeting_transcripts", []):
        speaker = (turn.get("speaker") or "Speaker ?").strip()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def _gold_from_meeting(meeting: dict) -> dict:
    """Use the first general_query answer as the meeting-level gold summary."""
    gq = meeting.get("general_query_list") or []
    summary = (gq[0].get("answer") if gq else "") or ""
    return {"summary": summary.strip(), "action_items": []}


def prepare(
    meetings: list[str] = DEFAULT_MEETINGS, benchmark: str = "qmsum"
) -> list[str]:
    raw_dir = RAW_DIR / benchmark
    gold_dir = GOLD_DIR / benchmark
    raw_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[str] = []
    for name in meetings:
        raw_path = raw_dir / f"{name}.txt"
        gold_path = gold_dir / f"{name}.json"
        if raw_path.exists() and gold_path.exists():
            print(f"[qmsum] {name}: cached, skipping download")
            prepared.append(name)
            continue

        print(f"[qmsum] downloading {name}.json")
        meeting = _download(name)
        raw_path.write_text(_to_transcript_text(meeting), encoding="utf-8")
        gold_path.write_text(
            json.dumps(_gold_from_meeting(meeting), indent=2), encoding="utf-8"
        )
        prepared.append(name)
        n_turns = len(meeting.get("meeting_transcripts", []))
        print(
            f"[qmsum] {name}: {n_turns} turns → {raw_path.name}, "
            f"gold → {gold_path.name}"
        )
    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meetings",
        nargs="*",
        default=DEFAULT_MEETINGS,
        help="QMSum meeting ids to download (default: a 4-meeting subset)",
    )
    parser.add_argument("--benchmark", default="qmsum")
    args = parser.parse_args()

    names = prepare(args.meetings, args.benchmark)
    print(f"[qmsum] prepared {len(names)} meetings under benchmark={args.benchmark!r}")
