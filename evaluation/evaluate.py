"""End-to-end evaluation runner.

Expects:
  data/raw/<benchmark>/<meeting_id>.{mp4,wav,txt}
  data/gold/<benchmark>/<meeting_id>.json   (keys: summary, action_items)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import GOLD_DIR, OUTPUT_DIR, RAW_DIR
from evaluation.metrics import (
    bertscore_corpus,
    fuzzy_match,
    rouge_corpus,
    score_action_items,
)
from src.input_processing import load_meeting
from src.report import build_report
from src.schemas import ActionItem


def run(benchmark: str, prompt_version: str = "v1") -> pd.DataFrame:
    rows: list[dict] = []
    raw_dir = RAW_DIR / benchmark
    gold_dir = GOLD_DIR / benchmark

    for raw_path in sorted(raw_dir.iterdir()):
        gold_path = gold_dir / f"{raw_path.stem}.json"
        if not gold_path.exists():
            continue
        gold = json.loads(gold_path.read_text())

        transcript = load_meeting(raw_path)
        report = build_report(transcript, prompt_version=prompt_version)

        pred_summary = report.summary.overview
        rouge = rouge_corpus([pred_summary], [gold["summary"]])
        bert = bertscore_corpus([pred_summary], [gold["summary"]])

        gold_items = [ActionItem(**a) for a in gold["action_items"]]
        ai = score_action_items(report.action_items, gold_items, match_fn=fuzzy_match)

        rows.append({"meeting": raw_path.stem, **rouge, **bert, **ai})

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"eval_{benchmark}_{prompt_version}.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="e.g. ami, qmsum, mock")
    parser.add_argument("--prompt", default="v1")
    args = parser.parse_args()

    df = run(args.benchmark, args.prompt)
    print(df.describe())
