"""End-to-end evaluation runner.

Expects:
  data/raw/<benchmark>/<meeting_id>.{mp4,wav,txt,srt,vtt}
  data/gold/<benchmark>/<meeting_id>.json
      (keys: ``summary`` [str], ``action_items`` [list, optional])

If ``action_items`` is missing or empty, action-item metrics are skipped
for that meeting (and omitted from aggregates) — useful for public
benchmarks like QMSum that only annotate summaries.

Can run multiple prompt versions in a single pass and emits a
side-by-side CSV + markdown so prompt variants can be compared directly.
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

SUMMARY_METRICS = ["rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1"]
ACTION_METRICS = ["precision", "recall", "f1"]


def _score_meeting(
    raw_path: Path, gold: dict, prompt_version: str
) -> dict[str, float | int | str]:
    """Run the pipeline once for ``prompt_version`` and compute all metrics."""
    transcript = load_meeting(raw_path)
    report = build_report(transcript, prompt_version=prompt_version)

    pred_summary = report.summary.overview
    gold_summary = gold.get("summary", "") or ""

    row: dict[str, float | int | str] = {"pred_summary": pred_summary}
    row.update(rouge_corpus([pred_summary], [gold_summary]))
    row.update(bertscore_corpus([pred_summary], [gold_summary]))

    raw_gold_items = gold.get("action_items") or []
    if raw_gold_items:
        gold_items = [ActionItem(**a) for a in raw_gold_items]
        row.update(
            score_action_items(report.action_items, gold_items, match_fn=fuzzy_match)
        )
        row["has_action_items"] = 1
    else:
        for k in ACTION_METRICS + ["tp", "fp", "fn"]:
            row[k] = None
        row["has_action_items"] = 0
    row["n_pred_action_items"] = len(report.action_items)
    return row


def run(benchmark: str, prompt_versions: list[str]) -> dict[str, pd.DataFrame]:
    """Run every prompt version over every meeting in ``benchmark``.

    Returns a dict of ``prompt_version → per-meeting DataFrame``. Also
    writes per-prompt CSVs and a combined side-by-side comparison file.
    """
    raw_dir = RAW_DIR / benchmark
    gold_dir = GOLD_DIR / benchmark
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw dir: {raw_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_prompt: dict[str, list[dict]] = {v: [] for v in prompt_versions}

    for raw_path in sorted(raw_dir.iterdir()):
        if raw_path.name.startswith("."):
            continue
        gold_path = gold_dir / f"{raw_path.stem}.json"
        if not gold_path.exists():
            print(f"[eval] skipping {raw_path.name} (no gold)")
            continue
        gold = json.loads(gold_path.read_text())

        for v in prompt_versions:
            print(f"[eval] {raw_path.stem} · prompt={v}")
            row = _score_meeting(raw_path, gold, v)
            row = {"meeting": raw_path.stem, **row}
            per_prompt[v].append(row)

    dfs: dict[str, pd.DataFrame] = {}
    for v, rows in per_prompt.items():
        df = pd.DataFrame(rows)
        out = OUTPUT_DIR / f"eval_{benchmark}_{v}.csv"
        df.to_csv(out, index=False)
        print(f"[eval] wrote {out}")
        dfs[v] = df

    if len(prompt_versions) > 1:
        _write_comparison(benchmark, dfs)

    return dfs


# ─── Comparison reporting ──────────────────────────────────────────────────
def _aggregate(df: pd.DataFrame) -> dict[str, float]:
    agg: dict[str, float] = {}
    for m in SUMMARY_METRICS:
        if m in df.columns and not df[m].dropna().empty:
            agg[m] = float(df[m].mean())
    ai_df = df[df.get("has_action_items", 0) == 1] if "has_action_items" in df else df
    for m in ACTION_METRICS:
        if m in ai_df.columns and not ai_df[m].dropna().empty:
            agg[m] = float(ai_df[m].mean())
    return agg


def _write_comparison(benchmark: str, dfs: dict[str, pd.DataFrame]) -> None:
    """Write side-by-side comparison CSV + a human-readable markdown table."""
    metric_cols = SUMMARY_METRICS + ACTION_METRICS
    merged: pd.DataFrame | None = None
    for v, df in dfs.items():
        cols = ["meeting"] + [m for m in metric_cols if m in df.columns]
        slim = df[cols].copy()
        slim.columns = ["meeting"] + [
            f"{m}_{v}" for m in slim.columns if m != "meeting"
        ]
        merged = slim if merged is None else merged.merge(slim, on="meeting", how="outer")
    assert merged is not None
    csv_path = OUTPUT_DIR / f"eval_{benchmark}_compare.csv"
    merged.to_csv(csv_path, index=False)
    print(f"[eval] wrote {csv_path}")

    versions = list(dfs.keys())
    lines = [f"# Evaluation comparison — benchmark `{benchmark}`", ""]
    lines.append("## Aggregate (mean over meetings)")
    lines.append("")
    header = ["metric"] + versions
    if len(versions) == 2:
        header.append(f"Δ ({versions[1]} − {versions[0]})")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    aggs = {v: _aggregate(df) for v, df in dfs.items()}
    for m in SUMMARY_METRICS + ACTION_METRICS:
        cells = [m]
        vals = []
        for v in versions:
            val = aggs[v].get(m)
            vals.append(val)
            cells.append("—" if val is None else f"{val:.4f}")
        if len(versions) == 2 and None not in vals:
            cells.append(f"{vals[1] - vals[0]:+.4f}")
        elif len(versions) == 2:
            cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Per-meeting (summary quality)", ""]
    lines.append("| meeting | metric | " + " | ".join(versions) + " |")
    lines.append("|" + "|".join(["---"] * (2 + len(versions))) + "|")
    first = next(iter(dfs.values()))
    for meeting in first["meeting"]:
        for m in SUMMARY_METRICS:
            row = [meeting, m]
            for v in versions:
                df = dfs[v]
                if m not in df.columns:
                    row.append("—")
                    continue
                hit = df.loc[df["meeting"] == meeting, m]
                row.append("—" if hit.empty or pd.isna(hit.iloc[0]) else f"{hit.iloc[0]:.4f}")
            lines.append("| " + " | ".join(row) + " |")

    md_path = OUTPUT_DIR / f"eval_{benchmark}_compare.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {md_path}")


# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="e.g. qmsum, ami, mock")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["v1", "v2"],
        help="One or more prompt versions to evaluate (default: v1 v2)",
    )
    args = parser.parse_args()

    dfs = run(args.benchmark, args.prompts)
    for v, df in dfs.items():
        print(f"\n=== prompt={v} ===")
        print(df.drop(columns=["pred_summary"], errors="ignore").describe())
