"""All evaluation metrics in one place.

- ROUGE-1/2/L   (for summary comparison vs gold)
- BERTScore      (semantic similarity for summaries)
- Fuzzy match    (rapidfuzz token-set ratio for action items)
- Semantic match (sentence-transformers for action items)
- P / R / F1     (greedy one-to-one matching over action items)
"""

from __future__ import annotations

from typing import Callable

from bert_score import BERTScorer
from rapidfuzz import fuzz
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from src.schemas import ActionItem

FUZZY_THRESHOLD = 0.70
SEMANTIC_THRESHOLD = 0.65

_rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)
_encoder: SentenceTransformer | None = None
_bert_scorer: BERTScorer | None = None


# ─── ROUGE ─────────────────────────────────────────────────────────────────
def rouge_score(prediction: str, reference: str) -> dict[str, float]:
    s = _rouge.score(reference, prediction)
    return {
        "rouge1_f": s["rouge1"].fmeasure,
        "rouge2_f": s["rouge2"].fmeasure,
        "rougeL_f": s["rougeL"].fmeasure,
    }


def rouge_corpus(predictions: list[str], references: list[str]) -> dict[str, float]:
    assert len(predictions) == len(references)
    agg = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    for p, r in zip(predictions, references):
        for k, v in rouge_score(p, r).items():
            agg[k] += v
    n = len(predictions) or 1
    return {k: v / n for k, v in agg.items()}


# ─── BERTScore ─────────────────────────────────────────────────────────────
def _get_bert_scorer(lang: str = "en") -> BERTScorer:
    """Cache the underlying RoBERTa model across calls.

    `bert_score.score` reloads the model every call, which both spams the
    "Some weights of RobertaModel were not initialized…" warning and adds
    ~5–10s of latency per meeting. Reusing one BERTScorer avoids both.
    """
    global _bert_scorer
    if _bert_scorer is None:
        _bert_scorer = BERTScorer(lang=lang, rescale_with_baseline=False)
    return _bert_scorer


def bertscore_corpus(
    predictions: list[str], references: list[str], lang: str = "en"
) -> dict[str, float]:
    P, R, F1 = _get_bert_scorer(lang).score(predictions, references)
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


# ─── Action item matching ──────────────────────────────────────────────────
def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def fuzzy_match(pred: ActionItem, gold: ActionItem) -> bool:
    """Same owner AND token-set ratio above threshold."""
    if pred.owner.lower().strip() != gold.owner.lower().strip():
        return False
    return fuzz.token_set_ratio(pred.task, gold.task) / 100.0 >= FUZZY_THRESHOLD


def semantic_match(pred: ActionItem, gold: ActionItem) -> bool:
    """Same owner AND semantic similarity above threshold."""
    if pred.owner.lower().strip() != gold.owner.lower().strip():
        return False
    model = _get_encoder()
    e1, e2 = model.encode([pred.task, gold.task], convert_to_tensor=True)
    return float(util.cos_sim(e1, e2)) >= SEMANTIC_THRESHOLD


# ─── Action item P/R/F1 ────────────────────────────────────────────────────
def score_action_items(
    predicted: list[ActionItem],
    gold: list[ActionItem],
    match_fn: Callable[[ActionItem, ActionItem], bool] = fuzzy_match,
) -> dict[str, float]:
    """Greedy one-to-one matching → precision / recall / F1."""
    matched_gold: set[int] = set()
    tp = 0
    for p in predicted:
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if match_fn(p, g):
                tp += 1
                matched_gold.add(i)
                break
    fp = len(predicted) - tp
    fn = len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }
