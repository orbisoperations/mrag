"""
Evaluation metrics.

Answer accuracy:
  - hotpot_em_f1(pred, gold) — port of the official HotpotQA token-level
    normalization (lowercase, strip articles + punctuation, whitespace
    tokenize, F1 over tokens; EM = normalized strings equal).
  - judge_correct — boolean accumulator for LLM-as-judge.

Retrieval:
  - retrieval_recall, retrieval_precision over chunk-index sets.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Set


# ---------------------------------------------------------------------------
# HotpotQA-style normalization (official)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        # If both are empty, EM is 1 / F1 is 1; otherwise 0. Per HotpotQA convention.
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)


def hotpot_em_f1(pred: str, gold: str) -> dict:
    return {"em": exact_match(pred, gold), "f1": f1_score(pred, gold)}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieval_recall(retrieved: Iterable[int], gold: Iterable[int]) -> float:
    gold_set: Set[int] = set(gold)
    if not gold_set:
        return 1.0  # No gold — vacuously satisfied.
    ret_set: Set[int] = set(retrieved)
    return len(gold_set & ret_set) / len(gold_set)


def retrieval_precision(retrieved: Iterable[int], gold: Iterable[int]) -> float:
    ret_set: Set[int] = set(retrieved)
    if not ret_set:
        return 0.0
    gold_set: Set[int] = set(gold)
    return len(gold_set & ret_set) / len(ret_set)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: List[dict]) -> dict:
    """Aggregate per-question records into mean metrics."""
    if not records:
        return {"n": 0}
    keys = ["em", "f1", "recall", "precision", "judge"]
    out: dict = {"n": len(records)}
    for k in keys:
        vals = [float(r[k]) for r in records if k in r and r[k] is not None]
        out[f"mean_{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out
