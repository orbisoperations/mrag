"""
Benchmark loaders for the mRAG experiment harness.

Every loader returns a list of `Question` dataclasses with a consistent schema:
  - id:               stable string id
  - query:            the question text
  - corpus:           list[str] of candidate passages (mRAG retrieves from these)
  - gold_answer:      the reference answer string
  - gold_support_ids: indices into `corpus` for the gold support paragraphs
  - meta:             dict of extras (question_type, level, ...)

HotpotQA + 2WikiMH supply a per-question distractor pool (10 passages each).
MultiHop-RAG supplies a shared corpus across all queries; we pass the full
corpus as `corpus` and let the experiment runner detect the shared-index case
to embed/PCA once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from datasets import load_dataset


@dataclass
class Question:
    id: str
    query: str
    corpus: List[str]
    gold_answer: str
    gold_support_ids: List[int]
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HotpotQA — distractor config
# ---------------------------------------------------------------------------

def _paragraph_from_sentences(sentences: Iterable[str]) -> str:
    return " ".join(s.strip() for s in sentences).strip()


def load_hotpotqa(n: int = 500, split: str = "validation", seed: int = 42) -> List[Question]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=False)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:n]

    out: List[Question] = []
    for i in idx:
        ex = ds[i]
        titles = ex["context"]["title"]
        sents_per_para = ex["context"]["sentences"]
        # One paragraph per (title, sentences) pair. Prepend the title so embeddings see entity.
        corpus = [f"{t}. {_paragraph_from_sentences(ss)}" for t, ss in zip(titles, sents_per_para)]
        # Gold support = titles in supporting_facts.
        sup_titles = set(ex["supporting_facts"]["title"])
        gold_ids = [j for j, t in enumerate(titles) if t in sup_titles]
        out.append(Question(
            id=str(ex["id"]),
            query=ex["question"],
            corpus=corpus,
            gold_answer=ex["answer"],
            gold_support_ids=gold_ids,
            meta={"type": ex["type"], "level": ex["level"]},
        ))
    return out


# ---------------------------------------------------------------------------
# MultiHop-RAG — shared corpus, query-level evidence linking
# ---------------------------------------------------------------------------

def load_multihop_rag(n: int = 500, seed: int = 42) -> List[Question]:
    queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")

    # Build the shared corpus once: each entry is "TITLE. BODY".
    corpus: List[str] = []
    title_to_id: dict[str, int] = {}
    url_to_id: dict[str, int] = {}
    for j, art in enumerate(corpus_ds):
        text = (art["title"] or "") + ". " + (art["body"] or "")
        corpus.append(text)
        if art["title"]:
            title_to_id[art["title"]] = j
        if art["url"]:
            url_to_id[art["url"]] = j

    rng = random.Random(seed)
    sample_idx = list(range(len(queries)))
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[:n]

    out: List[Question] = []
    for i in sample_idx:
        ex = queries[i]
        gold_ids: List[int] = []
        for ev in ex.get("evidence_list", []):
            # Evidence references the source article by URL or title.
            j = url_to_id.get(ev.get("url")) if ev.get("url") else None
            if j is None and ev.get("title"):
                j = title_to_id.get(ev["title"])
            if j is not None and j not in gold_ids:
                gold_ids.append(j)
        out.append(Question(
            id=f"mhrag-{i}",
            query=ex["query"],
            corpus=corpus,  # shared — runner will detect identity for caching.
            gold_answer=str(ex["answer"]),
            gold_support_ids=gold_ids,
            meta={"question_type": ex.get("question_type", "")},
        ))
    return out


# ---------------------------------------------------------------------------
# 2WikiMultiHopQA — per-question distractor pool
# ---------------------------------------------------------------------------

def load_2wiki(n: int = 500, split: str = "validation", seed: int = 42) -> List[Question]:
    ds = load_dataset("voidful/2WikiMultihopQA", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:n]

    out: List[Question] = []
    for i in idx:
        ex = ds[i]
        # context = list of [title, list[sentences]]
        titles = [c[0] for c in ex["context"]]
        corpus = [f"{c[0]}. {_paragraph_from_sentences(c[1])}" for c in ex["context"]]
        sup_titles = {sf[0] for sf in ex["supporting_facts"]}
        gold_ids = [j for j, t in enumerate(titles) if t in sup_titles]
        out.append(Question(
            id=str(ex["_id"]),
            query=ex["question"],
            corpus=corpus,
            gold_answer=str(ex["answer"]),
            gold_support_ids=gold_ids,
            meta={"type": ex["type"]},
        ))
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "hotpotqa": load_hotpotqa,
    "multihoprag": load_multihop_rag,
    "2wiki": load_2wiki,
}


def load_benchmark(name: str, n: int = 500, seed: int = 42) -> List[Question]:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark '{name}'. options: {list(BENCHMARKS)}")
    return BENCHMARKS[name](n=n, seed=seed)
