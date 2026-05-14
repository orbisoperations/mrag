"""
Experiment runner — runs (dataset × embedder × metric × strategy × d) cells
and appends per-question records to results.parquet.

Heavy work is cached:
  - embeddings (per text/model/dim/task)
  - PCA fits (per corpus/embedder/d)
  - ε thresholds (per corpus/embedder/d/metric)
  - generated answers (per prompt)
  - judge verdicts (per query/pred/gold)

So a re-run with a new metric never re-embeds; a re-run of the same cell is
~free.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmarks import Question, load_benchmark
from cache import PCACache, ThresholdCache, corpus_hash
from distances import build_z_metric, cosine_distance
from embeddings import Embedder, build_embedder
from evaluation import f1_score, exact_match, retrieval_precision, retrieval_recall
from llm import build_flat_prompt, build_mrag_prompt, build_no_context_prompt, generate_answer, llm_judge
from manifolds import fit_pca
from traversals import semantic_decay_traversal, continuous_random_walk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellConfig:
    dataset: str
    embedder: str
    metric: str            # one of distances.METRICS, or "naive" / "no_context"
    strategy: str          # "decay" | "ppr" | "naive" | "no_context"
    d: int = 32
    n: int = 500
    tau: float = 0.70      # X-space cosine cutoff for the semantic-drift bound
    gamma: float = 1.3     # semantic decay multiplier (>1 expands the cap per hop)
    max_hops: int = 2
    top_k_naive: int = 5
    top_k_ppr: int = 5
    min_anchors: int = 3   # top-K cosine anchors guaranteed in S^(0)
    gen_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash"
    seed: int = 42

    @property
    def label(self) -> str:
        return f"{self.dataset}|{self.embedder}|{self.metric}|{self.strategy}|d{self.d}"


# Modules used as caches.
_PCA = PCACache()
_THRESH = ThresholdCache()


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_epsilon(
    Z: np.ndarray,
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    quantile: float = 0.10,
    sample_size: int = 50,
    seed: int = 0,
) -> float:
    """Set ε to the `quantile`-th of pairwise Z-distances on a random sample.

    This makes ε comparable across metrics with different absolute scales.
    """
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    if n <= 1:
        return 1.0
    sample = Z if n <= sample_size else Z[rng.choice(n, sample_size, replace=False)]
    m = sample.shape[0]
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            pairs.append(dist_fn(sample[i], sample[j]))
    if not pairs:
        return 1.0
    return float(np.quantile(np.array(pairs), quantile))


# ---------------------------------------------------------------------------
# Per-question retrieval given a cell config
# ---------------------------------------------------------------------------

@dataclass
class RetrievalOutput:
    primary: List[int]     # S^(0) anchors in X
    structural: List[int]  # union of S^(t>=1)
    all_ids: List[int]     # primary + structural
    hops: dict             # {hop_t: list[int]} or {-1: [list of top-k]}


def _topk_cosine(q_vec: np.ndarray, X: np.ndarray, k: int) -> List[int]:
    sims = X @ q_vec
    idx = np.argsort(-sims)[:k]
    return idx.tolist()


def _select_anchors(
    q_vec: np.ndarray,
    X: np.ndarray,
    tau: float,
    min_anchors: int = 3,
) -> List[int]:
    """Anchor selection for S^(0).

    The paper specifies a strict cosine cutoff τ, but real corpora may produce
    zero matches under any fixed τ. We take whichever is larger of (top-K
    cosine, threshold-filtered set), guaranteeing the traversal always has
    something to start from.
    """
    dists = np.array([cosine_distance(q_vec, X[i]) for i in range(len(X))])
    threshold_idx = [i for i, d in enumerate(dists) if d <= tau]
    topk_idx = np.argsort(dists)[:min_anchors].tolist()
    return sorted(set(threshold_idx) | set(topk_idx))


def _build_ppr_transition(
    Z: np.ndarray,
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    beta: float = 1.0,
) -> np.ndarray:
    """Precompute the PPR transition matrix P from a Z-space metric.

    This is the O(N²) hot path. It only depends on (Z, metric), so we factor
    it out so it can be computed once per (corpus, embedder, metric) and
    reused across every query in a shared-corpus dataset.
    """
    N = len(Z)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            v = np.exp(-beta * dist_fn(Z[i], Z[j]))
            A[i, j] = v
            A[j, i] = v
    row_sums = A.sum(axis=1, keepdims=True)
    return np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums != 0)


def _ppr_topk(
    q_vec: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    tau: float,
    k: int,
    alpha: float = 0.85,
    beta: float = 1.0,
    iterations: int = 10,
    min_anchors: int = 3,
    P: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int]]:
    """Returns (primary anchor ids, structural top-k by PPR mass).

    If a precomputed transition matrix P is supplied, skip the O(N²) build.
    """
    N = len(X)
    anchors = _select_anchors(q_vec, X, tau, min_anchors=min_anchors)
    v = np.zeros(N)
    for i in anchors:
        v[i] = 1.0
    if v.sum() == 0:
        v[:] = 1.0 / N
    else:
        v /= v.sum()

    if P is None:
        P = _build_ppr_transition(Z, dist_fn, beta=beta)

    pi = v.copy()
    for _ in range(iterations):
        pi = (1 - alpha) * v + alpha * (P.T @ pi)

    order = np.argsort(-pi).tolist()
    top = order[:k]
    primary = [i for i in top if i in set(anchors)]
    structural = [i for i in top if i not in set(anchors)]
    return primary or anchors[:k], structural


def retrieve(
    cfg: CellConfig,
    question: Question,
    X: np.ndarray,
    q_vec: np.ndarray,
    Z: Optional[np.ndarray],
    z_metric_fn: Optional[Callable],
    epsilon: Optional[float],
    ppr_P: Optional[np.ndarray] = None,
) -> RetrievalOutput:
    if cfg.strategy == "no_context":
        return RetrievalOutput(primary=[], structural=[], all_ids=[], hops={})
    if cfg.strategy == "naive":
        top = _topk_cosine(q_vec, X, cfg.top_k_naive)
        return RetrievalOutput(primary=top, structural=[], all_ids=top, hops={0: top})
    if cfg.strategy == "decay":
        assert Z is not None and z_metric_fn is not None and epsilon is not None
        anchors = _select_anchors(q_vec, X, cfg.tau, min_anchors=3)
        hops = semantic_decay_traversal(
            q=q_vec,
            X=X, Z=Z,
            dist_X_fn=cosine_distance,
            dist_Z_fn=z_metric_fn,
            tau=cfg.tau,
            epsilon=epsilon,
            gamma=cfg.gamma,
            max_hops=cfg.max_hops,
            initial_set=frozenset(anchors),
        )
        primary = sorted(hops.get(0, frozenset()))
        structural = sorted({i for t, ids in hops.items() if t > 0 for i in ids})
        # Cap total context size so prompts stay sane.
        return RetrievalOutput(
            primary=primary[:cfg.top_k_naive],
            structural=structural[:cfg.top_k_naive * 2],
            all_ids=primary[:cfg.top_k_naive] + structural[:cfg.top_k_naive * 2],
            hops={int(k): list(v) for k, v in hops.items()},
        )
    if cfg.strategy == "ppr":
        assert Z is not None and z_metric_fn is not None
        primary, structural = _ppr_topk(
            q_vec, X, Z, z_metric_fn,
            tau=cfg.tau, k=cfg.top_k_ppr,
            P=ppr_P,
        )
        return RetrievalOutput(
            primary=primary,
            structural=structural,
            all_ids=primary + structural,
            hops={0: primary, 1: structural},
        )
    raise ValueError(f"unknown strategy: {cfg.strategy}")


# ---------------------------------------------------------------------------
# Embedding + PCA helpers (with caching)
# ---------------------------------------------------------------------------

def embed_corpus_and_query(
    embedder: Embedder,
    corpus: List[str],
    query: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X = embedder.embed(corpus, task="RETRIEVAL_DOCUMENT")
    q_vec = embedder.embed([query], task="RETRIEVAL_QUERY")[0]
    return X.astype(np.float64), q_vec.astype(np.float64)


def fit_pca_cached(corpus: List[str], embedder_name: str, X: np.ndarray, d: int) -> dict:
    c_h = corpus_hash(corpus)

    def _compute() -> dict:
        fit = fit_pca(X, d)
        return {
            "Z": fit.Z.astype(np.float64),
            "eigenvalues": fit.eigenvalues.astype(np.float64),
            "components": fit.components.astype(np.float64),
            "mean": fit.mean.astype(np.float64),
        }

    return _PCA.get_or_compute(c_h, embedder_name, d, _compute)


def threshold_cached(
    cfg: CellConfig,
    corpus_h: str,
    Z: np.ndarray,
    metric_fn: Callable,
) -> float:
    key = ThresholdCache.key(corpus_h, cfg.embedder, cfg.d, cfg.metric)
    hit = _THRESH.get(key)
    if hit is not None:
        return float(hit["epsilon"])
    eps = calibrate_epsilon(Z, metric_fn, quantile=0.10, sample_size=50, seed=cfg.seed)
    _THRESH.put(key, {"epsilon": eps, "metric": cfg.metric, "embedder": cfg.embedder, "d": cfg.d})
    return eps


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    cfg: CellConfig,
    questions: List[Question],
    embedder: Embedder,
    do_judge: bool = True,
    cell_index: Optional[int] = None,
    total_cells: Optional[int] = None,
) -> List[dict]:
    """Run one experimental cell. Returns per-question records.

    Caller is expected to pass the pre-built embedder so model weights are
    shared across cells. `cell_index` / `total_cells` are optional and used
    only to prefix the per-cell progress line.
    """
    records: List[dict] = []
    # Detect shared corpus (MultiHop-RAG) to PCA-fit once.
    first_corpus = questions[0].corpus
    shared = all(q.corpus is first_corpus or q.corpus == first_corpus for q in questions)

    cell_prefix = f"[{cell_index}/{total_cells}]" if cell_index is not None and total_cells is not None else "[cell]"
    print(f"\n{cell_prefix} START {cfg.label}  (n={len(questions)})", flush=True)
    cell_t0 = time.time()

    shared_state: Optional[dict] = None
    if shared and cfg.strategy not in ("no_context",):
        X = embedder.embed(first_corpus, task="RETRIEVAL_DOCUMENT").astype(np.float64)
        if cfg.strategy != "naive":
            pca = fit_pca_cached(first_corpus, embedder.name, X, cfg.d)
            metric_fn = build_z_metric(cfg.metric, pca["Z"], np.diag(1.0 / (pca["eigenvalues"] + 1e-8)), pca["eigenvalues"])
            corpus_h = corpus_hash(first_corpus)
            eps = threshold_cached(cfg, corpus_h, pca["Z"], metric_fn)
            # For PPR, precompute the transition matrix once per cell.
            ppr_P = _build_ppr_transition(pca["Z"], metric_fn, beta=1.0) if cfg.strategy == "ppr" else None
            shared_state = {"X": X, "Z": pca["Z"], "metric_fn": metric_fn, "epsilon": eps, "ppr_P": ppr_P}
        else:
            shared_state = {"X": X, "Z": None, "metric_fn": None, "epsilon": None, "ppr_P": None}

    bar = tqdm(
        questions,
        desc=f"  {cfg.metric}/{cfg.strategy}",
        leave=True,
        unit="q",
        ncols=88,
        mininterval=0.3,
        dynamic_ncols=False,
    )
    for q in bar:
        t0 = time.time()
        try:
            # 1. embed + project
            if cfg.strategy == "no_context":
                X, q_vec, Z, metric_fn, eps = None, None, None, None, None
            elif shared_state is not None:
                X = shared_state["X"]
                Z = shared_state["Z"]
                metric_fn = shared_state["metric_fn"]
                eps = shared_state["epsilon"]
                q_vec = embedder.embed([q.query], task="RETRIEVAL_QUERY")[0].astype(np.float64)
            else:
                X, q_vec = embed_corpus_and_query(embedder, q.corpus, q.query)
                if cfg.strategy == "naive":
                    Z, metric_fn, eps = None, None, None
                else:
                    pca = fit_pca_cached(q.corpus, embedder.name, X, cfg.d)
                    Z = pca["Z"]
                    metric_fn = build_z_metric(
                        cfg.metric, Z,
                        np.diag(1.0 / (pca["eigenvalues"] + 1e-8)),
                        pca["eigenvalues"],
                    )
                    eps = threshold_cached(cfg, corpus_hash(q.corpus), Z, metric_fn)

            # 2. retrieve
            retr = retrieve(cfg, q, X if X is not None else np.zeros((0,)), q_vec if q_vec is not None else np.zeros((0,)), Z, metric_fn, eps)

            # 3. prompt + generate
            if cfg.strategy == "no_context":
                prompt = build_no_context_prompt(q.query)
            elif cfg.strategy in ("decay", "ppr"):
                prompt = build_mrag_prompt(
                    q.query,
                    primary_chunks=[q.corpus[i] for i in retr.primary],
                    structural_chunks=[q.corpus[i] for i in retr.structural],
                )
            else:  # naive
                prompt = build_flat_prompt(q.query, [q.corpus[i] for i in retr.all_ids])

            pred = generate_answer(prompt, model=cfg.gen_model)

            # 4. score
            em = exact_match(pred, q.gold_answer)
            f1 = f1_score(pred, q.gold_answer)
            recall = retrieval_recall(retr.all_ids, q.gold_support_ids)
            precision = retrieval_precision(retr.all_ids, q.gold_support_ids)
            judge_correct: Optional[float] = None
            judge_rationale = ""
            if do_judge:
                judge = llm_judge(q.query, pred, q.gold_answer, model=cfg.judge_model)
                judge_correct = float(judge["correct"])
                judge_rationale = judge.get("rationale", "")

            records.append({
                "dataset": cfg.dataset,
                "embedder": cfg.embedder,
                "metric": cfg.metric,
                "strategy": cfg.strategy,
                "d": cfg.d,
                "qid": q.id,
                "query": q.query,
                "gold_answer": q.gold_answer,
                "predicted": pred,
                "em": em,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "judge": judge_correct,
                "judge_rationale": judge_rationale,
                "retrieved_ids": retr.all_ids,
                "primary_ids": retr.primary,
                "structural_ids": retr.structural,
                "gold_support_ids": q.gold_support_ids,
                "n_retrieved": len(retr.all_ids),
                "wall_s": time.time() - t0,
            })
        except Exception as exc:  # noqa: BLE001
            records.append({
                "dataset": cfg.dataset,
                "embedder": cfg.embedder,
                "metric": cfg.metric,
                "strategy": cfg.strategy,
                "d": cfg.d,
                "qid": q.id,
                "query": q.query,
                "gold_answer": q.gold_answer,
                "predicted": "",
                "em": 0.0,
                "f1": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "judge": 0.0,
                "judge_rationale": f"error: {exc}",
                "retrieved_ids": [],
                "primary_ids": [],
                "structural_ids": [],
                "gold_support_ids": q.gold_support_ids,
                "n_retrieved": 0,
                "wall_s": time.time() - t0,
            })

        # Live running averages shown in the tqdm bar postfix.
        n = len(records)
        running_f1 = sum(r["f1"] for r in records) / n
        running_judge = sum((r["judge"] or 0.0) for r in records) / n if do_judge else 0.0
        running_retr = sum(r["n_retrieved"] for r in records) / n
        bar.set_postfix({
            "F1": f"{running_f1:.2f}",
            "judge": f"{running_judge:.2f}",
            "retr": f"{running_retr:.1f}",
        })

    elapsed = time.time() - cell_t0
    n = len(records)
    mean_f1 = sum(r["f1"] for r in records) / max(n, 1)
    mean_em = sum(r["em"] for r in records) / max(n, 1)
    mean_judge = sum((r["judge"] or 0.0) for r in records) / max(n, 1) if do_judge else 0.0
    mean_recall = sum(r["recall"] for r in records) / max(n, 1)
    mean_retr = sum(r["n_retrieved"] for r in records) / max(n, 1)
    print(
        f"{cell_prefix} DONE  {cfg.label}  "
        f"F1={mean_f1:.3f} EM={mean_em:.3f} judge={mean_judge:.3f} "
        f"recall={mean_recall:.3f} retr={mean_retr:.1f}  ({elapsed:.1f}s)",
        flush=True,
    )
    return records


# ---------------------------------------------------------------------------
# Top-level grid runner
# ---------------------------------------------------------------------------

def write_records(records: List[dict], path: Path) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


def _cell_already_complete(out_path: Path, cfg: CellConfig, expected_n: int) -> bool:
    """Cheap skip-if-done check: parquet already has expected_n rows for this cell."""
    if not out_path.exists():
        return False
    try:
        df = pd.read_parquet(out_path, columns=["dataset", "embedder", "metric", "strategy", "d", "qid"])
    except Exception:  # noqa: BLE001
        return False
    sub = df[
        (df["dataset"] == cfg.dataset)
        & (df["embedder"] == cfg.embedder)
        & (df["metric"] == cfg.metric)
        & (df["strategy"] == cfg.strategy)
        & (df["d"] == cfg.d)
    ]
    return len(sub) >= expected_n


def run_grid(
    datasets: List[str],
    embedders: List[str],
    metrics: List[str],
    strategies: List[str],
    n: int = 500,
    d: int = 32,
    out_path: str = "results/results.parquet",
    judge: bool = True,
) -> Path:
    """Build the experiment grid and execute. Caches make re-runs cheap.

    Already-complete cells are skipped at the parquet level — so this is safe
    to re-run without producing duplicate rows.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load questions once per dataset.
    questions_by_ds = {ds: load_benchmark(ds, n=n) for ds in datasets}

    # Build embedders once. tfidf needs a fit_corpus; we fit on the union of corpora.
    embedder_cache: dict[str, Embedder] = {}

    def get_embedder(name: str, fit_corpus_hint: List[str]) -> Embedder:
        if name in embedder_cache:
            return embedder_cache[name]
        emb = build_embedder(name, fit_corpus=fit_corpus_hint)
        embedder_cache[name] = emb
        return emb

    seen_baseline: set[tuple] = set()

    # Total cell count for the progress prefix.
    total_cells = (
        len(datasets) * len(embedders) * (1 + len(metrics) * len(strategies))
        + len(datasets)  # no_context per dataset
    )
    cell_i = 0

    for ds_name in datasets:
        questions = questions_by_ds[ds_name]
        union_corpus = list({c for q in questions for c in q.corpus})
        print(f"\n=== dataset: {ds_name} (n={len(questions)}) ===", flush=True)

        for emb_name in embedders:
            embedder = get_embedder(emb_name, union_corpus)
            # naive RAG (cosine_x) — once per (dataset, embedder)
            key = (ds_name, emb_name, "naive", 0)
            if key not in seen_baseline:
                seen_baseline.add(key)
                cell_i += 1
                cfg = CellConfig(dataset=ds_name, embedder=emb_name, metric="cosine_x",
                                 strategy="naive", d=d, n=n)
                if not _cell_already_complete(out, cfg, n):
                    records = run_cell(cfg, questions, embedder, do_judge=judge,
                                        cell_index=cell_i, total_cells=total_cells)
                    write_records(records, out)
                else:
                    print(f"[{cell_i}/{total_cells}] SKIP  {cfg.label} (already complete)", flush=True)

            for metric in metrics:
                for strategy in strategies:
                    cell_i += 1
                    cfg = CellConfig(dataset=ds_name, embedder=emb_name, metric=metric,
                                     strategy=strategy, d=d, n=n)
                    if not _cell_already_complete(out, cfg, n):
                        records = run_cell(cfg, questions, embedder, do_judge=judge,
                                            cell_index=cell_i, total_cells=total_cells)
                        write_records(records, out)
                    else:
                        print(f"[{cell_i}/{total_cells}] SKIP  {cfg.label} (already complete)", flush=True)

        # no-context once per dataset.
        cell_i += 1
        cfg = CellConfig(dataset=ds_name, embedder="none", metric="none", strategy="no_context",
                         d=d, n=n)
        if not _cell_already_complete(out, cfg, n):
            any_embedder = next(iter(embedder_cache.values()))
            records = run_cell(cfg, questions, any_embedder, do_judge=judge,
                                cell_index=cell_i, total_cells=total_cells)
            write_records(records, out)
        else:
            print(f"[{cell_i}/{total_cells}] SKIP  {cfg.label} (already complete)", flush=True)

    return out
