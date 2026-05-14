"""
Unit tests for ManifoldRAG building blocks. Run with `pytest -q`.

These don't hit any APIs; everything uses TF-IDF + deterministic seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from distances import (
    METRICS, build_z_metric, cosine_distance,
    euclidean_distance, make_bhattacharyya_distance, make_negated_dot_distance,
    make_rbf_kernel_distance, make_mahalanobis_distance,
)
from evaluation import f1_score, exact_match, retrieval_precision, retrieval_recall
from manifolds import fit_pca


CORPUS = [
    "Cobalt is mined primarily in the Democratic Republic of Congo.",
    "Artisanal cobalt mining involves dangerous manual labor in open pits.",
    "The DRC produces over seventy percent of the global cobalt supply.",
    "Global supply chains link raw mineral extraction to consumer electronics.",
    "Mineral supply chains face increasing scrutiny for ethical sourcing.",
    "iPhone batteries rely on rechargeable lithium-ion cell technology.",
    "Apple sources battery components from multiple suppliers across Asia.",
    "Smartphone batteries degrade after hundreds of charge cycles.",
    "Photosynthesis converts sunlight into chemical energy inside plant cells.",
    "The Roman Empire fell in 476 CE after centuries of gradual decline.",
]


@pytest.fixture
def Z_fit():
    rng = np.random.default_rng(0)
    # Deterministic random embeddings — we don't need real semantics here.
    X = rng.standard_normal((len(CORPUS), 32)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return fit_pca(X, d=8)


def test_pca_orthonormality(Z_fit):
    W = Z_fit.components
    gram = W @ W.T
    assert np.allclose(gram, np.eye(W.shape[0]), atol=1e-8)
    assert np.allclose(Z_fit.Z.mean(axis=0), 0, atol=1e-8)


@pytest.mark.parametrize("name", METRICS)
def test_metric_self_distance_zero(name, Z_fit):
    """Most metrics satisfy d(x, x) ≈ 0. The negated-dot is offset by a constant
    so the smallest-possible value is the metric floor, not zero — we check that
    the self-pair gives the *minimum* of all pairwise values instead."""
    fn = build_z_metric(name, Z_fit.Z, Z_fit.inv_cov, Z_fit.eigenvalues)
    if name == "dot":
        self_d = fn(Z_fit.Z[0], Z_fit.Z[0])
        # self_d should not be larger than any cross-pair (since dot(z,z) >= dot(z,y))
        for j in range(1, len(Z_fit.Z)):
            cross = fn(Z_fit.Z[0], Z_fit.Z[j])
            assert self_d <= cross + 1e-8
        return
    for z in Z_fit.Z[:3]:
        d = fn(z, z)
        assert d >= -1e-8, f"{name} self-distance negative: {d}"
        assert d < 1e-3, f"{name} self-distance non-zero: {d}"


@pytest.mark.parametrize("name", METRICS)
def test_metric_non_negative(name, Z_fit):
    fn = build_z_metric(name, Z_fit.Z, Z_fit.inv_cov, Z_fit.eigenvalues)
    for i in range(len(Z_fit.Z)):
        for j in range(i + 1, len(Z_fit.Z)):
            d = fn(Z_fit.Z[i], Z_fit.Z[j])
            assert d >= -1e-8


def test_cosine_distance_basics():
    x = np.array([1.0, 0.0])
    y = np.array([1.0, 0.0])
    z = np.array([-1.0, 0.0])
    assert cosine_distance(x, y) == pytest.approx(0.0)
    assert cosine_distance(x, z) == pytest.approx(2.0)


def test_euclidean_distance_basics():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0)


def test_em_f1_normalization():
    """Official HotpotQA normalization (lowercase, strip articles + punctuation,
    whitespace-tokenize). Hyphens are *removed* without inserting whitespace —
    that's why we also run LLM-as-judge in the harness."""
    assert exact_match("The Beatles", "the beatles") == 1.0
    assert exact_match("Apple Inc.", "Apple") == 0.0
    assert f1_score("yes", "no") == 0.0
    assert f1_score("New York City", "New York") == pytest.approx(2 * (2 / 3 * 1.0) / (2 / 3 + 1.0))
    # Identity case (no punctuation) is exact.
    assert f1_score("Sam Bankman Fried", "Sam Bankman Fried") == pytest.approx(1.0)


def test_retrieval_metrics():
    assert retrieval_recall([1, 2, 3], [1, 2]) == 1.0
    assert retrieval_recall([1], [1, 2]) == 0.5
    assert retrieval_precision([1, 2, 3], [1]) == pytest.approx(1 / 3)
    assert retrieval_recall([], []) == 1.0
    assert retrieval_precision([], [1]) == 0.0


def test_traversal_metric_swap_changes_results():
    """Smoke check that swapping the metric actually changes retrieval."""
    from traversals import semantic_decay_traversal

    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 16)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    fit = fit_pca(X, d=4)
    q = X[0]

    # Two different Z-metrics, fixed thresholds.
    m1 = build_z_metric("mahalanobis", fit.Z, fit.inv_cov, fit.eigenvalues)
    m2 = build_z_metric("euclidean", fit.Z, fit.inv_cov, fit.eigenvalues)

    layers_1 = semantic_decay_traversal(
        q=q, X=X, Z=fit.Z,
        dist_X_fn=cosine_distance,
        dist_Z_fn=m1,
        tau=0.5, epsilon=2.0, gamma=1.2, max_hops=2,
    )
    layers_2 = semantic_decay_traversal(
        q=q, X=X, Z=fit.Z,
        dist_X_fn=cosine_distance,
        dist_Z_fn=m2,
        tau=0.5, epsilon=2.0, gamma=1.2, max_hops=2,
    )
    # The two retrievals don't have to differ on every input, but they should
    # differ on *something* over a random matrix this size. If they don't, the
    # metric system is suspiciously broken.
    flat_1 = sorted(i for ids in layers_1.values() for i in ids)
    flat_2 = sorted(i for ids in layers_2.values() for i in ids)
    assert flat_1 != flat_2 or len(flat_1) == 0


def test_smoke_run_end_to_end(tmp_path, monkeypatch):
    """End-to-end run on 2 TF-IDF questions, no API calls."""
    import benchmarks
    from embeddings import build_embedder
    from experiment import CellConfig, run_cell, write_records
    import llm

    # Stub LLM out.
    monkeypatch.setattr(llm, "generate_answer", lambda prompt, model="none": "unknown")
    monkeypatch.setattr(llm, "llm_judge", lambda *a, **kw: {"correct": False, "rationale": "stub"})
    import experiment as exp_mod
    monkeypatch.setattr(exp_mod, "generate_answer", lambda prompt, model="none": "unknown")
    monkeypatch.setattr(exp_mod, "llm_judge", lambda *a, **kw: {"correct": False, "rationale": "stub"})

    qs = benchmarks.load_benchmark("hotpotqa", n=2)
    fit_corpus = [c for q in qs for c in q.corpus]
    emb = build_embedder("tfidf", fit_corpus=fit_corpus, target_dim=64)

    cfg = CellConfig(dataset="hotpotqa", embedder="tfidf", metric="mahalanobis",
                     strategy="decay", d=8, n=2, gen_model="none")
    records = run_cell(cfg, qs, emb, do_judge=False)
    assert len(records) == 2
    for r in records:
        assert "f1" in r and "em" in r
        assert r["em"] == 0.0  # we stubbed out the LLM with "unknown"
