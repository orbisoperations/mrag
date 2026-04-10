"""
Sanity tests for ManifoldRAG building blocks.

Uses a deterministic hash-based bag-of-words embedder so the tests run with
only numpy — no model downloads. The synthetic corpus is designed so that
two clusters ("cobalt mining" and "iPhone batteries") are linked by a bridge
cluster ("supply chains"), mirroring the example from §2.2 of the paper.
"""

from __future__ import annotations

import numpy as np

from main import (
    build_index,
    cosine_distance,
    embed_chunks,
    embed_query,
    fit_pca,
    mahalanobis_distance,
    project_to_structural,
    query_semantic_decay,
    random_walk_traversal,
    semantic_decay_traversal,
)


# ---------------------------------------------------------------------------
# Toy deterministic embedder (no external dependencies)
# ---------------------------------------------------------------------------

def make_toy_embedder(dim: int = 128, vocab_dim: int = 4096, seed: int = 0):
    """
    Deterministic bag-of-words embedder via random projection.

    Each token is hashed to a vocab slot, then projected into R^dim with a
    fixed random Gaussian matrix. Output is L2-normalized so cosine distance
    behaves sensibly. Not as good as a real encoder, but adequate for
    verifying the pipeline math.
    """
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((vocab_dim, dim)) / np.sqrt(dim)

    def embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = hash(word) % vocab_dim
                out[i] += projection[h]
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out / norms

    return embed


# ---------------------------------------------------------------------------
# Synthetic corpus with a clear multi-hop bridge structure
# ---------------------------------------------------------------------------

CORPUS = [
    # --- Cluster A: Cobalt mining (indices 0-2) ---
    "Cobalt is mined primarily in the Democratic Republic of Congo.",
    "Artisanal cobalt mining involves dangerous manual labor in open pits.",
    "The DRC produces over seventy percent of the global cobalt supply.",
    # --- Bridge: Supply chains (indices 3-4) ---
    "Global supply chains link raw mineral extraction to consumer electronics assembly.",
    "Mineral supply chains face increasing scrutiny for ethical sourcing practices.",
    # --- Cluster B: iPhone batteries (indices 5-7) ---
    "iPhone batteries rely on rechargeable lithium-ion cell technology.",
    "Apple sources battery components from multiple suppliers across Asia.",
    "Smartphone batteries degrade after hundreds of charge and discharge cycles.",
    # --- Noise (indices 8-9) ---
    "Photosynthesis converts sunlight into chemical energy inside plant cells.",
    "The Roman Empire fell in 476 CE after centuries of gradual decline.",
]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_building_blocks():
    print("\n=== Building blocks ===")
    embed = make_toy_embedder(dim=128)

    X = embed_chunks(CORPUS, embed)
    assert X.shape == (len(CORPUS), 128)
    print(f"[OK] embed_chunks        -> X {X.shape}")

    components, mean, eigenvalues = fit_pca(X, d=8)
    assert components.shape == (8, 128)
    assert eigenvalues.shape == (8,)
    assert np.all(np.diff(eigenvalues) <= 0), "eigenvalues must be descending"
    assert np.all(eigenvalues >= -1e-9), "eigenvalues must be non-negative"
    print(f"[OK] fit_pca             -> top-3 λ = {eigenvalues[:3].round(4)}")

    Z = project_to_structural(X, components, mean)
    assert Z.shape == (len(CORPUS), 8)
    print(f"[OK] project_to_structural -> Z {Z.shape}")

    x_q = embed_query("mining minerals in Africa", embed)
    cos_dists = cosine_distance(x_q, X)
    assert cos_dists.shape == (len(CORPUS),)
    assert np.all(cos_dists >= -1e-9) and np.all(cos_dists <= 2 + 1e-9)
    print(f"[OK] cosine_distance     -> range [{cos_dists.min():.3f}, {cos_dists.max():.3f}]")
    print(f"     closest chunk: {CORPUS[cos_dists.argmin()]!r}")

    m_dists = mahalanobis_distance(Z[0], Z, eigenvalues)
    assert m_dists.shape == (len(CORPUS),)
    assert m_dists[0] < 1e-6, "self-distance should be 0"
    print(f"[OK] mahalanobis_distance -> self={m_dists[0]:.2e}, max={m_dists.max():.3f}")


def test_semantic_decay_traversal():
    print("\n=== Strategy 1: Semantic Decay Traversal ===")
    embed = make_toy_embedder(dim=128)
    index = build_index(CORPUS, embed, d=8)

    x_q = embed_query("cobalt mining", embed)
    layers = semantic_decay_traversal(
        x_q,
        index["X"], index["Z"], index["eigenvalues"],
        tau=0.6, gamma=0.9, epsilon=5.0, max_hops=2,
    )

    assert len(layers) >= 1
    # S^(0) must contain the direct cobalt-mining chunks
    assert any(i in layers[0] for i in [0, 1, 2]), "expected cobalt chunks in S^(0)"
    print(f"[OK] layers returned     -> {len(layers)}")
    for t, layer in enumerate(layers):
        print(f"     S^({t}) ({len(layer)} chunks):")
        for i in sorted(layer):
            print(f"       [{i}] {CORPUS[i]}")


def test_random_walk_traversal():
    print("\n=== Strategy 2: Personalized PageRank Random Walk ===")
    embed = make_toy_embedder(dim=128)
    index = build_index(CORPUS, embed, d=8)

    top_ids = random_walk_traversal(
        embed_query("cobalt mining", embed),
        index["X"], index["Z"], index["eigenvalues"],
        tau=0.6, alpha=0.85, beta=1.0, top_k=5,
    )

    assert len(top_ids) == 5
    assert len(set(top_ids.tolist())) == 5, "results should be unique"
    print(f"[OK] top-5 PPR results:")
    for rank, i in enumerate(top_ids, 1):
        print(f"     #{rank} [{i}] {CORPUS[i]}")


def test_end_to_end_prompt():
    print("\n=== End-to-end prompt assembly ===")
    embed = make_toy_embedder(dim=128)
    index = build_index(CORPUS, embed, d=8)

    prompt = query_semantic_decay(
        "cobalt mining",
        index,
        embed,
        tau=0.6, gamma=0.9, epsilon=5.0, max_hops=2,
    )

    assert "<primary_semantic_context>" in prompt
    assert "<structurally_linked_context>" in prompt
    assert "cobalt" in prompt.lower()
    print("[OK] prompt assembled")
    print("-" * 70)
    print(prompt)
    print("-" * 70)


def test_pca_axioms():
    """Verify the structural bottleneck axiom: PCA components are orthonormal
    and Z-space is mean-centered."""
    print("\n=== PCA sanity (orthonormal components, centered projection) ===")
    embed = make_toy_embedder(dim=128)
    X = embed_chunks(CORPUS, embed)
    components, mean, eigenvalues = fit_pca(X, d=8)

    # Orthonormal: W W^T ≈ I
    gram = components @ components.T
    assert np.allclose(gram, np.eye(8), atol=1e-8), "components not orthonormal"
    print(f"[OK] components orthonormal (||WW^T - I||_F = {np.linalg.norm(gram - np.eye(8)):.2e})")

    # Centered projection: Z should have ~zero mean
    Z = project_to_structural(X, components, mean)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-8)
    print(f"[OK] Z mean ≈ 0          (max |Z.mean| = {np.abs(Z.mean(axis=0)).max():.2e})")


if __name__ == "__main__":
    test_building_blocks()
    test_pca_axioms()
    test_semantic_decay_traversal()
    test_random_walk_traversal()
    test_end_to_end_prompt()
    print("\nAll tests passed.")
