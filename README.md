# ManifoldRAG (mRAG) — Experiment Harness

Implementation and empirical evaluation of the **ManifoldRAG** paper *(ManifoldRAG: Bypassing Explicit Ontologies with Dual-Manifold Structural Projections, March 2026)*. The paper proposes a zero-extraction multi-hop RAG that retrieves through two manifolds — a high-dim **Semantic** space X (cosine distance for anchors) and a low-dim **Structural** space Z built via PCA (Mahalanobis distance for traversal) — with two bounded traversal strategies: Semantic-Decay Beam Search and Continuous Random Walk with Restart.

This repo turns that proposal into a runnable benchmark:

- Tests **6 distance metrics** as substitutes for the paper's Mahalanobis-in-Z choice.
- Supports **4 embedding backends** (Gemini Embedding 2, two sentence-transformers, TF-IDF).
- Runs on **3 multi-hop QA benchmarks** (HotpotQA distractor, MultiHop-RAG, 2WikiMultiHopQA).
- Measures **answer accuracy** (EM, F1, Gemini-as-judge) and **retrieval quality** (context recall / precision).
- Produces a single self-contained HTML report under `results/report.html` with tables, plots, and an example gallery.

## Quick start

```bash
# Install deps
uv sync

# Sanity check — runs end-to-end on 5 questions with TF-IDF, no API calls.
uv run python main.py smoke

# Run the experiment grid (any subset of datasets/embedders/metrics/strategies)
uv run python main.py run \
  --datasets hotpotqa multihoprag 2wiki \
  --embedders all-MiniLM-L6-v2 \
  --metrics mahalanobis euclidean cosine dot bhattacharyya rbf \
  --strategies decay ppr \
  --n 50 \
  --d 16

# Build the consolidated HTML report
uv run python main.py analyze
open results/report.html

# Inspect the cache
uv run python main.py cache-stats
```

Requires `GEMINI_API_KEY` in `.env` for Gemini embeddings, answer generation, and LLM-as-judge.

## Experimental matrix

```
Benchmarks     × Embedders            × Z-distance metrics    × Retrieval modes
-------------------------------------------------------------------------------
HotpotQA-d     | TF-IDF               | Mahalanobis (paper)    | naive-RAG (cosine X)
MultiHop-RAG   | all-MiniLM-L6-v2     | Euclidean              | mRAG: semantic-decay
2WikiMH        | BAAI/bge-small-en    | Cosine                 | mRAG: random-walk PPR
               | gemini-embedding-2   | Dot product            | no-context (sanity)
                                      | Bhattacharyya
                                      | RBF kernel
```

X-space anchor selection is always cosine (paper §3.1). The ablation varies the Z-space metric. With caching, re-running the grid with a different metric or strategy never re-embeds.

## Distance metrics implemented

| Metric          | Implementation                                                                |
|-----------------|--------------------------------------------------------------------------------|
| Mahalanobis     | `sqrt((z_i - z_j)^T Λ^{-1} (z_i - z_j))` — paper Eq. 4                        |
| Euclidean       | `||z_i - z_j||_2`                                                             |
| Cosine          | `1 - (z_i · z_j) / (||z_i|| ||z_j||)`                                         |
| Dot product     | `max_dot - z_i · z_j` (offset to non-negative)                                |
| Bhattacharyya   | k-NN local Gaussians: `(1/8)Δ^T Σ_avg^{-1} Δ + (1/2) log(|Σ_avg|/√(|Σ_i||Σ_j|))` |
| RBF kernel      | `sqrt(2 - 2 exp(-γ ||z_i - z_j||²))`, γ from median-pairwise heuristic        |

All return non-negative scalars and plug into the existing `dist_Z_fn` slot in [traversals.py](traversals.py).

The traversal threshold `ε` is **calibrated per (corpus, embedder, d, metric)** to the 10th-percentile pairwise Z-distance — so metrics with wildly different absolute scales remain comparable.

## Embedding backends

| Name                  | Source                                  | Dim | Cached |
|-----------------------|-----------------------------------------|-----|--------|
| `tfidf`               | scikit-learn TF-IDF + Truncated SVD     | 256 | n/a    |
| `all-MiniLM-L6-v2`    | sentence-transformers                   | 384 | yes    |
| `bge-small-en-v1.5`   | BAAI/sentence-transformers              | 384 | yes    |
| `gemini-embedding-2`  | Google Gemini API                       | 768 | yes    |

Document and query embeddings are cached separately (Gemini emits different vectors for `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` task types).

## Benchmarks

| Loader            | HF source                          | Schema                                  |
|-------------------|------------------------------------|------------------------------------------|
| `hotpotqa`        | `hotpotqa/hotpot_qa` / `distractor` | per-q pool of 10 paragraphs, span answer |
| `multihoprag`     | `yixuantt/MultiHopRAG`              | shared corpus of 609 articles + 2,556 q  |
| `2wiki`           | `voidful/2WikiMultihopQA`           | per-q pool of 10 paragraphs, span answer |

All three loaders return the same `Question` dataclass: `id, query, corpus, gold_answer, gold_support_ids, meta`.

## Caching design

Caching is the primary cost lever. Layout (all under `.cache/`, gitignored):

```
.cache/
├── embeddings/{model}/{dim}/        # sqlite index + appendable float32 memmap
├── pca/                              # per (corpus, embedder, d) — .npz
├── thresholds/                       # per (corpus, embedder, d, metric) — .json
├── answers/                          # per (prompt, gen_model) — .json
└── judge/                            # per (query, pred, gold, judge_model) — .json
```

Key properties:

- **Content-addressed.** Cache keys are `sha1(text + model + dim + task)` for embeddings; the layout is order-independent.
- **Mixed hit/miss batches.** Embedding 100 chunks where 90 are cached sends only 10 to the API.
- **Cell-level skipping.** `python main.py run` reads the parquet, sees which (dataset, embedder, metric, strategy, d) cells already have N rows, and skips them.

After a first full run, a re-run that changes the metric or strategy never re-embeds and only pays for the LLM answer/judge calls on prompts that actually changed.

## Module layout

```
cache.py        # Persistent content-addressed cache (embeddings, PCA, thresholds, answers, judge)
embeddings.py   # TF-IDF / sentence-transformers / Gemini backends — common Embedder interface
distances.py    # 6 distance metrics + build_z_metric factory
manifolds.py    # PCA -> structural manifold Z + invertible artifacts
traversals.py   # Semantic-Decay Beam Search + Continuous Random Walk (PPR)
benchmarks.py   # HF dataset loaders -> common Question dataclass
evaluation.py   # HotpotQA EM/F1 + retrieval recall/precision
llm.py          # Gemini answer generation + LLM-as-judge, both cached
experiment.py   # Cell runner + threshold calibration + grid driver
analyze.py      # Tables + plots + standalone HTML report
main.py         # CLI: smoke / run / analyze / cache-stats
test_mrag.py    # pytest suite
```

## Requirements

- Python ≥ 3.13 (managed via `uv`)
- `GEMINI_API_KEY` in `.env` (only needed for Gemini embeddings / answer generation / judge)
- Roughly ~1 GB of disk for the full cache on 3 datasets at 500 q each (most of that is embeddings)

## Tests

```bash
uv run pytest -q
```

Tests cover: distance-metric signatures, self-distance, non-negativity, PCA orthonormality, EM/F1 normalization, retrieval metrics, traversal-metric-swap sanity, and a stubbed end-to-end smoke run.

## Out of scope

- LightRAG / Microsoft GraphRAG as the GraphRAG baseline. The paper lists these, but they require their own LLM-extraction pipelines and graph infrastructure. We compare against **naive cosine RAG** as the primary baseline; GraphRAG is a follow-up.
- UMAP as an alternative `Φ`. Trivial to add as a drop-in in [manifolds.py](manifolds.py) when needed.
- Non-English benchmarks.
