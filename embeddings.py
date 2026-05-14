"""
Embedding backends for the mRAG experiment harness.

Every backend is exposed as a `Embedder` dataclass with three things:
  - name (slug used for caching),
  - output_dim,
  - and an `embed(texts, task)` callable returning L2-normalized float32.

Backends:
  - TF-IDF      (deterministic, offline, fast — baseline)
  - SentenceTransformer (all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5)
  - Gemini Embedding 2 (cached via EmbeddingCache; rate-limited + retried)

The Gemini embedder uses google-genai's batch endpoint when available and
falls back to a small client-side concurrency budget otherwise.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from cache import EmbeddingCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms != 0).astype(np.float32, copy=False)


@dataclass
class Embedder:
    name: str
    output_dim: int
    embed_fn: Callable[[List[str], str], np.ndarray]  # (texts, task) -> (N, d) float32 L2

    def embed(self, texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return self.embed_fn(texts, task)


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def make_tfidf_embedder(fit_corpus: List[str], target_dim: int = 256) -> Embedder:
    """TF-IDF + SVD to a fixed dim so it plugs into the same pipeline."""
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max(target_dim * 8, 4096))
    raw = vectorizer.fit_transform(fit_corpus)
    n_features = raw.shape[1]
    d = min(target_dim, max(1, min(raw.shape) - 1))
    svd = TruncatedSVD(n_components=d, random_state=0)
    svd.fit(raw)

    def embed(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        V = vectorizer.transform(texts)
        Z = svd.transform(V)
        # Pad to target_dim with zeros if SVD had to shrink.
        if Z.shape[1] < target_dim:
            pad = np.zeros((Z.shape[0], target_dim - Z.shape[1]), dtype=Z.dtype)
            Z = np.hstack([Z, pad])
        return _l2_normalize(Z.astype(np.float32))

    _ = n_features  # silence linter
    return Embedder(name="tfidf-svd", output_dim=target_dim, embed_fn=embed)


# ---------------------------------------------------------------------------
# SentenceTransformer
# ---------------------------------------------------------------------------

_ST_MODELS: dict[str, object] = {}


def _get_st_model(model_name: str):
    if model_name not in _ST_MODELS:
        from sentence_transformers import SentenceTransformer
        _ST_MODELS[model_name] = SentenceTransformer(model_name)
    return _ST_MODELS[model_name]


def make_sentence_transformer_embedder(
    model_name: str,
    cache_enabled: bool = True,
) -> Embedder:
    """Wraps a sentence-transformers model with a content-addressed cache."""
    model = _get_st_model(model_name)
    dim = int(model.get_sentence_embedding_dimension())

    cache_doc: Optional[EmbeddingCache] = None
    cache_q: Optional[EmbeddingCache] = None
    if cache_enabled:
        cache_doc = EmbeddingCache(model=model_name, output_dim=dim, task_type="DOC")
        cache_q = EmbeddingCache(model=model_name, output_dim=dim, task_type="QUERY")

    def _raw_embed(texts: List[str]) -> np.ndarray:
        # ST handles batching internally; we just hand it the list.
        out = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(out.astype(np.float32))

    def embed(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        if cache_enabled:
            c = cache_q if task == "RETRIEVAL_QUERY" else cache_doc
            return c.get_or_compute(texts, _raw_embed)
        return _raw_embed(texts)

    return Embedder(name=model_name, output_dim=dim, embed_fn=embed)


# ---------------------------------------------------------------------------
# Gemini Embedding 2
# ---------------------------------------------------------------------------

_GEMINI_CLIENT = None


def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment / .env")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def make_gemini_embedder(
    model: str = "gemini-embedding-001",
    output_dim: int = 768,
    batch_size: int = 50,
    max_retries: int = 5,
    cache_enabled: bool = True,
) -> Embedder:
    """Gemini Embedding 2 with sha-keyed cache + exponential backoff.

    Notes:
      - The SDK accepts `output_dimensionality` to truncate the 3072-d default.
      - `task_type` distinguishes RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY embeddings;
        Gemini produces different vectors for the same text under different task
        types, so we cache them separately.
    """
    client = _get_gemini_client()
    from google.genai import types as gtypes

    cache_doc: Optional[EmbeddingCache] = None
    cache_q: Optional[EmbeddingCache] = None
    if cache_enabled:
        cache_doc = EmbeddingCache(model=model, output_dim=output_dim, task_type="RETRIEVAL_DOCUMENT")
        cache_q = EmbeddingCache(model=model, output_dim=output_dim, task_type="RETRIEVAL_QUERY")

    def _call_api(batch: List[str], task: str) -> np.ndarray:
        cfg = gtypes.EmbedContentConfig(
            task_type=task,
            output_dimensionality=output_dim,
        )
        for attempt in range(max_retries):
            try:
                resp = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=cfg,
                )
                vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
                return _l2_normalize(vecs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                transient = any(k in msg for k in ("429", "503", "500", "504", "resource_exhausted", "deadline"))
                if attempt == max_retries - 1 or not transient:
                    raise
                wait = (2 ** attempt) * 1.5
                print(f"  [gemini] transient error '{exc}' — retry in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # Models that silently drop all but the first item in a batched call (e.g.
    # gemini-embedding-2 in current SDK). Detected on first short response and
    # remembered for subsequent calls.
    _force_solo = {"value": False}

    def _raw_embed_for_task(task: str):
        def _raw(texts: List[str]) -> np.ndarray:
            out_rows: List[np.ndarray] = []
            if _force_solo["value"]:
                for t in texts:
                    out_rows.append(_call_api([t], task))
                return np.vstack(out_rows)
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                got = _call_api(batch, task)
                # Some Gemini models accept only one content per call — they
                # return a short batch silently. Detect and fall back to solo.
                if got.shape[0] != len(batch):
                    _force_solo["value"] = True
                    print(f"  [gemini] '{model}' returned {got.shape[0]}/{len(batch)} embeddings — falling back to one-at-a-time")
                    for t in batch:
                        out_rows.append(_call_api([t], task))
                else:
                    out_rows.append(got)
            return np.vstack(out_rows)
        return _raw

    def embed(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        c = cache_q if task == "RETRIEVAL_QUERY" else cache_doc
        raw = _raw_embed_for_task(task)
        if c is None:
            return raw(texts)
        return c.get_or_compute(texts, raw)

    return Embedder(name=model, output_dim=output_dim, embed_fn=embed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EMBEDDER_REGISTRY: dict[str, Callable[..., Embedder]] = {
    "tfidf": lambda fit_corpus, **kw: make_tfidf_embedder(fit_corpus, target_dim=kw.get("target_dim", 256)),
    "all-MiniLM-L6-v2": lambda fit_corpus=None, **kw: make_sentence_transformer_embedder("sentence-transformers/all-MiniLM-L6-v2"),
    "bge-small-en-v1.5": lambda fit_corpus=None, **kw: make_sentence_transformer_embedder("BAAI/bge-small-en-v1.5"),
    "gemini-embedding-001": lambda fit_corpus=None, **kw: make_gemini_embedder(model="gemini-embedding-001", output_dim=kw.get("output_dim", 768)),
    "gemini-embedding-2": lambda fit_corpus=None, **kw: make_gemini_embedder(model="gemini-embedding-2", output_dim=kw.get("output_dim", 768)),
}


def build_embedder(name: str, fit_corpus: Optional[List[str]] = None, **kwargs) -> Embedder:
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(f"unknown embedder '{name}'. options: {list(EMBEDDER_REGISTRY)}")
    return EMBEDDER_REGISTRY[name](fit_corpus=fit_corpus, **kwargs)
