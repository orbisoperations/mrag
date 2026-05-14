"""
Persistent, content-addressed cache for the mRAG experiment harness.

Caches:
  1. Embeddings — sqlite index + numpy memmap, keyed by sha1(text|model|dim|task).
  2. PCA fits  — npz, keyed by sha1(corpus|embedder|d).
  3. ε thresholds — json, keyed by sha1(corpus|embedder|d|metric).
  4. Generated answers — json, keyed by sha1(prompt|gen_model).
  5. LLM-judge verdicts — json, keyed by sha1(query|pred|gold|judge_model).

Embedding cache is the hot path. It supports:
  - O(1) lookup via sqlite.
  - Mixed hit/miss batches: only misses are sent to the embedder.
  - Crash safety: appends to vectors.npy as raw float32 bytes, index only
    commits after the append flushes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

CACHE_ROOT = Path(__file__).resolve().parent / ".cache"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha1(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def slug(text: str) -> str:
    """Filesystem-safe slug for model names etc."""
    return text.replace("/", "_").replace(":", "_").replace(" ", "_")


def corpus_hash(corpus: List[str]) -> str:
    """Stable hash of a corpus (order-sensitive)."""
    h = hashlib.sha1()
    for chunk in corpus:
        h.update(chunk.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Sqlite-indexed numpy memmap of float32 embeddings.

    One cache per (model, output_dim). Concurrent access from a single process
    is protected by a threading.Lock; cross-process is protected by sqlite's
    own transaction handling.
    """

    def __init__(self, model: str, output_dim: int, task_type: str = "default"):
        self.model = model
        self.output_dim = output_dim
        self.task_type = task_type
        self.dir = CACHE_ROOT / "embeddings" / slug(model) / str(output_dim)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "index.sqlite"
        self.vec_path = self.dir / "vectors.npy"
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS vectors (
                    key TEXT PRIMARY KEY,
                    offset INTEGER NOT NULL
                )"""
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_key ON vectors(key)")

    def _key(self, text: str) -> str:
        return _sha1(text, self.model, str(self.output_dim), self.task_type)

    def _row_bytes(self) -> int:
        return self.output_dim * 4  # float32

    def _read_row(self, offset: int) -> np.ndarray:
        with open(self.vec_path, "rb") as f:
            f.seek(offset * self._row_bytes())
            buf = f.read(self._row_bytes())
        return np.frombuffer(buf, dtype=np.float32).copy()

    def _append_rows(self, vectors: np.ndarray) -> int:
        """Append vectors to the memmap, return first offset."""
        assert vectors.dtype == np.float32, "expected float32"
        assert vectors.shape[1] == self.output_dim
        with open(self.vec_path, "ab") as f:
            start_byte = f.tell()
            f.write(vectors.tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
        return start_byte // self._row_bytes()

    def get_or_compute(
        self,
        texts: List[str],
        compute_fn: Callable[[List[str]], np.ndarray],
    ) -> np.ndarray:
        """Return embeddings for `texts` in order, computing only the misses.

        compute_fn takes a list of cache-miss texts and returns a 2D float32
        array of L2-normalized embeddings.
        """
        if not texts:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        result = np.zeros((len(texts), self.output_dim), dtype=np.float32)

        with self._lock, sqlite3.connect(self.db_path) as con:
            # 1. Look up all keys at once.
            placeholders = ",".join("?" * len(keys))
            rows = con.execute(
                f"SELECT key, offset FROM vectors WHERE key IN ({placeholders})",
                keys,
            ).fetchall()
            offset_by_key = dict(rows)

            # 2. Pull hits out of the memmap.
            miss_indices: List[int] = []
            for i, k in enumerate(keys):
                if k in offset_by_key:
                    result[i] = self._read_row(offset_by_key[k])
                else:
                    miss_indices.append(i)

            # 3. Compute misses (deduplicating within the batch).
            if miss_indices:
                miss_texts: List[str] = []
                miss_keys: List[str] = []
                seen: dict[str, int] = {}
                positions: List[int] = []
                for i in miss_indices:
                    k = keys[i]
                    if k in seen:
                        positions.append(seen[k])
                    else:
                        seen[k] = len(miss_texts)
                        positions.append(seen[k])
                        miss_texts.append(texts[i])
                        miss_keys.append(k)

                new_vecs = compute_fn(miss_texts).astype(np.float32, copy=False)
                assert new_vecs.shape == (len(miss_texts), self.output_dim), \
                    f"compute_fn returned {new_vecs.shape}, expected {(len(miss_texts), self.output_dim)}"

                first_offset = self._append_rows(new_vecs)

                # 4. Insert keys into sqlite + fill result.
                con.executemany(
                    "INSERT OR IGNORE INTO vectors (key, offset) VALUES (?, ?)",
                    [(k, first_offset + j) for j, k in enumerate(miss_keys)],
                )
                con.commit()

                for local_idx, i in enumerate(miss_indices):
                    result[i] = new_vecs[positions[local_idx]]

        return result

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as con:
            (count,) = con.execute("SELECT COUNT(*) FROM vectors").fetchone()
        size = self.vec_path.stat().st_size if self.vec_path.exists() else 0
        return {
            "model": self.model,
            "output_dim": self.output_dim,
            "task_type": self.task_type,
            "rows": count,
            "vectors_bytes": size,
        }


# ---------------------------------------------------------------------------
# PCA + threshold + answer + judge cache (simple key-value)
# ---------------------------------------------------------------------------

class _JsonCache:
    """A dumb per-key-file JSON cache. Use for small payloads."""

    def __init__(self, subdir: str):
        self.dir = CACHE_ROOT / subdir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(value, f)
        os.replace(tmp, p)

    def get_or_compute(self, key: str, compute_fn: Callable[[], dict]) -> dict:
        hit = self.get(key)
        if hit is not None:
            return hit
        value = compute_fn()
        self.put(key, value)
        return value


class PCACache:
    """Caches PCA fits as npz, keyed by (corpus_hash, embedder, d)."""

    def __init__(self):
        self.dir = CACHE_ROOT / "pca"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, corpus_h: str, embedder: str, d: int) -> Path:
        return self.dir / f"{corpus_h}__{slug(embedder)}__d{d}.npz"

    def get(self, corpus_h: str, embedder: str, d: int) -> Optional[dict]:
        p = self._path(corpus_h, embedder, d)
        if not p.exists():
            return None
        try:
            data = np.load(p)
            return {
                "Z": data["Z"],
                "eigenvalues": data["eigenvalues"],
                "components": data["components"],
                "mean": data["mean"],
            }
        except (OSError, KeyError):
            return None

    def put(self, corpus_h: str, embedder: str, d: int, payload: dict) -> None:
        p = self._path(corpus_h, embedder, d)
        # np.savez appends `.npz` to the filename automatically — write to a temp
        # path that already ends in `.npz` so we know the actual output path.
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, **payload)
        os.replace(tmp, p)

    def get_or_compute(
        self,
        corpus_h: str,
        embedder: str,
        d: int,
        compute_fn: Callable[[], dict],
    ) -> dict:
        hit = self.get(corpus_h, embedder, d)
        if hit is not None:
            return hit
        payload = compute_fn()
        self.put(corpus_h, embedder, d, payload)
        return payload


class ThresholdCache(_JsonCache):
    def __init__(self):
        super().__init__("thresholds")

    @staticmethod
    def key(corpus_h: str, embedder: str, d: int, metric: str) -> str:
        return _sha1(corpus_h, embedder, str(d), metric)[:24]


class AnswerCache(_JsonCache):
    def __init__(self):
        super().__init__("answers")

    @staticmethod
    def key(prompt: str, gen_model: str) -> str:
        return _sha1(prompt, gen_model)[:24]


class JudgeCache(_JsonCache):
    def __init__(self):
        super().__init__("judge")

    @staticmethod
    def key(query: str, pred: str, gold: str, judge_model: str) -> str:
        return _sha1(query, pred, gold, judge_model)[:24]


# ---------------------------------------------------------------------------
# Top-level inspection helper
# ---------------------------------------------------------------------------

def cache_stats() -> dict:
    """Walk the cache root and produce a summary dict."""
    out: dict = {"root": str(CACHE_ROOT), "embeddings": [], "pca": 0, "thresholds": 0, "answers": 0, "judge": 0}
    emb_root = CACHE_ROOT / "embeddings"
    if emb_root.exists():
        for model_dir in emb_root.iterdir():
            if not model_dir.is_dir():
                continue
            for dim_dir in model_dir.iterdir():
                if not dim_dir.is_dir():
                    continue
                db = dim_dir / "index.sqlite"
                vec = dim_dir / "vectors.npy"
                if not db.exists():
                    continue
                try:
                    with sqlite3.connect(db) as con:
                        (count,) = con.execute("SELECT COUNT(*) FROM vectors").fetchone()
                except sqlite3.DatabaseError:
                    count = 0
                size = vec.stat().st_size if vec.exists() else 0
                out["embeddings"].append({
                    "model": model_dir.name,
                    "dim": dim_dir.name,
                    "rows": count,
                    "bytes": size,
                })
    for kind in ("pca", "thresholds", "answers", "judge"):
        d = CACHE_ROOT / kind
        if d.exists():
            out[kind] = sum(1 for _ in d.iterdir())
    return out
