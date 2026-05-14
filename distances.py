"""
Distance metrics for the mRAG experiment ablation.

Anchor selection (X-space) is always cosine, per paper §3.1.
The Z-space metric is the ablation axis. All metrics share the signature
    (z_i, z_j) -> float
returning a non-negative scalar where "smaller = more similar".

Some metrics need precomputed state (local covariances, RBF gamma, dot-product
normalization stats) and so are constructed by `make_*_distance(...)` factory
functions that close over that state.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

DistFn = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------------------
# Cosine (X-space anchor + Z-space variant)
# ---------------------------------------------------------------------------

def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cosine distance. Eq. 3 of the paper."""
    n1, n2 = float(np.linalg.norm(x1)), float(np.linalg.norm(x2))
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    return float(1.0 - np.dot(x1, x2) / (n1 * n2))


# ---------------------------------------------------------------------------
# Mahalanobis (paper Eq. 4)
# ---------------------------------------------------------------------------

def make_mahalanobis_distance(inv_cov: np.ndarray) -> DistFn:
    """Mahalanobis with a fixed inverse covariance — paper's Z-space choice."""

    def distance(z1: np.ndarray, z2: np.ndarray) -> float:
        diff = z1 - z2
        q = float(diff @ inv_cov @ diff)
        return float(np.sqrt(max(q, 0.0)))

    return distance


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------

def euclidean_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    return float(np.linalg.norm(z1 - z2))


# ---------------------------------------------------------------------------
# Negated dot product (similarity flipped to distance)
# ---------------------------------------------------------------------------

def make_negated_dot_distance(Z: np.ndarray) -> DistFn:
    """`-(z_i · z_j)` shifted so it's non-negative.

    Pure dot product is unbounded below; we precompute the max possible dot
    in this Z space and use it as the offset so the metric stays >= 0.
    """
    max_dot = float(np.max(Z @ Z.T))

    def distance(z1: np.ndarray, z2: np.ndarray) -> float:
        return float(max_dot - z1 @ z2)

    return distance


# ---------------------------------------------------------------------------
# Bhattacharyya
# ---------------------------------------------------------------------------

def make_bhattacharyya_distance(
    Z: np.ndarray,
    k: int = 10,
    fallback_cov: Optional[np.ndarray] = None,
) -> DistFn:
    """
    Bhattacharyya distance between Gaussians fit to each Z-point's k-NN.

    For each point z_i we fit Σ_i = covariance of its k nearest neighbors in Z.
    Distance between two points then uses
        B = (1/8) (μ_i - μ_j)^T Σ_avg^{-1} (μ_i - μ_j)
            + (1/2) ln( |Σ_avg| / sqrt(|Σ_i| |Σ_j|) )
    with Σ_avg = (Σ_i + Σ_j) / 2.

    For very small / degenerate clusters this falls back to `fallback_cov`
    (typically Λ, the PCA eigenvalues), regularized with a tiny ridge.
    """
    N, d = Z.shape
    eye = np.eye(d) * 1e-6

    # Default fallback to identity-ish if PCA eigenvalues not provided.
    if fallback_cov is None:
        fallback_cov = np.eye(d)

    # Build k-NN local covariances. For tiny corpora k clamps to N-1.
    k = max(2, min(k, N - 1))
    sqd = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    nn_idx = np.argsort(sqd, axis=1)[:, :k]

    covs = np.empty((N, d, d), dtype=np.float64)
    log_dets = np.empty(N, dtype=np.float64)
    for i in range(N):
        neighbors = Z[nn_idx[i]]
        c = np.cov(neighbors, rowvar=False) if neighbors.shape[0] > 1 else fallback_cov
        c = np.atleast_2d(c) + eye
        # Guard against still-singular covariance.
        try:
            log_dets[i] = float(np.linalg.slogdet(c)[1])
        except np.linalg.LinAlgError:
            c = fallback_cov + eye
            log_dets[i] = float(np.linalg.slogdet(c)[1])
        covs[i] = c

    # Map z -> i by exact match (Z rows are unique per-corpus).
    z_lookup: dict[bytes, int] = {}
    for i, z in enumerate(Z):
        z_lookup[z.tobytes()] = i

    def distance(z1: np.ndarray, z2: np.ndarray) -> float:
        i = z_lookup.get(z1.tobytes())
        j = z_lookup.get(z2.tobytes())
        if i is None or j is None:
            # Query/calibration points not in the precomputed map — back off to
            # using the fallback covariance for both endpoints.
            sigma_avg = fallback_cov + eye
            diff = z1 - z2
            mahal_term = 0.125 * float(diff @ np.linalg.inv(sigma_avg) @ diff)
            return mahal_term  # log term cancels when Σ_i = Σ_j = fallback.

        c_i, c_j = covs[i], covs[j]
        sigma_avg = 0.5 * (c_i + c_j)
        diff = z1 - z2
        try:
            inv_avg = np.linalg.inv(sigma_avg)
        except np.linalg.LinAlgError:
            inv_avg = np.linalg.pinv(sigma_avg)
        mahal_term = 0.125 * float(diff @ inv_avg @ diff)
        try:
            slog_avg = float(np.linalg.slogdet(sigma_avg)[1])
        except np.linalg.LinAlgError:
            slog_avg = 0.0
        log_term = 0.5 * (slog_avg - 0.5 * (log_dets[i] + log_dets[j]))
        # The Bhattacharyya distance is non-negative for Gaussians; clip to be safe.
        return float(max(mahal_term + log_term, 0.0))

    return distance


# ---------------------------------------------------------------------------
# RBF-kernel distance (kernel trick)
# ---------------------------------------------------------------------------

def make_rbf_kernel_distance(Z: np.ndarray, gamma: Optional[float] = None) -> DistFn:
    """Distance in the implicit RBF feature space.

    ||φ(x) - φ(y)||² = K(x,x) + K(y,y) - 2K(x,y) = 2 - 2 exp(-γ ||x-y||²).
    γ defaults to 1 / median(pairwise ||·||²) (Silverman heuristic).
    """
    if gamma is None:
        # Subsample to keep this O(M²) cheap on large corpora.
        N = Z.shape[0]
        if N > 200:
            idx = np.random.default_rng(0).choice(N, 200, replace=False)
            S = Z[idx]
        else:
            S = Z
        diffs = S[:, None, :] - S[None, :, :]
        sqd = np.sum(diffs * diffs, axis=-1)
        med = float(np.median(sqd[sqd > 0])) if np.any(sqd > 0) else 1.0
        gamma = 1.0 / max(med, 1e-8)

    def distance(z1: np.ndarray, z2: np.ndarray) -> float:
        d2 = float(np.dot(z1 - z2, z1 - z2))
        val = 2.0 - 2.0 * np.exp(-gamma * d2)
        return float(np.sqrt(max(val, 0.0)))

    return distance


# ---------------------------------------------------------------------------
# Cosine in Z-space (separate from X-space cosine_distance for clarity)
# ---------------------------------------------------------------------------

def cosine_distance_z(z1: np.ndarray, z2: np.ndarray) -> float:
    """Cosine distance applied to Z-space points."""
    return cosine_distance(z1, z2)


# ---------------------------------------------------------------------------
# Factory: build any metric by name with the appropriate fit-time state
# ---------------------------------------------------------------------------

def build_z_metric(
    name: str,
    Z: np.ndarray,
    inv_cov: np.ndarray,
    eigenvalues: np.ndarray,
) -> DistFn:
    """Construct a Z-space distance function by name."""
    name = name.lower()
    if name == "mahalanobis":
        return make_mahalanobis_distance(inv_cov)
    if name == "euclidean":
        return euclidean_distance
    if name == "cosine":
        return cosine_distance_z
    if name in ("dot", "dot_product", "negated_dot"):
        return make_negated_dot_distance(Z)
    if name == "bhattacharyya":
        return make_bhattacharyya_distance(Z, k=10, fallback_cov=np.diag(eigenvalues))
    if name in ("rbf", "kernel", "rbf_kernel"):
        return make_rbf_kernel_distance(Z)
    raise ValueError(f"unknown Z metric: {name}")


METRICS = ["mahalanobis", "euclidean", "cosine", "dot", "bhattacharyya", "rbf"]
