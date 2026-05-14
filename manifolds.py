"""
Structural manifold projector — PCA from X (R^D) to Z (R^d).

Returns Z plus the artifacts needed for downstream metrics:
  - inv_cov: Σ_Z^{-1} (diagonal of 1/λ_k) for Mahalanobis.
  - eigenvalues: λ_k descending, for Bhattacharyya fallback covariance.
  - components: PCA basis (d × D), for projecting future queries.
  - mean: training-set mean used to center new points.

Backwards compatibility: callers that unpacked only `(Z, inv_cov)` still work
because we keep that as the first two return values.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA


@dataclass
class StructuralFit:
    Z: np.ndarray         # (N, d)
    inv_cov: np.ndarray   # (d, d) diagonal
    eigenvalues: np.ndarray
    components: np.ndarray  # (d, D)
    mean: np.ndarray      # (D,)

    def project(self, X_new: np.ndarray) -> np.ndarray:
        """Project a new high-dim point/array into Z-space using the fit basis."""
        if X_new.ndim == 1:
            return (X_new - self.mean) @ self.components.T
        return (X_new - self.mean) @ self.components.T


def fit_pca(X: np.ndarray, d: int) -> StructuralFit:
    """Fit PCA and return all the structural artifacts."""
    actual_d = max(1, min(d, X.shape[0], X.shape[1]))
    pca = PCA(n_components=actual_d)
    Z = pca.fit_transform(X)
    eigenvalues = pca.explained_variance_
    inv_cov = np.diag(1.0 / (eigenvalues + 1e-8))
    return StructuralFit(
        Z=Z.astype(np.float64),
        inv_cov=inv_cov,
        eigenvalues=eigenvalues.astype(np.float64),
        components=pca.components_.astype(np.float64),
        mean=pca.mean_.astype(np.float64),
    )


# Legacy API still used by main.py demo.
def make_pca_projector(d: int):
    def projector(X: np.ndarray):
        fit = fit_pca(X, d)
        return fit.Z, fit.inv_cov
    return projector
