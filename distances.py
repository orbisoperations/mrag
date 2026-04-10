import numpy as np
from typing import Callable


def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Eq 3: High-Dimensional Semantic Distance."""
    n1, n2 = np.linalg.norm(x1), np.linalg.norm(x2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(x1, x2) / (n1 * n2))


def make_mahalanobis_distance(
    inv_cov: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Eq 4: Low-Dimensional Structural Distance using a closure to carry the whitewashed space."""

    def distance(z1: np.ndarray, z2: np.ndarray) -> float:
        diff = z1 - z2
        # np.clip avoids math domain errors from microscopic negative floating-point imprecision
        return float(np.sqrt(np.clip(np.dot(np.dot(diff.T, inv_cov), diff), 0, None)))

    return distance
