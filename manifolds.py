import numpy as np
from typing import Callable, Tuple
from sklearn.decomposition import PCA


def make_pca_projector(d: int) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a pure function mapping X -> Z.
    Produces the structural manifold Z and its diagonal inverse covariance matrix.
    """

    def projector(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actual_d = min(d, X.shape[0], X.shape[1])
        pca = PCA(n_components=actual_d)
        Z = pca.fit_transform(X)

        # \Sigma_Z is diagonal with eigenvalues due to PCA feature orthogonality
        eigenvalues = pca.explained_variance_
        inv_cov = np.diag(1.0 / (eigenvalues + 1e-8))  # 1e-8 prevents division by zero
        return Z, inv_cov

    return projector
