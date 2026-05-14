import numpy as np
from typing import Callable, Set, Dict, FrozenSet


def semantic_decay_traversal(
    q: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    dist_X_fn: Callable[[np.ndarray, np.ndarray], float],
    dist_Z_fn: Callable[[np.ndarray, np.ndarray], float],
    tau: float,
    epsilon: float,
    gamma: float,
    max_hops: int,
    initial_set: FrozenSet[int] | None = None,
) -> Dict[int, FrozenSet[int]]:
    """Strategy 1: Bounded Beam Search with Semantic Decay.

    If `initial_set` is provided, use it directly as S^(0) (top-K cosine anchors
    from the caller). Otherwise, fall back to the paper's strict cosine cutoff.
    """

    if initial_set is not None:
        s_0 = initial_set
    else:
        s_0 = frozenset(i for i, x in enumerate(X) if dist_X_fn(q, x) <= tau)

    # Pure recursive traversal function (no mutable state lists)
    def traverse(
        t: int, current_s: FrozenSet[int], all_visited: FrozenSet[int]
    ) -> Dict[int, FrozenSet[int]]:
        if t >= max_hops or not current_s:
            return {}

        next_s = set()
        for i in current_s:
            for j in range(len(Z)):
                if j not in all_visited:
                    is_structurally_close = dist_Z_fn(Z[i], Z[j]) <= epsilon

                    # NOTE: Paper mathematically bounds via: tau * (gamma ** t)
                    # For semantic drift to be allowed, gamma > 1.0 expands the threshold.
                    is_semantically_bounded = dist_X_fn(q, X[j]) <= tau * (
                        gamma ** (t + 1)
                    )

                    if is_structurally_close and is_semantically_bounded:
                        next_s.add(j)

        next_frozen = frozenset(next_s)
        result = {t + 1: next_frozen} if next_frozen else {}
        if next_frozen:
            result.update(traverse(t + 1, next_frozen, all_visited | next_frozen))
        return result

    res = {0: s_0}
    res.update(traverse(0, s_0, s_0))
    return res


def continuous_random_walk(
    q: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    dist_X_fn: Callable[[np.ndarray, np.ndarray], float],
    dist_Z_fn: Callable[[np.ndarray, np.ndarray], float],
    tau: float,
    alpha: float = 0.85,
    beta: float = 1.0,
    iterations: int = 10,
) -> np.ndarray:
    """Strategy 2: Continuous Random Walk with Restart (Pure Power Iteration)"""
    N = len(X)
    v_q = np.array([1.0 if dist_X_fn(q, X[i]) <= tau else 0.0 for i in range(N)])
    v_q = v_q / v_q.sum() if v_q.sum() > 0 else np.ones(N) / N

    A = np.array(
        [[np.exp(-beta * dist_Z_fn(Z[i], Z[j])) for j in range(N)] for i in range(N)]
    )
    row_sums = A.sum(axis=1, keepdims=True)
    P = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums != 0)

    def power_iteration(pi: np.ndarray, step: int) -> np.ndarray:
        if step >= iterations:
            return pi
        return power_iteration((1 - alpha) * v_q + alpha * (P.T @ pi), step + 1)

    return power_iteration(v_q, 0)
