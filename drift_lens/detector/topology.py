"""
Topological drift detection via persistent homology.

This is the defensible moat — no other production drift tool does this.

Persistent homology captures the *shape* of the embedding space:
- H₀ features = connected components (clusters)
- H₁ features = loops / holes

By comparing persistence diagrams between baseline and current embeddings
using the Wasserstein distance, we detect structural changes (cluster
merging, splitting, hole formation) that statistical tests miss.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drift_lens.constants import (
    TOPOLOGY_MAX_DIM,
    TOPOLOGY_MAX_POINTS,
    TOPOLOGY_SCALE,
)


def _subsample(X: np.ndarray, max_points: int) -> np.ndarray:
    """
    Random subsample if X has more than *max_points* rows.

    Ripser's Vietoris-Rips computation is O(n³) in memory, so we must
    cap the input size for practical use.
    """
    if X.shape[0] <= max_points:
        return X
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], max_points, replace=False)
    return X[idx]


def _persistence_diagram(X: np.ndarray, max_dim: int) -> list[np.ndarray]:
    """
    Compute persistence diagrams using Ripser.

    Returns a list of arrays, one per homology dimension.
    Each array has shape ``(n_features, 2)`` with columns ``[birth, death]``.
    """
    from ripser import ripser as ripser_fn

    result = ripser_fn(X, maxdim=max_dim, metric="euclidean")
    return result["dgms"]


def _wasserstein_distance_diagrams(
    dgm1: np.ndarray, dgm2: np.ndarray
) -> float:
    """
    Wasserstein-1 distance between two persistence diagrams.

    Uses the simple matching approach: for each point in dgm1, find the
    closest point in dgm2 (or the diagonal).  This is an approximation
    to the full optimal-transport Wasserstein distance but runs in O(n·m)
    instead of O(n³) and is sufficient for drift detection.
    """
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    # Cost of projecting a point onto the diagonal (birth=death)
    def _diag_cost(pt: np.ndarray) -> float:
        return float(abs(pt[1] - pt[0]) / 2.0)

    total = 0.0

    if len(dgm1) == 0:
        for pt in dgm2:
            total += _diag_cost(pt)
        return total

    if len(dgm2) == 0:
        for pt in dgm1:
            total += _diag_cost(pt)
        return total

    # Remove infinite-death features (they represent the single connected
    # component that persists forever and would dominate the distance).
    finite1 = dgm1[np.isfinite(dgm1[:, 1])]
    finite2 = dgm2[np.isfinite(dgm2[:, 1])]

    if len(finite1) == 0 and len(finite2) == 0:
        return 0.0

    # Greedy nearest-neighbour matching + diagonal projection for leftovers.
    # This is a practical O(n·m) approximation.
    matched2 = set()
    for pt1 in finite1:
        best_cost = _diag_cost(pt1)
        best_j = -1
        for j, pt2 in enumerate(finite2):
            if j in matched2:
                continue
            cost = float(np.sum(np.abs(pt1 - pt2)))
            if cost < best_cost:
                best_cost = cost
                best_j = j
        total += best_cost
        if best_j >= 0:
            matched2.add(best_j)

    # Unmatched points in dgm2 are projected to the diagonal
    for j, pt2 in enumerate(finite2):
        if j not in matched2:
            total += _diag_cost(pt2)

    return total


def topology_compare(
    baseline: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float | None, dict[str, Any]]:
    """
    Compare two embedding sets using persistent homology.

    Parameters
    ----------
    baseline : np.ndarray
        Reference embeddings, shape ``(n, d)``.
    current : np.ndarray
        Current embeddings, shape ``(m, d)``.

    Returns
    -------
    score : float
        Normalised drift score in [0, 1].
    p_value : None
        Topological comparison has no natural p-value.
    details : dict
        Per-dimension Wasserstein distances, feature counts, etc.
    """
    base_sub = _subsample(baseline, TOPOLOGY_MAX_POINTS)
    curr_sub = _subsample(current, TOPOLOGY_MAX_POINTS)

    dgms_base = _persistence_diagram(base_sub, TOPOLOGY_MAX_DIM)
    dgms_curr = _persistence_diagram(curr_sub, TOPOLOGY_MAX_DIM)

    dim_distances: dict[str, float] = {}
    total_distance = 0.0

    for dim in range(TOPOLOGY_MAX_DIM + 1):
        dgm1 = dgms_base[dim] if dim < len(dgms_base) else np.empty((0, 2))
        dgm2 = dgms_curr[dim] if dim < len(dgms_curr) else np.empty((0, 2))
        d = _wasserstein_distance_diagrams(dgm1, dgm2)
        label = f"H{dim}"
        dim_distances[label] = d
        total_distance += d

    # Normalise to [0, 1]
    score = 1.0 - np.exp(-total_distance / TOPOLOGY_SCALE)

    details = {
        "total_wasserstein": total_distance,
        "per_dimension": dim_distances,
        "scale_factor": TOPOLOGY_SCALE,
        "baseline_subsampled": base_sub.shape[0],
        "current_subsampled": curr_sub.shape[0],
        "baseline_H0_features": len(dgms_base[0]) if len(dgms_base) > 0 else 0,
        "current_H0_features": len(dgms_curr[0]) if len(dgms_curr) > 0 else 0,
        "baseline_H1_features": len(dgms_base[1]) if len(dgms_base) > 1 else 0,
        "current_H1_features": len(dgms_curr[1]) if len(dgms_curr) > 1 else 0,
    }

    return float(score), None, details
