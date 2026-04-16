"""
Maximum Mean Discrepancy (MMD) drift detection.

MMD is a kernel-based two-sample test that measures the distance between two
probability distributions in a reproducing kernel Hilbert space (RKHS).
Unlike Fréchet distance, MMD makes NO Gaussian assumption and can detect
non-linear distribution shifts.

We use:
- RBF (Gaussian) kernel with **median heuristic** bandwidth selection
- Permutation test for p-value estimation
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances

from drift_lens.constants import MMD_NUM_PERMUTATIONS, MMD_SCALE


def _median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Median heuristic for RBF kernel bandwidth.

    σ = median of all pairwise distances in the combined sample.
    This is a widely-used, data-adaptive bandwidth that avoids manual tuning.
    """
    combined = np.vstack([X, Y])
    n = min(combined.shape[0], 1000)  # subsample for speed if large
    if combined.shape[0] > n:
        idx = np.random.default_rng(42).choice(combined.shape[0], n, replace=False)
        combined = combined[idx]
    dists = pairwise_distances(combined, metric="euclidean")
    # Use upper triangle only (no diagonal zeros)
    triu_idx = np.triu_indices_from(dists, k=1)
    median_dist = float(np.median(dists[triu_idx]))
    return max(median_dist, 1e-8)  # guard against zero


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """
    RBF (Gaussian) kernel: k(x, y) = exp(-||x - y||² / (2σ²)).
    """
    sq_dists = pairwise_distances(X, Y, metric="sqeuclidean")
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def _mmd_squared(
    K_XX: np.ndarray, K_XY: np.ndarray, K_YY: np.ndarray
) -> float:
    """
    Unbiased estimator of MMD²:
        MMD² = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]

    Uses the unbiased estimator that excludes diagonal terms (i ≠ j).
    """
    n = K_XX.shape[0]
    m = K_YY.shape[0]

    # Sum of off-diagonal entries
    sum_xx = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) if n > 1 else 0.0
    sum_yy = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) if m > 1 else 0.0
    sum_xy = K_XY.sum() / (n * m)

    return float(sum_xx - 2.0 * sum_xy + sum_yy)


def _permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
    observed_mmd: float,
    n_permutations: int,
) -> float:
    """
    Permutation test for MMD significance.

    Under H₀ (X and Y come from the same distribution), randomly shuffling
    the combined sample between the two groups should produce MMD values
    comparable to the observed one.  The p-value is the fraction of
    permuted MMDs that exceed the observed value.
    """
    rng = np.random.default_rng(42)
    combined = np.vstack([X, Y])
    n = X.shape[0]
    total = combined.shape[0]
    count_ge = 0

    for _ in range(n_permutations):
        perm = rng.permutation(total)
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]

        K_XX = _rbf_kernel(X_perm, X_perm, sigma)
        K_XY = _rbf_kernel(X_perm, Y_perm, sigma)
        K_YY = _rbf_kernel(Y_perm, Y_perm, sigma)

        mmd_perm = _mmd_squared(K_XX, K_XY, K_YY)
        if mmd_perm >= observed_mmd:
            count_ge += 1

    return (count_ge + 1) / (n_permutations + 1)  # +1 for continuity correction


def mmd_compare(
    baseline: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float | None, dict[str, Any]]:
    """
    Compare two embedding sets using Maximum Mean Discrepancy with
    RBF kernel and permutation-based p-value.

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
    p_value : float
        Permutation test p-value (small = significant drift).
    details : dict
        Raw MMD², bandwidth σ, permutation stats.
    """
    sigma = _median_bandwidth(baseline, current)

    K_XX = _rbf_kernel(baseline, baseline, sigma)
    K_XY = _rbf_kernel(baseline, current, sigma)
    K_YY = _rbf_kernel(current, current, sigma)

    raw_mmd = _mmd_squared(K_XX, K_XY, K_YY)
    # MMD² can be slightly negative due to the unbiased estimator; clip to 0
    raw_mmd = max(raw_mmd, 0.0)

    p_value = _permutation_test(
        baseline, current, sigma, raw_mmd, MMD_NUM_PERMUTATIONS
    )

    # Map to [0, 1] via exponential saturation
    score = 1.0 - np.exp(-raw_mmd / MMD_SCALE)

    details = {
        "raw_mmd_squared": raw_mmd,
        "bandwidth_sigma": sigma,
        "n_permutations": MMD_NUM_PERMUTATIONS,
        "p_value": p_value,
        "baseline_samples": baseline.shape[0],
        "current_samples": current.shape[0],
    }

    return float(score), float(p_value), details
