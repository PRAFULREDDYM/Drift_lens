"""
Fréchet Embedding Distance (FED).

The same idea as FID (Fréchet Inception Distance) for images, generalised to
arbitrary embedding spaces:

    FED = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)

This assumes the embedding distributions are roughly Gaussian.  It is fast,
interpretable, and works well as a first-pass drift detector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg

from drift_lens.constants import (
    COVARIANCE_REGULARISATION_EPS,
    EIGENVALUE_CLIP_THRESHOLD,
    FRECHET_SCALE,
    MAX_COVARIANCE_CONDITION,
)


def _regularise_covariance(sigma: np.ndarray) -> np.ndarray:
    """
    Add small diagonal regularisation when the covariance matrix is
    near-singular, preventing numerical blow-up in the matrix square root.
    """
    cond = np.linalg.cond(sigma)
    if cond > MAX_COVARIANCE_CONDITION:
        sigma = sigma + COVARIANCE_REGULARISATION_EPS * np.eye(sigma.shape[0])
    return sigma


def _stable_sqrtm(matrix: np.ndarray) -> np.ndarray:
    """
    Numerically stable matrix square root.

    Uses ``scipy.linalg.sqrtm`` then clips any small negative eigenvalues
    that arise from floating-point error in near-singular matrices.
    """
    sqrtm_result = linalg.sqrtm(matrix)

    # sqrtm can return complex results when the input has tiny negative eigenvalues.
    if np.iscomplexobj(sqrtm_result):
        # Keep only the real part; imaginary components from numerical noise
        # are negligibly small (< EIGENVALUE_CLIP_THRESHOLD).
        imag_max = np.max(np.abs(sqrtm_result.imag))
        if imag_max > EIGENVALUE_CLIP_THRESHOLD:
            # Fall back to eigen-decomposition for robustness
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.clip(eigenvalues, EIGENVALUE_CLIP_THRESHOLD, None)
            sqrtm_result = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        else:
            sqrtm_result = sqrtm_result.real

    return sqrtm_result


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """
    Compute the Fréchet distance between two multivariate Gaussians.

    Parameters
    ----------
    mu1, mu2 : np.ndarray
        Mean vectors, shape ``(d,)``.
    sigma1, sigma2 : np.ndarray
        Covariance matrices, shape ``(d, d)``.

    Returns
    -------
    float
        Raw (unnormalised) Fréchet distance.
    """
    diff = mu1 - mu2
    mean_term = diff @ diff  # ||μ₁ - μ₂||²

    sigma1 = _regularise_covariance(sigma1)
    sigma2 = _regularise_covariance(sigma2)

    # (Σ₁ · Σ₂)^½
    product = sigma1 @ sigma2
    sqrt_product = _stable_sqrtm(product)

    # Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)
    trace_term = np.trace(sigma1 + sigma2 - 2.0 * sqrt_product)

    # Clip tiny negatives from numerical error
    trace_term = max(0.0, float(trace_term))

    return float(mean_term + trace_term)


def frechet_compare(
    baseline: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float | None, dict[str, Any]]:
    """
    Compare two embedding sets using Fréchet Embedding Distance.

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
        FED is a distance metric, not a statistical test — no p-value.
    details : dict
        Contains ``raw_distance`` and ``scale_factor``.
    """
    mu1 = np.mean(baseline, axis=0)
    mu2 = np.mean(current, axis=0)

    # rowvar=False → each column is a variable, each row is an observation
    sigma1 = np.cov(baseline, rowvar=False)
    sigma2 = np.cov(current, rowvar=False)

    # Guard against 1-D covariance collapse when n_samples is small
    if sigma1.ndim == 0:
        sigma1 = np.array([[float(sigma1)]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[float(sigma2)]])

    raw_distance = _frechet_distance(mu1, sigma1, mu2, sigma2)

    # Map unbounded distance to [0, 1] via exponential saturation:
    #   score = 1 - exp(-raw / scale)
    score = 1.0 - np.exp(-raw_distance / FRECHET_SCALE)

    details = {
        "raw_distance": raw_distance,
        "scale_factor": FRECHET_SCALE,
        "mean_shift": float(np.linalg.norm(mu1 - mu2)),
        "baseline_samples": baseline.shape[0],
        "current_samples": current.shape[0],
    }

    return float(score), None, details
