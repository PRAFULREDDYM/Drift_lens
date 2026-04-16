"""
Unified drift detection interface.

Usage::

    detector = DriftDetector(method="frechet")  # or "mmd" or "topology"
    result = detector.compare(baseline_embeddings, current_embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from drift_lens.detector.frechet import frechet_compare
from drift_lens.detector.mmd import mmd_compare
from drift_lens.detector.topology import topology_compare
from drift_lens.logger import _to_numpy


@dataclass
class DriftResult:
    """
    Standardised output from any drift detection method.

    Attributes
    ----------
    drift_score : float
        Normalised score in [0, 1]. Higher means more drift.
    is_drift : bool
        Whether the score exceeds the detection threshold.
    p_value : float or None
        Statistical p-value where applicable (MMD permutation test).
    method : str
        Name of the detection method that produced this result.
    details : dict
        Method-specific diagnostics (raw distances, permutation stats, etc.).
    """

    drift_score: float
    is_drift: bool
    p_value: float | None
    method: str
    details: dict[str, Any] = field(default_factory=dict)


_METHOD_DISPATCH = {
    "frechet": frechet_compare,
    "mmd": mmd_compare,
    "topology": topology_compare,
}


class DriftDetector:
    """
    Unified interface for embedding drift detection.

    Parameters
    ----------
    method : str
        Detection method: ``"frechet"``, ``"mmd"``, or ``"topology"``.
    threshold : float
        Drift score above which ``is_drift`` is ``True``.  Default 0.3.
    """

    VALID_METHODS = tuple(_METHOD_DISPATCH.keys())

    def __init__(
        self,
        method: Literal["frechet", "mmd", "topology"] = "frechet",
        threshold: float = 0.3,
    ) -> None:
        if method not in _METHOD_DISPATCH:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {list(_METHOD_DISPATCH.keys())}"
            )
        self.method = method
        self.threshold = threshold
        self._compare_fn = _METHOD_DISPATCH[method]

    def compare(self, baseline: Any, current: Any) -> DriftResult:
        """
        Compare two embedding sets and return a drift result.

        Parameters
        ----------
        baseline : array-like
            Reference embedding set ``(n, d)``.
        current : array-like
            Current embedding set ``(m, d)``. Dimension ``d`` must match baseline.

        Returns
        -------
        DriftResult
            Normalised drift score, boolean flag, and diagnostics.

        Raises
        ------
        ValueError
            If embedding dimensions don't match.
        """
        baseline_arr = _to_numpy(baseline)
        current_arr = _to_numpy(current)

        if baseline_arr.shape[1] != current_arr.shape[1]:
            raise ValueError(
                f"Baseline embeddings have shape {baseline_arr.shape} but current "
                f"has shape {current_arr.shape}. Embedding dimension (axis 1) must match."
            )

        score, p_value, details = self._compare_fn(baseline_arr, current_arr)
        score = float(np.clip(score, 0.0, 1.0))

        return DriftResult(
            drift_score=score,
            is_drift=score >= self.threshold,
            p_value=p_value,
            method=self.method,
            details=details,
        )
