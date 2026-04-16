"""Tests for drift detection methods.

Core guarantees:
- Drift scores are in [0, 1].
- p-values (where applicable) are in [0, 1].
- Identical distributions → low drift.
- Shifted distributions → high drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from drift_lens.detector import DriftDetector, DriftResult


class TestDriftDetectorInterface:
    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            DriftDetector(method="invalid")

    def test_dimension_mismatch_raises(self) -> None:
        det = DriftDetector(method="frechet")
        a = np.ones((10, 32), dtype=np.float32)
        b = np.ones((10, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="Embedding dimension"):
            det.compare(a, b)


class TestFrechet:
    def test_score_in_range(
        self, baseline_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="frechet")
        result = det.compare(baseline_embeddings, shifted_embeddings)
        assert 0.0 <= result.drift_score <= 1.0

    def test_no_pvalue(
        self, baseline_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="frechet")
        result = det.compare(baseline_embeddings, shifted_embeddings)
        assert result.p_value is None

    def test_same_distribution_low_drift(
        self, baseline_embeddings: np.ndarray, same_distribution_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="frechet")
        result = det.compare(baseline_embeddings, same_distribution_embeddings)
        assert result.drift_score < 0.5

    def test_shifted_high_drift(
        self, baseline_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="frechet")
        result = det.compare(baseline_embeddings, shifted_embeddings)
        assert result.drift_score > 0.5
        assert result.is_drift


class TestMMD:
    def test_score_in_range(
        self, baseline_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="mmd")
        result = det.compare(baseline_embeddings, shifted_embeddings)
        assert 0.0 <= result.drift_score <= 1.0

    def test_pvalue_in_range(
        self, baseline_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="mmd")
        result = det.compare(baseline_embeddings, shifted_embeddings)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_shifted_low_pvalue(
        self, baseline_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="mmd")
        result = det.compare(baseline_embeddings, shifted_embeddings)
        assert result.p_value < 0.05

    def test_same_distribution_low_drift(
        self, baseline_embeddings: np.ndarray, same_distribution_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="mmd")
        result = det.compare(baseline_embeddings, same_distribution_embeddings)
        assert result.drift_score < 0.5


class TestTopology:
    def test_score_in_range(
        self, baseline_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        det = DriftDetector(method="topology")
        result = det.compare(baseline_embeddings, shifted_embeddings)
        assert 0.0 <= result.drift_score <= 1.0

    def test_structural_change_detected(
        self,
        baseline_embeddings: np.ndarray,
        different_structure_embeddings: np.ndarray,
    ) -> None:
        det = DriftDetector(method="topology")
        result = det.compare(baseline_embeddings, different_structure_embeddings)
        assert result.drift_score > 0.0
        assert result.details.get("total_wasserstein", 0) > 0


class TestDriftResult:
    def test_dataclass_fields(self) -> None:
        r = DriftResult(
            drift_score=0.5,
            is_drift=True,
            p_value=0.01,
            method="frechet",
            details={"raw_distance": 50.0},
        )
        assert r.drift_score == 0.5
        assert r.is_drift is True
        assert r.p_value == 0.01
        assert r.method == "frechet"
