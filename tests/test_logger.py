"""Tests for EmbeddingLogger."""

from __future__ import annotations

import numpy as np
import pytest

from drift_lens.logger import EmbeddingLogger, _to_numpy


class TestToNumpy:
    def test_numpy_passthrough(self) -> None:
        arr = np.ones((10, 32), dtype=np.float32)
        result = _to_numpy(arr)
        assert result.shape == (10, 32)
        assert result.dtype == np.float32

    def test_list_of_lists(self) -> None:
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = _to_numpy(data)
        assert result.shape == (2, 2)
        assert result.dtype == np.float32

    def test_1d_becomes_2d(self) -> None:
        arr = np.ones(32, dtype=np.float32)
        result = _to_numpy(arr)
        assert result.shape == (1, 32)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported embedding type"):
            _to_numpy("not an array")

    def test_3d_raises(self) -> None:
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="must be 2-D"):
            _to_numpy(arr)


class TestEmbeddingLogger:
    def test_log_and_load(self, tmp_path: object, baseline_embeddings: np.ndarray) -> None:
        logger = EmbeddingLogger(path=str(tmp_path), window="1d")
        path = logger.log(baseline_embeddings, metadata={"test": True})
        assert path is not None
        assert path.exists()

        loaded = logger.load_snapshot(path)
        assert loaded.shape == baseline_embeddings.shape
        np.testing.assert_allclose(loaded, baseline_embeddings, atol=1e-6)

    def test_dedup_skips_identical(self, tmp_path: object) -> None:
        arr = np.ones((5, 16), dtype=np.float32)
        logger = EmbeddingLogger(path=str(tmp_path), window="1d")
        first = logger.log(arr)
        second = logger.log(arr)
        assert first is not None
        assert second is None

    def test_list_snapshots(self, tmp_path: object) -> None:
        logger = EmbeddingLogger(path=str(tmp_path), window="1d")
        arr1 = np.ones((5, 16), dtype=np.float32)
        arr2 = np.zeros((5, 16), dtype=np.float32)
        logger.log(arr1, timestamp=1_000_000.0)
        logger.log(arr2, timestamp=1_100_000.0)
        files = logger.list_snapshots()
        assert len(files) == 2

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown window"):
            EmbeddingLogger(window="99h")
