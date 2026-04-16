"""Shared fixtures for drift-lens tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def baseline_embeddings(rng: np.random.Generator) -> np.ndarray:
    """200 embeddings of dim 64 from a single Gaussian."""
    return rng.normal(0, 1, size=(200, 64)).astype(np.float32)


@pytest.fixture
def same_distribution_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Another draw from the same distribution — should show low drift."""
    return rng.normal(0, 1, size=(200, 64)).astype(np.float32)


@pytest.fixture
def shifted_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Embeddings shifted by a large mean offset — should show high drift."""
    return rng.normal(5, 1, size=(200, 64)).astype(np.float32)


@pytest.fixture
def different_structure_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Two tight clusters instead of one blob — structural change."""
    c1 = rng.normal(-3, 0.3, size=(100, 64)).astype(np.float32)
    c2 = rng.normal(3, 0.3, size=(100, 64)).astype(np.float32)
    return np.vstack([c1, c2])
