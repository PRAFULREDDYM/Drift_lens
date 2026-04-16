"""
DriftProjector — dimensionality reduction for visualisation.

Wraps UMAP and PCA with a fit-on-baseline / transform-current pattern to
prevent data leakage from the current distribution into the projection.

Usage::

    projector = DriftProjector(method="umap", n_components=2)
    projector.fit(baseline_embeddings)
    projected_baseline = projector.transform(baseline_embeddings)
    projected_current = projector.transform(current_embeddings)
    projector.save("./projection_cache.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.decomposition import PCA

from drift_lens.constants import (
    PCA_WHITEN,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)
from drift_lens.logger import _to_numpy


class DriftProjector:
    """
    Dimensionality reduction fitted on baseline embeddings.

    Parameters
    ----------
    method : str
        ``"umap"`` or ``"pca"``.
    n_components : int
        Target dimensionality (typically 2 for scatter plots).
    """

    VALID_METHODS = ("umap", "pca")

    def __init__(
        self,
        method: Literal["umap", "pca"] = "umap",
        n_components: int = 2,
    ) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown projection method '{method}'. Choose from: {list(self.VALID_METHODS)}"
            )
        self.method = method
        self.n_components = n_components
        self._reducer: Any = None
        self._is_fitted = False

    def fit(self, embeddings: Any) -> "DriftProjector":
        """
        Fit the reducer on baseline embeddings.

        Parameters
        ----------
        embeddings : array-like
            Baseline embeddings ``(n, d)``.

        Returns
        -------
        self
        """
        arr = _to_numpy(embeddings)

        if self.method == "umap":
            import umap

            self._reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                metric=UMAP_METRIC,
                random_state=42,
            )
        else:
            self._reducer = PCA(
                n_components=self.n_components,
                whiten=PCA_WHITEN,
                random_state=42,
            )

        self._reducer.fit(arr)
        self._is_fitted = True
        return self

    def transform(self, embeddings: Any) -> np.ndarray:
        """
        Project embeddings using the fitted reducer.

        Parameters
        ----------
        embeddings : array-like
            Embeddings to project ``(m, d)``.

        Returns
        -------
        np.ndarray
            Projected coordinates ``(m, n_components)``.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Projector has not been fitted. Call .fit(baseline_embeddings) first."
            )
        arr = _to_numpy(embeddings)
        return self._reducer.transform(arr)

    def fit_transform(self, embeddings: Any) -> np.ndarray:
        """
        Fit on *embeddings* and return their projections.

        Parameters
        ----------
        embeddings : array-like
            Baseline embeddings ``(n, d)``.

        Returns
        -------
        np.ndarray
            Projected coordinates ``(n, n_components)``.
        """
        self.fit(embeddings)
        return self.transform(embeddings)

    def save(self, path: str | Path) -> Path:
        """
        Persist the fitted reducer to disk (pickle).

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            The written file path.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted projector.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "method": self.method,
                    "n_components": self.n_components,
                    "reducer": self._reducer,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DriftProjector":
        """
        Load a previously saved projector from disk.

        Parameters
        ----------
        path : str or Path
            Path to saved projector pickle.

        Returns
        -------
        DriftProjector
            Restored, fitted projector.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        proj = cls(method=data["method"], n_components=data["n_components"])
        proj._reducer = data["reducer"]
        proj._is_fitted = True
        return proj
