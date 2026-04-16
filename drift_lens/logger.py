"""
EmbeddingLogger — snapshot embeddings to disk as parquet files, chunked by time window.

Usage::

    logger = EmbeddingLogger(path="./drift_lens_data", window="1d")
    logger.log(embeddings=np.array(...), metadata={"model": "minilm", "source": "prod"})
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from drift_lens.constants import (
    SNAPSHOT_FILE_EXT,
    SNAPSHOT_FILE_PREFIX,
    WINDOW_SECONDS,
)


def _to_numpy(embeddings: Any) -> np.ndarray:
    """
    Convert embeddings from various formats to a 2-D numpy float32 array.

    Parameters
    ----------
    embeddings : array-like
        Numpy array, torch Tensor, or list of lists.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, embedding_dim)`` with dtype float32.

    Raises
    ------
    TypeError
        If the input type is not supported.
    ValueError
        If the resulting array is not 2-D.
    """
    if isinstance(embeddings, np.ndarray):
        arr = embeddings.astype(np.float32, copy=False)
    elif hasattr(embeddings, "detach"):
        # torch.Tensor path — avoids hard dependency on torch
        arr = embeddings.detach().cpu().numpy().astype(np.float32)
    elif isinstance(embeddings, (list, tuple)):
        arr = np.asarray(embeddings, dtype=np.float32)
    else:
        raise TypeError(
            f"Unsupported embedding type: {type(embeddings).__name__}. "
            "Pass a numpy array, torch Tensor, or list of lists."
        )

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2-D (n_samples, embedding_dim), got shape {arr.shape}."
        )
    return arr


def _embedding_hash(arr: np.ndarray) -> str:
    """SHA-256 hex digest of the raw embedding bytes for deduplication."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _window_bucket(timestamp: float, window_seconds: int) -> str:
    """
    Return an ISO-style bucket label for the time window containing *timestamp*.

    Example: ``"2024-03-15T00"`` for a 1-hour window starting at midnight.
    """
    bucket_start = int(timestamp // window_seconds) * window_seconds
    dt = datetime.fromtimestamp(bucket_start, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


class EmbeddingLogger:
    """
    Logs embedding snapshots to disk as parquet files, one per time window.

    Parameters
    ----------
    path : str or Path
        Directory where snapshot files are written.
    window : str
        Time window granularity. One of ``"1h"``, ``"6h"``, ``"1d"``, ``"7d"``.
    """

    def __init__(self, path: str | Path = "./drift_lens_data", window: str = "1d") -> None:
        if window not in WINDOW_SECONDS:
            raise ValueError(
                f"Unknown window '{window}'. Choose from: {list(WINDOW_SECONDS.keys())}"
            )
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.window = window
        self._window_seconds = WINDOW_SECONDS[window]
        self._seen_hashes: set[str] = set()

    def log(
        self,
        embeddings: Any,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> Path | None:
        """
        Persist an embedding batch to the appropriate time-window parquet file.

        Parameters
        ----------
        embeddings : array-like
            ``(n_samples, embedding_dim)`` embeddings as numpy array,
            torch Tensor, or list of lists.
        metadata : dict, optional
            Arbitrary key-value metadata stored alongside each row.
        timestamp : float, optional
            Unix epoch timestamp. Defaults to ``time.time()``.

        Returns
        -------
        Path or None
            Path to the written parquet file, or ``None`` if the batch was
            a duplicate (SHA-256 dedup).
        """
        arr = _to_numpy(embeddings)
        batch_hash = _embedding_hash(arr)

        if batch_hash in self._seen_hashes:
            return None
        self._seen_hashes.add(batch_hash)

        ts = timestamp if timestamp is not None else time.time()
        bucket = _window_bucket(ts, self._window_seconds)
        filename = f"{SNAPSHOT_FILE_PREFIX}{bucket}{SNAPSHOT_FILE_EXT}"
        filepath = self.path / filename

        meta_json = json.dumps(metadata or {})
        rows = []
        for i in range(arr.shape[0]):
            rows.append(
                {
                    "embedding": arr[i].tobytes(),
                    "timestamp": ts,
                    "metadata": meta_json,
                }
            )

        schema = pa.schema(
            [
                ("embedding", pa.binary()),
                ("timestamp", pa.float64()),
                ("metadata", pa.string()),
            ]
        )
        table = pa.table(
            {
                "embedding": [r["embedding"] for r in rows],
                "timestamp": [r["timestamp"] for r in rows],
                "metadata": [r["metadata"] for r in rows],
            },
            schema=schema,
        )

        if filepath.exists():
            existing = pq.read_table(filepath)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, filepath)
        return filepath

    def load_snapshot(self, filepath: str | Path) -> np.ndarray:
        """
        Read a parquet snapshot back into a numpy array.

        Parameters
        ----------
        filepath : str or Path
            Path to a ``.parquet`` snapshot file.

        Returns
        -------
        np.ndarray
            ``(n_samples, embedding_dim)`` float32 array.
        """
        table = pq.read_table(str(filepath))
        binary_col = table.column("embedding").to_pylist()
        arrays = [np.frombuffer(b, dtype=np.float32) for b in binary_col]
        return np.stack(arrays)

    def list_snapshots(self) -> list[Path]:
        """
        List all snapshot files in the logger directory, sorted by name (time).

        Returns
        -------
        list[Path]
            Sorted list of parquet file paths.
        """
        return sorted(self.path.glob(f"{SNAPSHOT_FILE_PREFIX}*{SNAPSHOT_FILE_EXT}"))
