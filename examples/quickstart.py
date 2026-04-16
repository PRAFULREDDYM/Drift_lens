"""
drift-lens quickstart — generate synthetic embeddings that demonstrate
the core value proposition: drift fires days BEFORE accuracy drops.

Scenario
--------
- Days 1-7:   3 tight, stable clusters — high accuracy, low drift.
- Days 8-11:  Clusters start shifting — drift score rises, accuracy unchanged.
- Days 12-14: New cluster appears, old cluster splits — accuracy collapses.

The point: drift-lens fires an alert on day 8.
           Accuracy doesn't drop until day 12.
           That's a 4-day early warning.

Run this script to generate demo data that the dashboard loads automatically::

    python examples/quickstart.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift_lens.logger import EmbeddingLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_DIM = 512
SAMPLES_PER_DAY = 200
NUM_DAYS = 14
DEMO_DATA_DIR = Path(__file__).resolve().parent / "demo_data"
SEED = 42


def _make_cluster(
    rng: np.random.Generator,
    center: np.ndarray,
    n: int,
    spread: float = 0.3,
) -> np.ndarray:
    """Generate *n* points around *center* with Gaussian noise."""
    return center + rng.normal(0, spread, size=(n, center.shape[0])).astype(np.float32)


def generate_demo_data() -> dict:
    """
    Generate 14 days of synthetic 512-D embeddings with progressive drift.

    Returns
    -------
    dict
        Keys: ``snapshots`` (list of arrays), ``drift_scores`` (list of floats),
        ``accuracy`` (list of floats), ``days`` (list of ints).
    """
    rng = np.random.default_rng(SEED)

    # Three stable cluster centers in 512-D space
    c1 = rng.normal(0, 1, EMBED_DIM).astype(np.float32)
    c2 = rng.normal(3, 1, EMBED_DIM).astype(np.float32)
    c3 = rng.normal(-3, 1, EMBED_DIM).astype(np.float32)

    # Fixed drift direction so shifts accumulate coherently across days
    # (regenerating per-day would average out in high dimensions).
    drift_dir = rng.normal(0, 1, EMBED_DIM).astype(np.float32)
    drift_dir = drift_dir / np.linalg.norm(drift_dir)

    snapshots: list[np.ndarray] = []
    drift_scores: list[float] = []
    accuracy: list[float] = []

    for day in range(1, NUM_DAYS + 1):
        n_per_cluster = SAMPLES_PER_DAY // 3

        if day <= 7:
            # Phase 1: Stable clusters
            shift = 0.0
            spread = 0.3
            extra_cluster = False
            acc = 0.94 + rng.normal(0, 0.005)
        elif day <= 11:
            # Phase 2: Gradual shift — clusters drift, but accuracy holds.
            # Effective mean shift ≈ 0.4*shift (averaged across 3 clusters).
            # Fréchet score = 1-exp(-raw/100); raw ≈ (0.4*shift)² + trace.
            # Need shift ≈ 12-20 so days 8-11 span scores 0.3-0.6.
            shift = (day - 7) * 8.0
            spread = 0.3 + (day - 7) * 0.15
            extra_cluster = False
            acc = 0.93 + rng.normal(0, 0.008)
        else:
            # Phase 3: Structural break — new cluster, old splits, accuracy tanks
            shift = 32.0 + (day - 11) * 5.0
            spread = 0.8 + (day - 11) * 0.2
            extra_cluster = True
            acc = 0.93 - (day - 11) * 0.08 + rng.normal(0, 0.01)

        drift_vec = drift_dir * shift

        pts1 = _make_cluster(rng, c1 + drift_vec * 0.5, n_per_cluster, spread)
        pts2 = _make_cluster(rng, c2 + drift_vec, n_per_cluster, spread)
        pts3 = _make_cluster(rng, c3 - drift_vec * 0.3, n_per_cluster, spread)

        day_embeddings = np.vstack([pts1, pts2, pts3])

        if extra_cluster:
            c4 = rng.normal(5, 1, EMBED_DIM).astype(np.float32)
            pts4 = _make_cluster(rng, c4, n_per_cluster // 2, spread * 0.8)
            day_embeddings = np.vstack([day_embeddings, pts4])

        snapshots.append(day_embeddings)
        accuracy.append(float(np.clip(acc, 0.0, 1.0)))

    # Pre-compute drift scores (Fréchet) against day-1 baseline
    from drift_lens.detector import DriftDetector

    detector = DriftDetector(method="frechet")
    baseline = snapshots[0]
    for snap in snapshots:
        result = detector.compare(baseline, snap)
        drift_scores.append(result.drift_score)

    return {
        "snapshots": snapshots,
        "drift_scores": drift_scores,
        "accuracy": accuracy,
        "days": list(range(1, NUM_DAYS + 1)),
    }


def save_demo_data(data: dict) -> Path:
    """
    Persist demo data to disk using EmbeddingLogger + a summary JSON.

    Parameters
    ----------
    data : dict
        Output of :func:`generate_demo_data`.

    Returns
    -------
    Path
        The demo data directory.
    """
    if DEMO_DATA_DIR.exists():
        import shutil
        shutil.rmtree(DEMO_DATA_DIR)

    snapshot_dir = DEMO_DATA_DIR / "snapshots"
    logger = EmbeddingLogger(path=str(snapshot_dir), window="1d")

    base_ts = 1_700_000_000.0  # ~2023-11-14, arbitrary fixed anchor
    day_seconds = 86400

    for i, (snap, day) in enumerate(zip(data["snapshots"], data["days"])):
        ts = base_ts + (day - 1) * day_seconds
        logger.log(
            embeddings=snap,
            metadata={"day": day, "source": "synthetic_demo"},
            timestamp=ts,
        )

    # Save pre-computed timeline so the dashboard loads instantly
    timeline = {
        "days": data["days"],
        "drift_scores": data["drift_scores"],
        "accuracy": data["accuracy"],
    }
    timeline_path = DEMO_DATA_DIR / "timeline.json"
    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)

    print(f"Demo data saved to {DEMO_DATA_DIR}")
    print(f"  Snapshots: {len(data['snapshots'])} days")
    print(f"  Timeline:  {timeline_path}")
    return DEMO_DATA_DIR


def main() -> None:
    """Generate and save demo data."""
    print("Generating synthetic embeddings (512-D, 14 days) ...")
    data = generate_demo_data()

    print("\nDrift scores by day:")
    for day, score, acc in zip(data["days"], data["drift_scores"], data["accuracy"]):
        bar = "#" * int(score * 40)
        flag = " << ALERT" if score >= 0.3 and day >= 8 else ""
        print(f"  Day {day:2d}: drift={score:.3f} acc={acc:.3f} |{bar}{flag}")

    save_demo_data(data)
    print("\nDone! Run the dashboard:")
    print("  streamlit run drift_lens/dashboard.py")


if __name__ == "__main__":
    main()
