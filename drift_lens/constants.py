"""
Constants for drift-lens.

All magic numbers and configuration defaults live here with explanations.
"""

# ---------------------------------------------------------------------------
# Time window parsing — maps human-readable window strings to seconds
# ---------------------------------------------------------------------------
WINDOW_SECONDS: dict[str, int] = {
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
    "7d": 604800,
}

# ---------------------------------------------------------------------------
# Drift score normalisation
# ---------------------------------------------------------------------------
# Fréchet distances are unbounded. We map them to [0, 1] using:
#   score = 1 - exp(-raw_distance / FRECHET_SCALE)
# A scale of 100 means a raw FED of ~230 maps to score ≈ 0.9.
FRECHET_SCALE: float = 100.0

# MMD scores are also unbounded; same exponential mapping.
MMD_SCALE: float = 0.1

# Topology (Wasserstein distance on persistence diagrams) scale factor.
TOPOLOGY_SCALE: float = 5.0

# ---------------------------------------------------------------------------
# MMD permutation test
# ---------------------------------------------------------------------------
# Number of permutations for the bootstrap null distribution.
MMD_NUM_PERMUTATIONS: int = 200

# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------
# When computing matrix square roots for Fréchet distance, eigenvalues
# below this threshold are clipped to zero to avoid complex results from
# near-singular covariance matrices.
EIGENVALUE_CLIP_THRESHOLD: float = 1e-10

# Maximum condition number before we regularise the covariance matrix.
MAX_COVARIANCE_CONDITION: float = 1e10

# Regularisation added to diagonal of covariance when ill-conditioned.
COVARIANCE_REGULARISATION_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Topology subsampling
# ---------------------------------------------------------------------------
# Ripser is O(n³) in memory. If the embedding set is larger than this,
# we subsample before computing persistence diagrams.
TOPOLOGY_MAX_POINTS: int = 500

# Maximum homology dimension to compute (H0 = components, H1 = loops).
TOPOLOGY_MAX_DIM: int = 1

# ---------------------------------------------------------------------------
# Alert severity thresholds — drift_score → severity label
# ---------------------------------------------------------------------------
SEVERITY_THRESHOLDS: dict[str, float] = {
    "low": 0.0,
    "medium": 0.3,
    "high": 0.5,
    "critical": 0.7,
}

# Default cooldown between repeated alerts for the same drift event (hours).
DEFAULT_COOLDOWN_HOURS: float = 6.0

# ---------------------------------------------------------------------------
# Projector defaults
# ---------------------------------------------------------------------------
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_METRIC: str = "cosine"

PCA_WHITEN: bool = False

# ---------------------------------------------------------------------------
# Dashboard / clustering
# ---------------------------------------------------------------------------
# Range of k to search when auto-selecting KMeans k via silhouette score.
KMEANS_K_MIN: int = 2
KMEANS_K_MAX: int = 10

# ---------------------------------------------------------------------------
# File conventions
# ---------------------------------------------------------------------------
SNAPSHOT_FILE_PREFIX: str = "embeddings_"
SNAPSHOT_FILE_EXT: str = ".parquet"
ALERTS_FILENAME: str = "alerts.jsonl"
PROJECTION_CACHE_FILENAME: str = "projection_cache.pkl"
