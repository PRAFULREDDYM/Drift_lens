"""
drift-lens: Your metrics lie. Your embeddings don't.

Detect embedding space drift days before accuracy drops,
with zero infra changes.
"""

__version__ = "0.1.0"

from drift_lens.logger import EmbeddingLogger
from drift_lens.detector import DriftDetector, DriftResult
from drift_lens.projector import DriftProjector
from drift_lens.alert import AlertEngine, Alert

__all__ = [
    "EmbeddingLogger",
    "DriftDetector",
    "DriftResult",
    "DriftProjector",
    "AlertEngine",
    "Alert",
]
