"""
AlertEngine — threshold-based alerting with cooldowns and audit trail.

Usage::

    alert_engine = AlertEngine(threshold=0.3, cooldown_hours=6)
    alert = alert_engine.check(drift_result)
    # alert.fired, alert.severity, alert.message, alert.recommended_action
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from drift_lens.constants import (
    ALERTS_FILENAME,
    DEFAULT_COOLDOWN_HOURS,
    SEVERITY_THRESHOLDS,
)
from drift_lens.detector import DriftResult


_RECOMMENDED_ACTIONS: dict[str, str] = {
    "low": "Continue monitoring. No action required.",
    "medium": (
        "Review recent data pipeline changes. "
        "Consider re-evaluating model on a fresh sample."
    ),
    "high": (
        "Significant embedding drift detected. "
        "Run evaluation on current production data and compare to baseline metrics."
    ),
    "critical": (
        "Critical drift — embeddings have shifted substantially. "
        "1) Halt auto-decisions if safety-critical. "
        "2) Re-evaluate model immediately. "
        "3) Consider fine-tuning or retraining on recent data."
    ),
}


def _severity_for_score(score: float) -> Literal["low", "medium", "high", "critical"]:
    """Map a drift score in [0, 1] to a severity label."""
    if score >= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    if score >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    if score >= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    return "low"


@dataclass
class Alert:
    """
    Represents a single alert event.

    Attributes
    ----------
    fired : bool
        Whether the alert actually fired (may be suppressed by cooldown).
    severity : str
        One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
    drift_score : float
        The drift score that triggered the alert.
    method : str
        Detection method name.
    message : str
        Human-readable alert summary.
    recommended_action : str
        Suggested next steps for the operator.
    timestamp : float
        Unix timestamp when the alert was evaluated.
    suppressed_by_cooldown : bool
        ``True`` if the alert would have fired but was within cooldown.
    """

    fired: bool
    severity: str
    drift_score: float
    method: str
    message: str
    recommended_action: str
    timestamp: float = field(default_factory=time.time)
    suppressed_by_cooldown: bool = False


class AlertEngine:
    """
    Evaluates drift results against thresholds and manages cooldowns.

    Parameters
    ----------
    threshold : float
        Drift score above which alerts fire. Default 0.3.
    cooldown_hours : float
        Minimum hours between successive alerts. Default 6.
    alert_dir : str or Path or None
        Directory for ``alerts.jsonl`` audit trail. ``None`` disables persistence.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
        alert_dir: str | Path | None = None,
    ) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_hours * 3600.0
        self._last_fire_time: float | None = None
        self._alert_dir: Path | None = None
        if alert_dir is not None:
            self._alert_dir = Path(alert_dir)
            self._alert_dir.mkdir(parents=True, exist_ok=True)

    def check(self, result: DriftResult) -> Alert:
        """
        Evaluate a drift result and optionally fire an alert.

        Parameters
        ----------
        result : DriftResult
            Output of ``DriftDetector.compare()``.

        Returns
        -------
        Alert
            Alert object with firing status, severity, and recommendations.
        """
        severity = _severity_for_score(result.drift_score)
        should_fire = result.drift_score >= self.threshold
        now = time.time()

        suppressed = False
        if should_fire and self._last_fire_time is not None:
            elapsed = now - self._last_fire_time
            if elapsed < self.cooldown_seconds:
                suppressed = True
                should_fire = False

        if should_fire:
            self._last_fire_time = now

        message = (
            f"Drift detected: score={result.drift_score:.3f} "
            f"({severity}) via {result.method}."
            if should_fire
            else f"Drift score={result.drift_score:.3f} ({severity}) — "
            + ("suppressed by cooldown." if suppressed else "below threshold.")
        )

        alert = Alert(
            fired=should_fire,
            severity=severity,
            drift_score=result.drift_score,
            method=result.method,
            message=message,
            recommended_action=_RECOMMENDED_ACTIONS[severity],
            timestamp=now,
            suppressed_by_cooldown=suppressed,
        )

        self._persist(alert)
        return alert

    def _persist(self, alert: Alert) -> None:
        """Append alert to the JSONL audit trail if a directory is configured."""
        if self._alert_dir is None:
            return
        filepath = self._alert_dir / ALERTS_FILENAME
        with open(filepath, "a") as f:
            f.write(json.dumps(asdict(alert)) + "\n")

    def load_history(self) -> list[dict]:
        """
        Load the full alert history from the JSONL audit trail.

        Returns
        -------
        list[dict]
            List of alert records, oldest first.
        """
        if self._alert_dir is None:
            return []
        filepath = self._alert_dir / ALERTS_FILENAME
        if not filepath.exists():
            return []
        records = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
