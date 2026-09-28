"""Retraining policy (Part 12): scheduled and/or triggered refits."""
from __future__ import annotations


class RetrainPolicy:
    """Decide whether to retrain.

    Parameters
    ----------
    every : int, optional
        Retrain when ``step`` is a positive multiple of ``every`` (scheduled).
    drift_monitor : DriftMonitor, optional
        Retrain when a supplied ``window`` breaches ``drift_threshold``.
    drift_threshold : float
        σ-distance threshold for the drift trigger.
    error_threshold : float, optional
        Retrain when a supplied ``recent_error`` exceeds this.
    """

    def __init__(self, every: int | None = None, drift_monitor=None,
                 drift_threshold: float = 2.0, error_threshold: float | None = None):
        self.every = every
        self.drift_monitor = drift_monitor
        self.drift_threshold = drift_threshold
        self.error_threshold = error_threshold

    def should_retrain(self, step: int | None = None, window=None,
                       recent_error: float | None = None) -> bool:
        if self.every and step is not None and step > 0 and step % self.every == 0:
            return True
        if self.drift_monitor is not None and window is not None:
            if abs(self.drift_monitor.score(window)) > self.drift_threshold:
                return True
        if self.error_threshold is not None and recent_error is not None:
            if recent_error > self.error_threshold:
                return True
        return False
