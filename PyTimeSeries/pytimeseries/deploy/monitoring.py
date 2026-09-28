"""Input-drift monitoring (Part 12): compare incoming data to the training
reference distribution."""
from __future__ import annotations

import numpy as np

from ..utils.validation import check_y


class DriftMonitor:
    """Flag when recent data drifts from a training reference.

    ``fit`` stores the reference mean/std; ``score`` returns a window's mean
    distance from it in reference-σ units; ``drift`` returns a boolean Series
    over a rolling window; ``ks_pvalue`` offers a distributional check.
    """

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def fit(self, reference):
        r = check_y(reference)
        self.ref_ = r.values.astype(float)
        self.ref_mu_ = float(r.mean())
        self.ref_sd_ = float(r.std()) or 1.0
        return self

    def score(self, window) -> float:
        w = check_y(window)
        return float((w.mean() - self.ref_mu_) / self.ref_sd_)

    def drift(self, series, window: int = 12):
        s = check_y(series)
        z = (s.rolling(window).mean() - self.ref_mu_) / self.ref_sd_
        return z.abs() > self.threshold

    def ks_pvalue(self, window) -> float:
        from scipy.stats import ks_2samp

        return float(ks_2samp(self.ref_, check_y(window).values).pvalue)
