"""Croston's method and relatives for demand full of zeros (Part 11).

Standard models chase the zeros; these forecast a smooth demand *rate* by
modelling demand size and timing separately.
"""
from __future__ import annotations

import numpy as np

from ..base import BaseForecaster


class Croston(BaseForecaster):
    """Croston's classic method (SES on demand sizes and inter-demand intervals)."""

    _tags = {"scitype": "univariate", "probabilistic": False}

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def _fit(self, y, X):
        d = np.asarray(y.values, dtype=float)
        nz = np.flatnonzero(d)
        if len(nz) == 0:
            self._rate = 0.0
            return
        sizes = d[nz]
        intervals = np.diff(np.concatenate(([-1], nz)))     # gap since previous demand
        a = self.alpha
        z, x = sizes[0], float(intervals[0])
        for i in range(1, len(nz)):
            z = a * sizes[i] + (1 - a) * z
            x = a * intervals[i] + (1 - a) * x
        self._rate = float(z / x) if x else 0.0

    def _predict(self, h, X):
        return np.repeat(self._rate, h)


class TSB(BaseForecaster):
    """Teunter–Syntetos–Babai: updates the demand *probability* every period,
    so it handles demand that can go obsolete."""

    _tags = {"scitype": "univariate", "probabilistic": False}

    def __init__(self, alpha_d: float = 0.1, alpha_p: float = 0.1):
        super().__init__()
        self.alpha_d = alpha_d
        self.alpha_p = alpha_p

    def _fit(self, y, X):
        d = np.asarray(y.values, dtype=float)
        nz = np.flatnonzero(d)
        z = float(d[nz].mean()) if len(nz) else 0.0
        p = float((d > 0).mean())
        for val in d:
            if val > 0:
                z = self.alpha_d * val + (1 - self.alpha_d) * z
                p = self.alpha_p * 1.0 + (1 - self.alpha_p) * p
            else:
                p = self.alpha_p * 0.0 + (1 - self.alpha_p) * p
        self._rate = p * z

    def _predict(self, h, X):
        return np.repeat(self._rate, h)


class ADIDA(BaseForecaster):
    """Aggregate-Disaggregate Intermittent Demand Approach: temporally aggregate
    to reduce intermittence, forecast the aggregate by SES, then disaggregate."""

    _tags = {"scitype": "univariate", "probabilistic": False}

    def __init__(self, alpha: float = 0.1, block: int | None = None):
        super().__init__()
        self.alpha = alpha
        self.block = block

    def _fit(self, y, X):
        d = np.asarray(y.values, dtype=float)
        nz = np.flatnonzero(d)
        k = self.block or (int(round(len(d) / len(nz))) if len(nz) else 1)
        k = max(1, k)
        self._k = k
        n_blocks = len(d) // k
        if n_blocks == 0:
            self._rate = float(d.mean())
            return
        agg = d[len(d) - n_blocks * k:].reshape(n_blocks, k).sum(axis=1)
        level = float(agg[0])
        for val in agg[1:]:
            level = self.alpha * val + (1 - self.alpha) * level
        self._rate = level / k

    def _predict(self, h, X):
        return np.repeat(self._rate, h)
