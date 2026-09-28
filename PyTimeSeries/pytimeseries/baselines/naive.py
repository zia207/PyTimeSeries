"""The mandatory reference forecasters (Part 6). A model that cannot beat
these has not earned its complexity."""
from __future__ import annotations

import numpy as np

from ..base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Repeat the last observed value."""

    def _fit(self, y, X):
        self._last = float(y.iloc[-1])

    def _predict(self, h, X):
        return np.repeat(self._last, h)


class SeasonalNaive(BaseForecaster):
    """Repeat the value from ``sp`` periods ago."""

    _tags = {"scale_free": False}

    def __init__(self, sp: int = 1):
        super().__init__()
        self.sp = sp

    def _fit(self, y, X):
        self._season = np.asarray(y.iloc[-self.sp:], dtype=float)

    def _predict(self, h, X):
        reps = int(np.ceil(h / self.sp))
        return np.tile(self._season, reps)[:h]


class DriftForecaster(BaseForecaster):
    """Extend the straight line from the first to the last training point."""

    def _fit(self, y, X):
        self._last = float(y.iloc[-1])
        self._slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)

    def _predict(self, h, X):
        return self._last + self._slope * np.arange(1, h + 1)


class MeanForecaster(BaseForecaster):
    """Forecast the historical mean."""

    def _fit(self, y, X):
        self._mean = float(np.mean(y))

    def _predict(self, h, X):
        return np.repeat(self._mean, h)
