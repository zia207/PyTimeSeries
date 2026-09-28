"""Unobserved Components / structural state-space model (Part 5) over
``statsmodels`` (core dependency). Probabilistic and exogenous-aware."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm

from ..base import BaseForecaster


class UnobservedComponents(BaseForecaster):
    """Level / trend / seasonal / cycle state-space model estimated by the
    Kalman filter. ``components_`` exposes the smoothed states."""

    _tags = {"probabilistic": True, "handles_exog": True, "scitype": "univariate"}

    def __init__(self, level: str = "local linear trend", seasonal: int | None = None,
                 cycle: bool = False, autoregressive: int | None = None):
        super().__init__()
        self.level = level
        self.seasonal = seasonal
        self.cycle = cycle
        self.autoregressive = autoregressive

    def _fit(self, y, X):
        from statsmodels.tsa.statespace.structural import UnobservedComponents as _UC

        self._res = _UC(
            y, level=self.level, seasonal=self.seasonal, cycle=self.cycle,
            autoregressive=self.autoregressive, exog=X,
        ).fit(disp=False)

    def _predict(self, h, X):
        return np.asarray(self._res.get_forecast(h, exog=X).predicted_mean, dtype=float)

    def _predict_quantiles(self, h, q: Sequence[float], X):
        fc = self._res.get_forecast(h, exog=X)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        se = np.asarray(fc.se_mean, dtype=float)
        return mean[:, None] + se[:, None] * norm.ppf(q)[None, :]

    @property
    def components_(self):
        """Smoothed state components (level/trend/seasonal/…)."""
        return self._res.states.smoothed
