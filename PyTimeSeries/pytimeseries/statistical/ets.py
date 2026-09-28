"""ETS / exponential smoothing (Part 3) — a thin, probabilistic wrapper over
``statsmodels`` ``ETSModel``.

Intervals and quantiles come from simulated future sample paths, so they are
correct for both additive and multiplicative error/seasonal specifications.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..base import BaseForecaster


class ETS(BaseForecaster):
    """Error-Trend-Seasonal exponential smoothing.

    Parameters
    ----------
    error, trend, seasonal : {"add", "mul", None}
        Component forms. ``trend``/``seasonal`` may be ``None``.
    damped_trend : bool
        Damp the trend for safer long horizons.
    seasonal_periods : int, optional
        Required when ``seasonal`` is set (e.g. 12 for monthly).
    initialization_method : str
        Passed through to ``ETSModel`` (default ``"estimated"``).
    n_sims : int
        Simulated paths used for quantiles/intervals.
    random_state : int
        Seed for the simulation, for reproducible intervals.
    """

    _tags = {"probabilistic": True, "scitype": "univariate"}

    def __init__(
        self,
        error: str = "add",
        trend: str | None = "add",
        damped_trend: bool = True,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        initialization_method: str = "estimated",
        n_sims: int = 1000,
        random_state: int = 0,
    ):
        super().__init__()
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.n_sims = n_sims
        self.random_state = random_state

    def _fit(self, y, X):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        self._model = ETSModel(
            y,
            error=self.error,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method=self.initialization_method,
        )
        self._res = self._model.fit(disp=False)

    def _predict(self, h, X):
        return np.asarray(self._res.forecast(h), dtype=float)

    def _predict_quantiles(self, h, q: Sequence[float], X):
        # Reproducible simulated paths: seed numpy locally, then restore state.
        state = np.random.get_state()
        np.random.seed(self.random_state)
        try:
            sims = self._res.simulate(nsimulations=h, repetitions=self.n_sims, anchor="end")
        finally:
            np.random.set_state(state)
        sims = np.asarray(sims, dtype=float)          # shape (h, n_sims)
        return np.quantile(sims, q, axis=1).T          # shape (h, len(q))
