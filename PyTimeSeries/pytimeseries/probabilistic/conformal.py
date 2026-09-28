"""Conformal prediction intervals (Part 12): turn ANY point forecaster into a
probabilistic one, using its own walk-forward backtest residuals — no
distributional assumptions."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..base import BaseForecaster
from ..evaluation.splitters import ExpandingWindow


class ConformalForecaster(BaseForecaster):
    """Wrap a base forecaster with calibrated, symmetric prediction intervals.

    On ``fit`` it backtests the base model to collect per-horizon absolute
    residuals, then refits the base on all data. Intervals use the empirical
    residual quantiles per horizon, so ``predict_interval``/``predict_quantiles``
    work for models that have no native uncertainty (trees, neural nets).

    Parameters
    ----------
    forecaster : BaseForecaster
        Any point (or probabilistic) forecaster; its point forecast is recalibrated.
    cv : splitter, optional
        A splitter with ``split(y)``. Defaults to ``ExpandingWindow(h, n_splits)``.
    h, n_splits : int
        Calibration horizon and number of folds when ``cv`` is not given.
    """

    _tags = {"probabilistic": True, "scitype": "univariate"}

    def __init__(self, forecaster: BaseForecaster, cv=None, h: int = 12, n_splits: int = 5):
        super().__init__()
        self.forecaster = forecaster
        self.cv = cv
        self.h = h
        self.n_splits = n_splits

    def _fit(self, y, X):
        cv = self.cv or ExpandingWindow(h=self.h, n_splits=self.n_splits)
        residuals = []
        for train_idx, test_idx in cv.split(y):
            fitted = self.forecaster.clone().fit(y.iloc[train_idx])
            pred = fitted.predict(len(test_idx)).values
            residuals.append(np.abs(y.iloc[test_idx].values - pred))
        if not residuals:
            raise ValueError("No calibration folds produced; series too short for the cv settings.")
        self._residuals = np.asarray(residuals)          # [n_folds, cal_h]
        self._base = self.forecaster.clone().fit(y)

    def _predict(self, h, X):
        return self._base.predict(h).values

    def _predict_quantiles(self, h, q: Sequence[float], X):
        mean = self._base.predict(h).values
        res = self._residuals
        cal_h = res.shape[1]
        out = np.empty((h, len(q)))
        for j, qq in enumerate(q):
            if np.isclose(qq, 0.5):
                out[:, j] = mean
                continue
            coverage = abs(2 * qq - 1)                    # e.g. 0.025/0.975 -> 0.95
            sign = 1.0 if qq > 0.5 else -1.0
            half = np.array([
                np.quantile(res[:, min(i, cal_h - 1)], coverage) for i in range(h)
            ])
            out[:, j] = mean + sign * half
        return out
