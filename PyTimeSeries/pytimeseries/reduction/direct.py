"""Direct multi-step reduction (Part 6): a separate regressor per horizon, so
prediction errors do not compound across steps."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster


def make_direct_matrix(y: pd.Series, n_lags: int, horizon: int):
    """Return ``(X, target)`` where each row's features are the ``n_lags`` most
    recent values at an origin ``t`` and the target is ``y[t + horizon]``."""
    cols = {f"lag_{k}": y.shift(k) for k in range(n_lags)}      # shift 0..L-1
    target = y.shift(-horizon)
    frame = pd.concat([pd.DataFrame(cols), target.rename("__t__")], axis=1).dropna()
    feat_cols = [f"lag_{k}" for k in range(n_lags)]
    return frame[feat_cols], frame["__t__"]


class DirectReductionForecaster(BaseForecaster):
    """One model per forecast horizon ``1..max_horizon``.

    Parameters
    ----------
    regressor : scikit-learn regressor, optional
        Defaults to ``LinearRegression``.
    lags : int
        Number of most-recent values used as features.
    max_horizon : int
        Largest horizon a model is trained for; ``predict(h)`` requires
        ``h <= max_horizon``.
    """

    _tags = {"scitype": "univariate", "probabilistic": False}

    def __init__(self, regressor=None, lags: int = 10, max_horizon: int = 24):
        super().__init__()
        self.regressor = regressor
        self.lags = lags
        self.max_horizon = max_horizon

    def _fit(self, y, X):
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        self._n = int(self.lags)
        base = self.regressor if self.regressor is not None else LinearRegression()
        self._models = {}
        for hz in range(1, self.max_horizon + 1):
            X_mat, target = make_direct_matrix(y, self._n, hz)
            if len(X_mat) == 0:
                raise ValueError(
                    f"Series too short (n={len(y)}) for lags={self._n} and horizon={hz}."
                )
            model = clone(base)
            model.fit(X_mat.values, target.values)
            self._models[hz] = model

    def _predict(self, h, X):
        if h > self.max_horizon:
            raise ValueError(
                f"h={h} exceeds max_horizon={self.max_horizon}; refit with a larger max_horizon."
            )
        arr = np.asarray(self._y, dtype=float)
        feat = arr[-1: -1 - self._n: -1].reshape(1, -1)        # [y_last, y_{last-1}, ...]
        return np.array([float(self._models[hz].predict(feat)[0]) for hz in range(1, h + 1)])
