"""The ML reduction (Parts 7-8): wrap any scikit-learn regressor as a
forecaster. ``make_reduction`` selects the multi-step strategy."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..base import BaseForecaster
from .tabularize import make_lag_matrix


class RecursiveReductionForecaster(BaseForecaster):
    """Fit a one-step regressor on lag features, then roll its own predictions
    forward (iterated strategy).

    Parameters
    ----------
    regressor : scikit-learn regressor, optional
        Defaults to ``LinearRegression``.
    lags : int or list of int
        ``int`` means lags ``1..lags``; a list uses exactly those lags.
    """

    _tags = {"scitype": "univariate", "probabilistic": False, "handles_exog": False}

    def __init__(self, regressor=None, lags=10):
        super().__init__()
        self.regressor = regressor
        self.lags = lags

    def _norm_lags(self) -> list[int]:
        if isinstance(self.lags, int):
            return list(range(1, self.lags + 1))
        return sorted(int(lag) for lag in self.lags)

    def _fit(self, y, X):
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        self._lags = self._norm_lags()
        base = self.regressor if self.regressor is not None else LinearRegression()
        self._reg = clone(base)
        X_mat, target = make_lag_matrix(y, self._lags)
        if len(X_mat) == 0:
            raise ValueError(
                f"Not enough observations ({len(y)}) for max lag {max(self._lags)}."
            )
        self._reg.fit(X_mat.values, target.values)

    def _predict(self, h, X):
        history = list(np.asarray(self._y, dtype=float))   # origin = current stored series
        preds = []
        for _ in range(h):
            feat = np.array([history[-lag] for lag in self._lags], dtype=float).reshape(1, -1)
            step = float(self._reg.predict(feat)[0])
            preds.append(step)
            history.append(step)
        return np.asarray(preds)


def make_reduction(regressor=None, strategy: str = "recursive", lags=10,
                   max_horizon: int = 24) -> BaseForecaster:
    """Turn ``regressor`` into a :class:`BaseForecaster`.

    Parameters
    ----------
    strategy : {"recursive", "direct"}
        ``recursive`` iterates a one-step model; ``direct`` fits one model per
        horizon (``max_horizon`` of them). ``dirrec`` arrives in a later version.
    """
    if strategy == "recursive":
        return RecursiveReductionForecaster(regressor=regressor, lags=lags)
    if strategy == "direct":
        from .direct import DirectReductionForecaster

        lags_int = lags if isinstance(lags, int) else max(lags)
        return DirectReductionForecaster(regressor=regressor, lags=lags_int,
                                         max_horizon=max_horizon)
    raise NotImplementedError(
        f"strategy={strategy!r} is not available yet; use 'recursive' or 'direct'."
    )
