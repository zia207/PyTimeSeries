"""Vector Autoregression (Part 5). A *multivariate* forecaster: ``fit`` takes a
DataFrame of (stationary) series; ``predict`` returns a DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster
from ..utils.validation import infer_freq, make_future_index


class VAR(BaseForecaster):
    """VAR(p) with automatic lag selection. Inputs should be stationary
    (difference or take returns first)."""

    _tags = {"scitype": "multivariate", "probabilistic": False, "handles_exog": False}

    def __init__(self, maxlags: int | None = None, ic: str = "aic"):
        super().__init__()
        self.maxlags = maxlags
        self.ic = ic

    def _fit(self, y, X):  # pragma: no cover - fit is overridden
        raise NotImplementedError("VAR.fit takes a DataFrame; use fit(df).")

    def _predict(self, h, X):  # pragma: no cover
        raise NotImplementedError

    def fit(self, df: pd.DataFrame):
        from statsmodels.tsa.api import VAR as _VAR

        self._cols = list(df.columns)
        model = _VAR(df)
        res = model.fit(maxlags=self.maxlags, ic=self.ic)
        if res.k_ar == 0:                       # a 0-lag VAR is a degenerate forecaster
            res = model.fit(1)                  # fall back to VAR(1)
        self._res = res
        self._p = self._res.k_ar
        self._tail = df.values[-self._p:] if self._p > 0 else df.values[-1:]
        self.cutoff_ = df.index[-1]
        self.freq_ = infer_freq(df[self._cols[0]]) if isinstance(df.index, pd.DatetimeIndex) else None
        self._is_fitted = True
        return self

    def predict(self, h: int):
        self._check_is_fitted()
        fc = self._res.forecast(self._tail, steps=int(h))
        idx = make_future_index(self.cutoff_, int(h), self.freq_)
        return pd.DataFrame(np.asarray(fc), index=idx, columns=self._cols)

    @property
    def lag_order_(self):
        return self._p
