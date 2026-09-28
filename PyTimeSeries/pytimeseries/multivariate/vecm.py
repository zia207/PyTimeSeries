"""Vector Error Correction Model (Part 5) for cointegrated (non-stationary but
tethered) series. ``fit`` takes a DataFrame of levels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster
from ..utils.validation import infer_freq, make_future_index


class VECM(BaseForecaster):
    """VECM with a given cointegration rank."""

    _tags = {"scitype": "multivariate", "probabilistic": False}

    def __init__(self, k_ar_diff: int = 1, coint_rank: int = 1, deterministic: str = "ci"):
        super().__init__()
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic

    def _fit(self, y, X):  # pragma: no cover
        raise NotImplementedError("VECM.fit takes a DataFrame; use fit(df).")

    def _predict(self, h, X):  # pragma: no cover
        raise NotImplementedError

    def fit(self, df: pd.DataFrame):
        from statsmodels.tsa.vector_ar.vecm import VECM as _VECM

        self._cols = list(df.columns)
        self._res = _VECM(df, k_ar_diff=self.k_ar_diff, coint_rank=self.coint_rank,
                          deterministic=self.deterministic).fit()
        self.cutoff_ = df.index[-1]
        self.freq_ = infer_freq(df[self._cols[0]]) if isinstance(df.index, pd.DatetimeIndex) else None
        self._is_fitted = True
        return self

    def predict(self, h: int):
        self._check_is_fitted()
        fc = self._res.predict(steps=int(h))
        idx = make_future_index(self.cutoff_, int(h), self.freq_)
        return pd.DataFrame(np.asarray(fc), index=idx, columns=self._cols)

    @property
    def alpha_(self):
        """Error-correction (adjustment) coefficients."""
        return self._res.alpha
