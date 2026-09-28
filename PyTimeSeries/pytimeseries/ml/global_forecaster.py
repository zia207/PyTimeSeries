"""Global (cross-series) forecasting (Part 8): one model trained across a whole
panel of series, with a series identifier as a feature and optional per-series
scaling so the model learns *shape*, not *level*."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster
from ..reduction.tabularize import make_lag_matrix
from ..utils.validation import infer_freq, make_future_index


class GlobalForecaster(BaseForecaster):
    """Train one regressor over many series and forecast them all.

    Accepts a **panel** in any of three shapes:
    - long ``DataFrame`` with ``id_col`` / ``time_col`` / ``target_col`` columns,
    - wide ``DataFrame`` (index = time, one column per series),
    - ``dict`` of ``{series_id: Series}``.

    ``predict(h)`` returns a long ``DataFrame`` ``[id_col, time_col, "forecast"]``
    with an ``h``-step forecast for every series.
    """

    _tags = {"scitype": "global", "probabilistic": False}

    def __init__(self, regressor=None, lags=12, scale: str | None = "local"):
        super().__init__()
        self.regressor = regressor
        self.lags = lags
        self.scale = scale

    # BaseForecaster abstract hooks are unused — fit/predict are overridden.
    def _fit(self, y, X):  # pragma: no cover
        raise NotImplementedError("GlobalForecaster overrides fit().")

    def _predict(self, h, X):  # pragma: no cover
        raise NotImplementedError("GlobalForecaster overrides predict().")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_long(panel, id_col, time_col, target_col) -> pd.DataFrame:
        if isinstance(panel, dict):
            parts = []
            for sid, s in panel.items():
                d = pd.DataFrame({time_col: s.index, target_col: np.asarray(s, dtype=float)})
                d[id_col] = sid
                parts.append(d)
            return pd.concat(parts, ignore_index=True)
        if isinstance(panel, pd.DataFrame):
            if id_col in panel.columns and target_col in panel.columns:
                return panel.copy()                                  # already long
            index_name = panel.index.name or "index"                # wide -> long
            long = panel.reset_index().melt(
                id_vars=index_name, var_name=id_col, value_name=target_col)
            return long.rename(columns={index_name: time_col})
        raise TypeError("panel must be a long/wide DataFrame or a dict of Series.")

    def _norm_lags(self) -> list[int]:
        return list(range(1, self.lags + 1)) if isinstance(self.lags, int) else sorted(self.lags)

    # ------------------------------------------------------------------ fit
    def fit(self, panel, id_col="unique_id", time_col="ds", target_col="y"):
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        self.id_col, self.time_col, self.target_col = id_col, time_col, target_col
        self._lags = self._norm_lags()
        long = self._to_long(panel, id_col, time_col, target_col)

        self._ids = list(pd.unique(long[id_col]))
        self._codes = {sid: i for i, sid in enumerate(self._ids)}
        self._scalers, self._hist, self._cutoff, self._freq = {}, {}, {}, {}

        frames = []
        maxlag = max(self._lags)
        for sid in self._ids:
            g = long[long[id_col] == sid].sort_values(time_col)
            s = pd.Series(np.asarray(g[target_col], dtype=float),
                          index=pd.Index(g[time_col]))
            if isinstance(s.index, pd.DatetimeIndex) and s.index.freq is None:
                f = pd.infer_freq(s.index)
                if f:
                    s = s.asfreq(f)
            mu, sd = (float(s.mean()), float(s.std())) if self.scale == "local" else (0.0, 1.0)
            sd = sd if sd and not np.isclose(sd, 0.0) else 1.0
            z = (s - mu) / sd

            X_mat, target = make_lag_matrix(z, self._lags)
            X_mat = X_mat.copy()
            X_mat["series_code"] = self._codes[sid]
            frames.append(pd.concat([X_mat, target.rename("__t__")], axis=1))

            self._scalers[sid] = (mu, sd)
            self._hist[sid] = list(np.asarray(z, dtype=float)[-maxlag:])
            self._cutoff[sid] = s.index[-1]
            self._freq[sid] = infer_freq(s) if isinstance(s.index, pd.DatetimeIndex) else None

        pooled = pd.concat(frames, ignore_index=True)
        self._feat_cols = [f"lag_{lag}" for lag in self._lags] + ["series_code"]
        base = self.regressor if self.regressor is not None else LinearRegression()
        self._reg = clone(base)
        self._reg.fit(pooled[self._feat_cols].values, pooled["__t__"].values)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, h: int):
        self._check_is_fitted()
        rows = []
        for sid in self._ids:
            code = self._codes[sid]
            mu, sd = self._scalers[sid]
            hist = list(self._hist[sid])
            out = []
            for _ in range(int(h)):
                feat = np.array([hist[-lag] for lag in self._lags] + [code],
                                dtype=float).reshape(1, -1)
                step = float(self._reg.predict(feat)[0])
                out.append(step)
                hist.append(step)
            values = np.asarray(out) * sd + mu
            index = make_future_index(self._cutoff[sid], int(h), self._freq[sid])
            for ts, val in zip(index, values):
                rows.append({self.id_col: sid, self.time_col: ts, "forecast": float(val)})
        return pd.DataFrame(rows)
