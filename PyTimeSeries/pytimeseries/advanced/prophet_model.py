"""Prophet — decomposable additive model (Part 5). Requires the ``[extra]`` extra."""
from __future__ import annotations

import contextlib
import io
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..base import BaseForecaster


class Prophet(BaseForecaster):
    """Trend + seasonality + holidays additive model (Meta's Prophet)."""

    _tags = {"probabilistic": True, "scitype": "univariate"}

    def __init__(self, growth: str = "linear", yearly_seasonality="auto",
                 weekly_seasonality=False, daily_seasonality=False,
                 seasonality_mode: str = "additive", interval_width: float = 0.8,
                 prophet_kwargs: dict | None = None):
        super().__init__()
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.prophet_kwargs = prophet_kwargs

    def _fit(self, y, X):
        try:
            from prophet import Prophet as _Prophet
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Prophet needs prophet — install: pip install 'pytimeseries[extra]'") from exc
        for lg in ("cmdstanpy", "prophet"):
            logging.getLogger(lg).setLevel(logging.ERROR)

        self._freqstr = getattr(self.freq_, "freqstr", None) or pd.infer_freq(y.index) or "D"
        self._model = _Prophet(
            growth=self.growth, yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality, daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode, interval_width=self.interval_width,
            **(self.prophet_kwargs or {}),
        )
        df = pd.DataFrame({"ds": y.index, "y": y.values})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self._model.fit(df)

    def _forecast_frame(self, h):
        future = self._model.make_future_dataframe(periods=h, freq=self._freqstr)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fc = self._model.predict(future)
        return fc.iloc[-h:]

    def _predict(self, h, X):
        return np.asarray(self._forecast_frame(h)["yhat"].values, dtype=float)

    def _predict_quantiles(self, h, q: Sequence[float], X):
        fc = self._forecast_frame(h)
        mean = fc["yhat"].values
        # back out a Gaussian s.e. from Prophet's interval, then map to quantiles
        z_iw = norm.ppf(0.5 + self.interval_width / 2)
        se = (fc["yhat_upper"].values - fc["yhat_lower"].values) / (2 * z_iw)
        return mean[:, None] + se[:, None] * norm.ppf(q)[None, :]
