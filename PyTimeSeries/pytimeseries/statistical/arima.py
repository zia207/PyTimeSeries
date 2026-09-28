"""ARIMA family (Part 4).

``ARIMA`` wraps ``statsmodels`` (non-seasonal, seasonal, and exogenous via the
``seasonal_order`` / ``X`` arguments — i.e. it also covers SARIMA/SARIMAX).
``AutoARIMA`` wraps ``pmdarima.auto_arima`` for automatic order selection.
Both are probabilistic (Gaussian predictive quantiles from the forecast s.e.).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm

from ..base import BaseForecaster


class ARIMA(BaseForecaster):
    """(S)ARIMA(X): ``order=(p,d,q)``, ``seasonal_order=(P,D,Q,m)``, optional ``trend``.

    Pass exogenous regressors through ``fit(y, X=...)`` and supply their future
    values to ``predict(h, X=...)``.
    """

    _tags = {"probabilistic": True, "handles_exog": True, "scitype": "univariate",
             "requires_stationary": False}

    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def _fit(self, y, X):
        from statsmodels.tsa.arima.model import ARIMA as _SM_ARIMA

        self._res = _SM_ARIMA(
            y, exog=X, order=self.order,
            seasonal_order=self.seasonal_order, trend=self.trend,
        ).fit()

    def _predict(self, h, X):
        fc = self._res.get_forecast(h, exog=X)
        return np.asarray(fc.predicted_mean, dtype=float)

    def _predict_quantiles(self, h, q: Sequence[float], X):
        fc = self._res.get_forecast(h, exog=X)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        se = np.asarray(fc.se_mean, dtype=float)
        z = norm.ppf(q)                                   # (len(q),)
        return mean[:, None] + se[:, None] * z[None, :]    # (h, len(q))


# Explicit aliases for discoverability; both are just ARIMA configurations.
class SARIMAX(ARIMA):
    """Alias for :class:`ARIMA` — emphasises seasonal + exogenous use."""


class AutoARIMA(BaseForecaster):
    """Automatic ARIMA order selection via ``pmdarima.auto_arima``.

    Requires the ``[ml]`` extra (``pip install 'pytimeseries[ml]'``).
    """

    _tags = {"probabilistic": True, "handles_exog": True, "scitype": "univariate"}

    def __init__(self, seasonal: bool = False, m: int = 1, auto_kwargs: dict | None = None):
        super().__init__()
        self.seasonal = seasonal
        self.m = m
        self.auto_kwargs = auto_kwargs

    def _fit(self, y, X):
        try:
            from pmdarima import auto_arima
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AutoARIMA needs pmdarima — install the extra: pip install 'pytimeseries[ml]'"
            ) from exc
        kwargs = dict(self.auto_kwargs or {})
        self._model = auto_arima(
            y, X=X, seasonal=self.seasonal, m=self.m,
            suppress_warnings=True, error_action="ignore", **kwargs,
        )

    def _predict(self, h, X):
        return np.asarray(self._model.predict(n_periods=h, X=X), dtype=float)

    def _predict_quantiles(self, h, q: Sequence[float], X):
        mean, conf = self._model.predict(n_periods=h, X=X, return_conf_int=True, alpha=0.05)
        mean = np.asarray(mean, dtype=float)
        se = (conf[:, 1] - conf[:, 0]) / (2 * norm.ppf(0.975))
        z = norm.ppf(q)
        return mean[:, None] + se[:, None] * z[None, :]

    @property
    def order_(self):
        """The selected (p, d, q) order."""
        return getattr(self._model, "order", None)
