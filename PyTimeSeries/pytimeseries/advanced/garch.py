"""GARCH volatility modelling (Part 5) over the ``arch`` backend. Requires the
``[extra]`` extra. Fit it on *returns*; its value is time-varying, calibrated
intervals rather than the (near-constant) mean forecast."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..base import BaseForecaster


class GARCH(BaseForecaster):
    """(G)ARCH conditional-variance model. ``predict`` returns the conditional
    mean; ``predict_interval`` uses the forecast variance for the bands."""

    _tags = {"probabilistic": True, "scitype": "univariate"}

    def __init__(self, mean: str = "Constant", vol: str = "GARCH", p: int = 1,
                 q: int = 1, dist: str = "normal", rescale: bool = False):
        super().__init__()
        self.mean = mean
        self.vol = vol
        self.p = p
        self.q = q
        self.dist = dist
        self.rescale = rescale

    def _fit(self, y, X):
        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover
            raise ImportError("GARCH needs arch — install: pip install 'pytimeseries[extra]'") from exc
        self._res = arch_model(y, mean=self.mean, vol=self.vol, p=self.p, q=self.q,
                               dist=self.dist, rescale=self.rescale).fit(disp="off")

    def _predict(self, h, X):
        fc = self._res.forecast(horizon=h, reindex=False)
        return np.asarray(fc.mean.values[-1], dtype=float)

    def _predict_quantiles(self, h, q: Sequence[float], X):
        from scipy.stats import norm, t as student_t

        fc = self._res.forecast(horizon=h, reindex=False)
        mean = np.asarray(fc.mean.values[-1], dtype=float)
        sd = np.sqrt(np.asarray(fc.variance.values[-1], dtype=float))
        if "t" in self.dist.lower() and "nu" in getattr(self._res, "params", {}):
            nu = float(self._res.params["nu"])
            z = student_t.ppf(q, nu) / np.sqrt(nu / (nu - 2)) if nu > 2 else norm.ppf(q)
        else:
            z = norm.ppf(q)
        return mean[:, None] + sd[:, None] * np.asarray(z)[None, :]

    @property
    def conditional_volatility_(self):
        """In-sample conditional volatility series."""
        return self._res.conditional_volatility
