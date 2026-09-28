"""The forecaster contract every model in pytimeseries implements.

One API across all families: ``fit`` / ``predict`` / ``predict_interval`` /
``predict_quantiles`` / ``update``. Subclasses implement only the two private
hooks ``_fit`` and ``_predict`` (and optionally ``_predict_quantiles`` for
native probabilistic output); the base class handles validation, the future
index, sklearn-style params/clone (so backtesting can refit fresh per fold),
and turning quantiles into intervals.
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd

from .exceptions import NotFittedError
from .utils.validation import check_y, infer_freq, make_future_index

#: Capability flags carried by every forecaster; subclasses override in ``_tags``.
DEFAULT_TAGS = {
    "scitype": "univariate",       # univariate | multivariate | global
    "probabilistic": False,        # provides native quantiles/intervals
    "requires_stationary": False,  # caller should difference first
    "handles_exog": False,         # uses the X argument
    "handles_missing": False,      # tolerates NaNs / gaps
    "scale_free": False,           # invariant to series scale
}


def _round_q(q) -> float:
    return round(float(q), 6)


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    _tags: dict = {}

    def __init__(self):
        self._is_fitted = False
        self._y: pd.Series | None = None
        self._X = None
        self.cutoff_ = None
        self.freq_ = None

    # ------------------------------------------------------------------ tags
    @classmethod
    def get_class_tags(cls) -> dict:
        tags = dict(DEFAULT_TAGS)
        for klass in reversed(cls.__mro__):
            tags.update(getattr(klass, "_tags", {}) or {})
        return tags

    def get_tags(self) -> dict:
        return self.get_class_tags()

    # ------------------------------------------- sklearn-style params / clone
    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in self._param_names()}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def clone(self) -> "BaseForecaster":
        """Return a fresh, unfitted copy with identical hyper-parameters."""
        return type(self)(**deepcopy(self.get_params()))

    # ------------------------------------------------------------- lifecycle
    def fit(self, y, X=None) -> "BaseForecaster":
        y = check_y(y)
        self._y = y
        self._X = X
        self.cutoff_ = y.index[-1]
        self.freq_ = infer_freq(y)
        self._fit(y, X)
        self._is_fitted = True
        return self

    def predict(self, h: int, X=None) -> pd.Series:
        self._check_is_fitted()
        h = int(h)
        fh = self._make_fh(h)
        yhat = np.asarray(self._predict(h, X), dtype=float).ravel()
        return pd.Series(yhat, index=fh, name=self._y.name)

    def predict_quantiles(self, h: int, q: Sequence[float] = (0.1, 0.5, 0.9), X=None) -> pd.DataFrame:
        self._check_is_fitted()
        if not self.get_tags()["probabilistic"]:
            raise NotImplementedError(
                f"{type(self).__name__} has no native quantiles. Wrap it with "
                "pytimeseries.ConformalForecaster for calibrated intervals."
            )
        q = [_round_q(x) for x in q]
        fh = self._make_fh(int(h))
        out = np.asarray(self._predict_quantiles(int(h), q, X), dtype=float)
        return pd.DataFrame(out, index=fh, columns=q)

    def predict_interval(self, h: int, level: Sequence[float] = (0.8, 0.95), X=None) -> pd.DataFrame:
        """Return ``median`` plus symmetric ``{level}_lower/{level}_upper`` bands."""
        level = list(level)
        needed = {0.5}
        for lv in level:
            a = _round_q((1 - lv) / 2)
            needed.update({a, _round_q(1 - a)})
        qdf = self.predict_quantiles(h, q=sorted(needed), X=X)
        out = pd.DataFrame(index=qdf.index)
        out["median"] = qdf[0.5]
        for lv in level:
            a = _round_q((1 - lv) / 2)
            out[f"{lv}_lower"] = qdf[a]
            out[f"{lv}_upper"] = qdf[_round_q(1 - a)]
        return out

    def update(self, y_new, refit: bool = False) -> "BaseForecaster":
        """Ingest newer observations. ``refit=True`` re-estimates parameters.

        With ``refit=False`` the stored history is extended (so models that read
        their forecast origin from the series, e.g. the reduction forecaster,
        move forward), while estimated parameters are left unchanged.
        """
        self._check_is_fitted()
        y_new = check_y(y_new)
        combined = pd.concat([self._y, y_new]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        if refit:
            return self.fit(combined)
        self._y = combined
        self.cutoff_ = combined.index[-1]
        return self

    # -------------------------------------------------- subclass entry points
    @abstractmethod
    def _fit(self, y: pd.Series, X): ...

    @abstractmethod
    def _predict(self, h: int, X): ...

    def _predict_quantiles(self, h: int, q: Sequence[float], X):  # optional
        raise NotImplementedError

    # ------------------------------------------------------------- internals
    def _make_fh(self, h: int) -> pd.Index:
        return make_future_index(self.cutoff_, h, self.freq_)

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first.")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._param_names())
        return f"{type(self).__name__}({params})"
