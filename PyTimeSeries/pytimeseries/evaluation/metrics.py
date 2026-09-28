"""Forecast error metrics (Part 6). Every function accepts ``**_`` so a common
caller can pass ``y_train``/``sp`` uniformly."""
from __future__ import annotations

import numpy as np


def _a(x):
    return np.asarray(x, dtype=float)


def mae(y_true, y_pred, **_):
    return float(np.mean(np.abs(_a(y_true) - _a(y_pred))))


def rmse(y_true, y_pred, **_):
    return float(np.sqrt(np.mean((_a(y_true) - _a(y_pred)) ** 2)))


def mape(y_true, y_pred, **_):
    t, p = _a(y_true), _a(y_pred)
    return float(100 * np.mean(np.abs((t - p) / t)))


def smape(y_true, y_pred, **_):
    t, p = _a(y_true), _a(y_pred)
    return float(100 * np.mean(np.abs(t - p) / ((np.abs(t) + np.abs(p)) / 2)))


def mase(y_true, y_pred, y_train=None, sp: int = 1, **_):
    """Mean Absolute Scaled Error: MAE scaled by the in-sample seasonal-naive MAE."""
    if y_train is None:
        raise ValueError("MASE needs y_train (the in-sample series).")
    yt = _a(y_train)
    denom = np.mean(np.abs(yt[sp:] - yt[:-sp]))
    return float(mae(y_true, y_pred) / denom)


def coverage(y_true, lower, upper, **_):
    """Percentage of actuals falling within [lower, upper]."""
    t, lo, up = _a(y_true), _a(lower), _a(upper)
    return float(100 * np.mean((t >= lo) & (t <= up)))


def pinball(y_true, q_pred, tau: float, **_):
    """Pinball / quantile loss at quantile ``tau``."""
    t, p = _a(y_true), _a(q_pred)
    diff = t - p
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


_REGISTRY = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape, "MASE": mase}


def get_metric(name: str):
    key = name.upper()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
