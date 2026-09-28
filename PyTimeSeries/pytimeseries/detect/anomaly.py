"""Anomaly detection via decomposition residuals (Part 11) — statsmodels core."""
from __future__ import annotations

import numpy as np

from ..utils.validation import check_y


def detect_anomalies(y, period: int = 12, z_thresh: float = 3.0, robust: bool = True):
    """Flag points whose STL residual is more than ``z_thresh`` σ from zero.

    Returns the flagged sub-Series (index = anomaly timestamps).
    """
    from statsmodels.tsa.seasonal import STL

    y = check_y(y)
    resid = STL(y, period=period, robust=robust).fit().resid
    sd = resid.std()
    z = (resid - resid.mean()) / (sd if sd else 1.0)
    return y[np.abs(z) > z_thresh]
