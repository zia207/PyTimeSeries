"""Tiny synthetic data generator so examples and tests are self-contained."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_series(n=180, freq="MS", start="2010-01-01", level=50.0, trend=0.1,
                season_amp=5.0, sp=12, noise=1.0, seed=0) -> pd.Series:
    """A trend + seasonal + noise monthly series with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = (
        level + trend * t
        + season_amp * np.sin(2 * np.pi * t / sp)
        + rng.normal(0, noise, n)
    )
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(values, index=idx, name="y")


def make_panel(n_series=4, n=120, freq="MS", start="2012-01-01", seed=0):
    """A synthetic long panel (``unique_id``/``ds``/``y``) of heterogeneous series
    — different levels, trends, and seasonal amplitudes — for global models."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_series):
        s = make_series(
            n=n, freq=freq, start=start,
            level=float(rng.uniform(20, 120)),
            trend=float(rng.uniform(-0.05, 0.15)),
            season_amp=float(rng.uniform(2, 8)),
            noise=float(rng.uniform(0.5, 2.0)),
            seed=int(rng.integers(0, 10_000)),
        )
        frames.append(pd.DataFrame({"unique_id": f"series_{i+1}", "ds": s.index, "y": s.values}))
    return pd.concat(frames, ignore_index=True)


def make_intermittent(n=120, p_demand=0.3, low=1, high=8, freq="MS",
                      start="2015-01-01", seed=1):
    """A sparse (mostly-zero) demand series for Croston/TSB/ADIDA."""
    rng = np.random.default_rng(seed)
    d = np.where(rng.random(n) < p_demand, rng.integers(low, high, n), 0).astype(float)
    return pd.Series(d, index=pd.date_range(start, periods=n, freq=freq), name="demand")


def make_multivariate(n=200, k=3, freq="MS", start="2000-01-01", seed=0):
    """A DataFrame of ``k`` cointegrated I(1) series (share a common trend)."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=n))
    cols = {}
    for j in range(k):
        idio = np.cumsum(rng.normal(size=n)) * 0.3
        cols[f"y{j + 1}"] = common * (1 + 0.1 * j) + idio + 50 + 10 * j
    return pd.DataFrame(cols, index=pd.date_range(start, periods=n, freq=freq))
