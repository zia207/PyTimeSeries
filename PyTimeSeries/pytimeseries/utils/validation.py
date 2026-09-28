"""Input validation and index helpers shared by every forecaster."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_y(y) -> pd.Series:
    """Coerce ``y`` to a clean, sorted, float :class:`pandas.Series`.

    Accepts a Series, a single-column DataFrame, or an array-like. A
    ``DatetimeIndex`` without an explicit frequency has one inferred and set,
    because the statistical backends (and future-index generation) need it.
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                f"Univariate y expected; got a DataFrame with {y.shape[1]} columns. "
                "Multivariate/global inputs arrive in a later version."
            )
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y, dtype=float))
    y = y.sort_index()
    if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
        freq = pd.infer_freq(y.index)
        if freq is not None:
            y = y.asfreq(freq)
    return y.astype(float)


def infer_freq(y: pd.Series):
    """Return the index frequency (a DateOffset/str) or ``None``."""
    if isinstance(y.index, pd.DatetimeIndex):
        return y.index.freq or pd.infer_freq(y.index)
    return None


def make_future_index(cutoff, h: int, freq) -> pd.Index:
    """Build the forecast horizon index: the ``h`` steps after ``cutoff``."""
    if isinstance(cutoff, pd.Timestamp) and freq is not None:
        return pd.date_range(start=cutoff, periods=h + 1, freq=freq)[1:]
    start = int(cutoff) + 1
    return pd.RangeIndex(start=start, stop=start + h)
