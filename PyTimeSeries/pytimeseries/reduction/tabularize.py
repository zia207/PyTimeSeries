"""Turn a series into a supervised (lag-features -> target) table (Part 7)."""
from __future__ import annotations

from typing import Sequence

import pandas as pd


def make_lag_matrix(y: pd.Series, lags: Sequence[int]):
    """Return ``(X, target)`` where ``X`` holds the requested lag columns.

    Rows with any missing lag are dropped, so every feature uses only past
    values — the built-in leakage guard.
    """
    cols = {f"lag_{lag}": y.shift(lag) for lag in lags}
    frame = pd.concat([pd.DataFrame(cols), y.rename("__target__")], axis=1).dropna()
    return frame[[f"lag_{lag}" for lag in lags]], frame["__target__"]
