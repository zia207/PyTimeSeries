"""Change-point detection via ``ruptures`` (Part 11). Requires ``[extra]``."""
from __future__ import annotations

import numpy as np

from ..utils.validation import check_y


def detect_change_points(y, n_bkps: int | None = 5, penalty: float | None = None,
                         model: str = "l2", min_size: int = 6):
    """Locate structural breaks. Give ``n_bkps`` for a fixed count (binary
    segmentation) or ``penalty`` for automatic selection (PELT).

    Returns a list of break timestamps.
    """
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("change-point detection needs ruptures — "
                          "install: pip install 'pytimeseries[extra]'") from exc
    y = check_y(y)
    signal = y.values.astype(float)
    if penalty is not None:
        bkps = rpt.Pelt(model=model, min_size=min_size).fit(signal).predict(pen=penalty)
    else:
        bkps = rpt.Binseg(model=model, min_size=min_size).fit(signal).predict(n_bkps=n_bkps)
    return [y.index[b - 1] for b in bkps if b < len(y)]
