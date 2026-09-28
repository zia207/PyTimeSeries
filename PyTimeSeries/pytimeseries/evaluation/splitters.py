"""Time-aware cross-validation splitters (Part 6). Test blocks always follow
their training data — no shuffling, ever."""
from __future__ import annotations

import numpy as np


class ExpandingWindow:
    """Walk-forward CV with a growing training window.

    Parameters
    ----------
    h : int
        Forecast horizon / test-block length.
    n_splits : int
        Number of folds.
    step : int, optional
        Gap between consecutive test blocks (default ``h``).
    min_train : int, optional
        Minimum training length (default ``max(2*h, h+1)``).
    """

    def __init__(self, h: int, n_splits: int = 5, step: int | None = None,
                 min_train: int | None = None):
        self.h = h
        self.n_splits = n_splits
        self.step = step
        self.min_train = min_train

    def split(self, y):
        n = len(y)
        h = self.h
        step = self.step or h
        min_train = self.min_train or max(2 * h, h + 1)
        folds = []
        for i in range(self.n_splits):
            test_end = n - i * step
            test_start = test_end - h
            train_end = test_start
            if test_start < 0 or train_end < min_train:
                break
            folds.append((np.arange(0, train_end), np.arange(test_start, test_end)))
        return list(reversed(folds))

    def get_n_splits(self, y):
        return len(self.split(y))


class SlidingWindow(ExpandingWindow):
    """Walk-forward CV with a fixed-length training window."""

    def __init__(self, h: int, window: int, n_splits: int = 5, step: int | None = None):
        super().__init__(h=h, n_splits=n_splits, step=step, min_train=window)
        self.window = window

    def split(self, y):
        out = []
        for train_idx, test_idx in super().split(y):
            train_idx = train_idx[-self.window:]
            out.append((train_idx, test_idx))
        return out
