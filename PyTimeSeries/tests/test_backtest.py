import numpy as np

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_backtest_attaches_baselines_and_scores():
    y = make_series(n=180)
    cv = pt.ExpandingWindow(h=12, n_splits=3)
    res = pt.evaluate(pt.ETS(seasonal="add", seasonal_periods=12), y, cv=cv,
                      metrics=("MASE", "RMSE"), sp=12)

    # model + 3 baselines all present
    assert "ETS" in res.summary.index
    for b in ["Naive", "SeasonalNaive(sp=12)", "Drift"]:
        assert b in res.summary.index
    # metric columns and finite values
    assert list(res.summary.columns) == ["MASE", "RMSE"]
    assert np.isfinite(res.summary.values).all()
    # per-fold table has 4 models x 3 folds x 2 metrics = 24 rows
    assert len(res.folds) == 4 * 3 * 2


def test_expanding_window_folds_are_ordered():
    y = make_series(n=100)
    cv = pt.ExpandingWindow(h=12, n_splits=4)
    folds = cv.split(y)
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert train_idx[-1] < test_idx[0]          # test strictly after train
        assert len(test_idx) == 12
