import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_lightgbm_reduction_runs():
    pytest.importorskip("lightgbm")
    y = make_series()
    f = pt.make_reduction(pt.ml.LightGBM(n_estimators=60), lags=12).fit(y)
    assert len(f.predict(6)) == 6


def test_xgboost_reduction_runs():
    pytest.importorskip("xgboost")
    y = make_series()
    f = pt.make_reduction(pt.ml.XGBoost(n_estimators=60), lags=12).fit(y)
    assert len(f.predict(6)) == 6


def test_boosting_backtests_with_baselines():
    pytest.importorskip("lightgbm")
    y = make_series(n=180)
    cv = pt.ExpandingWindow(h=12, n_splits=3)
    res = pt.evaluate(pt.make_reduction(pt.ml.LightGBM(n_estimators=80), lags=12),
                      y, cv=cv, metrics=("MASE",), sp=12)
    assert "RecursiveReductionForecaster" in res.summary.index
    assert "Naive" in res.summary.index
