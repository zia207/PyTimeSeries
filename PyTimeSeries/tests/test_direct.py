import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_direct_strategy_forecasts():
    y = make_series()
    f = pt.make_reduction(strategy="direct", lags=12, max_horizon=12).fit(y)
    assert len(f.predict(12)) == 12


def test_direct_horizon_guard():
    y = make_series()
    f = pt.make_reduction(strategy="direct", lags=6, max_horizon=6).fit(y)
    with pytest.raises(ValueError):
        f.predict(12)


def test_direct_matches_contract_index():
    y = make_series()
    f = pt.DirectReductionForecaster(lags=6, max_horizon=6).fit(y)
    fc = f.predict(6)
    assert fc.index[0] == y.index[-1] + y.index.freq
