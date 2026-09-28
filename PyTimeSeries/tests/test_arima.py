import numpy as np
import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_arima_point_and_interval():
    y = make_series()
    m = pt.ARIMA(order=(1, 1, 1)).fit(y)
    assert len(m.predict(12)) == 12
    pi = m.predict_interval(12, level=[0.8, 0.95])
    assert (pi["0.8_lower"] <= pi["0.8_upper"]).all()
    assert (pi["0.95_lower"] <= pi["0.8_lower"]).all()


def test_sarima_seasonal_order():
    y = make_series()
    m = pt.ARIMA(order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)).fit(y)
    assert len(m.predict(12)) == 12


def test_arima_with_exog():
    y = make_series()
    X = np.arange(len(y), dtype=float).reshape(-1, 1)
    m = pt.ARIMA(order=(1, 0, 0)).fit(y, X=X)
    X_future = np.arange(len(y), len(y) + 6, dtype=float).reshape(-1, 1)
    assert len(m.predict(6, X=X_future)) == 6


def test_auto_arima_selects_and_forecasts():
    pytest.importorskip("pmdarima")
    y = make_series(n=72)
    m = pt.AutoARIMA(seasonal=False).fit(y)
    assert len(m.predict(6)) == 6
    assert m.predict_interval(6, level=[0.9]).shape[0] == 6
    assert m.order_ is not None
