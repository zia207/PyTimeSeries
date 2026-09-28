import numpy as np
import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_conformal_makes_point_model_probabilistic():
    y = make_series(n=150)
    # a plain reduction forecaster has NO native intervals ...
    point = pt.make_reduction(lags=12)
    with pytest.raises(NotImplementedError):
        point.clone().fit(y).predict_quantiles(6)
    # ... but wrapped in Conformal it does.
    cf = pt.ConformalForecaster(point, h=12, n_splits=4).fit(y)
    assert cf.get_tags()["probabilistic"] is True
    pi = cf.predict_interval(12, level=[0.8, 0.95])
    assert list(pi.columns) == ["median", "0.8_lower", "0.8_upper", "0.95_lower", "0.95_upper"]


def test_conformal_intervals_are_ordered_and_nested():
    y = make_series(n=150)
    cf = pt.ConformalForecaster(pt.ETS(seasonal="add", seasonal_periods=12),
                                h=12, n_splits=4).fit(y)
    pi = cf.predict_interval(12, level=[0.8, 0.95])
    assert (pi["0.8_lower"] <= pi["median"]).all()
    assert (pi["median"] <= pi["0.8_upper"]).all()
    assert (pi["0.95_lower"] <= pi["0.8_lower"]).all()
    assert (pi["0.95_upper"] >= pi["0.8_upper"]).all()


def test_conformal_point_equals_base_point():
    y = make_series(n=150)
    base = pt.ETS(seasonal="add", seasonal_periods=12)
    cf = pt.ConformalForecaster(base, h=12, n_splits=4).fit(y)
    # median column is the base point forecast
    assert np.allclose(cf.predict(6).values, cf.predict_interval(6)["median"].values)
