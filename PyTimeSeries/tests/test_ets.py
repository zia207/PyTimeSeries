import numpy as np

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_ets_point_and_interval():
    y = make_series()
    m = pt.ETS(trend="add", seasonal="add", seasonal_periods=12).fit(y)

    fc = m.predict(24)
    assert len(fc) == 24

    pi = m.predict_interval(24, level=[0.8, 0.95])
    assert list(pi.columns) == ["median", "0.8_lower", "0.8_upper", "0.95_lower", "0.95_upper"]
    # bands ordered and nested
    assert (pi["0.8_lower"] <= pi["median"]).all()
    assert (pi["median"] <= pi["0.8_upper"]).all()
    assert (pi["0.95_lower"] <= pi["0.8_lower"]).all()
    assert (pi["0.95_upper"] >= pi["0.8_upper"]).all()


def test_ets_quantiles_probabilistic_tag():
    assert pt.ETS().get_tags()["probabilistic"] is True
