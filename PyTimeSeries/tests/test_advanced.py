import numpy as np
import pandas as pd
import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_unobserved_components():
    y = make_series(n=120)
    m = pt.UnobservedComponents(level="local linear trend", seasonal=12).fit(y)
    assert len(m.predict(12)) == 12
    pi = m.predict_interval(12, level=[0.9])
    assert (pi["0.9_lower"] <= pi["median"]).all()
    assert m.components_.shape[0] == len(y)


def test_garch():
    pytest.importorskip("arch")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_t(6, size=300),
                  index=pd.date_range("2000-01-01", periods=300, freq="D"))
    g = pt.GARCH(dist="t").fit(r)
    assert len(g.predict(5)) == 5
    pi = g.predict_interval(5, level=[0.95])
    assert (pi["0.95_upper"] >= pi["0.95_lower"]).all()
    assert g.conditional_volatility_ is not None


def test_prophet():
    pytest.importorskip("prophet")
    y = make_series(n=120)
    p = pt.Prophet(yearly_seasonality=True).fit(y)
    assert len(p.predict(12)) == 12
    pi = p.predict_interval(12, level=[0.8])
    assert (pi["0.8_upper"] >= pi["0.8_lower"]).all()
