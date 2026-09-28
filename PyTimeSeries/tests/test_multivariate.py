import numpy as np
import pytimeseries as pt
from pytimeseries.datasets import make_multivariate


def test_var_forecast():
    df = make_multivariate(n=200, k=3).diff().dropna()          # stationary
    v = pt.VAR(maxlags=6, ic="aic").fit(df)
    fc = v.predict(6)
    assert fc.shape == (6, 3)
    assert list(fc.columns) == list(df.columns)
    assert v.lag_order_ >= 1


def test_vecm_forecast():
    df = make_multivariate(n=200, k=3)                          # levels (cointegrated)
    m = pt.VECM(k_ar_diff=1, coint_rank=1).fit(df)
    fc = m.predict(6)
    assert fc.shape == (6, 3)
    assert m.alpha_.shape[0] == 3


def test_granger_and_cointegration():
    df = make_multivariate(n=200, k=3)
    g = pt.granger_causality(df.diff().dropna(), "y1", "y2", maxlag=4)
    assert {"pvalue", "granger_causal"} <= set(g)
    c = pt.cointegration_test(df, method="johansen")
    assert "rank" in c and isinstance(c["rank"], int)
