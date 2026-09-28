import numpy as np

import pytimeseries as pt
from pytimeseries.datasets import make_series
from pytimeseries.reduction import make_lag_matrix


def test_lag_matrix_has_no_leakage():
    y = make_series(n=30)
    X, target = make_lag_matrix(y, [1, 2, 12])
    # each feature row must equal the corresponding past values of the target
    for t in target.index[:5]:
        assert X.loc[t, "lag_1"] == y.shift(1).loc[t]
        assert X.loc[t, "lag_12"] == y.shift(12).loc[t]


def test_reduction_default_and_custom_regressor():
    y = make_series()
    f1 = pt.make_reduction(lags=12).fit(y)          # default LinearRegression
    assert len(f1.predict(6)) == 6

    from sklearn.ensemble import RandomForestRegressor
    f2 = pt.make_reduction(RandomForestRegressor(n_estimators=20, random_state=0), lags=12).fit(y)
    assert len(f2.predict(6)) == 6


def test_dirrec_strategy_not_yet():
    import pytest
    # 'direct' is implemented in v0.2; 'dirrec' is the remaining strategy.
    with pytest.raises(NotImplementedError):
        pt.make_reduction(strategy="dirrec")
