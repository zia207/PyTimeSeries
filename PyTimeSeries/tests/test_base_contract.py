import numpy as np
import pandas as pd
import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series
from pytimeseries.exceptions import NotFittedError

MODELS = [
    pt.NaiveForecaster(),
    pt.SeasonalNaive(sp=12),
    pt.DriftForecaster(),
    pt.MeanForecaster(),
    pt.ETS(seasonal="add", seasonal_periods=12),
    pt.make_reduction(lags=12),
]


@pytest.mark.parametrize("model", MODELS)
def test_predict_shape_and_index(model):
    y = make_series()
    m = model.clone().fit(y)
    fc = m.predict(12)
    assert isinstance(fc, pd.Series)
    assert len(fc) == 12
    assert fc.index[0] == y.index[-1] + y.index.freq   # first future step
    assert np.isfinite(fc.values).all()


@pytest.mark.parametrize("model", MODELS)
def test_not_fitted_raises(model):
    with pytest.raises(NotFittedError):
        model.clone().predict(3)


def test_clone_is_independent():
    m = pt.SeasonalNaive(sp=12)
    c = m.clone()
    assert c.get_params() == m.get_params()
    assert c is not m
