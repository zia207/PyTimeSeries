import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import pytimeseries as pt
from pytimeseries.datasets import make_series

ONESTEP = [
    pt.LSTMForecaster(epochs=30, hidden=16),
    pt.GRUForecaster(epochs=30, hidden=16),
    pt.RNNForecaster(epochs=30, hidden=16),
    pt.TCNForecaster(epochs=30, hidden=16),
    pt.MLPForecaster(epochs=30, hidden=32),
]


@pytest.mark.parametrize("model", ONESTEP)
def test_onestep_deep_predict(model):
    y = make_series(n=120)
    m = model.clone().fit(y)
    fc = m.predict(12)
    assert isinstance(fc, pd.Series)
    assert len(fc) == 12
    assert fc.index[0] == y.index[-1] + y.index.freq
    assert np.isfinite(fc.values).all()


def test_seq2seq_horizon_and_bounds():
    y = make_series(n=120)
    m = pt.Seq2SeqForecaster(horizon=12, epochs=40, hidden=24).fit(y)
    fc = m.predict(12)
    assert len(fc) == 12 and np.isfinite(fc.values).all()
    fc_short = m.predict(6)
    assert len(fc_short) == 6
    with pytest.raises(ValueError):
        m.predict(24)                          # beyond the fitted horizon


def test_deep_reproducible_with_seed():
    y = make_series(n=120)
    a = pt.LSTMForecaster(epochs=30, hidden=16, random_state=0).fit(y).predict(6).values
    b = pt.LSTMForecaster(epochs=30, hidden=16, random_state=0).fit(y).predict(6).values
    assert np.allclose(a, b)


def test_conformal_wraps_deep_model():
    y = make_series(n=120)
    cf = pt.ConformalForecaster(pt.LSTMForecaster(epochs=30, hidden=16), h=12, n_splits=3).fit(y)
    pi = cf.predict_interval(12, level=[0.9])
    assert (pi["0.9_lower"] <= pi["0.9_upper"]).all()
