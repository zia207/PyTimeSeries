import numpy as np

import pytimeseries as pt
from pytimeseries.datasets import make_intermittent


def test_croston_tsb_adida_rate():
    y = make_intermittent(n=120, p_demand=0.3, seed=1)
    for model in [pt.Croston(), pt.TSB(), pt.ADIDA()]:
        fc = model.fit(y).predict(6)
        assert len(fc) == 6
        assert np.all(fc.values >= 0)              # a smooth demand rate
        assert np.allclose(fc.values, fc.values[0])  # flat
