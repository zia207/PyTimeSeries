import numpy as np
import pytest

pytest.importorskip("neuralforecast")

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_nhits_fixed_horizon():
    y = make_series(n=120)
    m = pt.NHiTS(horizon=12, input_size=24, max_steps=30).fit(y)
    fc = m.predict(12)
    assert len(fc) == 12
    assert np.isfinite(fc.values).all()
    with pytest.raises(ValueError):
        m.predict(24)                          # beyond fitted horizon
