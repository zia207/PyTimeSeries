import numpy as np
import pandas as pd
import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series


def _panel():
    a = make_series(n=120, seed=1, trend=0.10, level=50)
    b = make_series(n=120, seed=2, trend=-0.05, level=80)
    c = make_series(n=120, seed=3, trend=0.02, level=30)
    return {"A": a, "B": b, "C": c}


def test_global_dict_panel():
    gf = pt.GlobalForecaster(lags=12).fit(_panel())
    fc = gf.predict(6)
    assert set(fc["unique_id"]) == {"A", "B", "C"}
    assert len(fc) == 3 * 6
    assert np.isfinite(fc["forecast"]).all()


def test_global_wide_panel():
    p = _panel()
    wide = pd.DataFrame({k: v for k, v in p.items()})
    gf = pt.GlobalForecaster(lags=12).fit(wide)
    fc = gf.predict(6)
    assert len(fc) == 3 * 6


def test_global_long_panel_and_lightgbm():
    pytest.importorskip("lightgbm")
    p = _panel()
    long = pd.concat(
        [pd.DataFrame({"unique_id": k, "ds": v.index, "y": v.values}) for k, v in p.items()],
        ignore_index=True)
    gf = pt.GlobalForecaster(pt.ml.LightGBM(n_estimators=80), lags=12).fit(long)
    fc = gf.predict(6)
    assert len(fc) == 3 * 6


def test_global_clone_preserves_params():
    gf = pt.GlobalForecaster(lags=8, scale=None)
    c = gf.clone()
    assert c.get_params() == {"regressor": None, "lags": 8, "scale": None}
