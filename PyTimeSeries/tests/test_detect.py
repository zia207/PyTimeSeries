import pytest

import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_anomalies_flag_spike():
    y = make_series(n=120).copy()
    y.iloc[60] += 50
    anoms = pt.detect_anomalies(y, period=12, z_thresh=3.0)
    assert y.index[60] in anoms.index


def test_change_points():
    pytest.importorskip("ruptures")
    y = make_series(n=120).copy()
    y.iloc[60:] += 30
    cps = pt.detect_change_points(y, n_bkps=1)
    assert len(cps) >= 1
