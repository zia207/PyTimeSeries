import numpy as np
import pytimeseries as pt
from pytimeseries.datasets import make_series


def test_save_load_roundtrip(tmp_path):
    y = make_series()
    m = pt.ETS(seasonal="add", seasonal_periods=12).fit(y)
    path = tmp_path / "m.joblib"
    pt.deploy.save(m, path, metadata={"note": "unit-test"})
    m2, meta = pt.deploy.load(path, with_metadata=True)
    assert np.allclose(m.predict(6).values, m2.predict(6).values)
    assert meta["note"] == "unit-test"
    assert meta["pytimeseries_version"] == pt.__version__


def test_batch_forecast(tmp_path):
    path = tmp_path / "m.joblib"
    pt.deploy.save(pt.ETS().fit(make_series()), path)
    fc = pt.deploy.batch_forecast(path, h=6, out_path=tmp_path / "out.csv")
    assert len(fc) == 6
    assert (tmp_path / "out.csv").exists()


def test_drift_monitor():
    ref = make_series(n=120, level=50, trend=0, season_amp=0, noise=1)
    dm = pt.deploy.DriftMonitor(threshold=2.0).fit(ref)
    calm = make_series(n=12, level=50, trend=0, season_amp=0, noise=1, seed=7)
    shifted = make_series(n=12, level=90, trend=0, season_amp=0, noise=1, seed=8)
    assert abs(dm.score(calm)) < 2.0
    assert abs(dm.score(shifted)) > 2.0
    assert bool(dm.drift(shifted, window=6).any())


def test_retrain_policy():
    pol = pt.deploy.RetrainPolicy(every=12)
    assert pol.should_retrain(step=12)
    assert not pol.should_retrain(step=7)
    dm = pt.deploy.DriftMonitor(threshold=2.0).fit(
        make_series(n=100, level=50, trend=0, season_amp=0, noise=1))
    pol2 = pt.deploy.RetrainPolicy(drift_monitor=dm, drift_threshold=2.0)
    shifted = make_series(n=12, level=90, trend=0, season_amp=0, noise=1, seed=9)
    assert pol2.should_retrain(window=shifted)
    assert not pol2.should_retrain(window=make_series(n=12, level=50, trend=0,
                                                      season_amp=0, noise=1, seed=10))
