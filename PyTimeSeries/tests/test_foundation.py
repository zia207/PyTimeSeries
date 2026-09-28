import pytest
import pytimeseries as pt
from pytimeseries.datasets import make_series

CLIENTS = [pt.foundation.Chronos, pt.foundation.TimesFM,
           pt.foundation.Moirai, pt.foundation.TimeGPT]


@pytest.mark.parametrize("Cls", CLIENTS)
def test_tags_and_zero_shot_fit(Cls):
    tags = Cls().get_tags()
    assert tags["probabilistic"] is True
    assert tags.get("foundation") is True
    # zero-shot "fit" stores context and must not train or error
    m = Cls().fit(make_series(n=60))
    assert m._is_fitted


@pytest.mark.parametrize("Cls", CLIENTS)
def test_predict_raises_clear_error_without_backend(Cls):
    # backends (chronos/timesfm/uni2ts/nixtla) are not installed -> informative error
    m = Cls().fit(make_series(n=60))
    with pytest.raises((ImportError, RuntimeError)):
        m.predict(6)
