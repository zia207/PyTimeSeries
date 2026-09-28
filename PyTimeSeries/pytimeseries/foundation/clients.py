"""Zero-shot foundation-model clients under the common forecaster contract.

``fit(y)`` stores the context (no training happens); ``predict``/
``predict_interval`` call the pretrained backend. Backends are imported lazily,
so importing pytimeseries never requires them; a missing backend raises a clear
``ImportError`` pointing at the ``[foundation]`` extra.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd

from ..base import BaseForecaster


class _FoundationForecaster(BaseForecaster):
    """Base for zero-shot models. Subclasses implement ``_load`` (return a ready
    backend or raise) and one of ``_predict`` / ``_predict_quantiles``."""

    _tags = {"probabilistic": True, "scitype": "univariate", "foundation": True}

    def __init__(self, model_name: str | None = None, num_samples: int = 100,
                 random_state: int = 0):
        super().__init__()
        self.model_name = model_name
        self.num_samples = num_samples
        self.random_state = random_state

    def _fit(self, y, X):
        pass                      # zero-shot: the fitted "state" is just the context

    def _load(self):              # pragma: no cover - backend/network required
        raise NotImplementedError

    def _predict(self, h, X):
        return self._predict_quantiles(h, [0.5], X)[:, 0]


class Chronos(_FoundationForecaster):
    """Amazon **Chronos** — tokenizes scaled values onto a T5 backbone."""

    def __init__(self, model_name: str = "amazon/chronos-t5-small",
                 num_samples: int = 100, random_state: int = 0):
        super().__init__(model_name, num_samples, random_state)

    def _load(self):
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "Chronos needs chronos-forecasting + torch — "
                "install: pip install 'pytimeseries[foundation]'"
            ) from exc
        try:
            return ChronosPipeline.from_pretrained(
                self.model_name, device_map="cpu", torch_dtype=torch.float32)
        except Exception as exc:  # pragma: no cover - needs network/HF cache
            raise RuntimeError(
                f"Could not load Chronos weights {self.model_name!r} "
                "(needs internet or a local Hugging Face cache)."
            ) from exc

    def _predict_quantiles(self, h, q: Sequence[float], X):
        import torch

        pipe = self._load()
        context = torch.tensor(np.asarray(self._y.values, dtype="float32"))
        samples = pipe.predict(context, prediction_length=int(h),
                               num_samples=self.num_samples)
        arr = np.asarray(samples)
        arr = arr[0] if arr.ndim == 3 else arr           # -> (num_samples, h)
        return np.quantile(arr, q, axis=0).T             # (h, len(q))


class TimesFM(_FoundationForecaster):
    """Google **TimesFM** — decoder-only, patched-input foundation model."""

    def __init__(self, model_name: str = "google/timesfm-1.0-200m",
                 num_samples: int = 100, random_state: int = 0):
        super().__init__(model_name, num_samples, random_state)

    def _load(self):
        try:
            import timesfm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TimesFM needs the timesfm package — "
                "install: pip install 'pytimeseries[foundation]'"
            ) from exc
        return timesfm

    def _predict_quantiles(self, h, q: Sequence[float], X):  # pragma: no cover
        tfm = self._load()
        model = tfm.TimesFm(context_len=len(self._y), horizon_len=int(h))
        model.load_from_checkpoint(self.model_name)
        point, quant = model.forecast([np.asarray(self._y.values, dtype="float32")],
                                      freq=[0])
        mean = np.asarray(point[0], dtype=float)
        # TimesFM returns deciles; interpolate to requested q around the mean spread
        spread = (np.asarray(quant[0])[:, -1] - np.asarray(quant[0])[:, 0]) / 2.0
        from scipy.stats import norm
        z = norm.ppf(q) / norm.ppf(0.9)
        return mean[:, None] + spread[:, None] * z[None, :]


class Moirai(_FoundationForecaster):
    """Salesforce **Moirai** — masked-encoder universal forecaster (uni2ts)."""

    def __init__(self, model_name: str = "Salesforce/moirai-1.0-R-small",
                 num_samples: int = 100, random_state: int = 0):
        super().__init__(model_name, num_samples, random_state)

    def _load(self):
        try:
            import uni2ts  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Moirai needs the uni2ts package — "
                "install: pip install 'pytimeseries[foundation]'"
            ) from exc
        return uni2ts

    def _predict_quantiles(self, h, q: Sequence[float], X):  # pragma: no cover
        raise RuntimeError(
            "Moirai requires uni2ts + Hugging Face weights; see the uni2ts docs "
            "for the exact predictor construction, then map its sample paths here."
        )


class TimeGPT(_FoundationForecaster):
    """Nixtla **TimeGPT** — served via API (needs an API key)."""

    _tags = {"probabilistic": True, "scitype": "univariate",
             "foundation": True, "handles_exog": True}

    def __init__(self, api_key: str | None = None, model: str = "timegpt-1",
                 freq: str | None = None):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self.freq = freq

    def _load(self):
        try:
            from nixtla import NixtlaClient
        except ImportError as exc:
            raise ImportError(
                "TimeGPT needs the nixtla client — "
                "install: pip install 'pytimeseries[foundation]'"
            ) from exc
        key = self.api_key or os.environ.get("NIXTLA_API_KEY")
        if not key:
            raise RuntimeError(
                "TimeGPT needs an API key: pass api_key=... or set NIXTLA_API_KEY.")
        return NixtlaClient(api_key=key)

    def _long(self):
        return pd.DataFrame({"unique_id": "series", "ds": self._y.index,
                             "y": self._y.values})

    def _freqstr(self):
        return self.freq or getattr(self.freq_, "freqstr", None) \
            or pd.infer_freq(self._y.index) or "D"

    def _predict(self, h, X):
        client = self._load()
        fc = client.forecast(df=self._long(), h=int(h), freq=self._freqstr(),
                             model=self.model)
        return np.asarray(fc[self.model.upper().replace("-", "")].values
                          if self.model.upper().replace("-", "") in fc
                          else fc.iloc[:, -1].values, dtype=float)

    def _predict_quantiles(self, h, q: Sequence[float], X):  # pragma: no cover
        client = self._load()
        levels = sorted({int(round(abs(2 * qq - 1) * 100)) for qq in q if qq != 0.5})
        fc = client.forecast(df=self._long(), h=int(h), freq=self._freqstr(),
                             model=self.model, level=levels or [80])
        mean = np.asarray(fc.iloc[:, 2].values, dtype=float)
        out = np.empty((int(h), len(q)))
        for j, qq in enumerate(q):
            if qq == 0.5:
                out[:, j] = mean
                continue
            lv = int(round(abs(2 * qq - 1) * 100))
            side = "hi" if qq > 0.5 else "lo"
            col = [c for c in fc.columns if f"{side}-{lv}" in c][0]
            out[:, j] = fc[col].values
        return out
