"""Modern architectures (Part 10) via ``neuralforecast`` — N-BEATS, N-HiTS,
PatchTST. Fixed-horizon wrappers around the univariate contract: they hold one
series internally as a length-1 panel. Requires the ``[dl]`` extra."""
from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
import pandas as pd

from ..base import BaseForecaster

_QUIET = ["lightning", "pytorch_lightning", "lightning.pytorch",
          "lightning.pytorch.utilities.rank_zero", "lightning_fabric"]


class _NeuralForecastModel(BaseForecaster):
    """Base wrapper. Subclasses set ``_nf_name`` to a ``neuralforecast`` model."""

    _tags = {"scitype": "univariate", "probabilistic": False}
    _nf_name: str | None = None

    def __init__(self, horizon: int = 24, input_size: int = 48, max_steps: int = 300,
                 random_state: int = 0, model_kwargs: dict | None = None):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size
        self.max_steps = max_steps
        self.random_state = random_state
        self.model_kwargs = model_kwargs

    def _fit(self, y, X):
        try:
            import torch
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NBEATS, NHITS, PatchTST
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Modern DL models need neuralforecast — install the extra: "
                "pip install 'pytimeseries[dl]'"
            ) from exc

        for lg in _QUIET:
            logging.getLogger(lg).setLevel(logging.ERROR)
        torch.manual_seed(self.random_state)

        cls = {"NBEATS": NBEATS, "NHITS": NHITS, "PatchTST": PatchTST}[self._nf_name]
        kwargs = dict(self.model_kwargs or {})
        model = cls(h=self.horizon, input_size=self.input_size, max_steps=self.max_steps,
                    scaler_type="standard", enable_progress_bar=False, logger=False, **kwargs)

        freq = getattr(self.freq_, "freqstr", None) or pd.infer_freq(y.index) or "D"
        self._nf = NeuralForecast(models=[model], freq=freq)
        long = pd.DataFrame({"unique_id": "series", "ds": y.index, "y": y.values})
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self._nf.fit(long)

    def _predict(self, h, X):
        if h > self.horizon:
            raise ValueError(
                f"{type(self).__name__} was fit for horizon {self.horizon}; "
                f"call predict(h) with h <= {self.horizon}."
            )
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            fc = self._nf.predict()
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        return np.asarray(fc[col].values[:h], dtype=float)


class NBEATS(_NeuralForecastModel):
    """N-BEATS — deep basis-expansion stacks."""
    _nf_name = "NBEATS"


class NHiTS(_NeuralForecastModel):
    """N-HiTS — multi-rate basis expansion for long horizons."""
    _nf_name = "NHITS"


class PatchTST(_NeuralForecastModel):
    """PatchTST — patched Transformer."""
    _nf_name = "PatchTST"
