"""From-scratch deep forecasters (Part 9). Point forecasters — wrap with
``ConformalForecaster`` for calibrated intervals."""
from __future__ import annotations

import numpy as np

from ..base import BaseForecaster
from ._torch_base import _TorchOneStepForecaster, make_windows


class MLPForecaster(_TorchOneStepForecaster):
    """Feed-forward MLP over the flattened look-back window."""

    def _make_module(self, L):
        from .modules import MLPModule
        return MLPModule(L, hidden=self.hidden)


class RNNForecaster(_TorchOneStepForecaster):
    """Vanilla RNN forecaster."""

    def _make_module(self, L):
        from .modules import RNNModule
        return RNNModule("RNN", hidden=self.hidden)


class LSTMForecaster(_TorchOneStepForecaster):
    """LSTM forecaster."""

    def _make_module(self, L):
        from .modules import RNNModule
        return RNNModule("LSTM", hidden=self.hidden)


class GRUForecaster(_TorchOneStepForecaster):
    """GRU forecaster."""

    def _make_module(self, L):
        from .modules import RNNModule
        return RNNModule("GRU", hidden=self.hidden)


class TCNForecaster(_TorchOneStepForecaster):
    """Temporal Convolutional Network (causal, dilated) forecaster."""

    def _make_module(self, L):
        from .modules import TCNModule
        return TCNModule(channels=self.hidden)


class Seq2SeqForecaster(BaseForecaster):
    """LSTM encoder + direct multi-output head — forecasts the whole horizon at
    once (no recursive error compounding). Fixed-horizon: fit with ``horizon``,
    then ``predict(h)`` with ``h <= horizon``."""

    _tags = {"scitype": "univariate", "probabilistic": False}

    def __init__(self, lags: int = 12, horizon: int = 24, hidden: int = 48,
                 epochs: int = 300, lr: float = 0.01, batch_size: int = 16,
                 patience: int = 30, val_frac: float = 0.15, random_state: int = 0):
        super().__init__()
        self.lags = lags
        self.horizon = horizon
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def _fit(self, y, X):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from .modules import Seq2SeqModule

        L, H = int(self.lags), int(self.horizon)
        self._mu = float(y.mean())
        self._sd = float(y.std()) or 1.0
        z = ((y.values - self._mu) / self._sd).astype("float32")

        xs, ys = [], []
        for i in range(len(z) - L - H + 1):
            xs.append(z[i:i + L])
            ys.append(z[i + L:i + L + H])
        if len(xs) < 8:
            raise ValueError(f"Series too short for lags={L} + horizon={H}.")
        Xw = np.stack(xs).astype("float32")
        Yw = np.stack(ys).astype("float32")
        n_val = max(1, int(len(Xw) * self.val_frac))

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        loader = DataLoader(
            TensorDataset(torch.tensor(Xw[:-n_val]), torch.tensor(Yw[:-n_val])),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        Xv_t, Yv_t = torch.tensor(Xw[-n_val:]), torch.tensor(Yw[-n_val:])

        model = Seq2SeqModule(hidden=self.hidden, horizon=H)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
        lossf = nn.MSELoss()

        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                lossf(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = lossf(model(Xv_t), Yv_t).item()
            sched.step(v)
            if v < best - 1e-6:
                best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        self._module = model
        self._L = L

    def _predict(self, h, X):
        import torch

        if h > self.horizon:
            raise ValueError(
                f"{type(self).__name__} was fit for horizon {self.horizon}; "
                f"call predict(h) with h <= {self.horizon}."
            )
        z = ((self._y.values - self._mu) / self._sd).astype("float32")
        self._module.eval()
        with torch.no_grad():
            x = torch.tensor(z[-self._L:]).unsqueeze(0)
            full = self._module(x).squeeze().numpy()
        return np.asarray(full[:h]) * self._sd + self._mu
