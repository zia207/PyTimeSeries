"""Shared training machinery for the one-step deep forecasters: windowing,
train-only scaling, mini-batch training with validation early stopping, and a
recursive multi-step rollout. ``torch`` is imported inside ``_fit`` only."""
from __future__ import annotations

import numpy as np

from ..base import BaseForecaster


def make_windows(z: np.ndarray, L: int):
    X = np.stack([z[i:i + L] for i in range(len(z) - L)]).astype("float32")
    Y = z[L:].astype("float32")
    return X, Y


class _TorchOneStepForecaster(BaseForecaster):
    """Base for LSTM/GRU/RNN/TCN/MLP forecasters. Subclasses implement
    ``_make_module(L)`` returning an ``nn.Module`` mapping ``[B, L] -> [B, 1]``."""

    _tags = {"scitype": "univariate", "probabilistic": False}

    def __init__(self, lags: int = 12, hidden: int = 32, epochs: int = 200,
                 lr: float = 0.01, batch_size: int = 32, patience: int = 30,
                 val_frac: float = 0.15, random_state: int = 0):
        super().__init__()
        self.lags = lags
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def _make_module(self, L: int):
        raise NotImplementedError

    def _fit(self, y, X):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        L = int(self.lags)
        self._mu = float(y.mean())
        self._sd = float(y.std()) or 1.0
        z = ((y.values - self._mu) / self._sd).astype("float32")

        Xw, Yw = make_windows(z, L)
        if len(Xw) < 8:
            raise ValueError(f"Series too short ({len(y)}) for lags={L}.")
        n_val = max(1, int(len(Xw) * self.val_frac))

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        Xtr, Ytr = Xw[:-n_val], Yw[:-n_val]
        Xv, Yv = Xw[-n_val:], Yw[-n_val:]
        loader = DataLoader(
            TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr).unsqueeze(1)),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        Xv_t, Yv_t = torch.tensor(Xv), torch.tensor(Yv).unsqueeze(1)

        model = self._make_module(L)
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

        model = self._module
        model.eval()
        z = ((self._y.values - self._mu) / self._sd).astype("float32")  # origin = stored series
        buf = list(z[-self._L:])
        out = []
        with torch.no_grad():
            for _ in range(int(h)):
                x = torch.tensor(np.array(buf[-self._L:], dtype="float32")).unsqueeze(0)
                step = float(model(x).squeeze())
                out.append(step)
                buf.append(step)
        return np.asarray(out) * self._sd + self._mu
