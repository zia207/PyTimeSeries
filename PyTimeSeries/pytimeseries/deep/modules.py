"""PyTorch modules for the from-scratch deep forecasters (Part 9). Imported
lazily (only when a deep model is fitted), so ``import torch`` never runs at
package-import time."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RNNModule(nn.Module):
    """Vanilla RNN / LSTM / GRU reading a length-L window, one-step head."""

    def __init__(self, kind: str = "LSTM", hidden: int = 32):
        super().__init__()
        rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = rnn(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: [B, L]
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1, :])


class MLPModule(nn.Module):
    """Order-blind MLP over the flattened window."""

    def __init__(self, L: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class _CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):                 # x: [B, C, T] -> causal (left-pad only)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNModule(nn.Module):
    """Two dilated causal convolutions -> one-step head."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.c1 = _CausalConv1d(1, channels, 3, dilation=1)
        self.c2 = _CausalConv1d(channels, channels, 3, dilation=2)
        self.act = nn.ReLU()
        self.head = nn.Linear(channels, 1)

    def forward(self, x):                 # x: [B, L]
        h = x.unsqueeze(1)
        h = self.act(self.c1(h))
        h = self.act(self.c2(h))
        return self.head(h[:, :, -1])


class Seq2SeqModule(nn.Module):
    """LSTM encoder + linear head emitting the whole horizon at once."""

    def __init__(self, hidden: int = 48, horizon: int = 24):
        super().__init__()
        self.enc = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, horizon))

    def forward(self, x):                 # x: [B, L] -> [B, horizon]
        _, (h, _) = self.enc(x.unsqueeze(-1))
        return self.head(h[-1])
