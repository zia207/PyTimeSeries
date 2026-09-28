"""Coherent reconciliation of hierarchical forecasts (Part 11).

Base forecasts made independently per node rarely add up; reconciliation makes
them coherent (parents = sum of children). Implemented from scratch with the
summing matrix S — bottom-up, top-down, and MinT/OLS.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def build_summing_matrix(bottom_ids: Sequence[str],
                         groups: Mapping[str, Sequence[str]] | None = None):
    """Return ``(node_ids, S)`` for a hierarchy.

    ``node_ids`` is ordered ``["Total", *groups, *bottom_ids]``; ``S`` has shape
    ``(n_nodes, n_bottom)`` and maps a bottom-level vector to every node.
    """
    bottom_ids = list(bottom_ids)
    b = len(bottom_ids)
    pos = {bid: i for i, bid in enumerate(bottom_ids)}

    node_ids = ["Total"]
    rows = [np.ones(b)]
    if groups:
        for gname, members in groups.items():
            row = np.zeros(b)
            for m in members:
                row[pos[m]] = 1.0
            node_ids.append(gname)
            rows.append(row)
    for bid in bottom_ids:
        e = np.zeros(b)
        e[pos[bid]] = 1.0
        node_ids.append(bid)
        rows.append(e)
    return node_ids, np.vstack(rows)


def reconcile(base: pd.DataFrame, S: np.ndarray, method: str = "ols") -> pd.DataFrame:
    """Reconcile ``base`` (rows = nodes matching ``S``, columns = horizon).

    ``method`` in ``{"bottom_up", "top_down", "ols"}``. Returns a coherent frame
    (same index/columns) where every parent equals the sum of its children.
    """
    Y = np.asarray(base.values, dtype=float)          # (n_nodes, H)
    n_nodes, b = S.shape
    if Y.shape[0] != n_nodes:
        raise ValueError(f"base has {Y.shape[0]} rows but S expects {n_nodes} nodes.")
    bottom_rows = np.arange(n_nodes - b, n_nodes)     # bottom nodes are the last b

    if method == "bottom_up":
        bhat = Y[bottom_rows]
    elif method == "top_down":
        total = Y[0]
        bsum = Y[bottom_rows].sum(axis=0)
        props = np.divide(Y[bottom_rows].mean(axis=1),
                          Y[bottom_rows].mean(axis=1).sum(),
                          out=np.full(b, 1.0 / b), where=Y[bottom_rows].mean(axis=1).sum() != 0)
        bhat = props[:, None] * total[None, :]
    elif method == "ols":
        G = np.linalg.pinv(S)                          # (b, n_nodes) = (S'S)^-1 S'
        bhat = G @ Y
    else:
        raise ValueError(f"Unknown method {method!r}; use bottom_up|top_down|ols.")

    reconciled = S @ bhat
    return pd.DataFrame(reconciled, index=base.index, columns=base.columns)


class HierarchicalReconciler:
    """Hold a hierarchy's summing matrix and reconcile base forecasts."""

    def __init__(self, bottom_ids, groups=None):
        self.node_ids, self.S = build_summing_matrix(bottom_ids, groups)

    def reconcile(self, base: pd.DataFrame, method: str = "ols") -> pd.DataFrame:
        return reconcile(base.loc[self.node_ids], self.S, method=method)
