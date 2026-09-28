"""Walk-forward backtesting engine (Part 6) — the *only* evaluation entry point.

Baselines are attached automatically, so no model is ever scored in a vacuum.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..baselines import DriftForecaster, NaiveForecaster, SeasonalNaive
from ..utils.validation import check_y
from .metrics import get_metric


@dataclass
class BacktestResult:
    """Aggregated (``summary``) and per-fold (``folds``) scores."""

    summary: pd.DataFrame
    folds: pd.DataFrame

    def __repr__(self):
        return f"BacktestResult(models={list(self.summary.index)})\n{self.summary}"


def backtest(forecaster, y, cv, metrics=("MASE", "RMSE"), sp: int = 1,
             baselines: bool = True) -> BacktestResult:
    """Backtest ``forecaster`` over ``cv`` folds, against automatic baselines.

    Returns a :class:`BacktestResult` whose ``summary`` is a model x metric
    table of mean scores (ascending in the first metric).
    """
    y = check_y(y)
    metric_fns = {m: get_metric(m) for m in metrics}

    entries = {type(forecaster).__name__: forecaster}
    if baselines:
        entries.setdefault("Naive", NaiveForecaster())
        entries.setdefault(f"SeasonalNaive(sp={sp})", SeasonalNaive(sp=sp))
        entries.setdefault("Drift", DriftForecaster())

    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(y)):
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        h = len(y_te)
        for label, model in entries.items():
            fitted = model.clone().fit(y_tr)
            y_pred = fitted.predict(h).values
            for mname, fn in metric_fns.items():
                rows.append({
                    "model": label, "fold": fold, "metric": mname,
                    "value": fn(y_te.values, y_pred, y_train=y_tr, sp=sp),
                })

    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["model", "metric"])["value"].mean()
        .unstack("metric")[list(metrics)]
        .sort_values(list(metrics)[0])
        .round(4)
    )
    return BacktestResult(summary=summary, folds=folds)


def evaluate(model, y, cv, metrics=("MASE", "RMSE"), sp: int = 1,
             baselines: bool = True) -> BacktestResult:
    """Alias for :func:`backtest` — the recommended one-liner."""
    return backtest(model, y, cv, metrics=metrics, sp=sp, baselines=baselines)
