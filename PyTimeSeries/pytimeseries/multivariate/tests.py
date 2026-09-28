"""Granger-causality and cointegration tests (Part 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def granger_causality(df: pd.DataFrame, cause: str, effect: str, maxlag: int = 4) -> dict:
    """Does the past of ``cause`` help predict ``effect``? (Predictive precedence,
    not mechanism.) Returns the F-test p-value and a boolean at the 5% level."""
    from statsmodels.tsa.api import VAR

    # Fit at exactly `maxlag` lags (not IC-selected) so the causality test always
    # has >= 1 lag to work with, and tests precedence at the intended horizon.
    res = VAR(df).fit(maxlag)
    test = res.test_causality(effect, [cause], kind="f")
    return {"cause": cause, "effect": effect, "pvalue": float(test.pvalue),
            "granger_causal": bool(test.pvalue < 0.05)}


def cointegration_test(df: pd.DataFrame, method: str = "johansen",
                       det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Test for a long-run equilibrium among the columns.

    ``method="johansen"`` returns the estimated cointegration rank;
    ``method="engle_granger"`` returns the pairwise p-value (2 columns)."""
    if method == "engle_granger":
        from statsmodels.tsa.stattools import coint

        if df.shape[1] != 2:
            raise ValueError("Engle–Granger expects exactly two columns.")
        stat, pval, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
        return {"method": "engle_granger", "pvalue": float(pval),
                "cointegrated": bool(pval < 0.05)}

    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    joh = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)
    rank = int(np.sum(joh.lr1 > joh.cvt[:, 1]))       # trace stat vs 95% critical
    return {"method": "johansen", "rank": rank,
            "trace_stat": joh.lr1.tolist(), "crit_95": joh.cvt[:, 1].tolist()}
