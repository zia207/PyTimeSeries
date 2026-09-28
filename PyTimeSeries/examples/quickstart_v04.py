"""PyTimeSeries v0.4 quickstart: advanced statistical & multivariate models,
hierarchical reconciliation, detection, and intermittent demand ([extra])."""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
import pytimeseries as pt
from pytimeseries.datasets import make_series, make_multivariate

y = make_series(n=200)

# --- Advanced statistical (Part 5) ---
uc = pt.UnobservedComponents(level="local linear trend", seasonal=12).fit(y)
print("UnobservedComponents 6-step:", uc.predict(6).round(2).tolist())
ret = 100 * np.log(y.clip(lower=1)).diff().dropna()
g = pt.GARCH(p=1, q=1).fit(ret)
print("GARCH conditional-vol forecast:", np.sqrt(
      g._res.forecast(horizon=3, reindex=False).variance.values[-1]).round(3).tolist())

# --- Multivariate (Part 5) ---
df = make_multivariate(n=200, k=3)
print("VAR lag order:", pt.VAR(maxlags=6).fit(df.diff().dropna()).lag_order_)
print("Granger y1->y2 p-value:", round(pt.granger_causality(df.diff().dropna(), "y1", "y2")["pvalue"], 3))
print("Johansen cointegration rank:", pt.cointegration_test(df)["rank"])
print("VECM 4-step:\n", pt.VECM(coint_rank=1).fit(df).predict(4).round(1))

# --- Hierarchical reconciliation (Part 11) ---
rec = pt.HierarchicalReconciler(bottom_ids=["A", "B", "C"])
base = pd.DataFrame(np.array([[10, 11], [4, 5], [3, 3], [2, 2]], float), index=rec.node_ids)
coherent = rec.reconcile(base, method="ols")            # MinT/OLS
print("\nReconciled (Total == A+B+C):",
      np.allclose(coherent.loc["Total"], coherent.loc[["A", "B", "C"]].sum()))

# --- Detection (Part 11) ---
shocked = y.copy(); shocked.iloc[120] += 50
print("Anomalies:", [d.date().isoformat() for d in pt.detect_anomalies(shocked, z_thresh=3).index])
print("Change-points:", [d.date().isoformat() for d in pt.detect_change_points(y, n_bkps=3)])

# --- Intermittent demand (Part 11) ---
rng = np.random.default_rng(0)
demand = pd.Series(np.where(rng.random(80) < 0.3, rng.integers(1, 6, 80), 0).astype(float),
                   index=pd.date_range("2015-01-01", periods=80, freq="MS"))
for M in (pt.Croston(), pt.TSB(), pt.ADIDA()):
    print(f"{type(M).__name__:8s} demand-rate forecast:", round(M.fit(demand).predict(3).iloc[0], 3))
