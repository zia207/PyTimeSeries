"""PyTimeSeries v0.2 quickstart: the ARIMA family, direct-vs-recursive
reduction with a gradient booster, and a global model over a panel."""
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pytimeseries as pt
from pytimeseries.datasets import make_series, make_panel

y = make_series(n=180)
cv = pt.ExpandingWindow(h=12, n_splits=5)

# 1) ARIMA family — automatic order selection, probabilistic output
auto = pt.AutoARIMA(seasonal=False).fit(y)
print("AutoARIMA order:", auto.order_)
print(pt.evaluate(pt.ARIMA(order=(1, 1, 1)), y, cv=cv, metrics=("MASE", "RMSE"), sp=12).summary)

# 2) Reduction with a booster — recursive vs direct, one-line swap
for strat in ("recursive", "direct"):
    f = pt.make_reduction(pt.ml.LightGBM(), strategy=strat, lags=12, max_horizon=12)
    res = pt.evaluate(f, y, cv=cv, metrics=("MASE",), sp=12)
    print(f"\nLightGBM [{strat}]  MASE={res.summary.loc[type(f).__name__, 'MASE']:.3f}")

# 3) Global model — ONE LightGBM across a whole panel
panel = make_panel(n_series=5, n=120)
gf = pt.GlobalForecaster(pt.ml.LightGBM(), lags=12).fit(panel)   # long/wide/dict all accepted
fc = gf.predict(12)
print("\nGlobal model: one fit ->", fc["unique_id"].nunique(), "series forecast,",
      len(fc), "rows")
print(fc.head(3).to_string(index=False))
