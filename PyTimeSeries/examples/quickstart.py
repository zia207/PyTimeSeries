"""PyTimeSeries v0.1 quickstart: swap model families behind one API,
backtest against automatic baselines, and get calibrated ETS intervals."""
import pytimeseries as pt
from pytimeseries.datasets import make_series
from sklearn.ensemble import RandomForestRegressor

y = make_series(n=180)                 # synthetic monthly series

# 1) Same CV, three families — one-line swap
cv = pt.ExpandingWindow(h=12, n_splits=5)
for model in [
    pt.ETS(trend="add", seasonal="add", seasonal_periods=12),
    pt.make_reduction(RandomForestRegressor(n_estimators=200, random_state=0), lags=12),
    pt.NaiveForecaster(),
]:
    res = pt.evaluate(model, y, cv=cv, metrics=("MASE", "RMSE"), sp=12)
    print(f"\n=== {type(model).__name__} (with baselines) ===")
    print(res.summary)

# 2) Probabilistic forecast from ETS
ets = pt.ETS(trend="add", seasonal="add", seasonal_periods=12).fit(y)
print("\nETS 24-step forecast with intervals:")
print(ets.predict_interval(24, level=[0.8, 0.95]).head())

# 3) Global (cross-series) model — one LightGBM across a panel of series
panel = {
    "A": make_series(n=140, seed=1, trend=0.10, level=50),
    "B": make_series(n=140, seed=2, trend=-0.05, level=80),
    "C": make_series(n=140, seed=3, trend=0.02, level=30),
}
try:
    gf = pt.GlobalForecaster(pt.ml.LightGBM(n_estimators=150), lags=12).fit(panel)
    print("\nGlobal LightGBM — 6-step forecast for every series:")
    print(gf.predict(6).head())
except ImportError:
    print("\n(install the [ml] extra to run the global LightGBM demo)")
