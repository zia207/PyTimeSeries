# PyTimeSeries

A unified, **tutorial-driven** time-series forecasting library — every method
from the 12-part *Time Series Analysis in Python* course behind **one API**,
from `NaiveForecaster` to `PatchTST`, with backtesting, probabilistic output,
and global (multi-series) training as first-class citizens.

> **Status: v0.5 — feature-complete.** Adds the **foundation-model** clients
> (`[foundation]`): `Chronos`, `TimesFM`, `Moirai`, `TimeGPT` (zero-shot, lazy
> backends), and the **deployment** layer (Part 12): `save`/`load` with metadata,
> `DriftMonitor`, `RetrainPolicy`, and serving helpers (`batch_forecast`, FastAPI/
> Streamlit templates). The full tutorial arc — Naive → foundation models →
> deployment — now lives behind one `BaseForecaster` contract.

## Design principles

1. **Baselines are mandatory** — `evaluate()` attaches naive/seasonal-naive/drift automatically.
2. **Walk-forward by default** — the only evaluation entry point is backtesting; no single-split footgun.
3. **Match capacity to data** — one API makes swapping families a one-line change.

## Documentation & tutorials

- **User guide (12 tutorials):** https://zia207.github.io/PyTimeSeries — each part demonstrates
  the package API on one shared dataset, scored with the same walk-forward backtest.
  Source notebooks live in [`Tutorials/`](Tutorials); rendered HTML in [`docs/Tutorials/`](docs/Tutorials).
- **Distributions:** `dist/pytimeseries-0.5.0-py3-none-any.whl` and `dist/pytimeseries-0.5.0.tar.gz`
  (`pip install dist/pytimeseries-0.5.0-py3-none-any.whl`).

## Install

```bash
pip install -e .            # core: numpy, pandas, scikit-learn, statsmodels
pip install -e ".[ml]"      # + lightgbm, xgboost, catboost, pmdarima   (v0.2)
pip install -e ".[dl]"      # + torch, neuralforecast                   (v0.3)
pip install -e ".[extra]"   # + arch, prophet, ruptures, hierarchicalforecast, tsfresh
```

## Quickstart

```python
import pytimeseries as pt
from pytimeseries.datasets import make_series

y = make_series(n=180)                       # synthetic monthly series
cv = pt.ExpandingWindow(h=12, n_splits=5)    # walk-forward CV

# swap families behind one API; baselines are added automatically
res = pt.evaluate(pt.ETS(seasonal="add", seasonal_periods=12), y, cv=cv,
                  metrics=("MASE", "RMSE"), sp=12)
print(res.summary)

# calibrated probabilistic forecast
pt.ETS(seasonal="add", seasonal_periods=12).fit(y).predict_interval(24, level=[0.8, 0.95])
```

### v0.2 — ARIMA family, boosting, and global models

```python
import pytimeseries as pt

# ARIMA / SARIMAX / automatic order selection
pt.ARIMA(order=(2, 1, 2), seasonal_order=(1, 0, 0, 12)).fit(y).predict_interval(12)
pt.AutoARIMA(seasonal=True, m=12).fit(y).predict(12)

# gradient boosting through the reduction (recursive or direct)
pt.make_reduction(pt.ml.LightGBM(), lags=12).fit(y).predict(12)
pt.make_reduction(pt.ml.XGBoost(), strategy="direct", lags=12, max_horizon=12).fit(y).predict(12)

# one global model trained across many series (long / wide / dict panels)
gf = pt.GlobalForecaster(pt.ml.LightGBM(), lags=12).fit(panel)   # panel = {id: Series}
gf.predict(12)      # -> long DataFrame of forecasts for every series
```

## New in v0.2 — ARIMA, boosting, and global models

```python
import pytimeseries as pt
from pytimeseries.datasets import make_series, make_panel

y = make_series(n=180)

# ARIMA family (probabilistic, exogenous-aware)
pt.AutoARIMA(seasonal=True, m=12).fit(y).predict_interval(12, level=[0.8, 0.95])

# reduction with a gradient booster — recursive OR direct, one-line swap
pt.make_reduction(pt.ml.LightGBM(), strategy="direct", lags=12, max_horizon=12).fit(y).predict(12)

# ONE model across a whole panel (long / wide / dict all accepted)
panel = make_panel(n_series=5)
fc = pt.GlobalForecaster(pt.ml.LightGBM(), lags=12).fit(panel).predict(12)   # long DataFrame
```

## New in v0.3 — deep learning + conformal intervals

```python
import pytimeseries as pt
from pytimeseries.datasets import make_series
y = make_series(n=180)

# from-scratch deep forecaster (torch) — same fit/predict contract
pt.LSTMForecaster(lags=12, epochs=200).fit(y).predict(12)

# calibrated intervals for ANY point model (trees, nets) from its backtest residuals
pt.ConformalForecaster(pt.LSTMForecaster(epochs=150), h=12, n_splits=5).fit(y) \
  .predict_interval(12, level=[0.8, 0.95])

# modern architecture via neuralforecast (fixed horizon)
pt.NHiTS(horizon=12, input_size=36, max_steps=300).fit(y).predict(12)
```

## New in v0.3 — deep learning & conformal intervals

```python
import pytimeseries as pt
from pytimeseries.datasets import make_series
y = make_series(n=180)

# from-scratch sequence models (torch) — same API
pt.deep.LSTMForecaster(lags=12).fit(y).predict(12)
pt.deep.TCNForecaster(lags=12).fit(y).predict(12)
pt.deep.Seq2SeqForecaster(lags=12, horizon=24).fit(y).predict(24)   # direct multi-step

# modern architectures via neuralforecast
pt.modern.NHiTS(horizon=24, input_size=48, max_steps=300).fit(y).predict(24)

# calibrated intervals for ANY point model (trees, nets, ...)
pt.ConformalForecaster(pt.deep.LSTMForecaster(), h=12, n_splits=5).fit(y).predict_interval(12, level=[0.8, 0.95])
```

## New in v0.5 — deployment & foundation models

```python
import pytimeseries as pt
from pytimeseries.datasets import make_series
y = make_series(n=180)

# deployment loop (Part 12)
m = pt.ETS(seasonal="add", seasonal_periods=12).fit(y)
pt.deploy.save(m, "model.joblib", metadata={"owner": "team"})
pt.deploy.batch_forecast("model.joblib", h=24)          # serving: batch job
mon = pt.deploy.DriftMonitor(threshold=2.0).fit(y[:120])
pol = pt.deploy.RetrainPolicy(every=12, drift_monitor=mon)
pol.should_retrain(step=12, window=y[-12:])

# zero-shot foundation models (needs [foundation])
pt.foundation.Chronos("amazon/chronos-t5-small").fit(y).predict_interval(24, level=[0.8, 0.95])
pt.foundation.TimeGPT(api_key="...").fit(y).predict(24)
```

## The forecaster contract

Every model implements the same interface (`pytimeseries.BaseForecaster`):

```
fit(y, X=None) -> self
predict(h, X=None) -> Series
predict_interval(h, level=[0.8, 0.95]) -> DataFrame
predict_quantiles(h, q=[0.1, 0.5, 0.9]) -> DataFrame
update(y_new, refit=False) -> self
```

Subclasses implement only `_fit` and `_predict` (plus `_predict_quantiles` for
native probabilistic output). `clone()`/`get_params()` mirror scikit-learn so
backtesting refits a fresh model on every fold.

## Layout

```
pytimeseries/
├── base.py          BaseForecaster ABC
├── baselines/       Naive, SeasonalNaive, Drift, Mean          (Part 6)
├── statistical/     ETS, ARIMA, SARIMAX, AutoARIMA             (Parts 3–4)
├── reduction/       make_reduction (recursive + direct)        (Parts 6–8)
├── ml/              LightGBM/XGBoost/CatBoost, GlobalForecaster (Part 8)
├── deep/            MLP/RNN/LSTM/GRU/TCN/Seq2Seq (torch)         (Part 9)
├── modern/          NBEATS/NHiTS/PatchTST (neuralforecast)       (Part 10)
├── probabilistic/   ConformalForecaster (any model -> intervals) (Part 12)
├── advanced/        GARCH, UnobservedComponents, Prophet        (Part 5)
├── multivariate/    VAR, VECM, Granger, cointegration           (Part 5)
├── hierarchical/    reconciliation (bottom-up/top-down/MinT)    (Part 11)
├── detect/          change-point & anomaly detection            (Part 11)
├── intermittent/    Croston, TSB, ADIDA                          (Part 11)
├── foundation/      Chronos, TimesFM, Moirai, TimeGPT (zero-shot)(Part 11)
├── deploy/          save/load, DriftMonitor, RetrainPolicy, serve (Part 12)
├── evaluation/      metrics, splitters, backtest engine        (Part 6)
├── datasets.py      synthetic generator
└── utils/           validation & index helpers
```

MIT licensed · author: Zia U. Ahmed · https://github.com/zia207/pytimeseries
