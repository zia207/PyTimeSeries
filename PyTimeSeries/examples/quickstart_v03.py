"""PyTimeSeries v0.3 quickstart: from-scratch deep forecasters, modern
architectures via neuralforecast, and conformal intervals for ANY point model."""
import warnings; warnings.simplefilter("ignore")
import pytimeseries as pt
from pytimeseries.datasets import make_series

y = make_series(n=180)

# 1) From-scratch deep forecaster (point) — same fit/predict contract
lstm = pt.LSTMForecaster(lags=12, hidden=24, epochs=120).fit(y)
print("LSTM 12-step:", lstm.predict(12).round(2).tolist())

# 2) Conformal intervals for a model with NO native uncertainty (the LSTM),
#    calibrated from its own walk-forward backtest residuals
cf = pt.ConformalForecaster(pt.LSTMForecaster(lags=12, hidden=24, epochs=80),
                            h=12, n_splits=3).fit(y)
print("\nConformal(LSTM) 12-step intervals:")
print(cf.predict_interval(12, level=[0.8, 0.95]).round(2).head())

# 3) Modern architecture via neuralforecast (fixed horizon)
nhits = pt.NHiTS(horizon=12, input_size=36, max_steps=200).fit(y)
print("\nN-HiTS 12-step:", nhits.predict(12).round(2).tolist())
