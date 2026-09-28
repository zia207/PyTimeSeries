"""PyTimeSeries v0.5 quickstart: the deployment loop + zero-shot foundation models.

The deployment parts run with core deps; the foundation clients need the
``[foundation]`` extra (model weights / API key), so they're shown but not run.
"""
import warnings; warnings.simplefilter("ignore")
import pytimeseries as pt
from pytimeseries.datasets import make_series

y = make_series(n=180)

# --- Deployment loop (Part 12) ---
model = pt.ETS(trend="add", seasonal="add", seasonal_periods=12).fit(y[:-24])
pt.deploy.save(model, "model.joblib", metadata={"owner": "forecasting-team"})
reloaded = pt.deploy.load("model.joblib")
print("reloaded forecasts identically:",
      bool((reloaded.predict(6).round(6) == model.predict(6).round(6)).all()))

fc = pt.deploy.batch_forecast("model.joblib", h=12)     # serving: batch job
print("batch forecast (first 3):", fc.round(2).head(3).tolist())

monitor = pt.deploy.DriftMonitor(threshold=2.0).fit(y[:120])
policy = pt.deploy.RetrainPolicy(every=12, drift_monitor=monitor, drift_threshold=2.0)
print("retrain @ step 12?", policy.should_retrain(step=12))
print("retrain on shifted window?",
      policy.should_retrain(window=make_series(n=12, level=200, trend=0, season_amp=0)))

# --- Zero-shot foundation models (needs [foundation] extra) ---
# ch = pt.foundation.Chronos("amazon/chronos-t5-small")
# ch.fit(y).predict_interval(24, level=[0.8, 0.95])       # no training — pretrained
# pt.foundation.TimeGPT(api_key="...").fit(y).predict(24)  # API-served

print("\nFastAPI serving template:\n" + pt.deploy.FASTAPI_TEMPLATE)
