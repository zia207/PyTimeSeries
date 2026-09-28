"""Serving helpers (Part 12): a runnable batch job, plus API/dashboard
templates you can drop into a service."""
from __future__ import annotations

from .persistence import load


def batch_forecast(model_path, h: int, out_path=None):
    """Load a persisted model, forecast ``h`` steps, optionally write a CSV."""
    model = load(model_path)
    fc = model.predict(int(h))
    if out_path is not None:
        fc.to_frame("forecast").to_csv(out_path)
    return fc


FASTAPI_TEMPLATE = '''\
# app.py — run: uvicorn app:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI
from pytimeseries.deploy import load

app = FastAPI()
model = load("model.joblib")            # loaded once at startup

@app.get("/forecast")
def forecast(h: int = 24):
    fc = model.predict(h)
    return {"dates": [d.isoformat() for d in fc.index], "forecast": fc.tolist()}
'''

STREAMLIT_TEMPLATE = '''\
# dashboard.py — run: streamlit run dashboard.py
import streamlit as st
from pytimeseries.deploy import load

model = load("model.joblib")
h = st.slider("Horizon", 1, 36, 24)
st.line_chart(model.predict(h))
'''
