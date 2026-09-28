"""Reproducible model persistence with metadata (Part 12)."""
from __future__ import annotations

import datetime as _dt

import joblib


def save(model, path, metadata: dict | None = None):
    """Persist a fitted forecaster plus metadata (version, timestamp, repr)."""
    import pytimeseries as _pt

    meta = {
        "pytimeseries_version": _pt.__version__,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "model_repr": repr(model),
        **(metadata or {}),
    }
    joblib.dump({"model": model, "metadata": meta}, path)
    return path


def load(path, with_metadata: bool = False):
    """Load a model saved with :func:`save`. Returns the model, or
    ``(model, metadata)`` when ``with_metadata=True``."""
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return (bundle["model"], bundle["metadata"]) if with_metadata else bundle["model"]
    return (bundle, {}) if with_metadata else bundle      # tolerate a bare dumped model
