"""Gradient-boosting regressor factories (the ``[ml]`` extra).

Each returns a configured, scikit-learn-compatible estimator to feed into
``make_reduction`` or ``GlobalForecaster``. Imports are lazy, with a clear
message if the extra is not installed.
"""
from __future__ import annotations


def _need(pkg: str):
    return ImportError(
        f"{pkg} is required — install the extra: pip install 'pytimeseries[ml]'"
    )


def LightGBM(**kwargs):
    """LightGBM regressor with forecasting-friendly defaults."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover
        raise _need("lightgbm") from exc
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
    params.update(kwargs)
    return LGBMRegressor(**params)


def XGBoost(**kwargs):
    """XGBoost regressor with forecasting-friendly defaults."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise _need("xgboost") from exc
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=3, verbosity=0)
    params.update(kwargs)
    return XGBRegressor(**params)


def CatBoost(**kwargs):
    """CatBoost regressor with forecasting-friendly defaults."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise _need("catboost") from exc
    params = dict(n_estimators=300, learning_rate=0.05, depth=3, verbose=0)
    params.update(kwargs)
    return CatBoostRegressor(**params)
