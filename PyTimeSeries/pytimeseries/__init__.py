"""PyTimeSeries — a unified, tutorial-driven forecasting library (v0.4)."""
from .base import BaseForecaster
from .baselines import DriftForecaster, MeanForecaster, NaiveForecaster, SeasonalNaive
from .statistical import ETS, ARIMA, SARIMAX, AutoARIMA
from .reduction import (
    RecursiveReductionForecaster, DirectReductionForecaster, make_reduction,
)
from .evaluation import ExpandingWindow, SlidingWindow, backtest, evaluate
from .evaluation import metrics
from . import ml
from .ml import GlobalForecaster
from .deep import (
    MLPForecaster, RNNForecaster, LSTMForecaster, GRUForecaster,
    TCNForecaster, Seq2SeqForecaster,
)
from .modern import NBEATS, NHiTS, PatchTST
from .probabilistic import ConformalForecaster
# --- v0.4: advanced statistical, multivariate, hierarchical, detect, intermittent ---
from . import advanced, multivariate, hierarchical, detect, intermittent
from .advanced import GARCH, UnobservedComponents, Prophet
from .multivariate import VAR, VECM, granger_causality, cointegration_test
from .hierarchical import build_summing_matrix, reconcile, HierarchicalReconciler
from .detect import detect_anomalies, detect_change_points
from .intermittent import Croston, TSB, ADIDA
from . import foundation, deploy
from .deploy import save, load, DriftMonitor, RetrainPolicy
from . import datasets

__version__ = "0.5.0"
__all__ = [
    "BaseForecaster",
    "NaiveForecaster", "SeasonalNaive", "DriftForecaster", "MeanForecaster",
    "ETS", "ARIMA", "SARIMAX", "AutoARIMA",
    "RecursiveReductionForecaster", "DirectReductionForecaster", "make_reduction",
    "GlobalForecaster", "ml",
    "MLPForecaster", "RNNForecaster", "LSTMForecaster", "GRUForecaster",
    "TCNForecaster", "Seq2SeqForecaster",
    "NBEATS", "NHiTS", "PatchTST",
    "ConformalForecaster",
    "GARCH", "UnobservedComponents", "Prophet",
    "VAR", "VECM", "granger_causality", "cointegration_test",
    "build_summing_matrix", "reconcile", "HierarchicalReconciler",
    "detect_anomalies", "detect_change_points",
    "Croston", "TSB", "ADIDA",
    "ExpandingWindow", "SlidingWindow", "backtest", "evaluate", "metrics",
    "foundation", "deploy", "save", "load", "DriftMonitor", "RetrainPolicy",
    "advanced", "multivariate", "hierarchical", "detect", "intermittent", "datasets",
]
