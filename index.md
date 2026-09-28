---
title: "Time Series Analysis in Python"
subtitle: "From Statistical Modeling to Machine Learning — a 12-part applied tutorial"
toc: false
---

A hands-on, code-first course that builds time-series forecasting from first
principles all the way to the state of the art. Every part is an **executed
Jupyter notebook** — read it here as a web page, or open the `.ipynb` to run it
yourself.

## One dataset, twelve lenses

The whole series works a single real dataset — **356 months of global commodity
prices (Jan 1992 – Aug 2021)**: fertilizers (urea, MP, TSP), crops (maize, rice,
wheat), and market indices (food, energy, VIX). Using one dataset throughout lets
each method be compared *honestly* against the last, under a consistent
**walk-forward backtesting** discipline.

## The parts

| Part | What it covers |
|------|----------------|
| **[1. Foundations & Data Handling](notebooks/part-01-foundations.ipynb)** | Handling time-indexed data: DatetimeIndex, resampling, rolling windows, gaps, and the diagnostic plots (plus an STL decomposition). |
| **[2. Statistical Properties & Decomposition](notebooks/part-02-stationarity.ipynb)** | Stationarity, ACF/PACF, unit-root tests (ADF, KPSS, Phillips–Perron), and transformations (log, Box–Cox, differencing). |
| **[3. Smoothing & Exponential Methods](notebooks/part-03-smoothing.ipynb)** | The first forecasters: moving averages, SES, Holt, Holt–Winters, and the ETS framework — with baselines and prediction intervals. |
| **[4. The ARIMA Family](notebooks/part-04-arima.ipynb)** | AR/MA/ARIMA/SARIMA/SARIMAX — Box–Jenkins identification from ACF/PACF, order selection by AIC/BIC and auto_arima, residual diagnostics. |
| **[5. Advanced Statistical & Multivariate Models](notebooks/part-05-advanced-multivariate.ipynb)** | GARCH volatility, state-space / Kalman, VAR, cointegration & VECM, Granger causality, and Prophet. |
| **[6. Evaluation, Backtesting & Baselines](notebooks/part-06-evaluation-backtesting.ipynb)** | Why a single split misleads: baselines, the metric zoo (MASE, pinball), walk-forward cross-validation, leakage traps, and multi-step strategies. |
| **[7. Feature Engineering & the ML Reduction](notebooks/part-07-feature-engineering.ipynb)** | The ML reduction — turning a series into a supervised feature matrix (lags, rolling stats, calendar/Fourier, exogenous), plus tsfresh. |
| **[8. Machine Learning Forecasting](notebooks/part-08-ml-forecasting.ipynb)** | Regularized linear models and gradient-boosted trees (XGBoost/LightGBM/CatBoost), and the pivotal global-vs-local model idea. |
| **[9. Deep Learning for Sequences](notebooks/part-09-deep-learning.ipynb)** | Sequence models from scratch in PyTorch: MLP, 1D-CNN/TCN, RNN/LSTM/GRU, and seq2seq — the first models to beat the baseline. |
| **[10. Modern Deep Forecasting Architectures](notebooks/part-10-modern-dl.ipynb)** | State-of-the-art architectures via neuralforecast — Transformers, N-BEATS/N-HiTS, PatchTST — probabilistic and global training. |
| **[11. Specialized Topics](notebooks/part-11-specialized-topics.ipynb)** | Hierarchical reconciliation, change-point & anomaly detection, intermittent demand (Croston), long horizons, and foundation models. |
| **[12. Deployment & Capstone](notebooks/part-12-deployment-capstone.ipynb)** | From notebook to system: reproducible pipelines, persistence, retraining, drift monitoring, serving — and the grand capstone benchmark. |

## How it's organized

- **Foundations (1–2)** — handling time-indexed data; stationarity and decomposition.
- **Classical forecasting (3–5)** — smoothing/ETS, the ARIMA family, and advanced statistical & multivariate models.
- **Rigorous evaluation (6)** — baselines, metrics, and backtesting: the spine that makes every comparison trustworthy.
- **Machine learning (7–8)** — the feature-engineering reduction, gradient boosting, and global models.
- **Deep learning (9–10)** — sequence models from scratch, then modern architectures with probabilistic, global training.
- **Frontiers & practice (11–12)** — hierarchical/anomaly/intermittent/foundation models, and the deployment loop.

## A recurring, honest theme

On this short, shock-driven series, **almost nothing reliably beat a simple drift
baseline** until a model was either given more data (Part 10's *global* N-HiTS) or
the right inductive bias (Part 9's LSTM). That is the discipline the series keeps
returning to:

1. **Always beat the baselines** — if you can't beat drift, you don't have a model.
2. **Trust walk-forward, not a single split** — the evaluation *procedure* decides the verdict.
3. **Match model capacity to data scale** — complexity is a cost you pay for scale you actually have.

## Running the notebooks

```bash
pip install -r requirements.txt      # see each notebook's imports
jupyter lab                          # open any notebook under notebooks/
```

The dataset (`monthly_price.csv`) sits alongside the notebooks, so they run
as-is. Standalone rendered pages are also in the [`html/`](html) folder.

## Building this website

```bash
quarto render        # builds the site into ./docs (ready for GitHub Pages)
quarto preview       # live local preview
```

Then enable **GitHub Pages → Deploy from branch → /docs** in your repository
settings.
