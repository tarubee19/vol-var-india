"""Forecast evaluation: robust loss functions and Diebold-Mariano comparison.

The evaluation target is the squared close-to-close return, an unbiased but noisy
proxy for the day's variance. QLIKE and MSE are *robust* losses (Patton, 2011):
they rank forecasts correctly even when the proxy is noisy, as long as the proxy
is conditionally unbiased. That is why no realized-variance measure is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from volvar.config import BENCHMARK_MODEL, MODEL_ORDER

LOSSES = ("qlike", "mse")


def qlike(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    """QLIKE loss ``ln(h) + y/h`` (lower is better; minimised in expectation at ``h = E[y]``)."""
    h = np.asarray(h, dtype=float)
    return np.log(h) + np.asarray(y, dtype=float) / h


def mse(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(h, dtype=float) - np.asarray(y, dtype=float)) ** 2


def loss_fn(name: str):
    if name == "qlike":
        return qlike
    if name == "mse":
        return mse
    raise ValueError(f"unknown loss '{name}'")


def aligned_wide(forecasts: pd.DataFrame, returns: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Wide frame on the *common* out-of-sample sample: one column per model plus ``r2``.

    Rows are indexed by ``(ticker, date)`` and kept only when every requested model
    has a forecast, so all models are scored on exactly the same observations.
    """
    wide = forecasts.pivot_table(index=["ticker", "date"], columns="model", values="var")
    wide = wide[[m for m in models if m in wide.columns]].dropna()
    r2 = returns.set_index(["ticker", "date"])["r2"]
    wide = wide.join(r2, how="inner")
    return wide.sort_index()


def loss_table(wide: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Mean QLIKE and MSE per model on the sample in ``wide``."""
    models = models or [m for m in MODEL_ORDER if m in wide.columns]
    y = wide["r2"].to_numpy()
    rows = {}
    for m in models:
        h = wide[m].to_numpy()
        rows[m] = {"qlike": float(qlike(h, y).mean()), "mse": float(mse(h, y).mean()), "n": len(wide)}
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int)
    return out


def _hac_mean_test(d: pd.Series, lags: int = 5) -> tuple[float, float, float]:
    """Mean, t-statistic and two-sided p-value of ``d`` using Newey-West errors."""
    d = d.dropna()
    if len(d) < 30 or float(d.std()) == 0.0:
        return float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan")
    fit = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0]), float(fit.pvalues[0])


def dm_vs_benchmark(
    wide: pd.DataFrame,
    models: list[str] | None = None,
    benchmark: str = BENCHMARK_MODEL,
    loss: str = "qlike",
    lags: int = 5,
) -> pd.DataFrame:
    """Diebold-Mariano style test of each model against ``benchmark`` (pooled over tickers).

    The loss differential ``L(benchmark) - L(model)`` is averaged across tickers on each
    date, then tested for a zero mean with Newey-West standard errors. A positive
    ``mean_diff`` means the model has lower loss than the benchmark.
    """
    models = [m for m in (models or MODEL_ORDER) if m in wide.columns and m != benchmark]
    f = loss_fn(loss)
    y = wide["r2"]
    base_loss = pd.Series(f(wide[benchmark].to_numpy(), y.to_numpy()), index=wide.index)
    rows = {}
    for m in models:
        m_loss = pd.Series(f(wide[m].to_numpy(), y.to_numpy()), index=wide.index)
        diff = (base_loss - m_loss).groupby(level="date").mean()
        mean, t, p = _hac_mean_test(diff, lags)
        rows[m] = {"mean_diff": mean, "t_stat": t, "p_value": p, "benchmark": benchmark, "loss": loss}
    return pd.DataFrame(rows).T


def per_ticker_losses(
    wide: pd.DataFrame, loss: str = "qlike", models: list[str] | None = None
) -> pd.DataFrame:
    """Mean loss per ticker and model (rows: ticker, columns: model)."""
    f = loss_fn(loss)
    models = models or [m for m in MODEL_ORDER if m in wide.columns]
    out = {}
    for ticker, g in wide.groupby(level="ticker"):
        y = g["r2"].to_numpy()
        out[ticker] = {m: float(f(g[m].to_numpy(), y).mean()) for m in models}
    return pd.DataFrame(out).T.rename_axis("ticker")


def best_model_counts(per_ticker: pd.DataFrame) -> pd.Series:
    """How many tickers each model wins (lowest mean loss)."""
    return per_ticker.idxmin(axis=1).value_counts().reindex(per_ticker.columns, fill_value=0)


def market_regimes(returns: pd.DataFrame, window: int = 22) -> pd.Series:
    """Tercile label (calm/normal/turbulent) from trailing market variance known the day before."""
    wide = returns.pivot(index="date", columns="ticker", values="ret")
    mkt = wide.mean(axis=1, skipna=True)
    trailing = (mkt**2).rolling(window, min_periods=window).mean().shift(1)
    q1, q2 = trailing.quantile([1 / 3, 2 / 3])
    labels = pd.Series("normal", index=trailing.index)
    labels[trailing <= q1] = "calm"
    labels[trailing > q2] = "turbulent"
    labels[trailing.isna()] = np.nan
    return labels.rename("regime")


def loss_by_regime(
    wide: pd.DataFrame,
    regimes: pd.Series,
    loss: str = "qlike",
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Mean loss per model within each volatility regime."""
    f = loss_fn(loss)
    models = models or [m for m in MODEL_ORDER if m in wide.columns]
    dates = wide.index.get_level_values("date")
    reg = regimes.reindex(dates).to_numpy()
    out = {}
    for label in ("calm", "normal", "turbulent"):
        mask = reg == label
        if mask.sum() == 0:
            continue
        sub = wide[mask]
        y = sub["r2"].to_numpy()
        out[label] = {m: float(f(sub[m].to_numpy(), y).mean()) for m in models}
    return pd.DataFrame(out).T.rename_axis("regime")


def loss_in_window(
    wide: pd.DataFrame, start: str, end: str, models: list[str] | None = None
) -> pd.DataFrame | None:
    """Loss table restricted to a date window (e.g. the COVID crash); ``None`` if empty."""
    dates = wide.index.get_level_values("date")
    sub = wide[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if len(sub) < 50:
        return None
    return loss_table(sub, models)
