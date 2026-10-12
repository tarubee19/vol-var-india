"""One-day Value-at-Risk and Expected Shortfall from a variance forecast.

Two ways of turning a volatility forecast ``sigma_t`` into a loss quantile:

* **normal** - assume standardized returns are standard normal.
* **fhs** - filtered historical simulation: standardize past returns by the model's own
  past forecasts (``z_j = r_j / sigma_j``) and read the empirical lower quantile and tail
  mean from a trailing window. This keeps the model's volatility dynamics but lets the
  data decide the tail shape, which matters because equity returns are fat-tailed.

VaR and ES are reported as **positive percent losses**. For day ``t`` only returns up to
``t-1`` are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def normal_var_es(sigma: pd.Series, level: float) -> tuple[pd.Series, pd.Series]:
    """VaR and ES under normality: ``VaR = sigma * z``, ``ES = sigma * phi(z) / (1 - level)``."""
    _check_level(level)
    z = stats.norm.ppf(level)
    es_mult = stats.norm.pdf(z) / (1.0 - level)
    return (sigma * z).rename("var"), (sigma * es_mult).rename("es")


def fhs_var_es(
    ret: pd.Series,
    variance: pd.Series,
    level: float,
    window: int = 500,
    min_obs: int = 250,
) -> tuple[pd.Series, pd.Series]:
    """Filtered-historical-simulation VaR and ES for each date in ``variance``.

    Args:
        ret: percent returns indexed by date.
        variance: one-day-ahead variance forecasts (percent squared), indexed by target date.
        level: confidence level, e.g. 0.99.
        window: trailing number of standardized returns used for the empirical tail.
        min_obs: minimum standardized returns required before a VaR is produced.
    """
    _check_level(level)
    if min_obs > window:
        raise ValueError("min_obs cannot exceed window")
    var = variance.dropna()
    sigma = np.sqrt(var.to_numpy())
    r = ret.reindex(var.index).to_numpy(dtype=float)
    z = r / sigma
    n = len(z)
    var_out = np.full(n, np.nan)
    es_out = np.full(n, np.nan)
    tail = 1.0 - level
    for i in range(min_obs, n):
        w = z[max(0, i - window) : i]  # strictly before day i
        w = w[np.isfinite(w)]
        if len(w) < min_obs:
            continue
        q = np.quantile(w, tail)
        tail_vals = w[w <= q]
        var_out[i] = -sigma[i] * q
        es_out[i] = -sigma[i] * tail_vals.mean()
    idx = var.index
    return pd.Series(var_out, index=idx, name="var"), pd.Series(es_out, index=idx, name="es")


def var_es(
    ret: pd.Series,
    variance: pd.Series,
    level: float,
    method: str = "fhs",
    window: int = 500,
    min_obs: int = 250,
) -> tuple[pd.Series, pd.Series]:
    """Dispatch to the chosen method (``"fhs"`` or ``"normal"``)."""
    if method == "normal":
        return normal_var_es(np.sqrt(variance.dropna()), level)
    if method == "fhs":
        return fhs_var_es(ret, variance, level, window, min_obs)
    raise ValueError("method must be 'fhs' or 'normal'")


def _check_level(level: float) -> None:
    if not 0.5 < level < 1.0:
        raise ValueError("level must be in (0.5, 1)")
