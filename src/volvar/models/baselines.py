"""Benchmark variance forecasters: rolling historical variance and RiskMetrics EWMA."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_variance(r2: pd.Series, window: int = 250) -> pd.Series:
    """Mean of the previous ``window`` squared returns (zero-mean assumption)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    return r2.rolling(window, min_periods=window).mean().shift(1).rename("hist")


def ewma_variance(r2: pd.Series, lam: float = 0.94, init_window: int = 30) -> pd.Series:
    """RiskMetrics EWMA: ``s_t = lam * s_{t-1} + (1 - lam) * r2_{t-1}``.

    ``s_t`` is the forecast for day ``t``. The recursion is seeded with the mean of
    the first ``init_window`` squared returns, and forecasts inside that seed
    window are left as NaN so no future information leaks into early values.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    x = r2.to_numpy(dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n <= init_window:
        return pd.Series(out, index=r2.index, name="ewma")
    out[init_window] = x[:init_window].mean()
    for t in range(init_window + 1, n):
        out[t] = lam * out[t - 1] + (1.0 - lam) * x[t - 1]
    return pd.Series(out, index=r2.index, name="ewma")
