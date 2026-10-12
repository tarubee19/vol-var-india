"""HAR-style regression forecaster using range-based (Garman-Klass) predictors.

The Heterogeneous AutoRegressive idea (Corsi, 2009) predicts next-day variance
from daily, weekly and monthly averages of past variance. Here the predictors
are Garman-Klass range variances plus the lagged squared return, and the target
is the next-day squared return so it is evaluated on the same footing as every
other model. Coefficients are re-estimated by OLS every ``refit_every`` days
using only targets that are already known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["gk_d", "gk_w", "gk_m", "r2_d"]


def har_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Predictors known at the close of day ``t`` (aligned to day ``t``)."""
    gk, r2 = panel["gk"], panel["r2"]
    return pd.DataFrame(
        {
            "gk_d": gk,
            "gk_w": gk.rolling(5, min_periods=5).mean(),
            "gk_m": gk.rolling(22, min_periods=22).mean(),
            "r2_d": r2,
        }
    )


def har_variance(
    panel: pd.DataFrame,
    min_train: int = 1000,
    refit_every: int = 63,
    floor_series: pd.Series | None = None,
) -> pd.Series:
    """Walk-forward HAR forecast for each out-of-sample day.

    Args:
        panel: frame with ``gk`` and ``r2`` columns (see :func:`volvar.data.make_panel`).
        floor_series: optional positive series (e.g. historical variance); forecasts are
            floored at 10% of it so the variance stays strictly positive.
    """
    n = len(panel)
    if n <= min_train + 1:
        raise ValueError(f"need more than min_train+1={min_train + 1} observations, got {n}")

    x_all = har_features(panel).shift(1)  # features for target day t come from day t-1
    y_all = panel["r2"]
    out = np.full(n, np.nan)

    t0 = min_train
    while t0 < n:
        t1 = min(t0 + refit_every, n)
        x_tr = x_all.iloc[:t0].dropna()
        y_tr = y_all.loc[x_tr.index]
        design = np.column_stack([np.ones(len(x_tr)), x_tr.to_numpy()])
        coef, *_ = np.linalg.lstsq(design, y_tr.to_numpy(), rcond=None)
        x_te = x_all.iloc[t0:t1]
        pred = np.column_stack([np.ones(len(x_te)), x_te.to_numpy()]) @ coef
        out[t0:t1] = pred
        t0 = t1

    s = pd.Series(out, index=panel.index, name="har")
    floor = 0.1 * floor_series.reindex(s.index) if floor_series is not None else 1e-4
    return s.clip(lower=floor)
