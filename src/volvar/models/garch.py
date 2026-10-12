"""GARCH-family forecasters with walk-forward re-estimation.

Parameters are re-estimated on an expanding window every ``refit_every`` days.
Between refits the conditional variance is *filtered* forward with the fixed
parameters using only observed returns, which gives a true one-step-ahead
forecast for every out-of-sample day without look-ahead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model

Kind = Literal["garch", "gjr"]


@dataclass
class GarchOutput:
    variance: pd.Series
    refits: list[dict] = field(default_factory=list)

    @property
    def last_params(self) -> dict:
        return self.refits[-1]["params"] if self.refits else {}

    @property
    def n_failed_fits(self) -> int:
        return sum(1 for r in self.refits if not r["converged"])


def _spec(returns: pd.Series, kind: Kind, dist: str):
    o = 1 if kind == "gjr" else 0
    return arch_model(returns, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False)


def garch_variance(
    ret: pd.Series,
    kind: Kind = "garch",
    min_train: int = 1000,
    refit_every: int = 63,
    dist: str = "t",
) -> GarchOutput:
    """Walk-forward one-day-ahead variance from GARCH(1,1) or GJR-GARCH(1,1,1).

    Args:
        ret: percent log returns.
        kind: ``"garch"`` or ``"gjr"`` (asymmetric response to negative returns).
        min_train: observations used before the first forecast.
        refit_every: re-estimate parameters this often (trading days).
        dist: innovation distribution (``"t"`` by default for fat tails).
    """
    if kind not in ("garch", "gjr"):
        raise ValueError("kind must be 'garch' or 'gjr'")
    n = len(ret)
    if n <= min_train + 1:
        raise ValueError(f"need more than min_train+1={min_train + 1} observations, got {n}")

    out = np.full(n, np.nan)
    refits: list[dict] = []
    params = None

    t0 = min_train
    while t0 < n:
        t1 = min(t0 + refit_every, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _spec(ret.iloc[:t0], kind, dist).fit(disp="off", show_warning=False)
        converged = res.convergence_flag == 0 and np.all(np.isfinite(res.params.to_numpy()))
        if converged:
            params = res.params
        elif params is None:
            raise RuntimeError(f"{kind} failed to converge on the first fit")
        refits.append(
            {
                "date": ret.index[t0],
                "converged": bool(converged),
                "params": {k: float(v) for k, v in params.items()},
            }
        )
        # Filter through observed data up to t1 with the fixed parameters.
        fixed = _spec(ret.iloc[:t1], kind, dist).fix(params)
        out[t0:t1] = fixed.conditional_volatility.to_numpy()[t0:t1] ** 2
        t0 = t1

    name = "gjr" if kind == "gjr" else "garch"
    return GarchOutput(variance=pd.Series(out, index=ret.index, name=name), refits=refits)


def persistence(params: dict) -> float:
    """alpha + beta (+ gamma/2 for GJR): how slowly volatility shocks decay."""
    a = params.get("alpha[1]", 0.0)
    b = params.get("beta[1]", 0.0)
    g = params.get("gamma[1]", 0.0)
    return float(a + b + 0.5 * g)
