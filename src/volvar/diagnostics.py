"""Statistical diagnostics on return series.

These tests justify the modelling choices: returns should be stationary (ADF/KPSS)
while *squared* returns are autocorrelated (Ljung-Box) and show ARCH effects
(Engle's LM test), which is the volatility clustering GARCH-type models exploit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


def adf_pvalue(x: pd.Series) -> float:
    """Augmented Dickey-Fuller p-value (H0: unit root)."""
    return float(adfuller(x.dropna(), autolag="AIC")[1])


def kpss_pvalue(x: pd.Series) -> float:
    """KPSS p-value (H0: level-stationary). statsmodels clips p to [0.01, 0.1]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kpss(x.dropna(), regression="c", nlags="auto")[1])


def arch_lm_pvalue(x: pd.Series, lags: int = 5) -> float:
    """Engle's ARCH-LM test p-value (H0: no ARCH effects)."""
    centered = x.dropna() - x.dropna().mean()
    return float(het_arch(centered, nlags=lags)[1])


def ljung_box_pvalue(x: pd.Series, lags: int = 10) -> float:
    """Ljung-Box p-value at ``lags`` (H0: no autocorrelation up to that lag)."""
    res = acorr_ljungbox(x.dropna(), lags=[lags], return_df=True)
    return float(res["lb_pvalue"].iloc[0])


def describe_returns(ret: pd.Series) -> dict[str, float]:
    """One row of diagnostics for a return series (percent log returns)."""
    r = ret.dropna()
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        "n": int(len(r)),
        "mean_pct": float(r.mean()),
        "std_pct": float(r.std(ddof=1)),
        "ann_vol_pct": float(r.std(ddof=1) * np.sqrt(252)),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "jarque_bera_p": float(jb_p),
        "adf_p_returns": adf_pvalue(r),
        "kpss_p_returns": kpss_pvalue(r),
        "ljungbox_p_returns": ljung_box_pvalue(r),
        "ljungbox_p_squared": ljung_box_pvalue(r**2),
        "arch_lm_p": arch_lm_pvalue(r),
    }


def diagnostics_table(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Diagnostics for every ticker, one row per ticker."""
    rows = {t: describe_returns(p["ret"]) for t, p in panels.items()}
    return pd.DataFrame(rows).T.rename_axis("ticker")
