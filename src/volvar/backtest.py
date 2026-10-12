"""VaR backtests: Kupiec, Christoffersen and the Basel traffic light.

A VaR model with confidence ``level`` should be exceeded on a fraction ``p = 1 - level``
of days (*unconditional coverage*) and the exceedances should not cluster
(*independence*). Both properties are tested; together they form *conditional coverage*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def violations(ret: pd.Series, var: pd.Series) -> np.ndarray:
    """Indicator array: 1 where the realised loss exceeds the VaR (``ret < -VaR``)."""
    r = ret.reindex(var.index)
    return (r.to_numpy() < -var.to_numpy()).astype(int)


def kupiec_pof(x: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR statistic and p-value (chi-square, 1 dof)."""
    if n <= 0:
        return float("nan"), float("nan")
    ll_null = (n - x) * np.log(1.0 - p) + xlogy(x, p)
    pi = x / n
    ll_alt = xlogy(n - x, 1.0 - pi) + xlogy(x, pi)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(v: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence LR test on a 0/1 violation sequence.

    Returns ``(nan, nan)`` when the sequence has too few violations to test.
    """
    v = np.asarray(v, dtype=int)
    if len(v) < 2 or v.sum() < 2:
        return float("nan"), float("nan")
    prev, cur = v[:-1], v[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return float("nan"), float("nan")
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_cc(v: np.ndarray, p: float) -> tuple[float, float]:
    """Conditional coverage LR = Kupiec POF + independence (chi-square, 2 dof)."""
    v = np.asarray(v, dtype=int)
    lr_pof, _ = kupiec_pof(int(v.sum()), len(v), p)
    lr_ind, _ = christoffersen_independence(v)
    if np.isnan(lr_ind):
        return float("nan"), float("nan")
    lr = lr_pof + lr_ind
    return float(lr), float(stats.chi2.sf(lr, 2))


def basel_zone(exceptions_250d: int) -> str:
    """Basel traffic-light zone for 250 days of 99% VaR: green 0-4, yellow 5-9, red 10+."""
    if exceptions_250d <= 4:
        return "green"
    if exceptions_250d <= 9:
        return "yellow"
    return "red"


def worst_rolling_exceptions(v: np.ndarray, window: int = 250) -> int:
    """Largest number of violations in any ``window``-day span (the whole sample if shorter)."""
    v = np.asarray(v, dtype=int)
    if len(v) <= window:
        return int(v.sum())
    csum = np.concatenate([[0], np.cumsum(v)])
    return int((csum[window:] - csum[:-window]).max())


def zone_shares(v: np.ndarray, window: int = 250) -> dict[str, float]:
    """Share of rolling ``window``-day spans falling in each Basel zone (99% VaR convention).

    For a well-calibrated 99% model the expected shares are roughly 89% green, 11% yellow
    and under 1% red, so this is a fairer summary than the single worst window.
    """
    v = np.asarray(v, dtype=int)
    if len(v) < window:
        return {"green": float("nan"), "yellow": float("nan"), "red": float("nan")}
    csum = np.concatenate([[0], np.cumsum(v)])
    counts = csum[window:] - csum[:-window]
    zones = np.array([basel_zone(int(c)) for c in counts])
    return {z: float((zones == z).mean()) for z in ("green", "yellow", "red")}


def es_ratio(ret: pd.Series, var: pd.Series, es: pd.Series) -> float:
    """Mean realised loss on violation days divided by mean predicted ES on those days.

    A well-calibrated ES gives a ratio near 1; above 1 means the tail is underestimated.
    """
    r = ret.reindex(var.index)
    mask = r.to_numpy() < -var.to_numpy()
    if mask.sum() == 0:
        return float("nan")
    return float((-r[mask]).mean() / es[mask].mean())


def backtest_var(ret: pd.Series, var: pd.Series, es: pd.Series, level: float) -> dict:
    """Full backtest summary for one VaR series (already restricted to the test sample)."""
    p = 1.0 - level
    v = violations(ret, var)
    n, x = len(v), int(v.sum())
    lr_uc, p_uc = kupiec_pof(x, n, p)
    lr_ind, p_ind = christoffersen_independence(v)
    lr_cc, p_cc = christoffersen_cc(v, p)
    worst = worst_rolling_exceptions(v)
    is99 = abs(level - 0.99) < 1e-9
    latest = int(v[-250:].sum()) if len(v) else 0
    shares = zone_shares(v) if is99 else {"green": float("nan"), "yellow": float("nan"), "red": float("nan")}
    # pass/fail: use conditional coverage when it is defined, otherwise unconditional coverage
    p_decision = p_cc if not np.isnan(p_cc) else p_uc
    return {
        "n": n,
        "violations": x,
        "violation_rate": x / n if n else float("nan"),
        "expected_rate": p,
        "kupiec_p": p_uc,
        "independence_p": p_ind,
        "cc_p": p_cc,
        "es_ratio": es_ratio(ret, var, es),
        "mean_var_pct": float(var.mean()),
        "worst_250d_exceptions": worst,
        "latest_250d_exceptions": latest,
        "basel_zone": basel_zone(latest) if is99 else "n/a",  # zone of the most recent 250 days
        "pct_windows_green": shares["green"],
        "pct_windows_red": shares["red"],
        "passes": bool(p_decision > 0.05),
        "lr_uc": lr_uc,
        "lr_ind": lr_ind,
        "lr_cc": lr_cc,
    }
