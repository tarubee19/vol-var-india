"""Portfolio Value-at-Risk, Expected Shortfall, risk attribution and stress tests.

The single-stock study answers "which model forecasts each stock's volatility best?". A risk
desk then asks about the *book*: how much can the portfolio lose tomorrow, which positions
drive that number, and what would past crises or simple shocks do to it today?

The book is a daily-rebalanced **equal-weight portfolio of every stock in the universe** (the
index, ``NIFTY50``, is excluded and only used for betas and comparisons), valued at a notional
of INR 1 crore so that results can be read in rupees.

Four one-day-ahead portfolio VaR/ES models are run walk-forward and backtested on a common
out-of-sample window:

* ``hs`` - plain historical simulation on the last 500 portfolio returns (the classic bank model).
* ``ewma_normal`` - RiskMetrics EWMA covariance matrix (lambda 0.94), normal quantile.
* ``ewma_fhs`` - the same covariance, with filtered historical simulation for the tail.
* ``lgbm_fhs`` - bottom-up: each stock's own LightGBM variance forecast from the main study
  combined with the EWMA correlation matrix, then filtered historical simulation. It tests
  whether better single-stock forecasts make a better *portfolio* VaR.

Conventions: returns are **simple returns in percent** (portfolio returns are a weighted sum of
simple returns); the variance forecasts of the main study are for log returns, which differ from
simple returns by a negligible amount at a daily horizon. For day ``t`` only data up to ``t-1``
is used. VaR and ES are positive percent losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from volvar.backtest import backtest_var
from volvar.risk import fhs_var_es

INDEX_TICKERS = frozenset({"NIFTY50"})
NOTIONAL_INR = 1e7  # INR 1 crore
PORTFOLIO_LEVELS = (0.95, 0.975, 0.99)  # 97.5% is the Basel FRTB Expected Shortfall level
PORTFOLIO_MODELS = ("hs", "ewma_normal", "ewma_fhs", "lgbm_fhs")
PORTFOLIO_LABELS = {
    "hs": "Historical simulation (500d)",
    "ewma_normal": "EWMA covariance, normal",
    "ewma_fhs": "EWMA covariance, FHS",
    "lgbm_fhs": "LightGBM vols + EWMA correlation, FHS",
}

# Sector tags for hypothetical shocks; tickers not listed are left unshocked.
SECTORS = {
    "HDFCBANK": "banks",
    "ICICIBANK": "banks",
    "SBIN": "banks",
    "KOTAKBANK": "banks",
    "AXISBANK": "banks",
    "TCS": "it",
    "INFY": "it",
}

# Historical stress windows. A window covers the returns dated strictly after ``start``
# (the last close before the shock) up to and including ``end``.
HISTORICAL_SCENARIOS = (
    ("Brexit referendum result", "2016-06-23", "2016-06-24"),
    ("Demonetisation announcement", "2016-11-08", "2016-11-21"),
    ("IL&FS / NBFC credit scare", "2018-08-28", "2018-10-26"),
    ("COVID-19 crash", "2020-02-19", "2020-03-23"),
    ("Russia invades Ukraine", "2022-02-23", "2022-02-24"),
    ("2024 general election results day", "2024-06-03", "2024-06-04"),
    ("Global sell-off (yen carry unwind)", "2024-08-02", "2024-08-05"),
)


@dataclass
class PortfolioResult:
    """Everything the pipeline writes for the portfolio study."""

    returns: pd.Series  # portfolio simple returns (%)
    var_series: dict[tuple[str, float], tuple[pd.Series, pd.Series]]  # (model, level) -> (VaR, ES)
    backtest: pd.DataFrame
    next_day: pd.DataFrame
    contributions: pd.DataFrame
    stress: pd.DataFrame
    summary: dict


# ---------------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------------


def stock_tickers(panels: dict[str, pd.DataFrame]) -> list[str]:
    """Tickers that are stocks (the index is excluded)."""
    return [t for t in panels if t.upper() not in INDEX_TICKERS]


def simple_returns_pct(panels: dict[str, pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    """Date x ticker frame of simple returns in percent, on dates where every ticker trades."""
    wide = pd.DataFrame({t: 100.0 * np.expm1(panels[t]["ret"] / 100.0) for t in tickers})
    return wide.dropna(how="any").sort_index()


def equal_weights(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    if n == 0:
        raise ValueError("portfolio needs at least one stock")
    return pd.Series(np.full(n, 1.0 / n), index=tickers, name="weight")


def ewma_covariances(returns: pd.DataFrame, lam: float = 0.94, init: int = 60) -> np.ndarray:
    """One-day-ahead RiskMetrics covariance matrices, shape ``(T + 1, N, N)``.

    Element ``t`` (``t < T``) is the forecast for the day of ``returns.iloc[t]`` and uses only
    rows ``< t``; the extra last element is the forecast for the day after the sample ends.
    The first ``init`` rows are seeded with their sample covariance (and are not used for any
    out-of-sample number because the backtests start much later).
    """
    x = returns.to_numpy(dtype=float)
    t_n, n = x.shape
    if t_n < init + 2:
        raise ValueError(f"need more than {init + 2} rows to start the EWMA covariance")
    out = np.empty((t_n + 1, n, n))
    seed = np.cov(x[:init].T, ddof=1).reshape(n, n)
    out[: init + 1] = seed
    cov = seed
    for t in range(init, t_n):
        r = x[t][:, None]
        cov = lam * cov + (1.0 - lam) * (r @ r.T)
        out[t + 1] = cov
    return out


def portfolio_variance(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``w' S_t w`` for a stack of covariance matrices ``S`` (shape ``(T, N, N)``)."""
    return np.einsum("i,tij,j->t", w, cov, w)


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.einsum("tii->ti", cov))
    return cov / (sd[:, :, None] * sd[:, None, :])


def historical_var_es(port: pd.Series, level: float, window: int = 500) -> tuple[pd.Series, pd.Series]:
    """Plain historical-simulation VaR/ES from the previous ``window`` portfolio returns."""
    r = port.to_numpy(dtype=float)
    n = len(r)
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    tail = 1.0 - level
    for i in range(window, n):
        w = r[i - window : i]
        q = np.quantile(w, tail)
        var[i] = -q
        es[i] = -w[w <= q].mean()
    return pd.Series(var, index=port.index, name="var"), pd.Series(es, index=port.index, name="es")


def euler_contributions(cov: np.ndarray, w: pd.Series, level: float = 0.99) -> pd.DataFrame:
    """Euler decomposition of a normal portfolio VaR into per-position components.

    ``component_i = w_i (S w)_i / sigma_p``; the components add up to ``sigma_p`` exactly, so
    the percentage contributions add up to 100%.
    """
    wv = w.to_numpy(dtype=float)
    sigma_p = float(np.sqrt(wv @ cov @ wv))
    z = stats.norm.ppf(level)
    marginal = cov @ wv / sigma_p
    component = wv * marginal
    vols = np.sqrt(np.diag(cov))
    df = pd.DataFrame(
        {
            "weight": wv,
            "vol_daily_pct": vols,
            "standalone_var_pct": z * wv * vols,
            "component_var_pct": z * component,
            "pct_of_portfolio_var": component / sigma_p,
        },
        index=w.index,
    )
    df["component_var_inr"] = df["component_var_pct"] / 100.0 * NOTIONAL_INR
    df.index.name = "ticker"
    return df.sort_values("pct_of_portfolio_var", ascending=False)


def betas_to_index(panels: dict[str, pd.DataFrame], tickers: list[str], index: str = "NIFTY50",
                   window: int = 250) -> pd.Series | None:
    """OLS beta of each stock's return on the index over the last ``window`` days."""
    if index not in panels:
        return None
    idx = panels[index]["ret"]
    out = {}
    for t in tickers:
        pair = pd.concat([panels[t]["ret"], idx], axis=1, join="inner").dropna().tail(window)
        if len(pair) < 60:
            continue
        x, y = pair.iloc[:, 1].to_numpy(), pair.iloc[:, 0].to_numpy()
        out[t] = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    return pd.Series(out, name="beta")


# ---------------------------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------------------------


def _lgbm_variances(forecasts: pd.DataFrame | None, tickers: list[str], index: pd.Index) -> pd.DataFrame | None:
    if forecasts is None or forecasts.empty:
        return None
    f = forecasts[(forecasts["model"] == "lgbm") & forecasts["ticker"].isin(tickers)]
    if f.empty:
        return None
    wide = f.pivot(index="date", columns="ticker", values="var").reindex(index=index, columns=tickers)
    return wide


def portfolio_variance_forecasts(
    rets: pd.DataFrame,
    w: pd.Series,
    forecasts: pd.DataFrame | None = None,
    lam: float = 0.94,
    next_lgbm: pd.Series | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-day portfolio variance forecasts for the covariance-based models.

    Returns a frame indexed by ``rets.index`` plus one extra row for the next business day,
    with columns ``ewma`` and (when LightGBM forecasts exist) ``lgbm``, and the stack of EWMA
    covariance matrices.
    """
    covs = ewma_covariances(rets, lam)
    target = rets.index[-1] + pd.offsets.BDay(1)
    idx = rets.index.append(pd.DatetimeIndex([target]))
    wv = w.to_numpy(dtype=float)
    out = pd.DataFrame({"ewma": portfolio_variance(covs, wv)}, index=idx)
    lgbm = _lgbm_variances(forecasts, list(w.index), rets.index)
    if lgbm is not None:
        if next_lgbm is not None:
            lgbm.loc[target] = next_lgbm.reindex(w.index).to_numpy()
        else:
            lgbm.loc[target] = np.nan
        corr = correlation_from_cov(covs)
        sd = np.sqrt(lgbm.to_numpy(dtype=float))
        a = wv[None, :] * sd
        out["lgbm"] = np.einsum("ti,tij,tj->t", a, corr, a)
    return out, covs


def run_portfolio_study(
    panels: dict[str, pd.DataFrame],
    forecasts: pd.DataFrame | None = None,
    next_day_var: pd.DataFrame | None = None,
    oos_start: pd.Timestamp | None = None,
    label: str = "real",
    lam: float = 0.94,
    fhs_window: int = 500,
    fhs_min_obs: int = 250,
    hs_window: int = 500,
    levels: tuple[float, ...] = PORTFOLIO_LEVELS,
) -> PortfolioResult:
    """Run the walk-forward portfolio VaR study, attribution and stress tests."""
    tickers = stock_tickers(panels)
    rets = simple_returns_pct(panels, tickers)
    w = equal_weights(tickers)
    port = (rets @ w).rename("portfolio")
    target = rets.index[-1] + pd.offsets.BDay(1)

    next_lgbm = None
    if next_day_var is not None and len(next_day_var):
        nd = next_day_var[next_day_var["model"] == "lgbm"].set_index("ticker")["var"]
        if set(tickers) <= set(nd.index):
            next_lgbm = nd
    pvar, covs = portfolio_variance_forecasts(rets, w, forecasts, lam, next_lgbm)

    # extend the return series with a placeholder for the next day (never used as information)
    port_ext = pd.concat([port, pd.Series([np.nan], index=pd.DatetimeIndex([target]))])

    series: dict[tuple[str, float], tuple[pd.Series, pd.Series]] = {}
    for level in levels:
        series[("hs", level)] = historical_var_es(port_ext, level, hs_window)
        sig = np.sqrt(pvar["ewma"])
        z = stats.norm.ppf(level)
        series[("ewma_normal", level)] = (
            (sig * z).rename("var"),
            (sig * stats.norm.pdf(z) / (1 - level)).rename("es"),
        )
        series[("ewma_fhs", level)] = fhs_var_es(port_ext, pvar["ewma"], level, fhs_window, fhs_min_obs)
        if "lgbm" in pvar:
            series[("lgbm_fhs", level)] = fhs_var_es(
                port_ext, pvar["lgbm"].dropna(), level, fhs_window, fhs_min_obs
            )

    # common out-of-sample window: every model has a value, and (if given) after oos_start
    common = None
    for (_, _), (v, _) in series.items():
        idx = v.loc[: rets.index[-1]].dropna().index
        common = idx if common is None else common.intersection(idx)
    if oos_start is not None:
        common = common[common >= pd.Timestamp(oos_start)]
    rows = []
    for (model, level), (v, e) in series.items():
        row = backtest_var(port, v.loc[common], e.loc[common], level)
        row.update({"model": model, "label": PORTFOLIO_LABELS[model], "level": level})
        rows.append(row)
    bt = pd.DataFrame(rows)
    bt = bt[["model", "label", "level", *[c for c in bt.columns if c not in ("model", "label", "level")]]]

    # next-day VaR / ES in percent and rupees
    nd_rows = []
    for (model, level), (v, e) in series.items():
        if target in v.index and np.isfinite(v.loc[target]):
            nd_rows.append(
                {
                    "model": model,
                    "label": PORTFOLIO_LABELS[model],
                    "level": level,
                    "var_pct": float(v.loc[target]),
                    "es_pct": float(e.loc[target]),
                    "var_inr": float(v.loc[target]) / 100.0 * NOTIONAL_INR,
                    "es_inr": float(e.loc[target]) / 100.0 * NOTIONAL_INR,
                    "as_of": rets.index[-1],
                    "target_date": target,
                }
            )
    next_day = pd.DataFrame(nd_rows)

    contrib = euler_contributions(covs[-1], w, 0.99)
    stress = stress_tests(panels, tickers, w, port, series.get(("ewma_fhs", 0.99)), label)

    summary = _summary(bt, next_day, contrib, stress, port, common, tickers)
    return PortfolioResult(port, series, bt, next_day, contrib, stress, summary)


def stress_tests(
    panels: dict[str, pd.DataFrame],
    tickers: list[str],
    w: pd.Series,
    port: pd.Series,
    var99: tuple[pd.Series, pd.Series] | None,
    label: str,
) -> pd.DataFrame:
    """Historical scenarios replayed on the current book, plus hypothetical shocks.

    Historical scenarios are only run on real data (their dates mean nothing for simulated
    series). Each row reports the portfolio P&L in percent and rupees on the INR 1 crore book,
    the Nifty 50 move over the same window when available, and the 99% VaR breaches.
    """
    rows: list[dict] = []
    idx_ret = None
    if "NIFTY50" in panels:
        idx_ret = 100.0 * np.expm1(panels["NIFTY50"]["ret"] / 100.0)
    if label == "real":
        for name, start, end in HISTORICAL_SCENARIOS:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            if s < port.index[0] or e > port.index[-1]:
                continue
            r = port[(port.index > s) & (port.index <= e)]
            if r.empty:
                continue
            growth = np.cumprod(1.0 + r.to_numpy() / 100.0)
            pnl = 100.0 * (growth[-1] - 1.0)
            peak = np.maximum.accumulate(np.concatenate([[1.0], growth]))
            mdd = 100.0 * float((np.concatenate([[1.0], growth]) / peak - 1.0).min())
            row = {
                "scenario": name,
                "type": "historical",
                "window": f"{r.index[0]:%Y-%m-%d} to {r.index[-1]:%Y-%m-%d}",
                "days": len(r),
                "pnl_pct": pnl,
                "pnl_inr": pnl / 100.0 * NOTIONAL_INR,
                "max_drawdown_pct": mdd,
                "worst_day_pct": float(r.min()),
            }
            if idx_ret is not None:
                ir = idx_ret[(idx_ret.index > s) & (idx_ret.index <= e)]
                if len(ir):
                    row["nifty50_pct"] = 100.0 * (np.prod(1.0 + ir.to_numpy() / 100.0) - 1.0)
            if var99 is not None:
                v = var99[0].reindex(r.index)
                ok = v.notna()
                if ok.any():
                    row["var99_breaches"] = int((r[ok] < -v[ok]).sum())
                    row["worst_day_vs_var99"] = float(-r.min() / v.loc[r.idxmin()]) if np.isfinite(
                        v.loc[r.idxmin()]
                    ) else np.nan
            rows.append(row)

    # hypothetical, instantaneous shocks on today's book
    betas = betas_to_index(panels, tickers)
    if betas is not None and len(betas) == len(tickers):
        shock = -10.0
        pnl = float((w * betas.reindex(w.index) * shock).sum())
        rows.append(_hypo("Nifty 50 falls 10% (beta-mapped)", pnl))
    sector_names = {"banks": ("Banks fall 15%", -15.0), "it": ("IT services fall 12%", -12.0)}
    for sector, (name, size) in sector_names.items():
        members = [t for t in tickers if SECTORS.get(t) == sector]
        if members:
            rows.append(_hypo(name, float(w[members].sum() * size)))
    if not rows:
        return pd.DataFrame(columns=["scenario", "type", "window", "days", "pnl_pct", "pnl_inr"])
    return pd.DataFrame(rows)


def _hypo(name: str, pnl_pct: float) -> dict:
    return {
        "scenario": name,
        "type": "hypothetical",
        "window": "instantaneous",
        "days": 0,
        "pnl_pct": pnl_pct,
        "pnl_inr": pnl_pct / 100.0 * NOTIONAL_INR,
    }


def _summary(bt, next_day, contrib, stress, port, common, tickers) -> dict:
    def pick(model, level, col):
        r = bt[(bt.model == model) & np.isclose(bt.level, level)]
        if r.empty:
            return None
        val = r.iloc[0][col]
        return val.item() if hasattr(val, "item") else val

    out = {
        "n_stocks": len(tickers),
        "weights": "equal",
        "notional_inr": NOTIONAL_INR,
        "oos_start": common.min() if len(common) else None,
        "oos_end": common.max() if len(common) else None,
        "n_oos_days": int(len(common)),
        "annual_vol_pct": float(port.std() * np.sqrt(252)),
        "backtest_99": {
            m: {
                "violation_rate": pick(m, 0.99, "violation_rate"),
                "kupiec_p": pick(m, 0.99, "kupiec_p"),
                "cc_p": pick(m, 0.99, "cc_p"),
                "passes": pick(m, 0.99, "passes"),
                "basel_zone": pick(m, 0.99, "basel_zone"),
                "es_ratio": pick(m, 0.99, "es_ratio"),
            }
            for m in bt.model.unique()
        },
        "top_contributors": contrib.head(3)["pct_of_portfolio_var"].to_dict(),
        "diversification_benefit": float(
            1.0 - contrib["component_var_pct"].sum() / contrib["standalone_var_pct"].sum()
        ),
    }
    if len(next_day):
        nd = next_day[(next_day.model == "ewma_fhs") & np.isclose(next_day.level, 0.99)]
        if len(nd):
            out["next_day_var99_ewma_fhs_pct"] = float(nd.iloc[0]["var_pct"])
            out["next_day_var99_ewma_fhs_inr"] = float(nd.iloc[0]["var_inr"])
            out["next_day_target"] = nd.iloc[0]["target_date"]
    if len(stress):
        h = stress[stress["type"] == "historical"]
        if len(h):
            worst = h.loc[h["pnl_pct"].idxmin()]
            out["worst_historical_scenario"] = {"scenario": worst["scenario"], "pnl_pct": worst["pnl_pct"]}
    return out
