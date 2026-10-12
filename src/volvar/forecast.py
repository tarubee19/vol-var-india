"""Run every volatility model walk-forward over a universe of tickers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from volvar.config import ExperimentConfig
from volvar.models.baselines import ewma_variance, historical_variance
from volvar.models.garch import garch_variance, persistence
from volvar.models.har import har_variance
from volvar.models.ml import lgbm_variance


@dataclass
class ForecastResult:
    """Long-format forecasts plus model metadata."""

    forecasts: pd.DataFrame  # columns: date, ticker, model, var
    returns: pd.DataFrame  # columns: date, ticker, ret, r2
    garch_params: dict[str, dict[str, dict]] = field(default_factory=dict)  # model -> ticker -> params
    garch_failed_fits: dict[str, int] = field(default_factory=dict)


def _garch_job(ticker: str, ret: pd.Series, cfg: ExperimentConfig) -> tuple[str, dict]:
    out = {}
    for kind in ("garch", "gjr"):
        if kind in cfg.models:
            res = garch_variance(ret, kind=kind, min_train=cfg.min_train, refit_every=cfg.refit_every)
            out[kind] = res
    return ticker, out


def run_forecasts(
    panels: dict[str, pd.DataFrame], cfg: ExperimentConfig | None = None, n_jobs: int = 2
) -> ForecastResult:
    """Produce walk-forward one-day-ahead variance forecasts for every enabled model."""
    cfg = cfg or ExperimentConfig()
    pieces: list[pd.DataFrame] = []

    def add(model: str, ticker: str, s: pd.Series) -> None:
        s = s.dropna()
        pieces.append(
            pd.DataFrame({"date": s.index, "ticker": ticker, "model": model, "var": s.to_numpy()})
        )

    hist_cache: dict[str, pd.Series] = {}
    for ticker, p in panels.items():
        hist = historical_variance(p["r2"], cfg.hist_window)
        hist_cache[ticker] = hist
        if "hist" in cfg.models:
            add("hist", ticker, hist)
        if "ewma" in cfg.models:
            add("ewma", ticker, ewma_variance(p["r2"], cfg.ewma_lambda))
        if "har" in cfg.models:
            har = har_variance(p, cfg.min_train, cfg.refit_every, floor_series=hist)
            add("har", ticker, har)

    garch_params: dict[str, dict[str, dict]] = {"garch": {}, "gjr": {}}
    failed = {"garch": 0, "gjr": 0}
    if "garch" in cfg.models or "gjr" in cfg.models:
        jobs = Parallel(n_jobs=n_jobs)(
            delayed(_garch_job)(t, p["ret"], cfg) for t, p in panels.items()
        )
        for ticker, outputs in jobs:
            for kind, res in outputs.items():
                add(kind, ticker, res.variance)
                params = dict(res.last_params)
                params["persistence"] = persistence(params)
                garch_params[kind][ticker] = params
                failed[kind] += res.n_failed_fits

    if "lgbm" in cfg.models:
        for ticker, s in lgbm_variance(
            panels, cfg.min_train, cfg.refit_every, cfg.ewma_lambda, cfg.lgbm_params, cfg.seed
        ).items():
            add("lgbm", ticker, s)

    forecasts = pd.concat(pieces, ignore_index=True)
    returns = pd.concat(
        [
            pd.DataFrame({"date": p.index, "ticker": t, "ret": p["ret"].to_numpy(), "r2": p["r2"].to_numpy()})
            for t, p in panels.items()
        ],
        ignore_index=True,
    )
    return ForecastResult(
        forecasts=forecasts,
        returns=returns,
        garch_params={k: v for k, v in garch_params.items() if v},
        garch_failed_fits={k: v for k, v in failed.items() if k in cfg.models},
    )


def extend_panels(
    panels: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, list[str]]:
    """Append one placeholder row (the next business day) to every up-to-date panel.

    All features and forecasts for a given day depend only on *earlier* rows, so the
    placeholder values never influence the forecast made for that day. Tickers whose
    data stops before the latest date in the universe are skipped and returned in the
    third element.
    """
    as_of = max(p.index[-1] for p in panels.values())
    target = as_of + pd.offsets.BDay(1)
    extended: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for ticker, p in panels.items():
        if p.index[-1] != as_of:
            skipped.append(ticker)
            continue
        row = pd.DataFrame(
            {
                "ret": [0.0],
                "r2": [0.0],
                "gk": [float(p["gk"].iloc[-22:].mean())],
                "volume": [float(p["volume"].iloc[-1])],
            },
            index=pd.DatetimeIndex([target], name=p.index.name),
        )
        extended[ticker] = pd.concat([p, row])
    return extended, target, skipped


def next_day_variances(
    panels: dict[str, pd.DataFrame], cfg: ExperimentConfig | None = None
) -> pd.DataFrame:
    """Variance forecast for the first trading day after the data ends, for every model.

    Models are re-estimated on *all* available history, exactly like the last refit of the
    walk-forward run. Returns columns ``ticker, model, var, as_of, target_date``.
    """
    cfg = cfg or ExperimentConfig()
    ext, target, _ = extend_panels(panels)
    as_of = target - pd.offsets.BDay(1)
    rows: list[dict] = []

    def add(model: str, ticker: str, value: float) -> None:
        rows.append(
            {"ticker": ticker, "model": model, "var": float(value), "as_of": as_of, "target_date": target}
        )

    for ticker, p in ext.items():
        n_real = len(p) - 1
        hist = historical_variance(p["r2"], cfg.hist_window)
        if "hist" in cfg.models and np.isfinite(hist.iloc[-1]):
            add("hist", ticker, hist.iloc[-1])
        if "ewma" in cfg.models:
            add("ewma", ticker, ewma_variance(p["r2"], cfg.ewma_lambda).iloc[-1])
        if "har" in cfg.models:
            add("har", ticker, har_variance(p, n_real, 1, floor_series=hist).iloc[-1])
        for kind in ("garch", "gjr"):
            if kind in cfg.models:
                out = garch_variance(p["ret"], kind=kind, min_train=n_real, refit_every=1)
                add(kind, ticker, out.variance.iloc[-1])

    if "lgbm" in cfg.models and ext:
        n_cal = len(pd.concat([p["ret"] for p in ext.values()], axis=1).index) - 1
        res = lgbm_variance(ext, n_cal, 1, cfg.ewma_lambda, cfg.lgbm_params, cfg.seed)
        for ticker, s in res.items():
            if target in s.index:
                add("lgbm", ticker, s.loc[target])
    return pd.DataFrame(rows)


def wide_forecasts(result: ForecastResult, ticker: str) -> pd.DataFrame:
    """Date-indexed frame of one ticker's forecasts, one column per model."""
    f = result.forecasts[result.forecasts["ticker"] == ticker]
    return f.pivot(index="date", columns="model", values="var").sort_index()
