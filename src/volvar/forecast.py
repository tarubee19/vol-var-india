"""Run every volatility model walk-forward over a universe of tickers."""

from __future__ import annotations

from dataclasses import dataclass, field

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


def wide_forecasts(result: ForecastResult, ticker: str) -> pd.DataFrame:
    """Date-indexed frame of one ticker's forecasts, one column per model."""
    f = result.forecasts[result.forecasts["ticker"] == ticker]
    return f.pivot(index="date", columns="model", values="var").sort_index()
