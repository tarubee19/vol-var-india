"""End-to-end run: data -> diagnostics -> walk-forward forecasts -> evaluation -> VaR backtests.

Everything the API, dashboard, assistant and README need is written to ``results_dir``:

* small, committable summaries (``*.csv``, ``summary.json``, ``meta.json``, figures)
* larger working files (``forecasts.parquet``, ``returns.parquet``), git-ignored
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from volvar import __version__
from volvar.backtest import backtest_var
from volvar.config import BENCHMARK_MODEL, MODEL_LABELS, ExperimentConfig
from volvar.data import load_universe, make_panels
from volvar.diagnostics import diagnostics_table
from volvar.evaluation import (
    aligned_wide,
    best_model_counts,
    dm_vs_benchmark,
    loss_by_regime,
    loss_in_window,
    loss_table,
    market_regimes,
    per_ticker_losses,
)
from volvar.forecast import ForecastResult, next_day_variances, run_forecasts
from volvar.portfolio import run_portfolio_study, stock_tickers
from volvar.risk import var_es

METHODS = ("fhs", "normal")
COVID_WINDOW = ("2020-02-15", "2020-06-30")


def detect_dataset_label(data_dir: Path, tickers: list[str]) -> str:
    """``simulated`` when the folder is marked SIMULATED or all tickers are SIM*; else ``real``."""
    marker = Path(data_dir) / "README.txt"
    if marker.exists() and "SIMULATED" in marker.read_text().upper():
        return "simulated"
    return "simulated" if all(t.startswith("SIM") for t in tickers) else "real"


def _variance_series(result: ForecastResult, ticker: str, model: str) -> pd.Series:
    f = result.forecasts
    sub = f[(f["ticker"] == ticker) & (f["model"] == model)]
    return sub.set_index("date")["var"].sort_index()


def compute_var_backtests(
    result: ForecastResult,
    panels: dict[str, pd.DataFrame],
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, dict]:
    """Backtest VaR/ES for every ticker x model x method x level on a common sample.

    Returns the long results table and a dict of ``(ticker, model, method, level) -> (var, es)``
    series for plotting.
    """
    rows: list[dict] = []
    series: dict[tuple, tuple[pd.Series, pd.Series]] = {}
    for ticker, panel in panels.items():
        ret = panel["ret"]
        variances = {m: _variance_series(result, ticker, m) for m in cfg.models}
        variances = {m: v for m, v in variances.items() if len(v)}
        for level in cfg.var_levels:
            computed = {
                (m, method): var_es(ret, v, level, method, cfg.fhs_window, cfg.fhs_min_obs)
                for m, v in variances.items()
                for method in METHODS
            }
            common = None
            for (_, method), (var, _) in computed.items():
                if method == "fhs":
                    idx = var.dropna().index
                    common = idx if common is None else common.intersection(idx)
            if common is None or len(common) < 100:
                continue
            for (m, method), (var, es) in computed.items():
                v, e = var.loc[common], es.loc[common]
                row = backtest_var(ret, v, e, level)
                row.update({"ticker": ticker, "model": m, "method": method, "level": level})
                rows.append(row)
                series[(ticker, m, method, level)] = (v, e)
    table = pd.DataFrame(rows)
    first = ["ticker", "model", "method", "level"]
    return table[first + [c for c in table.columns if c not in first]], series


def summarize_var(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-ticker backtests into one row per model/method/level."""
    g = table.groupby(["model", "method", "level"])
    out = g.agg(
        n_tickers=("ticker", "nunique"),
        mean_violation_rate=("violation_rate", "mean"),
        expected_rate=("expected_rate", "first"),
        pass_rate=("passes", "mean"),
        mean_es_ratio=("es_ratio", "mean"),
        mean_worst_250d=("worst_250d_exceptions", "mean"),
        share_latest_green=(
            "basel_zone",
            lambda s: float((s == "green").mean()) if (s != "n/a").any() else np.nan,
        ),
        mean_pct_windows_green=("pct_windows_green", "mean"),
        mean_pct_windows_red=("pct_windows_red", "mean"),
    )
    return out.reset_index()


def compute_next_day_var(
    result: ForecastResult,
    panels: dict[str, pd.DataFrame],
    next_var: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Next-day VaR and ES per ticker/model/method/level from the next-day variance forecasts."""
    rows = []
    for rec in next_var.itertuples(index=False):
        oos = _variance_series(result, rec.ticker, rec.model)
        if oos.empty:
            continue
        ext = pd.concat([oos, pd.Series([rec.var], index=pd.DatetimeIndex([rec.target_date]))])
        ret = panels[rec.ticker]["ret"]
        for level in cfg.var_levels:
            for method in METHODS:
                var, es = var_es(ret, ext, level, method, cfg.fhs_window, cfg.fhs_min_obs)
                v, e = var.iloc[-1], es.iloc[-1]
                if np.isfinite(v) and np.isfinite(e):
                    rows.append(
                        {
                            "ticker": rec.ticker,
                            "model": rec.model,
                            "method": method,
                            "level": level,
                            "var_pct": float(v),
                            "es_pct": float(e),
                            "as_of": rec.as_of,
                            "target_date": rec.target_date,
                        }
                    )
    return pd.DataFrame(rows)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    return obj


def run_pipeline(
    data_dir: str | Path,
    results_dir: str | Path,
    cfg: ExperimentConfig | None = None,
    tickers: list[str] | None = None,
    label: str | None = None,
    n_jobs: int = 2,
    figures: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the whole study and write all outputs; returns the headline summary dict."""
    cfg = cfg or ExperimentConfig()
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.time() - t_start:6.1f}s] {msg}", flush=True)

    universe = load_universe(data_dir, tickers)
    panels = make_panels(universe)
    label = label or detect_dataset_label(data_dir, list(panels))
    log(f"loaded {len(panels)} series ({label} data) from {data_dir}")

    diag = diagnostics_table(panels)
    diag.to_csv(results_dir / "diagnostics.csv")
    log("diagnostics done")

    result = run_forecasts(panels, cfg, n_jobs=n_jobs)
    result.forecasts.to_parquet(results_dir / "forecasts.parquet", index=False)
    result.returns.to_parquet(results_dir / "returns.parquet", index=False)
    (results_dir / "garch_params.json").write_text(json.dumps(_json_safe(result.garch_params), indent=2))
    log(f"forecasts done ({len(result.forecasts):,} rows)")

    models = list(cfg.models)
    wide = aligned_wide(result.forecasts, result.returns, models)
    dates = wide.index.get_level_values("date")
    oos_start, oos_end = dates.min(), dates.max()

    lt = loss_table(wide, models)
    dm_q = dm_vs_benchmark(wide, models, BENCHMARK_MODEL, "qlike")
    dm_m = dm_vs_benchmark(wide, models, BENCHMARK_MODEL, "mse")
    comp = lt.join(dm_q[["mean_diff", "t_stat", "p_value"]].rename(
        columns={"mean_diff": "qlike_gain_vs_ewma", "t_stat": "dm_t", "p_value": "dm_p"}
    ))
    comp.insert(0, "label", [MODEL_LABELS.get(m, m) for m in comp.index])
    comp.rename_axis("model").to_csv(results_dir / "model_comparison.csv")
    pd.concat([dm_q.assign(loss_fn="qlike"), dm_m.assign(loss_fn="mse")]).rename_axis("model").to_csv(
        results_dir / "dm_tests.csv"
    )
    pt = per_ticker_losses(wide, "qlike", models)
    pt.to_csv(results_dir / "loss_by_ticker.csv")
    regimes = market_regimes(result.returns)
    reg = loss_by_regime(wide, regimes, "qlike", models)
    reg.to_csv(results_dir / "loss_by_regime.csv")
    yr = dates.year
    by_year = pd.DataFrame(
        {
            int(y): {m: float(np.mean(_qlike_np(wide.loc[yr == y, m], wide.loc[yr == y, "r2"]))) for m in models}
            for y in sorted(set(yr))
        }
    ).T.rename_axis("year")
    by_year.to_csv(results_dir / "loss_by_year.csv")
    covid = loss_in_window(wide, *COVID_WINDOW, models) if label == "real" else None
    if covid is not None:
        covid.rename_axis("model").to_csv(results_dir / "loss_covid_window.csv")
    log("evaluation done")

    var_table, var_series = compute_var_backtests(result, panels, cfg)
    var_table.to_csv(results_dir / "var_backtest.csv", index=False)
    var_sum = summarize_var(var_table)
    var_sum.to_csv(results_dir / "var_summary.csv", index=False)
    log(f"VaR backtests done ({len(var_table)} rows)")

    next_var = next_day_variances(panels, cfg)
    next_var = next_var.assign(
        vol_daily_pct=np.sqrt(next_var["var"]), vol_annual_pct=np.sqrt(next_var["var"] * 252)
    )
    next_var.to_csv(results_dir / "next_day.csv", index=False)
    next_var_risk = compute_next_day_var(result, panels, next_var, cfg)
    next_var_risk.to_csv(results_dir / "next_day_var.csv", index=False)
    log("next-day forecasts done")

    portfolio = None
    if len(stock_tickers(panels)) >= 2:
        portfolio = run_portfolio_study(
            panels,
            result.forecasts,
            next_var,
            oos_start=oos_start,
            label=label,
            lam=cfg.ewma_lambda,
            fhs_window=cfg.fhs_window,
            fhs_min_obs=cfg.fhs_min_obs,
        )
        portfolio.backtest.to_csv(results_dir / "portfolio_backtest.csv", index=False)
        portfolio.next_day.to_csv(results_dir / "portfolio_next_day.csv", index=False)
        portfolio.contributions.to_csv(results_dir / "portfolio_contributions.csv")
        portfolio.stress.to_csv(results_dir / "stress_tests.csv", index=False)
        portfolio.returns.rename("ret").rename_axis("date").to_frame().to_parquet(
            results_dir / "portfolio_returns.parquet"
        )
        log(f"portfolio study done ({len(portfolio.backtest)} backtests, {len(portfolio.stress)} stress rows)")

    best = lt["qlike"].idxmin()
    fhs99 = var_sum[(var_sum["method"] == "fhs") & (np.isclose(var_sum["level"], 0.99))].set_index("model")
    summary = {
        "dataset": label,
        "n_tickers": len(panels),
        "n_oos_observations": int(len(wide)),
        "oos_start": oos_start,
        "oos_end": oos_end,
        "models": models,
        "benchmark": BENCHMARK_MODEL,
        "best_model_by_qlike": best,
        "pooled_qlike": lt["qlike"].to_dict(),
        "pooled_mse": lt["mse"].to_dict(),
        "dm_vs_benchmark_qlike": dm_q[["mean_diff", "t_stat", "p_value"]].to_dict("index"),
        "tickers_won_by_model": best_model_counts(pt).to_dict(),
        "var_fhs99_pass_rate": fhs99["pass_rate"].to_dict(),
        "var_fhs99_violation_rate": fhs99["mean_violation_rate"].to_dict(),
        "garch_failed_fits": result.garch_failed_fits,
    }
    if portfolio is not None:
        summary["portfolio"] = portfolio.summary
    (results_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))

    meta = {
        "dataset": label,
        "tickers": list(panels),
        "data_start": min(p.index.min() for p in panels.values()),
        "data_end": max(p.index.max() for p in panels.values()),
        "oos_start": oos_start,
        "oos_end": oos_end,
        "next_day_target": next_var["target_date"].iloc[0] if len(next_var) else None,
        "config": {k: v for k, v in asdict(cfg).items()},
        "versions": {
            "volvar": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (results_dir / "meta.json").write_text(json.dumps(_json_safe(meta), indent=2))

    if figures:
        from volvar.reporting import make_figures

        make_figures(results_dir, result, panels, var_series, cfg, label, portfolio=portfolio)
        log("figures written")

    log("pipeline complete")
    return summary


def _qlike_np(h: pd.Series, y: pd.Series) -> np.ndarray:
    return np.log(h.to_numpy()) + y.to_numpy() / h.to_numpy()
