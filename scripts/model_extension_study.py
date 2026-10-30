#!/usr/bin/env python
"""Extension study: is the pooled LightGBM really the best forecaster, and why?

The main study (scripts/run_pipeline.py) compares six models against EWMA with pairwise
Diebold-Mariano tests. This script asks the follow-up questions an interviewer would:

1. MORE MODELS, SAME PROTOCOL. Every learner below uses the same 15 scale-free features, the
   same ratio-to-EWMA target, the same pooled walk-forward refits (every 63 days, only rows
   whose targets were already known) and the same out-of-sample dates:
     - Ridge on the same features (is the gain just a better linear combination?)
     - Ridge on log features + log target (a "HAR-X" style linear model)
     - XGBoost, Random forest, MLP (sklearn)
     - LightGBM trained on the log ratio (MSE in logs is closer to QLIKE than MSE in levels)
   plus EGARCH(1,1)-t per ticker (asymmetric GARCH in logs).
2. MODEL CONFIDENCE SET (Hansen, Lunde & Nason 2011). Pairwise DM tests against one benchmark
   do not say which models are *jointly* indistinguishable from the best. The MCS does: at
   90% confidence it keeps the set of models that cannot be rejected as worst.
3. HYPERPARAMETER SENSITIVITY of the shipped LightGBM settings (num_leaves, rounds, learning
   rate, min_child_samples, regularisation) - chosen on the FIRST half of the out-of-sample
   period and checked on the second half, so the check is not circular.
4. FEATURE ATTRIBUTION. Gain importance and drop-a-feature-group ablations (own squared
   returns, Garman-Klass range, signed returns/leverage, market, volume, weekday).

Usage:
    python scripts/model_extension_study.py --data-dir data/raw --out results/extension
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volvar.config import ExperimentConfig  # noqa: E402
from volvar.data import load_universe, make_panels  # noqa: E402
from volvar.evaluation import qlike  # noqa: E402
from volvar.forecast import run_forecasts  # noqa: E402
from volvar.models.ml import FEATURE_COLUMNS, RATIO_CLIP, build_ml_frame  # noqa: E402

GROUPS = {
    "own squared returns": ["f_r2_1", "f_r2_5", "f_r2_22", "f_r2_66"],
    "Garman-Klass range": ["f_gk_1", "f_gk_5", "f_gk_22"],
    "signed returns (leverage)": ["f_ret_1", "f_neg_1", "f_ret_5"],
    "market": ["f_mkt_r2_1", "f_mkt_r2_22"],
    "volume": ["f_vol"],
    "level + weekday": ["f_logbase", "f_dow"],
}


# ----------------------------------------------------------------------------- learners
def make_learner(kind: str, params: dict | None = None):
    """Returns (model_factory, target_transform, inverse_transform, uses_log_features)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        p = {**ExperimentConfig().lgbm_params, **(params or {})}
        p.update(random_state=7, deterministic=True, force_row_wise=True)
        return (lambda: LGBMRegressor(**p)), None, None, False
    if kind == "lgbm_log":
        from lightgbm import LGBMRegressor
        p = {**ExperimentConfig().lgbm_params, **(params or {})}
        p.update(random_state=7, deterministic=True, force_row_wise=True)
        # E[ratio] from a log model needs the variance correction exp(mu + s2/2); s2 is estimated per refit
        return (lambda: LGBMRegressor(**p)), "log", None, False
    if kind == "ridge":
        return (lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))), None, None, False
    if kind == "ridge_log":
        return (lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))), "log", None, True
    if kind == "xgb":
        from xgboost import XGBRegressor
        return (lambda: XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=200,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=2,
                                     random_state=7)), None, None, False
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return (lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=200, max_features=0.5,
                                              n_jobs=2, random_state=7)), None, None, False
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return (lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3,
                                                                     max_iter=200, early_stopping=True,
                                                                     random_state=7))), None, None, False
    raise ValueError(kind)


def _log_features(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for c in x.columns:
        if c.startswith(("f_r2", "f_gk", "f_mkt")):
            x[c] = np.log(np.clip(x[c], 1e-4, None))
    return x


def pooled_walk_forward(frame: pd.DataFrame, kind: str, features: list[str], params: dict | None = None,
                        min_train: int = 1000, refit_every: int = 63, importance: list | None = None) -> pd.DataFrame:
    """Same protocol as volvar.models.ml.lgbm_variance, any learner. Returns date,ticker,var."""
    factory, target_tf, _, log_x = make_learner(kind, params)
    usable = frame.dropna(subset=FEATURE_COLUMNS + ["base", "y"]).sort_values("date")
    calendar = np.sort(frame["date"].unique())
    preds = []
    t0 = min_train
    while t0 < len(calendar):
        t1 = min(t0 + refit_every, len(calendar))
        start, stop = calendar[t0], calendar[t1] if t1 < len(calendar) else None
        train = usable[usable["date"] < start]
        test = usable[usable["date"] >= start]
        if stop is not None:
            test = test[test["date"] < stop]
        if len(train) > 500 and len(test) > 0:
            xtr, xte = train[features], test[features]
            if log_x:
                xtr, xte = _log_features(xtr), _log_features(xte)
            y = train["y"].to_numpy()
            model = factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if target_tf == "log":
                    ly = np.log(np.clip(y, 1e-2, None))   # zero returns would give log(0); floor at 1% of the EWMA level
                    model.fit(xtr, ly)
                    resid_var = float(np.var(ly - model.predict(xtr)))
                    ratio = np.exp(model.predict(xte) + resid_var / 2)
                else:
                    model.fit(xtr, y)
                    ratio = model.predict(xte)
            if importance is not None and hasattr(model, "booster_"):
                importance.append(pd.Series(model.booster_.feature_importance("gain"), index=features))
            ratio = np.clip(ratio, *RATIO_CLIP)
            preds.append(pd.DataFrame({"date": test["date"], "ticker": test["ticker"], "var": test["base"] * ratio}))
        t0 = t1
    return pd.concat(preds, ignore_index=True)


def egarch_forecasts(panels, min_train=1000, refit_every=63) -> pd.DataFrame:
    from arch import arch_model
    rows = []
    for ticker, p in panels.items():
        ret = p["ret"].dropna()
        out = pd.Series(np.nan, index=ret.index)
        t0 = min_train
        while t0 < len(ret):
            t1 = min(t0 + refit_every, len(ret))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(ret.iloc[:t0], mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
                res = am.fit(disp="off", show_warning=False)
                full = arch_model(ret.iloc[:t1], mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
                fc = full.fix(res.params).forecast(horizon=1, start=t0 - 1, reindex=False)
            v = fc.variance.iloc[:, 0].to_numpy()
            idx = ret.index[t0:t1]
            out.loc[idx] = v[: len(idx)]
            t0 = t1
        rows.append(pd.DataFrame({"date": out.index, "ticker": ticker, "var": out.to_numpy()}).dropna())
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- evaluation
def to_wide(pieces: dict[str, pd.DataFrame], returns: pd.DataFrame) -> pd.DataFrame:
    w = returns.set_index(["date", "ticker"])[["r2"]]
    for name, df in pieces.items():
        w = w.join(df.set_index(["date", "ticker"])["var"].rename(name), how="inner")
    return w.dropna()


def mcs(wide: pd.DataFrame, models: list[str], size: float = 0.10, seed: int = 0) -> pd.DataFrame:
    """Model Confidence Set on daily cross-sectional mean QLIKE losses (arch implementation)."""
    from arch.bootstrap import MCS
    y = wide["r2"].to_numpy()
    losses = pd.DataFrame({m: qlike(wide[m].to_numpy(), y) for m in models}, index=wide.index)
    daily = losses.groupby(level="date").mean()
    m = MCS(daily, size=size, reps=2000, block_size=10, method="R", bootstrap="stationary", seed=seed)
    m.compute()
    pv = m.pvalues.rename(columns={"Pvalue": "mcs_pvalue"})
    pv["in_90pct_mcs"] = pv.index.isin(m.included)
    return pv


def mean_qlike(wide, m, mask=None):
    w = wide if mask is None else wide[mask]
    return float(qlike(w[m].to_numpy(), w["r2"].to_numpy()).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "extension"))
    ap.add_argument("--skip-egarch", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    cfg = ExperimentConfig()
    panels = make_panels(load_universe(a.data_dir))
    log(f"{len(panels)} series")
    base = run_forecasts(panels, cfg, n_jobs=2)            # the six original models
    pieces = {m: g[["date", "ticker", "var"]] for m, g in base.forecasts.groupby("model")}
    log("original six models done")
    frame = build_ml_frame(panels, cfg.ewma_lambda)

    imp: list = []
    pieces["lgbm"] = pooled_walk_forward(frame, "lgbm", FEATURE_COLUMNS, importance=imp)
    for kind in ("lgbm_log", "ridge", "ridge_log", "xgb", "rf", "mlp"):
        pieces[kind] = pooled_walk_forward(frame, kind, FEATURE_COLUMNS)
        log(f"{kind} done")
    if not a.skip_egarch:
        pieces["egarch"] = egarch_forecasts(panels, cfg.min_train, cfg.refit_every)
        log("egarch done")

    wide = to_wide(pieces, base.returns)
    models = [c for c in wide.columns if c != "r2"]
    dates = wide.index.get_level_values("date")
    table = pd.DataFrame({m: {"qlike": mean_qlike(wide, m)} for m in models}).T
    table["gain_vs_ewma_pct"] = 100 * (table.loc["ewma", "qlike"] - table["qlike"]) / table.loc["ewma", "qlike"]
    table = table.join(mcs(wide, models)).sort_values("qlike")
    table.to_csv(out / "extended_model_comparison.csv")
    log("\n" + table.round(4).to_string())

    # ---- hyperparameter sensitivity: rank on the first half of the OOS period, confirm on the second
    mid = dates.min() + (dates.max() - dates.min()) / 2
    first, second = dates < mid, dates >= mid
    grid = {"shipped (leaves 8, 200 x 0.03, min_child 200, lambda 10)": {}, "num_leaves 4": {"num_leaves": 4},
            "num_leaves 32": {"num_leaves": 32}, "num_leaves 128, min_child 20": {"num_leaves": 128, "min_child_samples": 20},
            "100 rounds": {"n_estimators": 100}, "500 rounds": {"n_estimators": 500},
            "learning_rate 0.1": {"learning_rate": 0.1}, "min_child_samples 50": {"min_child_samples": 50},
            "min_child_samples 1000": {"min_child_samples": 1000}, "reg_lambda 0": {"reg_lambda": 0.0}}
    sens = []
    for name, params in grid.items():
        p = pooled_walk_forward(frame, "lgbm", FEATURE_COLUMNS, params)
        w = to_wide({"x": p}, base.returns).join(wide[[]], how="inner")
        d = w.index.get_level_values("date")
        sens.append({"setting": name, "qlike_first_half": mean_qlike(w, "x", d < mid),
                     "qlike_second_half": mean_qlike(w, "x", d >= mid), "qlike_all": mean_qlike(w, "x")})
        log(f"  {name:55s} {sens[-1]['qlike_first_half']:.4f} / {sens[-1]['qlike_second_half']:.4f}")
    pd.DataFrame(sens).to_csv(out / "lgbm_sensitivity.csv", index=False)

    # ---- feature attribution
    gain = pd.concat(imp, axis=1).mean(axis=1)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    gain.rename("gain_share").to_csv(out / "lgbm_gain_importance.csv")
    ablation = [{"dropped": "none", "qlike": mean_qlike(wide, "lgbm")}]
    for g, cols in GROUPS.items():
        feats = [c for c in FEATURE_COLUMNS if c not in cols]
        p = pooled_walk_forward(frame, "lgbm", feats)
        w = to_wide({"x": p}, base.returns).join(wide[[]], how="inner")
        ablation.append({"dropped": g, "qlike": mean_qlike(w, "x")})
        log(f"  drop {g:30s} {ablation[-1]['qlike']:.4f}")
    abl = pd.DataFrame(ablation)
    abl["qlike_increase_pct"] = 100 * (abl["qlike"] - abl.loc[0, "qlike"]) / abl.loc[0, "qlike"]
    abl.to_csv(out / "feature_group_ablation.csv", index=False)
    meta = {"oos_start": str(dates.min().date()), "oos_end": str(dates.max().date()), "n_forecasts": int(len(wide)),
            "split_for_sensitivity": str(pd.Timestamp(mid).date()), "runtime_s": time.time() - t_start}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log("done")


if __name__ == "__main__":
    main()
