"""Pooled LightGBM forecaster that learns a correction to the EWMA forecast.

Instead of predicting raw variance (very noisy, heavy-tailed) the model predicts
the *ratio* ``r2_{t+1} / ewma_{t+1}`` from scale-free features. The final forecast is
``ewma * predicted_ratio``, which keeps the forecast positive and anchored to a
sensible baseline. One model is trained on all tickers together and re-estimated
every ``refit_every`` days using only rows whose targets were already known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from volvar.models.baselines import ewma_variance

FEATURE_COLUMNS = [
    "f_r2_1",
    "f_r2_5",
    "f_r2_22",
    "f_r2_66",
    "f_gk_1",
    "f_gk_5",
    "f_gk_22",
    "f_ret_1",
    "f_neg_1",
    "f_ret_5",
    "f_logbase",
    "f_vol",
    "f_mkt_r2_1",
    "f_mkt_r2_22",
    "f_dow",
]
RATIO_CLIP = (0.2, 4.0)
TARGET_CLIP = 20.0


def _market_features(panels: dict[str, pd.DataFrame], lam: float) -> pd.DataFrame:
    """Equal-weighted market variance features, indexed by target date."""
    rets = pd.concat({t: p["ret"] for t, p in panels.items()}, axis=1)
    mkt = rets.mean(axis=1, skipna=True).dropna()
    mkt_r2 = mkt**2
    base = ewma_variance(mkt_r2, lam=lam)
    return pd.DataFrame(
        {
            "f_mkt_r2_1": mkt_r2.shift(1) / base,
            "f_mkt_r2_22": mkt_r2.rolling(22, min_periods=22).mean().shift(1) / base,
        }
    )


def build_ml_frame(panels: dict[str, pd.DataFrame], lam: float = 0.94) -> pd.DataFrame:
    """Long frame, one row per (ticker, target date), with features known the day before."""
    mkt = _market_features(panels, lam)
    frames = []
    for ticker, p in panels.items():
        r2, ret, gk = p["r2"], p["ret"], p["gk"]
        base = ewma_variance(r2, lam=lam)
        sd = np.sqrt(base)
        vol = p["volume"].replace(0.0, np.nan)
        vol_med = vol.rolling(22, min_periods=10).median()
        f = pd.DataFrame(
            {
                "f_r2_1": r2.shift(1) / base,
                "f_r2_5": r2.rolling(5, min_periods=5).mean().shift(1) / base,
                "f_r2_22": r2.rolling(22, min_periods=22).mean().shift(1) / base,
                "f_r2_66": r2.rolling(66, min_periods=66).mean().shift(1) / base,
                "f_gk_1": gk.shift(1) / base,
                "f_gk_5": gk.rolling(5, min_periods=5).mean().shift(1) / base,
                "f_gk_22": gk.rolling(22, min_periods=22).mean().shift(1) / base,
                "f_ret_1": ret.shift(1) / sd,
                "f_neg_1": ret.clip(upper=0.0).shift(1) / sd,
                "f_ret_5": ret.rolling(5, min_periods=5).sum().shift(1) / sd,
                "f_logbase": np.log(base),
                "f_vol": np.log(vol.shift(1) / vol_med.shift(1)).fillna(0.0),
            },
            index=p.index,
        )
        f = f.join(mkt, how="left")
        f["f_dow"] = f.index.dayofweek.astype(float)
        f["base"] = base
        f["r2"] = r2
        f["y"] = (r2 / base).clip(upper=TARGET_CLIP)
        f["ticker"] = ticker
        f.index.name = "date"
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True)


def lgbm_variance(
    panels: dict[str, pd.DataFrame],
    min_train: int = 1000,
    refit_every: int = 63,
    lam: float = 0.94,
    params: dict | None = None,
    seed: int = 7,
) -> dict[str, pd.Series]:
    """Walk-forward pooled LightGBM forecasts; returns ``{ticker: variance Series}``."""
    frame = build_ml_frame(panels, lam)
    usable = frame.dropna(subset=FEATURE_COLUMNS + ["base", "y"]).sort_values("date")
    calendar = np.sort(frame["date"].unique())
    if len(calendar) <= min_train + 1:
        raise ValueError("not enough history for the requested min_train")

    model_params = dict(params or {})
    model_params.update({"random_state": seed, "deterministic": True, "force_row_wise": True})

    preds: list[pd.DataFrame] = []
    t0 = min_train
    while t0 < len(calendar):
        t1 = min(t0 + refit_every, len(calendar))
        start, stop = calendar[t0], calendar[t1] if t1 < len(calendar) else None
        train = usable[usable["date"] < start]
        test = usable[usable["date"] >= start]
        if stop is not None:
            test = test[test["date"] < stop]
        if len(train) > 500 and len(test) > 0:
            model = LGBMRegressor(**model_params)
            model.fit(train[FEATURE_COLUMNS], train["y"])
            ratio = np.clip(model.predict(test[FEATURE_COLUMNS]), *RATIO_CLIP)
            preds.append(
                pd.DataFrame(
                    {"date": test["date"], "ticker": test["ticker"], "var": test["base"] * ratio}
                )
            )
        t0 = t1

    if not preds:
        raise ValueError("no out-of-sample LightGBM forecasts were produced")
    allp = pd.concat(preds, ignore_index=True)
    out = {}
    for ticker, g in allp.groupby("ticker"):
        out[ticker] = g.set_index("date")["var"].rename("lgbm").sort_index()
    return out
