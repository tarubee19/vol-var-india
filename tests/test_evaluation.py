import numpy as np
import pandas as pd
import pytest

from volvar.evaluation import (
    aligned_wide,
    best_model_counts,
    dm_vs_benchmark,
    loss_by_regime,
    loss_in_window,
    loss_table,
    market_regimes,
    mse,
    per_ticker_losses,
    qlike,
)


def test_qlike_minimised_at_true_variance():
    rng = np.random.default_rng(0)
    true_var = 2.0
    y = true_var * rng.standard_normal(200_000) ** 2
    grid = np.linspace(0.5, 4.0, 36)
    losses = [qlike(np.full_like(y, h), y).mean() for h in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(true_var, abs=0.15)


def test_mse_basic():
    assert mse(np.array([2.0]), np.array([5.0]))[0] == 9.0


def _wide(n_days=400, tickers=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for t in tickers:
        true_var = 1.0 + 0.5 * np.sin(np.arange(n_days) / 30.0)
        y = true_var * rng.standard_normal(n_days) ** 2
        rows.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "date": dates,
                    "good": true_var,  # the true variance
                    "bad": np.full(n_days, 3.0),  # a flat, badly scaled forecast
                    "ewma": np.full(n_days, 1.0),
                    "r2": y,
                }
            )
        )
    return pd.concat(rows).set_index(["ticker", "date"]).sort_index()


def test_loss_table_ranks_true_variance_best():
    t = loss_table(_wide(), ["good", "bad", "ewma"])
    assert t["qlike"].idxmin() == "good"
    assert t.loc["bad", "qlike"] > t.loc["ewma", "qlike"]


def test_dm_detects_clear_difference_and_ignores_identical():
    w = _wide(n_days=800)
    clear = dm_vs_benchmark(w, ["good", "bad"], benchmark="ewma", loss="qlike")
    assert clear.loc["bad", "mean_diff"] < 0 and clear.loc["bad", "p_value"] < 0.01
    w["same"] = w["ewma"]
    same = dm_vs_benchmark(w, ["same"], benchmark="ewma")
    assert same.loc["same", "p_value"] != same.loc["same", "p_value"] or same.loc["same", "p_value"] > 0.5


def test_per_ticker_and_counts():
    pt = per_ticker_losses(_wide(), models=["good", "bad", "ewma"])
    assert list(pt.index) == ["A", "B"]
    counts = best_model_counts(pt)
    assert counts["good"] == 2


def test_aligned_wide_uses_common_sample():
    dates = pd.bdate_range("2023-01-02", periods=10)
    f = pd.concat(
        [
            pd.DataFrame({"date": dates, "ticker": "X", "model": "ewma", "var": 1.0}),
            pd.DataFrame({"date": dates[3:], "ticker": "X", "model": "garch", "var": 1.2}),
        ]
    )
    r = pd.DataFrame({"date": dates, "ticker": "X", "ret": 1.0, "r2": 1.0})
    w = aligned_wide(f, r, ["ewma", "garch"])
    assert len(w) == 7  # only dates where both models forecast
    assert {"ewma", "garch", "r2"} <= set(w.columns)


def test_regimes_and_window():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-04", periods=300)
    vol = np.where(np.arange(300) > 150, 3.0, 0.5)
    rets = pd.DataFrame(
        {"date": dates, "ticker": "A", "ret": rng.standard_normal(300) * vol}
    )
    rets["r2"] = rets["ret"] ** 2
    reg = market_regimes(rets)
    assert reg.dropna().isin(["calm", "normal", "turbulent"]).all()
    assert reg.iloc[-1] == "turbulent" and reg.iloc[60] == "calm"
    w = _wide(n_days=300, tickers=("A",))
    w = w.set_index(w.index.set_levels(dates, level="date", verify_integrity=False))
    by = loss_by_regime(w, reg, models=["good", "bad"])
    assert set(by.index) <= {"calm", "normal", "turbulent"}
    assert loss_in_window(w, "2021-01-04", "2021-06-30", ["good", "bad"]) is not None
    assert loss_in_window(w, "2021-02-01", "2021-02-10", ["good", "bad"]) is None  # < 50 obs
    assert loss_in_window(w, "2030-01-01", "2030-02-01") is None
