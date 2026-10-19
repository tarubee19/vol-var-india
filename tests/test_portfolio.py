"""Portfolio VaR, attribution and stress tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from volvar.portfolio import (
    NOTIONAL_INR,
    equal_weights,
    euler_contributions,
    ewma_covariances,
    historical_var_es,
    portfolio_variance,
    run_portfolio_study,
    simple_returns_pct,
    stock_tickers,
    stress_tests,
)


@pytest.fixture(scope="module")
def rets():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=400)
    common = rng.standard_normal(400)
    data = {f"S{i}": common * 0.8 + rng.standard_normal(400) * (0.5 + 0.2 * i) for i in range(4)}
    return pd.DataFrame(data, index=idx)


def test_index_is_excluded_from_the_book(small_panels):
    panels = dict(small_panels)
    panels["NIFTY50"] = small_panels["SIM01"]
    assert "NIFTY50" not in stock_tickers(panels)
    assert set(stock_tickers(panels)) == set(small_panels)


def test_simple_returns_match_log_returns(small_panels):
    wide = simple_returns_pct(small_panels, list(small_panels))
    log_r = small_panels["SIM01"]["ret"].reindex(wide.index)
    np.testing.assert_allclose(wide["SIM01"], 100 * np.expm1(log_r / 100))


def test_ewma_covariance_uses_only_past_rows(rets):
    base = ewma_covariances(rets)
    shocked = rets.copy()
    shocked.iloc[300:] *= 50.0  # change the future only
    after = ewma_covariances(shocked)
    np.testing.assert_allclose(base[: 301], after[: 301])  # forecasts up to day 300 unchanged
    assert not np.allclose(base[302], after[302])
    assert base.shape == (len(rets) + 1, 4, 4)


def test_ewma_covariance_recursion(rets):
    lam = 0.94
    covs = ewma_covariances(rets, lam=lam, init=60)
    r = rets.iloc[100].to_numpy()[:, None]
    np.testing.assert_allclose(covs[101], lam * covs[100] + (1 - lam) * r @ r.T)


def test_portfolio_variance_single_asset_reduces_to_asset_variance(rets):
    covs = ewma_covariances(rets[["S0"]])
    np.testing.assert_allclose(portfolio_variance(covs, np.array([1.0])), covs[:, 0, 0])


def test_euler_contributions_add_up(rets):
    covs = ewma_covariances(rets)
    w = equal_weights(list(rets.columns))
    c = euler_contributions(covs[-1], w, 0.99)
    sigma_p = float(np.sqrt(w.to_numpy() @ covs[-1] @ w.to_numpy()))
    assert c["pct_of_portfolio_var"].sum() == pytest.approx(1.0)
    assert c["component_var_pct"].sum() == pytest.approx(stats.norm.ppf(0.99) * sigma_p)
    # diversification: the portfolio VaR is below the sum of stand-alone VaRs
    assert c["component_var_pct"].sum() < c["standalone_var_pct"].sum()
    assert c["component_var_inr"].sum() == pytest.approx(c["component_var_pct"].sum() / 100 * NOTIONAL_INR)


def test_historical_var_matches_empirical_quantile():
    s = pd.Series(np.arange(1.0, 601.0) - 300.0, index=pd.bdate_range("2020-01-01", periods=600))
    var, es = historical_var_es(s, 0.99, window=500)
    window = s.iloc[99:599].to_numpy()  # the 500 returns before the last day
    q = np.quantile(window, 0.01)
    assert var.iloc[-1] == pytest.approx(-q)
    assert es.iloc[-1] == pytest.approx(-window[window <= q].mean())
    assert var.iloc[:500].isna().all()


def test_hypothetical_sector_shocks_use_weights():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    panels = {
        t: pd.DataFrame({"ret": rng.standard_normal(300)}, index=idx)
        for t in ("HDFCBANK", "SBIN", "TCS", "ITC")
    }
    w = equal_weights(list(panels))
    port = pd.Series(rng.standard_normal(300), index=idx)
    out = stress_tests(panels, list(panels), w, port, None, label="simulated").set_index("scenario")
    assert out.loc["Banks fall 15%", "pnl_pct"] == pytest.approx(-15 * 0.5)
    assert out.loc["IT services fall 12%", "pnl_pct"] == pytest.approx(-12 * 0.25)
    assert "historical" not in set(out["type"])  # dates are meaningless on simulated data


def test_portfolio_study_on_simulated_universe(small_panels):
    res = run_portfolio_study(small_panels, label="simulated", fhs_min_obs=100, hs_window=250)
    bt = res.backtest
    assert {"hs", "ewma_normal", "ewma_fhs"} <= set(bt["model"])
    assert set(np.round(bt["level"], 3)) == {0.95, 0.975, 0.99}
    assert bt["n"].nunique() == 1  # one common out-of-sample window
    r99 = bt[np.isclose(bt.level, 0.99)]
    assert (r99["violation_rate"] < 0.05).all()
    assert len(res.next_day) >= 9
    assert (res.next_day["var_pct"] > 0).all()
    assert (res.next_day["es_pct"] >= res.next_day["var_pct"] - 1e-9).all()


def test_pipeline_writes_portfolio_outputs(pipeline_run):
    summary, out = pipeline_run
    for name in ("portfolio_backtest.csv", "portfolio_next_day.csv", "portfolio_contributions.csv",
                 "stress_tests.csv"):
        assert (out / name).exists(), name
    saved = json.loads((out / "summary.json").read_text())
    assert saved["portfolio"]["n_stocks"] == 3
    assert 0 < saved["portfolio"]["diversification_benefit"] < 1
    assert (out / "figures" / "fig_risk_contrib.png").exists()


def test_portfolio_endpoint(pipeline_run):
    from fastapi.testclient import TestClient

    from volvar.api import create_app

    with TestClient(create_app(results_dir=pipeline_run[1])) as client:
        r = client.get("/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "simulated"
    assert body["next_day"] and body["backtest"] and body["contributions"]
    shares = sum(c["pct_of_portfolio_var"] for c in body["contributions"])
    assert shares == pytest.approx(1.0)
