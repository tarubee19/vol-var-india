import numpy as np
import pandas as pd
import pytest
from scipy import stats

from volvar.backtest import (
    backtest_var,
    basel_zone,
    christoffersen_cc,
    christoffersen_independence,
    es_ratio,
    kupiec_pof,
    violations,
    worst_rolling_exceptions,
)
from volvar.risk import fhs_var_es, normal_var_es, var_es


def test_normal_var_es_known_values():
    sigma = pd.Series([1.0, 2.0])
    v, e = normal_var_es(sigma, 0.99)
    assert v.iloc[0] == pytest.approx(2.3263, abs=1e-3)
    assert e.iloc[0] == pytest.approx(2.6652, abs=1e-3)
    assert v.iloc[1] == pytest.approx(2 * v.iloc[0])
    assert (e > v).all()  # ES always exceeds VaR


def test_fhs_matches_normal_for_gaussian_data_and_uses_no_future():
    rng = np.random.default_rng(1)
    n = 3000
    idx = pd.bdate_range("2015-01-01", periods=n)
    sigma = pd.Series(1.0 + 0.5 * np.abs(np.sin(np.arange(n) / 50)), index=idx)
    ret = pd.Series(sigma.to_numpy() * rng.standard_normal(n), index=idx)
    var, es = fhs_var_es(ret, sigma**2, 0.99, window=1000, min_obs=250)
    assert var.iloc[:250].isna().all()
    ratio = (var / sigma).dropna().mean()
    assert ratio == pytest.approx(stats.norm.ppf(0.99), rel=0.06)
    assert (es.dropna() > var.dropna()).all()
    # changing today's return must not change today's VaR (only tomorrow's)
    ret2 = ret.copy()
    ret2.iloc[1500] = -10.0 * sigma.iloc[1500]  # an extreme loss on day 1500
    var2, es2 = fhs_var_es(ret2, sigma**2, 0.99, window=1000, min_obs=250)
    assert var2.iloc[1500] == pytest.approx(var.iloc[1500])
    assert es2.iloc[1500] == pytest.approx(es.iloc[1500])
    assert es2.iloc[1501] > es.iloc[1501]  # the loss enters the tail from the next day on


def test_var_es_dispatch_and_validation():
    idx = pd.bdate_range("2020-01-01", periods=400)
    var = pd.Series(1.0, index=idx)
    ret = pd.Series(np.random.default_rng(0).standard_normal(400), index=idx)
    v, _ = var_es(ret, var, 0.95, method="normal")
    assert v.iloc[0] == pytest.approx(stats.norm.ppf(0.95))
    with pytest.raises(ValueError):
        var_es(ret, var, 0.95, method="magic")
    with pytest.raises(ValueError):
        normal_var_es(var, 1.5)


def test_kupiec_known_value_and_edges():
    lr, p = kupiec_pof(x=10, n=250, p=0.01)
    pi = 10 / 250
    expected = -2 * ((240 * np.log(0.99) + 10 * np.log(0.01)) - (240 * np.log(1 - pi) + 10 * np.log(pi)))
    assert lr == pytest.approx(expected)
    assert p < 0.001
    lr0, p0 = kupiec_pof(x=0, n=250, p=0.01)
    assert lr0 > 0 and 0 < p0 < 1  # zero violations in 250 days is still plausible at 1%
    lr_ok, p_ok = kupiec_pof(x=3, n=300, p=0.01)
    assert lr_ok == pytest.approx(0.0, abs=1e-9) and p_ok == pytest.approx(1.0)
    assert np.isnan(kupiec_pof(0, 0, 0.01)[0])


def test_independence_flags_clustering():
    clustered = np.zeros(1000, dtype=int)
    for start in (100, 300, 500, 700, 900):
        clustered[start : start + 4] = 1  # 20 violations in 5 tight clusters
    _, p_clustered = christoffersen_independence(clustered)
    assert p_clustered < 0.001
    rng = np.random.default_rng(2)
    rand = (rng.random(5000) < 0.02).astype(int)
    _, p_random = christoffersen_independence(rand)
    assert p_random > 0.05
    assert np.isnan(christoffersen_independence(np.zeros(100, dtype=int))[0])
    lr_cc, p_cc = christoffersen_cc(clustered, 0.02)
    assert p_cc < 0.001 and lr_cc > 0


def test_basel_zones_and_rolling_window():
    assert [basel_zone(k) for k in (0, 4, 5, 9, 10, 25)] == [
        "green", "green", "yellow", "yellow", "red", "red",
    ]
    v = np.zeros(600, dtype=int)
    v[300:307] = 1
    assert worst_rolling_exceptions(v, 250) == 7
    assert worst_rolling_exceptions(np.ones(10, dtype=int), 250) == 10


def test_backtest_var_well_calibrated_vs_too_low():
    rng = np.random.default_rng(4)
    n = 4000
    idx = pd.bdate_range("2012-01-01", periods=n)
    ret = pd.Series(rng.standard_normal(n), index=idx)
    good_var = pd.Series(stats.norm.ppf(0.99), index=idx)
    es = pd.Series(stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, index=idx)
    good = backtest_var(ret, good_var, es, 0.99)
    assert good["passes"] and good["violation_rate"] == pytest.approx(0.01, abs=0.006)
    assert good["es_ratio"] == pytest.approx(1.0, abs=0.25)
    bad = backtest_var(ret, good_var * 0.6, es * 0.6, 0.99)
    assert not bad["passes"] and bad["basel_zone"] == "red"
    assert backtest_var(ret, good_var, es, 0.95)["basel_zone"] == "n/a"


def test_violations_and_es_ratio_alignment():
    idx = pd.bdate_range("2020-01-01", periods=5)
    ret = pd.Series([-3.0, 0.5, -0.2, -4.0, 1.0], index=idx)
    var = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0], index=idx)
    es = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0], index=idx)
    assert violations(ret, var).tolist() == [1, 0, 0, 1, 0]
    assert es_ratio(ret, var, es) == pytest.approx(((3.0 + 4.0) / 2) / 3.0)
