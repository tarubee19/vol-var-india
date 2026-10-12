import numpy as np
import pandas as pd
import pytest

from volvar.models.baselines import ewma_variance, historical_variance
from volvar.models.garch import garch_variance, persistence
from volvar.models.har import har_features, har_variance
from volvar.models.ml import FEATURE_COLUMNS, build_ml_frame, lgbm_variance

MIN_TRAIN = 600
REFIT = 126


def test_historical_uses_only_past_data():
    r2 = pd.Series(np.arange(1.0, 301.0))
    h = historical_variance(r2, window=250)
    assert h.iloc[:250].isna().all()
    assert h.iloc[250] == pytest.approx(r2.iloc[:250].mean())
    # perturbing today's value must not change today's forecast
    r2b = r2.copy()
    r2b.iloc[260] = 1e6
    assert historical_variance(r2b, 250).iloc[260] == pytest.approx(h.iloc[260])


def test_ewma_recursion_and_no_lookahead():
    rng = np.random.default_rng(0)
    r2 = pd.Series(rng.standard_normal(200) ** 2)
    e = ewma_variance(r2, lam=0.94, init_window=30)
    assert e.iloc[:30].isna().all()
    assert e.iloc[31] == pytest.approx(0.94 * e.iloc[30] + 0.06 * r2.iloc[30])
    r2b = r2.copy()
    r2b.iloc[100] = 1e3
    assert ewma_variance(r2b, 0.94, 30).iloc[100] == pytest.approx(e.iloc[100])
    assert ewma_variance(r2b, 0.94, 30).iloc[101] > e.iloc[101]


def test_ewma_rejects_bad_lambda():
    with pytest.raises(ValueError):
        ewma_variance(pd.Series([1.0] * 50), lam=1.2)


@pytest.mark.parametrize("kind", ["garch", "gjr"])
def test_garch_forecasts_are_positive_and_oos_only(one_panel, kind):
    out = garch_variance(one_panel["ret"], kind=kind, min_train=MIN_TRAIN, refit_every=REFIT)
    v = out.variance
    assert v.iloc[:MIN_TRAIN].isna().all()
    assert v.iloc[MIN_TRAIN:].notna().all()
    assert (v.dropna() > 0).all()
    assert len(out.refits) >= 5
    assert 0.5 < persistence(out.last_params) < 1.05


def test_garch_forecast_ignores_future_returns(one_panel):
    r = one_panel["ret"].copy()
    base = garch_variance(r, "gjr", MIN_TRAIN, REFIT).variance
    cut = MIN_TRAIN + 200
    r2 = r.copy()
    r2.iloc[cut + 50 :] = r2.iloc[cut + 50 :] * 5  # distort only data after the check date
    changed = garch_variance(r2, "gjr", MIN_TRAIN, REFIT).variance
    # forecasts strictly before the distortion starts (and before any refit sees it) are identical
    np.testing.assert_allclose(base.iloc[MIN_TRAIN:cut].to_numpy(), changed.iloc[MIN_TRAIN:cut].to_numpy())


def test_garch_input_validation(one_panel):
    with pytest.raises(ValueError):
        garch_variance(one_panel["ret"], kind="egarch")
    with pytest.raises(ValueError):
        garch_variance(one_panel["ret"].iloc[:200], min_train=1000)


def test_har_features_and_forecast(one_panel):
    feats = har_features(one_panel)
    assert list(feats.columns) == ["gk_d", "gk_w", "gk_m", "r2_d"]
    hist = historical_variance(one_panel["r2"], 250)
    h = har_variance(one_panel, MIN_TRAIN, REFIT, floor_series=hist)
    assert h.iloc[:MIN_TRAIN].isna().all()
    assert (h.dropna() > 0).all()


def test_lgbm_pooled_forecasts(small_panels):
    out = lgbm_variance(
        small_panels,
        min_train=MIN_TRAIN,
        refit_every=REFIT,
        params={"n_estimators": 30, "num_leaves": 4, "min_child_samples": 50, "verbosity": -1, "n_jobs": 1},
    )
    assert set(out) == set(small_panels)
    for s in out.values():
        assert (s > 0).all() and s.notna().all()
        assert len(s) > 300


def test_ml_frame_features_have_no_lookahead(small_panels):
    frame = build_ml_frame(small_panels)
    row = frame[(frame.ticker == "SIM01")].dropna(subset=FEATURE_COLUMNS).iloc[100]
    panel = small_panels["SIM01"]
    pos = panel.index.get_loc(row["date"])
    # the first squared-return feature must be yesterday's r2 divided by today's EWMA base
    assert row["f_r2_1"] * row["base"] == pytest.approx(panel["r2"].iloc[pos - 1])
    assert row["r2"] == pytest.approx(panel["r2"].iloc[pos])
