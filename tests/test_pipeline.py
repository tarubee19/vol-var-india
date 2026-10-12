import json

import numpy as np
import pandas as pd
import pytest

from volvar.forecast import extend_panels, next_day_variances
from volvar.models.ml import FEATURE_COLUMNS, build_ml_frame
from volvar.pipeline import detect_dataset_label


def test_pipeline_writes_all_outputs(pipeline_run):
    summary, out = pipeline_run
    expected = [
        "diagnostics.csv", "model_comparison.csv", "dm_tests.csv", "loss_by_ticker.csv",
        "loss_by_regime.csv", "loss_by_year.csv", "var_backtest.csv", "var_summary.csv",
        "next_day.csv", "next_day_var.csv", "summary.json", "meta.json", "forecasts.parquet",
        "figures/fig_model_comparison.png", "figures/fig_vol_forecast.png",
        "figures/fig_var_backtest.png", "figures/fig_violation_heatmap.png",
    ]
    for name in expected:
        assert (out / name).exists(), name
    assert summary["dataset"] == "simulated"
    assert not (out / "loss_covid_window.csv").exists()  # never produced for simulated data


def test_pipeline_results_are_sane(pipeline_run, small_cfg):
    summary, out = pipeline_run
    comp = pd.read_csv(out / "model_comparison.csv", index_col="model")
    assert set(comp.index) == set(small_cfg.models)
    assert comp["qlike"].notna().all() and (comp["n"] == comp["n"].iloc[0]).all()
    bt = pd.read_csv(out / "var_backtest.csv")
    assert {"fhs", "normal"} == set(bt["method"]) and {0.95, 0.99} == set(bt["level"])
    assert bt["violation_rate"].between(0, 0.2).all()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["dataset"] == "simulated" and meta["next_day_target"]
    nxt = pd.read_csv(out / "next_day.csv")
    assert (nxt["var"] > 0).all() and set(nxt["model"]) == set(small_cfg.models)
    risk = pd.read_csv(out / "next_day_var.csv")
    assert (risk["es_pct"] >= risk["var_pct"]).all()


def test_detect_dataset_label(tmp_path):
    assert detect_dataset_label(tmp_path, ["SIM01", "SIM02"]) == "simulated"
    assert detect_dataset_label(tmp_path, ["RELIANCE", "TCS"]) == "real"
    (tmp_path / "README.txt").write_text("SIMULATED DATA")
    assert detect_dataset_label(tmp_path, ["RELIANCE"]) == "simulated"


def test_next_day_placeholder_cannot_influence_its_own_forecast(small_panels):
    ext, target, skipped = extend_panels(small_panels)
    assert not skipped and target > max(p.index[-2] for p in ext.values())
    frame_a = build_ml_frame(ext)
    tweaked = {t: p.copy() for t, p in ext.items()}
    for p in tweaked.values():
        p.iloc[-1, p.columns.get_loc("ret")] = 7.5
        p.iloc[-1, p.columns.get_loc("r2")] = 7.5**2
        p.iloc[-1, p.columns.get_loc("gk")] = 50.0
        p.iloc[-1, p.columns.get_loc("volume")] = 1.0
    frame_b = build_ml_frame(tweaked)
    last_a = frame_a[frame_a["date"] == target].set_index("ticker")[FEATURE_COLUMNS + ["base"]]
    last_b = frame_b[frame_b["date"] == target].set_index("ticker")[FEATURE_COLUMNS + ["base"]]
    pd.testing.assert_frame_equal(last_a, last_b)


def test_next_day_ewma_matches_hand_calculation(small_panels, small_cfg):
    out = next_day_variances(small_panels, small_cfg)
    p = small_panels["SIM01"]
    from volvar.models.baselines import ewma_variance

    s = ewma_variance(p["r2"], 0.94)
    expected = 0.94 * s.iloc[-1] + 0.06 * p["r2"].iloc[-1]
    got = out[(out.ticker == "SIM01") & (out.model == "ewma")]["var"].iloc[0]
    assert got == pytest.approx(expected)
    assert np.isfinite(out["var"]).all() and (out["var"] > 0).all()
