"""Shared fixtures: small simulated universes so tests stay fast."""

from __future__ import annotations

import pytest

from volvar.config import ExperimentConfig
from volvar.data import make_panels
from volvar.pipeline import run_pipeline
from volvar.synthetic import simulate_universe, write_universe


@pytest.fixture(scope="session")
def small_universe():
    return simulate_universe(n_days=1500, n_tickers=3, seed=5)


@pytest.fixture(scope="session")
def small_panels(small_universe):
    return make_panels(small_universe)


@pytest.fixture(scope="session")
def one_panel(small_panels):
    return small_panels["SIM01"]


@pytest.fixture(scope="session")
def small_cfg():
    """Cheap experiment settings for tests (short history, tiny LightGBM)."""
    return ExperimentConfig(
        min_train=600,
        refit_every=126,
        lgbm_params={
            "n_estimators": 30,
            "num_leaves": 4,
            "min_child_samples": 50,
            "verbosity": -1,
            "n_jobs": 1,
        },
    )


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, small_universe, small_cfg):
    """One full pipeline run on the small simulated universe, shared by all tests."""
    data_dir = tmp_path_factory.mktemp("data")
    out_dir = tmp_path_factory.mktemp("results")
    write_universe(small_universe, data_dir)
    summary = run_pipeline(data_dir, out_dir, cfg=small_cfg, n_jobs=1, verbose=False)
    return summary, out_dir
