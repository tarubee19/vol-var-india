"""Central configuration: paths and default experiment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


DATA_RAW = _path_from_env("VOLVAR_DATA_DIR", ROOT / "data" / "raw")
DATA_SYNTH = ROOT / "data" / "synthetic"
RESULTS_DIR = _path_from_env("VOLVAR_RESULTS_DIR", ROOT / "results")
KB_DIR = ROOT / "docs" / "kb"

TRADING_DAYS = 252
VAR_LEVELS = (0.95, 0.99)
MODEL_ORDER = ("hist", "ewma", "garch", "gjr", "har", "lgbm")
MODEL_LABELS = {
    "hist": "Historical (250d)",
    "ewma": "EWMA (lambda=0.94)",
    "garch": "GARCH(1,1)-t",
    "gjr": "GJR-GARCH(1,1,1)-t",
    "har": "HAR (Garman-Klass)",
    "lgbm": "LightGBM (pooled)",
}
BENCHMARK_MODEL = "ewma"


@dataclass
class ExperimentConfig:
    """Settings for one walk-forward run."""

    min_train: int = 1000  # trading days before the first out-of-sample forecast
    refit_every: int = 63  # re-estimate parameters roughly every quarter
    hist_window: int = 250
    ewma_lambda: float = 0.94
    fhs_window: int = 500  # trailing window for filtered-historical-simulation quantiles
    fhs_min_obs: int = 250
    var_levels: tuple[float, ...] = VAR_LEVELS
    models: tuple[str, ...] = MODEL_ORDER
    seed: int = 7
    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 8,
            "min_child_samples": 200,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
            "verbosity": -1,
            "n_jobs": 2,
        }
    )
