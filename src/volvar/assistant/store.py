"""Read-only view over the files written by ``run_pipeline``.

The assistant never recomputes anything: every number it can quote comes from one of these
tables, which is what makes the answers checkable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import MODEL_ORDER

REQUIRED_FILES = (
    "meta.json",
    "next_day.csv",
    "next_day_var.csv",
    "model_comparison.csv",
    "var_backtest.csv",
    "var_summary.csv",
)


class ResultsNotFoundError(FileNotFoundError):
    """Raised when the results directory has not been produced by ``run_pipeline`` yet."""


class ResultsStore:
    """Loads the pipeline outputs once and answers lookups against them."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        missing = [f for f in REQUIRED_FILES if not (self.results_dir / f).exists()]
        if missing:
            raise ResultsNotFoundError(
                f"{self.results_dir} is missing {', '.join(missing)}. "
                "Run `python scripts/run_pipeline.py` first."
            )
        self.meta: dict = json.loads((self.results_dir / "meta.json").read_text())
        self.next_day = pd.read_csv(self.results_dir / "next_day.csv")
        self.next_day_var = pd.read_csv(self.results_dir / "next_day_var.csv")
        self.model_comparison = pd.read_csv(self.results_dir / "model_comparison.csv")
        self.var_backtest = pd.read_csv(self.results_dir / "var_backtest.csv")
        self.var_summary = pd.read_csv(self.results_dir / "var_summary.csv")

    # -- identity ---------------------------------------------------------------------------
    @property
    def dataset(self) -> str:
        return str(self.meta.get("dataset", "unknown"))

    @property
    def is_simulated(self) -> bool:
        return self.dataset == "simulated"

    @property
    def dataset_note(self) -> str:
        if self.is_simulated:
            return "SIMULATED data (pipeline validation only, not market performance)"
        if self.dataset == "real":
            return "real daily market data"
        return f"{self.dataset} data"

    @property
    def tickers(self) -> list[str]:
        return [str(t) for t in self.meta.get("tickers", sorted(self.next_day.ticker.unique()))]

    @property
    def models(self) -> list[str]:
        present = set(self.next_day.model.astype(str))
        return [m for m in MODEL_ORDER if m in present]

    @property
    def best_model(self) -> str:
        """Model with the lowest out-of-sample QLIKE (used when the user names none)."""
        row = self.model_comparison.sort_values("qlike").iloc[0]
        return str(row["model"])

    # -- lookups ----------------------------------------------------------------------------
    def resolve_ticker(self, name: str) -> str | None:
        wanted = name.strip().upper().replace(" ", "")
        for t in self.tickers:
            if t.upper() == wanted:
                return t
        return None
