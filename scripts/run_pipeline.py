"""Run the full study: diagnostics, walk-forward forecasts, evaluation, VaR backtests, figures."""

from __future__ import annotations

import argparse
import json

from volvar.config import ExperimentConfig
from volvar.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="folder of <TICKER>.csv files")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tickers", nargs="*", default=None, help="subset of tickers (default: all)")
    parser.add_argument("--min-train", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=63)
    parser.add_argument("--label", choices=["real", "simulated"], default=None,
                        help="dataset label (auto-detected by default)")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig(min_train=args.min_train, refit_every=args.refit_every)
    summary = run_pipeline(
        args.data_dir,
        args.results_dir,
        cfg=cfg,
        tickers=args.tickers,
        label=args.label,
        n_jobs=args.n_jobs,
        figures=not args.no_figures,
    )
    print(json.dumps({k: summary[k] for k in ("dataset", "n_tickers", "n_oos_observations",
                                              "best_model_by_qlike")}, indent=2, default=str))


if __name__ == "__main__":
    main()
