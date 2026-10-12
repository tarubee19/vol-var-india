"""Generate simulated OHLCV CSVs (tickers SIM01...) for development and CI."""

from __future__ import annotations

import argparse

from volvar.synthetic import simulate_universe, write_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--days", type=int, default=2600)
    parser.add_argument("--tickers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    universe = simulate_universe(n_days=args.days, n_tickers=args.tickers, seed=args.seed)
    names = write_universe(universe, args.out)
    print(f"Wrote {len(names)} simulated series ({args.days} days each) to {args.out}")


if __name__ == "__main__":
    main()
