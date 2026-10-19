"""Download daily OHLCV for Nifty 50 constituents and the Nifty 50 index.

Run this on a machine with normal internet access:

    pip install yfinance pandas
    python scripts/fetch_data.py

It writes one CSV per symbol into ``data/raw/`` with the columns
``Date,Open,High,Low,Close,Volume`` (split/dividend adjusted via yfinance
``auto_adjust=True``). Then zip the folder and use it with the pipeline:

    python scripts/run_pipeline.py --data-dir data/raw

Data comes from Yahoo Finance through the ``yfinance`` library. It is meant
for research and education; check Yahoo's terms before any other use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Large, long-listed Nifty 50 names with stable tickers. The index is fetched
# as ^NSEI. Tickers that changed recently (mergers/demergers) are avoided.
DEFAULT_SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "ASIANPAINT.NS",
    "^NSEI",
]


def clean_name(symbol: str) -> str:
    """File-system friendly name: ``RELIANCE.NS`` -> ``RELIANCE``, ``^NSEI`` -> ``NIFTY50``."""
    if symbol == "^NSEI":
        return "NIFTY50"
    return symbol.replace(".NS", "").replace("^", "")


def fetch_one(symbol: str, start: str, end: str | None, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
            if df is not None and len(df) > 0:
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df.index.name = "Date"
                df = df[~df.index.duplicated(keep="last")].sort_index()
                return df.dropna(subset=["Open", "High", "Low", "Close"])
        except Exception as exc:  # network / parsing problems
            last_err = exc
        time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {symbol}: {last_err}")


def drop_incomplete_session(
    df: pd.DataFrame, now: pd.Timestamp | None = None, settle: str = "16:00"
) -> pd.DataFrame:
    """Drop today's bar if the download runs before the NSE session has closed and settled.

    Yahoo serves a live, partial bar for the current day during market hours (NSE closes at
    15:30 IST). Using it would mix an intraday price into a daily study, so it is removed unless
    the run happens after ``settle`` India time.
    """
    now = now if now is not None else pd.Timestamp.now(tz="Asia/Kolkata")
    if now.tzinfo is not None:
        now = now.tz_convert("Asia/Kolkata").tz_localize(None)
    if len(df) and df.index[-1] == now.normalize() and now.time() < pd.Timestamp(settle).time():
        return df.iloc[:-1]
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date (default: today)")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--keep-incomplete", action="store_true",
                        help="keep today's bar even if the market has not closed yet")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for sym in args.symbols:
        name = clean_name(sym)
        try:
            df = fetch_one(sym, args.start, args.end)
            if not args.keep_incomplete:
                df = drop_incomplete_session(df)
        except RuntimeError as exc:
            print(f"[FAIL] {sym}: {exc}")
            failed.append(sym)
            continue
        path = out / f"{name}.csv"
        df.to_csv(path)
        print(f"[ OK ] {sym:<15} {len(df):>5} rows  {df.index.min().date()} -> {df.index.max().date()}  -> {path}")

    if failed:
        print(f"\n{len(failed)} symbol(s) failed: {', '.join(failed)}")
        return 1
    print(f"\nDone. Zip the folder '{out}' and share it, or run the pipeline on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
