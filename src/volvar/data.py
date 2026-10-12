"""Loading, validating and transforming daily OHLCV data.

Conventions used everywhere in the project
------------------------------------------
* Returns are **log returns in percent** (``100 * ln(C_t / C_{t-1})``).
* Variances are in **percent squared**.
* A forecast series is indexed by the date of the return it forecasts and
  uses only information available at the close of the previous trading day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class DataError(ValueError):
    """Raised when a price file is missing, malformed or fails validation."""


def validate_ohlcv(df: pd.DataFrame, name: str = "series") -> pd.DataFrame:
    """Check and lightly clean an OHLCV frame (lower-case columns, DatetimeIndex).

    Rows with non-positive prices or an impossible high/low ordering are dropped
    and counted; a series that loses more than 2% of its rows is rejected.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{name}: missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError(f"{name}: index must be a DatetimeIndex")

    out = df[list(REQUIRED_COLUMNS)].astype(float).copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    n0 = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])

    prices_ok = (out[["open", "high", "low", "close"]] > 0).all(axis=1)
    range_ok = (out["high"] >= out["low"]) & (out["high"] >= out[["open", "close"]].max(axis=1) * 0.999) & (
        out["low"] <= out[["open", "close"]].min(axis=1) * 1.001
    )
    out = out[prices_ok & range_ok]
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)

    if n0 == 0 or len(out) < 0.98 * n0:
        raise DataError(f"{name}: too many invalid rows ({n0 - len(out)} of {n0})")
    if len(out) < 300:
        raise DataError(f"{name}: only {len(out)} valid rows; need at least 300")
    return out


def load_ticker(path: str | Path) -> pd.DataFrame:
    """Read one CSV (``Date,Open,High,Low,Close,Volume``) and validate it."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col]).set_index(date_col)
    df.index = df.index.normalize()
    df.index.name = "date"
    return validate_ohlcv(df, name=path.stem)


def load_universe(data_dir: str | Path, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every ``*.csv`` in ``data_dir`` (or just the requested tickers)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataError(f"data directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.csv"))
    if tickers:
        wanted = {t.upper() for t in tickers}
        files = [f for f in files if f.stem.upper() in wanted]
    if not files:
        raise DataError(f"no CSV files found in {data_dir}")
    return {f.stem.upper(): load_ticker(f) for f in files}


def log_returns_pct(close: pd.Series) -> pd.Series:
    """Daily log returns in percent; the first observation is dropped."""
    return (100.0 * np.log(close).diff()).dropna().rename("ret")


def garman_klass_var_pct2(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass range-based daily variance estimate (percent squared).

    ``0.5*ln(H/L)^2 - (2 ln2 - 1) * ln(C/O)^2``. It ignores the overnight gap,
    so it is used only as a predictor, never as the evaluation target.
    """
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return (1e4 * gk).clip(lower=1e-6).rename("gk")


def make_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker working frame: ``ret``, ``r2`` (squared return), ``gk``, ``volume``."""
    ret = log_returns_pct(df["close"])
    panel = pd.DataFrame({"ret": ret})
    panel["r2"] = panel["ret"] ** 2
    panel["gk"] = garman_klass_var_pct2(df).reindex(panel.index)
    panel["volume"] = df["volume"].reindex(panel.index)
    return panel.dropna(subset=["ret", "gk"])


def make_panels(universe: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {t: make_panel(df) for t, df in universe.items()}
