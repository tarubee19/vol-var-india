import numpy as np
import pandas as pd
import pytest

from volvar.data import (
    DataError,
    garman_klass_var_pct2,
    load_ticker,
    load_universe,
    log_returns_pct,
    make_panel,
    validate_ohlcv,
)
from volvar.synthetic import simulate_universe, write_universe


def _ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx
    )


def test_log_returns_are_percent_and_drop_first():
    close = pd.Series([100.0, 101.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))
    r = log_returns_pct(close)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(100 * np.log(1.01))


def test_garman_klass_is_positive_and_scaled():
    gk = garman_klass_var_pct2(_ohlcv())
    assert (gk > 0).all()
    # ~1% daily vol -> variance around 1 %^2 order of magnitude
    assert 0.0001 < gk.mean() < 5


def test_validate_rejects_missing_columns():
    df = _ohlcv().drop(columns=["volume"])
    with pytest.raises(DataError, match="missing columns"):
        validate_ohlcv(df)


def test_validate_rejects_too_short():
    with pytest.raises(DataError, match="at least 300"):
        validate_ohlcv(_ohlcv(n=100))


def test_validate_drops_few_bad_rows_but_rejects_many():
    df = _ohlcv()
    df.iloc[5, df.columns.get_loc("low")] = -1.0  # one bad row is dropped
    out = validate_ohlcv(df)
    assert len(out) == len(df) - 1
    bad = _ohlcv()
    bad.iloc[:50, bad.columns.get_loc("close")] = 0.0
    with pytest.raises(DataError, match="too many invalid rows"):
        validate_ohlcv(bad)


def test_csv_roundtrip(tmp_path):
    uni = simulate_universe(n_days=500, n_tickers=2, seed=1)
    write_universe(uni, tmp_path)
    loaded = load_universe(tmp_path)
    assert set(loaded) == {"SIM01", "SIM02"}
    one = load_ticker(tmp_path / "SIM01.csv")
    assert list(one.columns) == ["open", "high", "low", "close", "volume"]
    assert one.index.is_monotonic_increasing
    np.testing.assert_allclose(one["close"].to_numpy(), uni["SIM01"]["close"].to_numpy(), rtol=1e-9)


def test_load_universe_errors(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_universe(tmp_path / "nope")
    with pytest.raises(DataError, match="no CSV files"):
        load_universe(tmp_path)


def test_make_panel_columns(small_universe):
    panel = make_panel(small_universe["SIM01"])
    assert {"ret", "r2", "gk", "volume"} <= set(panel.columns)
    np.testing.assert_allclose(panel["r2"], panel["ret"] ** 2)
    assert not panel[["ret", "gk"]].isna().any().any()


def _fetch_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.py"
    spec = importlib.util.spec_from_file_location("fetch_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_partial_intraday_bar_is_dropped():
    import pandas as pd

    fetch = _fetch_module()
    idx = pd.to_datetime(["2026-10-15", "2026-10-16", "2026-10-19"])
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    during = pd.Timestamp("2026-10-19 13:18", tz="Asia/Kolkata")
    after = pd.Timestamp("2026-10-19 18:15", tz="Asia/Kolkata")
    assert fetch.drop_incomplete_session(df, now=during).index[-1] == pd.Timestamp("2026-10-16")
    assert len(fetch.drop_incomplete_session(df, now=after)) == 3
    next_day = pd.Timestamp("2026-10-20 09:00", tz="Asia/Kolkata")
    assert len(fetch.drop_incomplete_session(df, now=next_day)) == 3
