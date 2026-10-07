from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from quant.data.binance_vision import (ChecksumError, merge_frames, parse_funding_zip, parse_klines_zip,
                                       verify_checksum)
from quant.data.market_data import MarketData
from quant.data.store import ParquetStore
from quant.data.validation import reindex_to_grid, validate_klines
from tests.conftest import frame_from_arrays

HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"


def kline_rows(start_ms: int, n: int, scale: int = 1) -> str:
    lines = []
    for i in range(n):
        t = (start_ms + i * 3_600_000) * scale
        lines.append(f"{t},100.0,101.0,99.0,100.5,10.0,{t + 3_599_999 * scale},1005.0,42,5.0,502.5,0")
    return "\n".join(lines) + "\n"


def zipped(text: str, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


@pytest.mark.parametrize("header,scale", [(True, 1), (False, 1), (True, 1000)])
def test_parse_klines_handles_header_and_microseconds(header, scale):
    text = (HEADER if header else "") + kline_rows(1_704_067_200_000, 3, scale)
    df = parse_klines_zip(zipped(text))
    assert list(df.index) == list(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    assert df["close"].iloc[0] == 100.5 and df["trades"].iloc[0] == 42
    assert df.index.name == "ts"


def test_parse_funding():
    text = "calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.00037409\n1704096000000,8,-0.0001\n"
    df = parse_funding_zip(zipped(text))
    assert df["funding_rate"].tolist() == [0.00037409, -0.0001]


def test_checksum_verification():
    data = b"payload"
    good = hashlib.sha256(data).hexdigest() + "  file.zip"
    verify_checksum(data, good)
    with pytest.raises(ChecksumError):
        verify_checksum(b"tampered", good)


def test_merge_drops_exact_duplicates_but_keeps_conflicts():
    a = parse_klines_zip(zipped(kline_rows(1_704_067_200_000, 3)))
    b = a.copy()
    c = a.iloc[[1]].copy()
    c["close"] = 999.0
    merged = merge_frames([a, b, c])
    assert len(merged) == 4
    report = validate_klines(merged, "1h", "X")
    assert report.conflicting_duplicates == 1
    assert not report.is_usable


def test_validation_flags_gaps_and_ohlc_violations():
    df = frame_from_arrays(100 + np.arange(100.0))
    df = df.drop(df.index[[10, 11, 12]])
    df.loc[df.index[50], "high"] = df.loc[df.index[50], "low"] - 1
    report = validate_klines(df, "1h", "X")
    assert report.missing_bars == 3
    assert report.largest_gap == pd.Timedelta(hours=3)
    assert report.ohlc_violations == 1
    assert not report.is_usable
    gridded = reindex_to_grid(df, "1h")
    assert len(gridded) == 100 and gridded["close"].isna().sum() == 3


def test_validation_passes_clean_series():
    report = validate_klines(frame_from_arrays(100 + np.arange(500.0)), "1h", "X")
    assert report.is_usable, report.summary()


def test_validation_rejects_off_grid_timestamps():
    df = frame_from_arrays(100 + np.arange(10.0))
    df.index = df.index + pd.Timedelta(minutes=1)
    assert validate_klines(df, "1h").misaligned_timestamps == 10


def test_market_data_alignment_and_funding_mapping():
    a = frame_from_arrays(100 + np.arange(10.0), start="2024-01-01 00:00")
    b = frame_from_arrays(50 + np.arange(5.0), start="2024-01-01 03:00")
    funding = pd.DataFrame({"funding_rate": [0.001, 0.002, 0.003]},
                           index=pd.DatetimeIndex(["2024-01-01 00:00:00.004", "2024-01-01 08:00", "2024-01-01 08:30"], tz="UTC"))
    data = MarketData.from_frames({"A": a, "B": b}, "1h", {"A": funding})
    assert data.symbols == ["A", "B"] and len(data) == 10
    assert data.close["B"].iloc[:3].isna().all() and data.close["B"].iloc[8:].isna().all()
    assert data.funding.loc[data.index[0], "A"] == pytest.approx(0.001)
    assert data.funding.loc[data.index[8], "A"] == pytest.approx(0.005)  # two settlements in one bar
    assert data.head(4).close.shape == (4, 2)
    assert len(data.slice("2024-01-01 02:00", "2024-01-01 05:00")) == 3


def test_parquet_store_roundtrip(tmp_path):
    store = ParquetStore(tmp_path)
    df = frame_from_arrays(100 + np.arange(24.0))
    store.write_klines(df, "um", "BTCUSDT", "1h")
    back = store.read_klines("um", "BTCUSDT", "1h", start="2024-01-01 05:00")
    assert len(back) == 19 and back.index[0] == pd.Timestamp("2024-01-01 05:00", tz="UTC")
    assert store.symbols("um", "1h") == ["BTCUSDT"]
    with pytest.raises(ValueError):
        store.write_klines(df.iloc[::-1], "um", "BAD", "1h")


def test_mask_stale_runs_removes_frozen_post_delisting_bars():
    from quant.data.validation import mask_stale_runs
    df = frame_from_arrays(100 + np.arange(200.0))
    frozen = df.index[150:]
    df.loc[frozen, ["open", "high", "low", "close"]] = 249.5
    df.loc[frozen, ["volume", "quote_volume"]] = 0.0
    df.loc[df.index[50:52], ["open", "high", "low", "close"]] = 150.0  # 2-bar maintenance: kept
    df.loc[df.index[50:52], ["volume", "quote_volume"]] = 0.0
    df.loc[df.index[80:120], ["open", "high", "low", "close"]] = 180.0  # 40-bar halt: masked
    df.loc[df.index[80:120], ["volume", "quote_volume"]] = 0.0
    cleaned, n = mask_stale_runs(df, min_run=24)
    assert len(cleaned) == 150 and n == 90
    assert cleaned["close"].iloc[80:120].isna().all()
    assert cleaned["close"].iloc[50:52].notna().all()


def test_tradable_excludes_series_after_data_ends():
    a = frame_from_arrays(100 + np.arange(100.0))
    b = frame_from_arrays(50 + np.arange(40.0))
    data = MarketData.from_frames({"A": a, "B": b}, "1h")
    tradable = data.tradable["B"]
    assert tradable.iloc[39] and tradable.iloc[39 + 24]
    assert not tradable.iloc[39 + 25:].any()


def test_metrics_cleaning_and_validation():
    from quant.data.market_data import clean_metrics, resample_metrics
    from quant.data.validation import validate_metrics
    idx = pd.date_range("2024-01-01", periods=36, freq="5min", tz="UTC", name="ts")
    df = pd.DataFrame({"open_interest": 100.0, "open_interest_value": 1e6, "global_account_ls": 1.5,
                       "taker_buy_sell_ratio": 1.0}, index=idx)
    df.loc[idx[3], "open_interest"] = 0.0          # corrupt snapshot
    df.loc[idx[4], "global_account_ls"] = -1.0
    report = validate_metrics(df, "X")
    assert report.coverage == pytest.approx(1.0)
    assert report.non_positive["open_interest"] == 1 and report.non_positive["global_account_ls"] == 1
    cleaned = clean_metrics(df)
    assert cleaned["open_interest"].isna().sum() == 1
    assert np.isfinite(cleaned["open_interest"].pct_change(fill_method=None).dropna()).all()
    hourly = resample_metrics(cleaned, "1h")
    assert len(hourly) == 3 and hourly["open_interest"].notna().all()
