"""The live feed must hand the strategy the same positioning snapshot the backtest used.

Verified against the exchange on 2026-10-06: the REST endpoints stamp a value with the END of its
period, the Binance Vision archive stamps the same value with the START of its 5-minute period.
Binning REST stamps like archive stamps fed the live strategy data one bar older than research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.data.market_data import (clean_metrics, metrics_available_at, resample_metrics,
                                    snapshots_to_bars)
from quant.data.store import ParquetStore
from quant.live.feed import BinancePublicFeed, _read_cache, _write_cache

FIVE = pd.Timedelta(minutes=5)


def archive(days: int = 12, seed: int = 0) -> pd.DataFrame:
    """Five-minute rows in the archive convention (stamp = start of the period)."""
    idx = pd.date_range("2026-09-01", periods=days * 288, freq="5min", tz="UTC", name="ts").as_unit("ns")
    rng = np.random.default_rng(seed)
    ratio = 1.5 + np.cumsum(rng.normal(0, 0.004, len(idx)))
    return pd.DataFrame({"global_account_ls": np.clip(ratio, 0.2, None),
                         "open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))),
                         "taker_buy_sell_ratio": 1.0 + rng.normal(0, 0.05, len(idx))}, index=idx)


def test_snapshot_binning_equals_the_research_resampling():
    rows = archive()
    research = resample_metrics(rows, "4h")
    via_snapshots = snapshots_to_bars(metrics_available_at(rows), "4h")
    common = research.index.intersection(via_snapshots.index)[1:-1]
    pd.testing.assert_frame_equal(research.loc[common], via_snapshots.loc[common][research.columns])


def test_hourly_storage_keeps_the_same_bar_values():
    """The store keeps metrics resampled to 1h; going through it must not change a 4h bar."""
    rows = archive()
    hourly = resample_metrics(rows, "1h")
    direct = resample_metrics(rows, "4h")["global_account_ls"]
    through_store = snapshots_to_bars(metrics_available_at(hourly), "4h")["global_account_ls"]
    common = direct.index.intersection(through_store.index)[1:-1]
    pd.testing.assert_series_equal(direct.loc[common], through_store.loc[common], check_freq=False)


def test_a_rest_point_belongs_to_the_bar_that_ends_at_its_stamp():
    """REST stamp 08:00 is the snapshot at the close of the 04:00 bar, not the first of the 08:00 bar."""
    stamps = pd.DatetimeIndex(["2026-10-06 04:00", "2026-10-06 08:00"], tz="UTC").as_unit("ns")
    rest = pd.DataFrame({"global_account_ls": [1.72, 1.75]}, index=stamps)
    bars = snapshots_to_bars(rest, "4h")["global_account_ls"]
    assert bars.loc[pd.Timestamp("2026-10-06 04:00", tz="UTC")] == 1.75
    assert bars.loc[pd.Timestamp("2026-10-06 00:00", tz="UTC")] == 1.72


class FakeRest:
    """Serves the exchange's view of the same data: stamp = end of period (archive stamp + 5 min)."""

    def __init__(self, rows: pd.DataFrame):
        self.snapshots = metrics_available_at(rows, FIVE)
        self.calls: list[tuple] = []

    def metrics(self, symbol, start, end, period="5m", fields=None):
        self.calls.append((symbol, period, pd.Timestamp(start), pd.Timestamp(end)))
        window = self.snapshots[(self.snapshots.index >= start) & (self.snapshots.index <= end)]
        step = pd.Timedelta(period)
        out = window[(window.index - pd.Timestamp(0, tz="UTC")) % step == pd.Timedelta(0)]
        return out[[c for c in out.columns if fields is None or c in fields]]


@pytest.mark.parametrize("store_days,period_expected", [(11.5, {"5m"}), (8, {"1h", "5m"})])
def test_live_feed_reproduces_the_research_frame(tmp_path, store_days, period_expected):
    rows = archive(days=12)
    now = rows.index[-1] + FIVE + pd.Timedelta(seconds=70)        # one minute after the last bar closes
    store = ParquetStore(tmp_path / "data")
    stored = rows[rows.index < rows.index[0] + pd.Timedelta(days=store_days)]
    store.write_metrics(resample_metrics(clean_metrics(stored), "1h"), "um", "AAAUSDT")
    rest = FakeRest(rows)
    feed = BinancePublicFeed(store, tmp_path / "cache", rest=rest)

    live = feed.metrics_history(["AAAUSDT"], rows.index[0], now, "4h")["global_account_ls"]["AAAUSDT"]
    research = resample_metrics(clean_metrics(rows), "4h")["global_account_ls"]

    decision_bar = now.floor("4h") - pd.Timedelta(hours=4)
    assert live.loc[decision_bar] == pytest.approx(research.loc[decision_bar], rel=1e-6)   # the bar being traded
    common = research.index.intersection(live.index)[1:]
    recent = common[common >= now - pd.Timedelta(hours=36)]
    np.testing.assert_allclose(live.loc[recent], research.loc[recent], rtol=1e-6)
    older = common[common < now - pd.Timedelta(hours=44)]
    # the 1h bridge only matters for bars inside the gap, and there it lands on the same top-of-hour snapshot
    np.testing.assert_allclose(live.loc[older], research.loc[older], rtol=1e-6)
    assert {c[1] for c in rest.calls} == period_expected

    # steady state: the second step asks for the few minutes it is missing, at 5 minutes, once
    rest.calls.clear()
    again = feed.metrics_history(["AAAUSDT"], rows.index[0], now + pd.Timedelta(minutes=1), "4h")
    assert [c[1] for c in rest.calls] in ([], ["5m"])
    assert again["global_account_ls"]["AAAUSDT"].loc[decision_bar] == pytest.approx(research.loc[decision_bar], rel=1e-6)


def test_cache_writes_are_atomic_and_a_corrupt_cache_is_rebuilt(tmp_path):
    path = tmp_path / "c" / "x.parquet"
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"))
    _write_cache(frame, path)
    assert not list(path.parent.glob("*.tmp")) and len(_read_cache(path)) == 2
    path.write_bytes(b"not a parquet file")                  # what a half-written file looks like
    assert _read_cache(path).empty and not path.exists()
