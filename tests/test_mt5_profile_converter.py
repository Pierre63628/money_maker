from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mt5_conv", ROOT / "scripts" / "mt5_profile_from_specs.py")
mt5_conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt5_conv)


def test_swap_conversion_handles_points_and_annual_percent():
    # points mode: -200 points of 0.01 on a 70,000 price -> 2 USD per night -> 0.286 bps/day
    assert mt5_conv.swap_bps_per_day(-200, 1, 0.01, 70_000) == pytest.approx(0.2857, abs=1e-3)
    # annual percent mode: -10.5% a year -> 2.877 bps/day, independent of the price
    assert mt5_conv.swap_bps_per_day(-10.5, 5, 0.01, 3_500) == pytest.approx(2.877, abs=1e-3)
    # a positive swap is a credit, not a cost
    assert mt5_conv.swap_bps_per_day(5.0, 5, 0.01, 100) < 0
    assert mt5_conv.swap_bps_per_day(0.0, 0, 0.01, 100) == 0.0


def test_base_currency_uses_the_field_then_falls_back_to_the_name():
    assert mt5_conv.base_currency(pd.Series({"symbol": "BTCUSD.a", "currency_base": "XBT"})) == "BTC"
    assert mt5_conv.base_currency(pd.Series({"symbol": "ETHUSDT_ecn", "currency_base": ""})) == "ETH"
    assert mt5_conv.base_currency(pd.Series({"symbol": "SOLUSD", "currency_base": None})) == "SOL"
