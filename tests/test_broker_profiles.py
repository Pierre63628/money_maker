from __future__ import annotations

import json

import pytest

from quant.brokers.mt5 import MT5Broker, MT5NotConfigured
from quant.brokers.profiles import TYPICAL_CRYPTO_CFD, BrokerProfile


def test_profile_from_json_builds_costs(tmp_path):
    spec = {
        "name": "demo",
        "commission_bps_per_side": 1.5,
        "slippage_bps": 3.0,
        "max_leverage": 2.0,
        "default": {"half_spread_bps": 12.0, "swap_bps_per_day": 3.0},
        "instruments": {
            "BTCUSD": {"source": "BTCUSDT", "half_spread_bps": 3.0, "swap_long_bps_per_day": 1.0,
                       "swap_short_bps_per_day": 5.0, "min_notional": 250.0},
            "ETHUSD": {"source": "ETHUSDT"},
        },
    }
    path = tmp_path / "broker.json"
    path.write_text(json.dumps(spec))
    profile = BrokerProfile.from_json(path)
    assert profile.source_symbols == ["BTCUSDT", "ETHUSDT"]
    assert profile.instruments["ETHUSD"].half_spread_bps == 12.0      # falls back to the default
    assert profile.instruments["BTCUSD"].financing_bps_per_day == 3.0  # average of both sides
    cfg = profile.backtest_config()
    assert cfg.costs.half_spread_for("BTCUSDT") == 3.0
    assert cfg.costs.financing_bps_for("ETHUSDT") == 3.0
    assert cfg.costs.fee_bps == 1.5 and cfg.costs.funding is False
    assert cfg.execution.mode == "taker" and cfg.max_gross_leverage == 2.0
    assert cfg.min_trade_notional == 250.0


def test_placeholder_profile_and_disabled_adapter():
    assert len(TYPICAL_CRYPTO_CFD.instruments) == 10
    assert TYPICAL_CRYPTO_CFD.backtest_config().costs.financing_bps_for("BTCUSDT") == 2.0
    with pytest.raises(MT5NotConfigured):
        MT5Broker()
