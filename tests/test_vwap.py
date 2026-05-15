from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.vwap import vwap, vwap_deviation_pct
from amms.strategy.vwap_strategy import VwapStrategy


def _bar(i: int, close: float, vol: float = 1000.0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, close + 1, close - 1, close, vol)


def _bars(closes: list[float], vol: float = 1000.0) -> list[Bar]:
    return [_bar(i, c, vol) for i, c in enumerate(closes)]


# ---------------------------------------------------------------------------
# vwap() tests
# ---------------------------------------------------------------------------

def test_vwap_equal_prices_and_volumes() -> None:
    bars = _bars([100.0] * 5)
    v = vwap(bars)
    # typical = (100+1+99)/3 = 100; all equal → VWAP = 100
    assert v == pytest.approx(100.0, abs=0.1)


def test_vwap_none_on_empty() -> None:
    assert vwap([]) is None


def test_vwap_none_on_zero_volume() -> None:
    bars = _bars([100.0] * 5, vol=0.0)
    assert vwap(bars) is None


def test_vwap_respects_window() -> None:
    bars = _bars([100.0] * 10 + [200.0] * 10)
    v_all = vwap(bars)
    v_last10 = vwap(bars, n=10)
    assert v_last10 is not None
    assert v_last10 > v_all  # last 10 bars are at 200


def test_vwap_deviation_zero_at_vwap() -> None:
    bars = _bars([100.0] * 20)
    # typical price = (100+1+99)/3 ≈ 100
    # price also ≈ 100 → deviation ≈ 0
    dev = vwap_deviation_pct(100.0, bars, n=20)
    assert dev is not None
    assert abs(dev) < 0.5  # should be near 0


def test_vwap_deviation_positive_when_above() -> None:
    bars = _bars([100.0] * 20)
    dev = vwap_deviation_pct(110.0, bars, n=20)
    assert dev is not None
    assert dev > 0


def test_vwap_deviation_negative_when_below() -> None:
    bars = _bars([100.0] * 20)
    dev = vwap_deviation_pct(90.0, bars, n=20)
    assert dev is not None
    assert dev < 0


# ---------------------------------------------------------------------------
# VwapStrategy tests
# ---------------------------------------------------------------------------

def test_default_params() -> None:
    s = VwapStrategy()
    assert s.window == 20
    assert s.buy_deviation == -1.5
    assert s.sell_deviation == 1.5


def test_lookback() -> None:
    s = VwapStrategy(window=15)
    assert s.lookback == 15


def test_hold_on_insufficient_history() -> None:
    s = VwapStrategy()
    sig = s.evaluate("X", _bars([100.0] * 5))
    assert sig.kind == "hold"


def test_buy_when_price_below_vwap_threshold() -> None:
    s = VwapStrategy(window=20, buy_deviation=-1.0)
    # Set price well below VWAP
    bars = _bars([100.0] * 20)
    # Monkey-patch last close to be well below VWAP
    low_bars = bars[:-1] + [Bar("X", "1D", "2026-01-21", 97.0, 98.0, 96.0, 97.0, 1000.0)]
    sig = s.evaluate("X", low_bars)
    assert sig.kind == "buy"
    assert sig.score > 0


def test_sell_when_price_above_vwap_threshold() -> None:
    s = VwapStrategy(window=20, sell_deviation=1.0)
    bars = _bars([100.0] * 20)
    high_bars = bars[:-1] + [Bar("X", "1D", "2026-01-21", 103.0, 104.0, 102.0, 103.0, 1000.0)]
    sig = s.evaluate("X", high_bars)
    assert sig.kind == "sell"


def test_validation_bad_buy_deviation() -> None:
    with pytest.raises(ValueError, match="buy_deviation"):
        VwapStrategy(buy_deviation=1.0)


def test_validation_bad_sell_deviation() -> None:
    with pytest.raises(ValueError, match="sell_deviation"):
        VwapStrategy(sell_deviation=-1.0)


def test_registered_in_strategy_module() -> None:
    from amms.strategy import build_strategy
    s = build_strategy("vwap", {})
    assert s.name == "vwap"
