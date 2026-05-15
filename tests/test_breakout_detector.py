"""Tests for amms.analysis.breakout_detector."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.breakout_detector import BreakoutSignal, detect


def _bar(sym: str, open_: float, high: float, low: float, close: float,
         volume: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               open_, high, low, close, volume)


def _flat_bars(sym: str, n: int, price: float = 100.0,
               volume: float = 100_000) -> list[Bar]:
    """Bars with small range and consistent volume."""
    bars = []
    for i in range(n):
        bars.append(_bar(sym, price, price * 1.005, price * 0.995, price, volume, i))
    return bars


def _squeeze_bars(sym: str, n: int) -> list[Bar]:
    """Bars with shrinking range (squeeze)."""
    bars = []
    price = 100.0
    for i in range(n):
        rng = max(0.1, 2.0 - i * 0.05)  # shrinking range
        bars.append(_bar(sym, price, price + rng, price - rng, price, 100_000, i))
    return bars


def _breakout_up_bars(sym: str, n: int) -> list[Bar]:
    """Normal bars followed by a big up move with high volume."""
    bars = _flat_bars(sym, n - 1, price=100.0, volume=100_000)
    # Last bar: breaks above recent high with high volume
    last = _bar(sym, 100.0, 105.0, 99.5, 105.0, 400_000, n)
    return bars + [last]


def _breakout_down_bars(sym: str, n: int) -> list[Bar]:
    """Normal bars followed by a big down move with high volume."""
    bars = _flat_bars(sym, n - 1, price=100.0, volume=100_000)
    last = _bar(sym, 100.0, 100.5, 94.5, 94.5, 400_000, n)
    return bars + [last]


class TestDetect:
    def test_returns_none_insufficient(self):
        bars = _flat_bars("AAPL", 10)
        assert detect(bars, lookback=20) is None

    def test_returns_result(self):
        bars = _flat_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        assert isinstance(result, BreakoutSignal)

    def test_symbol_preserved(self):
        bars = _flat_bars("TSLA", 30)
        result = detect(bars)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_no_signal_for_flat_bars(self):
        bars = _flat_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        assert result.signal in ("none", "squeeze")

    def test_squeeze_detected(self):
        bars = _squeeze_bars("AAPL", 35)
        result = detect(bars)
        assert result is not None
        assert result.is_squeeze is True

    def test_breakout_up_detected(self):
        bars = _breakout_up_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        assert result.signal == "breakout_up"

    def test_breakout_down_detected(self):
        bars = _breakout_down_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        assert result.signal == "breakout_down"

    def test_confidence_range(self):
        bars = _breakout_up_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        assert 0 <= result.confidence <= 100

    def test_breakout_level_set_on_breakout(self):
        bars = _breakout_up_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        if result.signal == "breakout_up":
            assert result.breakout_level is not None

    def test_breakout_level_none_on_squeeze(self):
        bars = _squeeze_bars("AAPL", 35)
        result = detect(bars)
        assert result is not None
        if result.signal == "squeeze":
            assert result.breakout_level is None

    def test_volume_ratio_above_1_on_breakout(self):
        bars = _breakout_up_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        if result.signal == "breakout_up":
            assert result.volume_ratio > 1.0

    def test_range_compression_on_squeeze(self):
        bars = _squeeze_bars("AAPL", 35)
        result = detect(bars)
        assert result is not None
        if result.is_squeeze:
            assert result.range_compression < 1.0

    def test_current_price_last_close(self):
        bars = _flat_bars("AAPL", 30, price=123.0)
        result = detect(bars)
        assert result is not None
        assert result.current_price == pytest.approx(123.0, abs=1.0)

    def test_confidence_zero_for_no_signal(self):
        bars = _flat_bars("AAPL", 30)
        result = detect(bars)
        assert result is not None
        if result.signal == "none":
            assert result.confidence == 0.0

    def test_bars_used_correct(self):
        bars = _flat_bars("AAPL", 35)
        result = detect(bars, lookback=20)
        assert result is not None
        assert result.bars_used == 35
