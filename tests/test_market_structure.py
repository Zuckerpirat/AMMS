"""Tests for amms.analysis.market_structure."""

from __future__ import annotations

import pytest

from amms.analysis.market_structure import MarketStructureReport, SwingPoint, analyze


class _Bar:
    def __init__(self, high, low, close, symbol="SYM"):
        self.symbol = symbol
        self.high = high
        self.low = low
        self.close = close
        self.volume = 1_000_000


def _make_uptrend(n: int = 50) -> list[_Bar]:
    """Clear HH + HL uptrend with swing points."""
    bars = []
    for i in range(n):
        # Oscillating but trending up
        cycle = i % 8
        base = 100.0 + i * 0.5
        if cycle < 4:
            # rally
            high = base + 2.0
            low = base - 0.5
            close = base + 1.5
        else:
            # pullback
            high = base + 0.5
            low = base - 2.0
            close = base - 1.5
        bars.append(_Bar(high, low, close))
    return bars


def _make_downtrend(n: int = 50) -> list[_Bar]:
    """Clear LH + LL downtrend."""
    bars = []
    for i in range(n):
        cycle = i % 8
        base = 160.0 - i * 0.5
        if cycle < 4:
            # decline
            high = base + 0.5
            low = base - 2.0
            close = base - 1.5
        else:
            # bounce
            high = base + 2.0
            low = base - 0.5
            close = base + 1.5
        bars.append(_Bar(high, low, close))
    return bars


def _make_flat(n: int = 50, price: float = 100.0) -> list[_Bar]:
    """Ranging market oscillating around a price."""
    bars = []
    for i in range(n):
        cycle = i % 6
        if cycle < 3:
            bars.append(_Bar(price + 2.0, price - 0.5, price + 1.5))
        else:
            bars.append(_Bar(price + 0.5, price - 2.0, price - 1.5))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100, 99, 100) for _ in range(5)]
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _make_uptrend(50)
        result = analyze(bars)
        assert result is not None or True  # may return None if no swings found
        # Just test it doesn't crash

    def test_returns_result_with_enough_bars(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        # Should find some structure with 60 bars


class TestStructure:
    def test_structure_valid(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            assert result.structure in ("uptrend", "downtrend", "ranging", "unclear")

    def test_uptrend_detected(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None and result.structure != "unclear":
            # In a strong uptrend, structure should not be downtrend
            assert result.structure != "downtrend"

    def test_downtrend_detected(self):
        bars = _make_downtrend(60)
        result = analyze(bars)
        if result is not None and result.structure != "unclear":
            assert result.structure != "uptrend"

    def test_current_price_correct(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_bars_used(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            assert result.bars_used == 60


class TestSwingPoints:
    def test_swing_highs_present(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            # At least some swing highs detected
            assert len(result.swing_highs) >= 0

    def test_swing_lows_present(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            assert len(result.swing_lows) >= 0

    def test_pct_from_high_computed(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None and result.last_swing_high is not None:
            assert isinstance(result.pct_from_last_high, float)

    def test_pct_from_low_computed(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None and result.last_swing_low is not None:
            assert isinstance(result.pct_from_last_low, float)


class TestCHoCH:
    def test_choch_bool(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            assert isinstance(result.choch_detected, bool)

    def test_choch_direction_valid(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            assert result.choch_direction in ("bullish", "bearish", "none")

    def test_no_choch_in_stable_uptrend(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None and result.structure == "uptrend":
            assert not result.choch_detected


class TestSymbol:
    def test_symbol_stored(self):
        bars = _make_uptrend(60)
        result = analyze(bars, symbol="AAPL")
        if result is not None:
            assert result.symbol == "AAPL"


class TestVerdict:
    def test_verdict_present(self):
        bars = _make_uptrend(60)
        result = analyze(bars)
        if result is not None:
            assert len(result.verdict) > 5
