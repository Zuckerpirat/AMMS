"""Tests for amms.analysis.swing_points."""

from __future__ import annotations

import pytest

from amms.analysis.swing_points import SwingPoint, SwingReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float = None):
        self.high = high
        self.low = low
        self.close = close if close is not None else (high + low) / 2


def _flat(n: int = 30, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 1.0, price - 1.0, price)] * n


def _uptrend_with_swings(n: int = 60) -> list[_Bar]:
    """Zigzag upward: clear HH/HL pattern."""
    bars = []
    price = 100.0
    for i in range(n):
        if i % 10 < 5:
            price += 1.0
        else:
            price -= 0.3
        price = max(price, 1.0)
        bars.append(_Bar(price + 1.5, price - 0.5, price))
    return bars


def _downtrend_with_swings(n: int = 60) -> list[_Bar]:
    """Zigzag downward: clear LH/LL pattern."""
    bars = []
    price = 200.0
    for i in range(n):
        if i % 10 < 5:
            price -= 1.0
        else:
            price += 0.3
        price = max(price, 1.0)
        bars.append(_Bar(price + 0.5, price - 1.5, price))
    return bars


def _oscillating(n: int = 60) -> list[_Bar]:
    """Up/down swings at similar levels — sideways."""
    bars = []
    base = 100.0
    for i in range(n):
        phase = i % 20
        if phase < 10:
            price = base + 5.0
            bars.append(_Bar(price + 2.0, price - 2.0, price))
        else:
            price = base - 5.0
            bars.append(_Bar(price + 2.0, price - 2.0, price))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(8)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(20))
        assert result is not None
        assert isinstance(result, SwingReport)

    def test_returns_none_no_high_low(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar()] * 20) is None


class TestSwingPoints:
    def test_swing_highs_are_swing_points(self):
        result = analyze(_uptrend_with_swings(60))
        assert result is not None
        for sp in result.swing_highs:
            assert isinstance(sp, SwingPoint)
            assert sp.kind == "high"

    def test_swing_lows_are_swing_points(self):
        result = analyze(_uptrend_with_swings(60))
        assert result is not None
        for sp in result.swing_lows:
            assert isinstance(sp, SwingPoint)
            assert sp.kind == "low"

    def test_swing_prices_positive(self):
        result = analyze(_uptrend_with_swings(60))
        assert result is not None
        for sp in result.swing_highs + result.swing_lows:
            assert sp.price > 0

    def test_uptrend_has_swings(self):
        result = analyze(_uptrend_with_swings(60))
        assert result is not None
        assert result.total_swings > 0


class TestMarketStructure:
    def test_structure_valid(self):
        for bars in [_flat(30), _uptrend_with_swings(60), _downtrend_with_swings(60)]:
            result = analyze(bars)
            if result:
                assert result.structure in {"uptrend", "downtrend", "sideways", "unknown"}

    def test_uptrend_structure(self):
        result = analyze(_uptrend_with_swings(80))
        assert result is not None
        assert result.structure in {"uptrend", "sideways"}

    def test_downtrend_structure(self):
        result = analyze(_downtrend_with_swings(80))
        assert result is not None
        assert result.structure in {"downtrend", "sideways"}

    def test_hh_hl_booleans(self):
        result = analyze(_flat(30))
        assert result is not None
        assert isinstance(result.hh, bool)
        assert isinstance(result.hl, bool)
        assert isinstance(result.lh, bool)
        assert isinstance(result.ll, bool)


class TestSupportResistance:
    def test_support_below_price(self):
        result = analyze(_uptrend_with_swings(60))
        assert result is not None
        if result.nearest_support is not None:
            assert result.nearest_support <= result.current_price

    def test_resistance_above_price(self):
        result = analyze(_downtrend_with_swings(60))
        assert result is not None
        if result.nearest_resistance is not None:
            assert result.nearest_resistance >= result.current_price

    def test_support_distance_non_negative(self):
        result = analyze(_uptrend_with_swings(80))
        assert result is not None
        if result.support_distance_pct is not None:
            assert result.support_distance_pct >= 0

    def test_resistance_distance_non_negative(self):
        result = analyze(_downtrend_with_swings(80))
        assert result is not None
        if result.resistance_distance_pct is not None:
            assert result.resistance_distance_pct >= 0


class TestRecentPriorPoints:
    def test_recent_high_is_swing_point(self):
        result = analyze(_uptrend_with_swings(60))
        assert result is not None
        if result.recent_high is not None:
            assert isinstance(result.recent_high, SwingPoint)

    def test_recent_low_is_swing_point(self):
        result = analyze(_uptrend_with_swings(60))
        assert result is not None
        if result.recent_low is not None:
            assert isinstance(result.recent_low, SwingPoint)


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(30)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 30

    def test_current_price_positive(self):
        result = analyze(_flat(30, price=100.0))
        assert result is not None
        assert result.current_price > 0

    def test_symbol_stored(self):
        result = analyze(_flat(30), symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_total_swings_non_negative(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.total_swings >= 0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(20))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_structure(self):
        result = analyze(_flat(20))
        assert result is not None
        text = result.verdict.lower()
        assert "structure" in text or "swing" in text
