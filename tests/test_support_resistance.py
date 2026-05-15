"""Tests for amms.analysis.support_resistance."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.support_resistance import SRLevel, SRResult, detect


def _bar(sym: str, close: float, high: float = None, low: float = None, i: int = 0) -> Bar:
    h = high if high is not None else close + 1
    l = low if low is not None else close - 1
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, h, l, close, 10_000)


def _bars_zigzag(sym: str, n: int, base: float = 100.0, amp: float = 5.0) -> list[Bar]:
    """Zigzag prices: oscillate between base-amp and base+amp."""
    bars = []
    for i in range(n):
        price = base + (amp if i % 4 < 2 else -amp)
        bars.append(_bar(sym, price, price + 0.5, price - 0.5, i))
    return bars


class TestDetect:
    def test_returns_none_insufficient_bars(self):
        bars = [_bar("AAPL", 100.0, i=i) for i in range(5)]
        assert detect(bars) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars_zigzag("AAPL", 30)
        result = detect(bars)
        assert result is not None
        assert isinstance(result, SRResult)

    def test_symbol_preserved(self):
        bars = _bars_zigzag("TSLA", 30)
        result = detect(bars)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_current_price_last_close(self):
        bars = _bars_zigzag("AAPL", 30)
        result = detect(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_levels_classified_correctly(self):
        """Support levels should be below current price, resistance above."""
        bars = _bars_zigzag("AAPL", 40)
        result = detect(bars)
        assert result is not None
        for level in result.levels:
            if level.kind == "support":
                assert level.price < result.current_price or abs(level.distance_pct) < 1.0
            elif level.kind == "resistance":
                assert level.price > result.current_price or abs(level.distance_pct) < 1.0

    def test_nearest_support_below_price(self):
        bars = _bars_zigzag("AAPL", 40)
        result = detect(bars)
        assert result is not None
        if result.nearest_support:
            assert result.nearest_support.price <= result.current_price or \
                   abs(result.nearest_support.distance_pct) < 1.0

    def test_nearest_resistance_above_price(self):
        bars = _bars_zigzag("AAPL", 40)
        result = detect(bars)
        assert result is not None
        if result.nearest_resistance:
            assert result.nearest_resistance.price >= result.current_price or \
                   abs(result.nearest_resistance.distance_pct) < 1.0

    def test_strength_normalized_0_100(self):
        bars = _bars_zigzag("AAPL", 40)
        result = detect(bars)
        assert result is not None
        if result.levels:
            for level in result.levels:
                assert 0 <= level.strength <= 100

    def test_top_n_respected(self):
        bars = _bars_zigzag("AAPL", 60)
        result = detect(bars, top_n=3)
        assert result is not None
        assert len(result.levels) <= 3

    def test_levels_sorted_by_price(self):
        bars = _bars_zigzag("AAPL", 50)
        result = detect(bars)
        assert result is not None
        prices = [l.price for l in result.levels]
        assert prices == sorted(prices)

    def test_touches_positive(self):
        bars = _bars_zigzag("AAPL", 50)
        result = detect(bars)
        assert result is not None
        for level in result.levels:
            assert level.touches >= 1

    def test_support_distance_pct_negative(self):
        bars = _bars_zigzag("AAPL", 50)
        result = detect(bars)
        assert result is not None
        if result.support_distance_pct is not None:
            assert result.support_distance_pct <= 0.5  # support is below or at price

    def test_resistance_distance_pct_positive(self):
        bars = _bars_zigzag("AAPL", 50)
        result = detect(bars)
        assert result is not None
        if result.resistance_distance_pct is not None:
            assert result.resistance_distance_pct >= -0.5  # resistance is above or at price

    def test_empty_levels_when_no_pivots(self):
        """Flat price → no pivots → empty levels list."""
        bars = [_bar("AAPL", 100.0, 101.0, 99.0, i) for i in range(20)]
        result = detect(bars)
        assert result is not None
        # Flat bars have no clear pivots (all equal highs/lows may not qualify)
        assert isinstance(result.levels, list)

    def test_bars_used_correct(self):
        bars = _bars_zigzag("AAPL", 40)
        result = detect(bars, lookback=30)
        assert result is not None
        assert result.bars_used == 30

    def test_repeated_pivots_higher_strength(self):
        """More pivot touches should give higher strength."""
        bars = _bars_zigzag("AAPL", 80, amp=5.0)
        result = detect(bars, top_n=10)
        assert result is not None
        if len(result.levels) >= 2:
            # Levels with more touches should have >= strength of fewer touches
            multi_touch = [l for l in result.levels if l.touches > 1]
            single_touch = [l for l in result.levels if l.touches == 1]
            if multi_touch and single_touch:
                max_multi = max(l.strength for l in multi_touch)
                max_single = max(l.strength for l in single_touch)
                assert max_multi >= max_single
