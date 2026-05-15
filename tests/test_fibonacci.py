"""Tests for Fibonacci retracement and extension levels."""

from __future__ import annotations

from amms.data.bars import Bar
from amms.features.fibonacci import FibResult, fibonacci


def _bar(high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, 1000.0)


def _uptrend(n: int = 20) -> list[Bar]:
    return [_bar(100.0 + i + 1, 99.0 + i, 100.0 + i, i) for i in range(n)]


def _downtrend(n: int = 20) -> list[Bar]:
    return [_bar(200.0 - i + 1, 199.0 - i, 200.0 - i, i) for i in range(n)]


class TestFibonacci:
    def test_returns_none_for_too_few_bars(self) -> None:
        assert fibonacci([_bar(102.0, 98.0, 100.0, i) for i in range(3)]) is None

    def test_returns_result_for_enough_bars(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        assert isinstance(result, FibResult)

    def test_swing_high_gte_swing_low(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        assert result.swing_high >= result.swing_low

    def test_levels_sorted_ascending(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        prices = [lv.price for lv in result.levels]
        assert prices == sorted(prices)

    def test_all_ratio_levels_present(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        ratios = {lv.ratio for lv in result.levels}
        assert 0.382 in ratios
        assert 0.618 in ratios
        assert 0.5 in ratios

    def test_direction_uptrend(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        assert result.direction == "uptrend"

    def test_direction_downtrend(self) -> None:
        result = fibonacci(_downtrend(20))
        assert result is not None
        assert result.direction == "downtrend"

    def test_nearest_support_below_price(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        if result.nearest_support:
            assert result.nearest_support.price < result.current_price

    def test_nearest_resistance_above_price(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        if result.nearest_resistance:
            assert result.nearest_resistance.price > result.current_price

    def test_extension_levels_included(self) -> None:
        result = fibonacci(_uptrend(20))
        assert result is not None
        ext = [lv for lv in result.levels if lv.is_extension]
        assert len(ext) > 0

    def test_lookback_respects_parameter(self) -> None:
        bars = _uptrend(30)
        r1 = fibonacci(bars, lookback=10)
        r2 = fibonacci(bars, lookback=25)
        assert r1 is not None and r2 is not None
        # Shorter lookback → smaller range
        assert (r1.swing_high - r1.swing_low) <= (r2.swing_high - r2.swing_low)
