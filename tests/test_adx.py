"""Tests for ADX (Average Directional Index) feature."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.adx import ADXResult, adx


def _bar(high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, 1000.0)


def _trending_up(n: int = 50) -> list[Bar]:
    """Strongly trending upward bars."""
    bars = []
    for i in range(n):
        base = 100.0 + i * 1.0
        bars.append(_bar(base + 1.5, base - 0.5, base + 0.5, i))
    return bars


def _ranging(n: int = 50) -> list[Bar]:
    """Sideways/ranging bars alternating up and down."""
    bars = []
    for i in range(n):
        if i % 2 == 0:
            bars.append(_bar(102.0, 98.0, 100.5, i))
        else:
            bars.append(_bar(101.0, 99.0, 99.5, i))
    return bars


class TestAdx:
    def test_returns_none_if_insufficient_bars(self) -> None:
        bars = [_bar(101.0, 99.0, 100.0, i) for i in range(20)]
        assert adx(bars, 14) is None

    def test_returns_adx_result_for_enough_bars(self) -> None:
        result = adx(_trending_up(50), 14)
        assert result is not None
        assert isinstance(result, ADXResult)

    def test_adx_bounded_0_to_100(self) -> None:
        result = adx(_trending_up(50), 14)
        assert result is not None
        assert 0.0 <= result.adx <= 100.0

    def test_di_values_non_negative(self) -> None:
        result = adx(_trending_up(50), 14)
        assert result is not None
        assert result.plus_di >= 0.0
        assert result.minus_di >= 0.0

    def test_strong_uptrend_bullish_direction(self) -> None:
        result = adx(_trending_up(60), 14)
        assert result is not None
        assert result.direction == "bullish"

    def test_strong_downtrend_bearish_direction(self) -> None:
        bars = []
        for i in range(60):
            base = 200.0 - i * 1.0
            bars.append(_bar(base + 0.5, base - 1.5, base - 0.5, i))
        result = adx(bars, 14)
        assert result is not None
        assert result.direction == "bearish"

    def test_ranging_market_low_adx(self) -> None:
        result = adx(_ranging(60), 14)
        assert result is not None
        assert result.adx < 30  # ranging → low ADX

    def test_trend_strength_label_present(self) -> None:
        valid = {"none", "emerging", "strong", "very_strong", "extreme"}
        result = adx(_trending_up(50), 14)
        assert result is not None
        assert result.trend_strength in valid

    def test_raises_for_period_less_than_2(self) -> None:
        with pytest.raises(ValueError):
            adx(_trending_up(50), period=1)

    def test_minimum_bars_requirement(self) -> None:
        """Exactly 2*period+1 bars should return a result."""
        bars = _trending_up(29)
        result = adx(bars, 14)
        assert result is not None  # 29 >= 14*2+1

    def test_one_bar_short_returns_none(self) -> None:
        bars = _trending_up(28)
        result = adx(bars, 14)
        assert result is None  # 28 < 14*2+1
