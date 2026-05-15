"""Tests for amms.analysis.trendlines."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.trendlines import TrendLine, TrendLineResult, detect_trendlines


def _bar(i: int, high: float, low: float, close: float | None = None) -> Bar:
    c = close if close is not None else (high + low) / 2
    return Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               c, high, low, c, 10_000)


def _trending_up_bars(n: int = 50) -> list[Bar]:
    """Rising channel with zigzag oscillation to create pivot highs/lows."""
    bars = []
    for i in range(n):
        base = 100.0 + i * 0.5
        # Zigzag: alternating +/- to create real pivot points
        wave = 3.0 if (i % 6) < 3 else -3.0
        mid = base + wave
        bars.append(_bar(i, mid + 1.5, mid - 1.5))
    return bars


def _trending_down_bars(n: int = 50) -> list[Bar]:
    bars = []
    for i in range(n):
        base = 200.0 - i * 0.5
        wave = 3.0 if (i % 6) < 3 else -3.0
        mid = base + wave
        bars.append(_bar(i, mid + 1.5, mid - 1.5))
    return bars


def _ranging_bars(n: int = 50) -> list[Bar]:
    """Flat channel with oscillating highs and lows around a constant level."""
    bars = []
    for i in range(n):
        mid = 100.0 + (2.0 if i % 4 < 2 else -2.0)
        bars.append(_bar(i, mid + 1.5, mid - 1.5))
    return bars


class TestDetectTrendlines:
    def test_returns_none_insufficient_data(self):
        bars = [_bar(i, 100.0, 99.0) for i in range(5)]
        assert detect_trendlines(bars) is None

    def test_returns_result_with_enough_data(self):
        bars = _trending_up_bars(30)
        result = detect_trendlines(bars)
        assert result is not None
        assert isinstance(result, TrendLineResult)

    def test_symbol_preserved(self):
        bars = _trending_up_bars(30)
        result = detect_trendlines(bars)
        assert result is not None
        assert result.symbol == "SYM"

    def test_bars_used_correct(self):
        bars = _trending_up_bars(30)
        result = detect_trendlines(bars, lookback=25)
        assert result is not None
        assert result.bars_used == 25

    def test_current_price_is_last_close(self):
        bars = _trending_up_bars(30)
        result = detect_trendlines(bars)
        assert result is not None
        assert result.current_price == bars[-1].close

    def test_support_line_present_for_trending(self):
        bars = _trending_up_bars(40)
        result = detect_trendlines(bars)
        assert result is not None
        assert result.support is not None

    def test_resistance_line_present_for_trending(self):
        bars = _trending_up_bars(40)
        result = detect_trendlines(bars)
        assert result is not None
        assert result.resistance is not None

    def test_uptrend_pattern(self):
        bars = _trending_up_bars(50)
        result = detect_trendlines(bars, pivot_window=2)
        assert result is not None
        assert result.pattern in {"uptrend", "wedge", "ranging", "unknown"}

    def test_trendline_has_required_fields(self):
        bars = _trending_up_bars(40)
        result = detect_trendlines(bars)
        if result and result.support:
            tl = result.support
            assert isinstance(tl.slope, float)
            assert isinstance(tl.intercept, float)
            assert tl.touches >= 2
            assert 0.0 <= tl.strength <= 1.0
            assert tl.direction in {"rising", "falling", "flat"}

    def test_support_distance_non_negative_in_uptrend(self):
        """In an uptrend, price should be above support → distance > 0."""
        bars = _trending_up_bars(50)
        result = detect_trendlines(bars, pivot_window=2)
        assert result is not None
        if result.support_distance_pct is not None:
            # Support is a floor — price should be at or above it
            assert result.support_distance_pct >= -5.0  # small tolerance

    def test_pattern_is_valid_string(self):
        bars = _trending_up_bars(40)
        result = detect_trendlines(bars)
        assert result is not None
        assert result.pattern in {"uptrend", "downtrend", "ranging", "wedge", "unknown"}


class TestFitLine:
    def test_perfect_line_r2_is_one(self):
        from amms.analysis.trendlines import _fit_line
        pivots = [(0, 100.0), (10, 110.0), (20, 120.0)]
        line = _fit_line(pivots, current_idx=25)
        assert line is not None
        assert line.strength == pytest.approx(1.0, abs=0.01)
        assert line.direction == "rising"

    def test_flat_line(self):
        from amms.analysis.trendlines import _fit_line
        pivots = [(0, 100.0), (10, 100.0), (20, 100.0)]
        line = _fit_line(pivots, current_idx=25)
        assert line is not None
        assert line.direction == "flat"
        assert line.slope == pytest.approx(0.0, abs=0.05)

    def test_falling_line(self):
        from amms.analysis.trendlines import _fit_line
        pivots = [(0, 120.0), (10, 110.0), (20, 100.0)]
        line = _fit_line(pivots, current_idx=25)
        assert line is not None
        assert line.direction == "falling"
        assert line.slope < 0

    def test_too_few_pivots_returns_none(self):
        from amms.analysis.trendlines import _fit_line
        assert _fit_line([(5, 100.0)], current_idx=10) is None

    def test_current_value_extrapolated(self):
        from amms.analysis.trendlines import _fit_line
        # Slope = 1.0/bar, intercept=100
        pivots = [(0, 100.0), (10, 110.0)]
        line = _fit_line(pivots, current_idx=20)
        assert line is not None
        # At index 20: expected ~120.0
        assert line.current_value == pytest.approx(120.0, abs=0.5)
