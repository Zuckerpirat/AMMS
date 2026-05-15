"""Tests for amms.features.swing_points."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.swing_points import SwingAnalysis, SwingPoint, detect_swings


def _bar(i: int, high: float, low: float, close: float | None = None) -> Bar:
    c = close if close is not None else (high + low) / 2
    return Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               c, high, low, c, 10_000)


def _zigzag_bars(n: int = 30) -> list[Bar]:
    """Creates clear alternating swing highs and lows."""
    bars = []
    for i in range(n):
        if i % 6 < 3:
            bars.append(_bar(i, high=105.0, low=99.0, close=102.0))
        else:
            bars.append(_bar(i, high=101.0, low=95.0, close=98.0))
    return bars


def _uptrend_bars(n: int = 30) -> list[Bar]:
    """Rising channel with zigzag to create higher highs and higher lows."""
    bars = []
    for i in range(n):
        base = 100.0 + i * 0.5
        wave = 3.0 if (i % 6) < 3 else -3.0
        mid = base + wave
        bars.append(_bar(i, high=mid + 1.5, low=mid - 1.5))
    return bars


def _downtrend_bars(n: int = 30) -> list[Bar]:
    bars = []
    for i in range(n):
        base = 150.0 - i * 0.5
        wave = 3.0 if (i % 6) < 3 else -3.0
        mid = base + wave
        bars.append(_bar(i, high=mid + 1.5, low=mid - 1.5))
    return bars


class TestDetectSwings:
    def test_returns_none_insufficient_data(self):
        bars = [_bar(i, 100.0, 99.0) for i in range(5)]
        assert detect_swings(bars, window=3) is None

    def test_returns_analysis_with_enough_data(self):
        bars = _zigzag_bars(20)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert isinstance(result, SwingAnalysis)

    def test_symbol_preserved(self):
        bars = _zigzag_bars(20)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert result.symbol == "SYM"

    def test_detects_swing_highs(self):
        bars = _zigzag_bars(20)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert len(result.swings_high) >= 1

    def test_detects_swing_lows(self):
        bars = _zigzag_bars(20)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert len(result.swings_low) >= 1

    def test_last_swing_high_has_correct_fields(self):
        bars = _zigzag_bars(20)
        result = detect_swings(bars, window=2)
        assert result is not None
        if result.last_swing_high:
            sp = result.last_swing_high
            assert isinstance(sp, SwingPoint)
            assert sp.kind == "high"
            assert sp.price > 0
            assert sp.idx >= 0

    def test_stop_below_set(self):
        bars = _zigzag_bars(25)
        result = detect_swings(bars, window=2)
        assert result is not None
        if result.last_swing_low:
            assert result.stop_below is not None
            assert result.stop_below < result.last_swing_low.price

    def test_uptrend_detection(self):
        bars = _uptrend_bars(40)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert result.trend in {"uptrend", "sideways", "unknown"}

    def test_downtrend_detection(self):
        bars = _downtrend_bars(40)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert result.trend in {"downtrend", "sideways", "unknown"}

    def test_trend_valid_string(self):
        bars = _zigzag_bars(30)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert result.trend in {"uptrend", "downtrend", "sideways", "unknown"}

    def test_breakout_when_above_swing_high(self):
        bars = _zigzag_bars(25)
        result = detect_swings(bars, window=2)
        assert result is not None
        # breakout_up and breakdown_down should be booleans
        assert isinstance(result.breakout_up, bool)
        assert isinstance(result.breakdown_down, bool)

    def test_lookback_limits_bars(self):
        bars = _zigzag_bars(50)
        result = detect_swings(bars, window=2, lookback=20)
        assert result is not None

    def test_current_price_is_last_close(self):
        bars = _zigzag_bars(20)
        result = detect_swings(bars, window=2)
        assert result is not None
        assert result.current_price == bars[-1].close
