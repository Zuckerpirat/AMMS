"""Tests for amms.features.williams_r."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.williams_r import WilliamsRResult, williams_r


def _bar(close: float, high: float | None = None, low: float | None = None, i: int = 0) -> Bar:
    h = high if high is not None else close + 1.0
    l = low if low is not None else close - 1.0
    return Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, h, l, close, 10_000)


def _bars(closes: list[float]) -> list[Bar]:
    return [_bar(c, i=i) for i, c in enumerate(closes)]


class TestWilliamsR:
    def test_returns_none_insufficient_data(self):
        bars = _bars([100.0] * 10)
        assert williams_r(bars, period=14) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars([100.0] * 20)
        result = williams_r(bars, period=14, smooth=3)
        assert result is not None
        assert isinstance(result, WilliamsRResult)

    def test_value_in_range(self):
        bars = _bars([100.0 + i * 0.1 for i in range(30)])
        result = williams_r(bars, period=14)
        assert result is not None
        assert -100.0 <= result.value <= 0.0

    def test_overbought_when_close_near_high(self):
        """Close at or near the highest high → %R near 0 → overbought."""
        bars = []
        for i in range(20):
            bars.append(Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
                            100.0, 100.0, 99.0, 100.0, 10_000))
        result = williams_r(bars, period=14, smooth=1)
        assert result is not None
        assert result.value >= -20.0
        assert result.zone == "overbought"
        assert result.signal == "sell"

    def test_oversold_when_close_near_low(self):
        """Close at or near the lowest low → %R near -100 → oversold."""
        bars = []
        for i in range(20):
            bars.append(Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
                            100.0, 101.0, 100.0, 100.0, 10_000))
        result = williams_r(bars, period=14, smooth=1)
        assert result is not None
        assert result.value <= -80.0
        assert result.zone == "oversold"
        assert result.signal == "buy"

    def test_neutral_mid_range(self):
        """Close exactly at midpoint → %R near -50 → neutral."""
        bars = []
        for i in range(20):
            bars.append(Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
                            100.0, 102.0, 98.0, 100.0, 10_000))
        result = williams_r(bars, period=14, smooth=1)
        assert result is not None
        assert result.zone == "neutral"
        assert result.signal == "none"

    def test_flat_range_returns_minus_50(self):
        """When high == low (no range), value = -50."""
        bars = []
        for i in range(20):
            bars.append(Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
                            100.0, 100.0, 100.0, 100.0, 10_000))
        result = williams_r(bars, period=14, smooth=1)
        assert result is not None
        assert result.value == -50.0

    def test_smoothed_differs_with_smooth_period(self):
        prices = [100.0 + i * 0.5 for i in range(30)]
        bars = [Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
                    prices[i], prices[i] + 2.0, prices[i] - 2.0, prices[i], 10_000)
                for i in range(30)]
        r1 = williams_r(bars, period=14, smooth=1)
        r3 = williams_r(bars, period=14, smooth=3)
        assert r1 is not None and r3 is not None
        # smoothed value with smooth=1 should equal raw value
        assert r1.smoothed == r1.value

    def test_period_stored(self):
        bars = _bars([100.0] * 20)
        result = williams_r(bars, period=10, smooth=1)
        assert result is not None
        assert result.period == 10

    def test_minimum_bars_exact(self):
        """Exactly period + smooth - 1 bars should work."""
        bars = _bars([100.0] * 16)  # period=14, smooth=3 → need 16
        result = williams_r(bars, period=14, smooth=3)
        assert result is not None
