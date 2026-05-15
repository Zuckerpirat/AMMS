"""Tests for amms.features.cci."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.cci import CCIResult, cci


def _bar(high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, high, low, close, 10_000)


def _bars_uniform(n: int, high: float = 101.0, low: float = 99.0, close: float = 100.0) -> list[Bar]:
    return [_bar(high, low, close, i) for i in range(n)]


class TestCCI:
    def test_returns_none_insufficient(self):
        bars = _bars_uniform(10)
        assert cci(bars, period=20) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars_uniform(25)
        result = cci(bars)
        assert result is not None
        assert isinstance(result, CCIResult)

    def test_zero_cci_when_typical_equals_mean(self):
        """All identical typical prices → mean = typical → CCI = 0."""
        bars = _bars_uniform(25)
        result = cci(bars, period=20)
        assert result is not None
        assert result.value == 0.0
        assert result.zone == "neutral"

    def test_positive_cci_when_price_above_mean(self):
        """Rising close prices → last typical > mean → positive CCI."""
        bars = [_bar(100.0 + i + 1, 100.0 + i - 1, 100.0 + i, i) for i in range(25)]
        result = cci(bars, period=20)
        assert result is not None
        assert result.value > 0

    def test_negative_cci_when_price_below_mean(self):
        """Falling close prices → last typical < mean → negative CCI."""
        bars = [_bar(200.0 - i + 1, 200.0 - i - 1, 200.0 - i, i) for i in range(25)]
        result = cci(bars, period=20)
        assert result is not None
        assert result.value < 0

    def test_overbought_signal(self):
        """Strong upward spike → CCI > 100 → overbought + sell."""
        # 20 bars at 100, then spike to 200
        bars = _bars_uniform(19, 101, 99, 100) + [_bar(201, 199, 200, 19)]
        result = cci(bars, period=20)
        assert result is not None
        assert result.value > 100
        assert result.zone == "overbought"
        assert result.signal == "sell"

    def test_oversold_signal(self):
        """Strong downward spike → CCI < -100 → oversold + buy."""
        bars = _bars_uniform(19, 101, 99, 100) + [_bar(11, 9, 10, 19)]
        result = cci(bars, period=20)
        assert result is not None
        assert result.value < -100
        assert result.zone == "oversold"
        assert result.signal == "buy"

    def test_period_stored(self):
        bars = _bars_uniform(25)
        result = cci(bars, period=14)
        assert result is not None
        assert result.period == 14

    def test_exact_minimum_data(self):
        """Exactly period bars should work."""
        bars = _bars_uniform(20)
        result = cci(bars, period=20)
        assert result is not None

    def test_mad_zero_gives_zero_cci(self):
        """When all typical prices are equal → MAD=0 → CCI=0."""
        bars = [_bar(100.0, 100.0, 100.0, i) for i in range(20)]
        result = cci(bars, period=20)
        assert result is not None
        assert result.value == 0.0
