"""Tests for amms.features.roc."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.roc import ROCResult, MultiROC, roc, multi_roc


def _bar(close: float, i: int = 0) -> Bar:
    return Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close + 1, close - 1, close, 10_000)


def _bars(closes: list[float]) -> list[Bar]:
    return [_bar(c, i) for i, c in enumerate(closes)]


class TestROC:
    def test_returns_none_insufficient(self):
        bars = _bars([100.0] * 10)
        assert roc(bars, period=10) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars([100.0] * 15)
        result = roc(bars, period=10)
        assert result is not None
        assert isinstance(result, ROCResult)

    def test_zero_roc_flat_price(self):
        bars = _bars([100.0] * 15)
        result = roc(bars, period=10)
        assert result is not None
        assert result.value == 0.0
        assert result.momentum == "flat"

    def test_positive_roc_rising(self):
        # 10 bars at 100, then last bar at 110 → ROC = 10%
        bars = _bars([100.0] * 10 + [110.0])
        result = roc(bars, period=10)
        assert result is not None
        assert result.value == pytest.approx(10.0, abs=0.01)
        assert result.momentum == "strong_up"

    def test_negative_roc_falling(self):
        bars = _bars([100.0] * 10 + [90.0])
        result = roc(bars, period=10)
        assert result is not None
        assert result.value == pytest.approx(-10.0, abs=0.01)
        assert result.momentum == "strong_down"

    def test_mild_positive_is_up(self):
        bars = _bars([100.0] * 10 + [102.0])
        result = roc(bars, period=10)
        assert result is not None
        assert result.momentum == "up"

    def test_mild_negative_is_down(self):
        bars = _bars([100.0] * 10 + [98.0])
        result = roc(bars, period=10)
        assert result is not None
        assert result.momentum == "down"

    def test_period_stored(self):
        bars = _bars([100.0] * 25)
        result = roc(bars, period=20)
        assert result is not None
        assert result.period == 20

    def test_zero_close_then_returns_none(self):
        bars = _bars([0.0] + [100.0] * 10)
        result = roc(bars, period=10)
        assert result is None

    def test_exact_minimum_bars(self):
        bars = _bars([100.0] * 11)
        result = roc(bars, period=10)
        assert result is not None


class TestMultiROC:
    def test_returns_multi_roc(self):
        bars = _bars([100.0] * 60)
        result = multi_roc(bars)
        assert isinstance(result, MultiROC)

    def test_all_none_when_insufficient(self):
        bars = _bars([100.0] * 5)
        result = multi_roc(bars)
        assert result.short is None
        assert result.medium is None
        assert result.long is None
        assert result.overall == "flat"

    def test_accelerating_up(self):
        """All three periods show strong upward momentum."""
        # 55 bars at 100, last bar at 115 → all ROC values are +15%
        closes = [100.0] * 55 + [115.0]
        bars = _bars(closes)
        result = multi_roc(bars)
        assert result.short is not None
        assert result.overall == "accelerating_up"

    def test_accelerating_down(self):
        """All three periods show strong downward momentum."""
        closes = [100.0] * 55 + [85.0]
        bars = _bars(closes)
        result = multi_roc(bars)
        assert result.overall == "accelerating_down"

    def test_flat_when_all_near_zero(self):
        bars = _bars([100.0] * 60)
        result = multi_roc(bars)
        assert result.overall == "flat"

    def test_short_medium_present_when_enough_data(self):
        bars = _bars([100.0 + i * 0.1 for i in range(25)])
        result = multi_roc(bars)
        assert result.short is not None
        assert result.medium is not None
        assert result.long is None  # need 51 bars for period=50
