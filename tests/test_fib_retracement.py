"""Tests for amms.analysis.fib_retracement."""

from __future__ import annotations

import pytest

from amms.analysis.fib_retracement import FibLevel, FibRetracementReport, analyze


class _Bar:
    def __init__(self, close: float, high: float | None = None, low: float | None = None):
        self.close = close
        self.high = high if high is not None else close + 0.5
        self.low = low if low is not None else close - 0.5


def _uptrend_with_pullback(n: int = 60) -> list[_Bar]:
    """Rises to peak, then pulls back 38%."""
    bars = []
    # Rise from 100 to 150
    for i in range(n // 2):
        price = 100.0 + i * (50.0 / (n // 2))
        bars.append(_Bar(price))
    # Pull back 38.2% of 50 = 19.1
    peak = 150.0
    retrace = 0.382 * 50.0
    for i in range(n // 2):
        price = peak - retrace * (i / (n // 2))
        bars.append(_Bar(price))
    return bars


def _downtrend(n: int = 60) -> list[_Bar]:
    """Drops from 150 to 100."""
    bars = []
    for i in range(n):
        price = 150.0 - i * (50.0 / n)
        bars.append(_Bar(price))
    return bars


def _flat(n: int = 60, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _flat(20)
        assert analyze(bars) is None

    def test_returns_result_enough_bars(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, FibRetracementReport)


class TestLevels:
    def test_eleven_levels_returned(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        # 7 retracement + 4 extension = 11
        assert len(result.levels) == 11

    def test_levels_are_fib_level(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        for l in result.levels:
            assert isinstance(l, FibLevel)

    def test_level_kind_retracement_or_extension(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        for l in result.levels:
            assert l.kind in {"retracement", "extension"}

    def test_zero_percent_level_at_swing_high(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        zero_level = next(l for l in result.levels if l.ratio == 0.0 and l.kind == "retracement")
        assert abs(zero_level.price - result.swing_high) < 0.01


class TestSwingPoints:
    def test_swing_high_above_low(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert result.swing_high > result.swing_low

    def test_swing_high_idx_valid(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert 0 <= result.swing_high_idx < len(bars)

    def test_swing_low_idx_valid(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert 0 <= result.swing_low_idx < len(bars)


class TestTrend:
    def test_trend_direction_valid(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert result.trend_direction in {"up", "down"}


class TestNearestLevels:
    def test_nearest_level_present(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert result.nearest_level is not None

    def test_nearest_support_below_current(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        if result.nearest_support:
            assert result.nearest_support.price <= result.current_price + 0.01

    def test_nearest_resistance_above_current(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        if result.nearest_resistance:
            assert result.nearest_resistance.price >= result.current_price - 0.01


class TestRetracementDepth:
    def test_retracement_depth_in_range(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.retracement_depth <= 100.0


class TestMetrics:
    def test_current_price_positive(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert result.current_price > 0

    def test_bars_used_correct(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 60

    def test_symbol_stored(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"


class TestVerdict:
    def test_verdict_present(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_fib(self):
        bars = _uptrend_with_pullback(60)
        result = analyze(bars)
        assert result is not None
        assert "fib" in result.verdict.lower() or "retracement" in result.verdict.lower()
