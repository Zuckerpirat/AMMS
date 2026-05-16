"""Tests for amms.analysis.williams_fractal."""

from __future__ import annotations

import math
import pytest

from amms.analysis.williams_fractal import FractalPoint, WilliamsFractalReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high  = high
        self.low   = low
        self.close = close


MIN_BARS = 40


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 1.0, price - 1.0, price) for _ in range(n)]


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price))
        price += step
    return bars


def _wavy(n: int = MIN_BARS, price: float = 100.0, amp: float = 5.0) -> list[_Bar]:
    """Sinusoidal to create clear fractals."""
    bars = []
    for i in range(n):
        center = price + amp * math.sin(2 * math.pi * i / 10)
        bars.append(_Bar(center + 1.0, center - 1.0, center))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(10)) is None

    def test_returns_result_min_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, WilliamsFractalReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            high = low = 1.0
        assert analyze([_Bad()] * MIN_BARS) is None


class TestFractals:
    def test_fractals_are_fractal_points(self):
        result = analyze(_wavy(100))
        assert result is not None
        for f in result.up_fractals + result.down_fractals:
            assert isinstance(f, FractalPoint)

    def test_fractal_types_valid(self):
        result = analyze(_wavy(100))
        assert result is not None
        for f in result.up_fractals:
            assert f.fractal_type == "up"
        for f in result.down_fractals:
            assert f.fractal_type == "down"

    def test_max_fractals_limit(self):
        result = analyze(_wavy(100), max_fractals=3)
        assert result is not None
        assert len(result.up_fractals) <= 3
        assert len(result.down_fractals) <= 3

    def test_up_fractals_prices_positive(self):
        result = analyze(_wavy(100))
        assert result is not None
        for f in result.up_fractals:
            assert f.price > 0

    def test_down_fractals_prices_positive(self):
        result = analyze(_wavy(100))
        assert result is not None
        for f in result.down_fractals:
            assert f.price > 0


class TestSupportResistance:
    def test_nearest_resistance_above_price(self):
        result = analyze(_flat(60))
        assert result is not None
        if result.nearest_resistance is not None:
            assert result.nearest_resistance > result.current_price

    def test_nearest_support_below_price(self):
        result = analyze(_flat(60))
        assert result is not None
        if result.nearest_support is not None:
            assert result.nearest_support < result.current_price

    def test_resistance_pct_positive(self):
        result = analyze(_flat(60))
        assert result is not None
        if result.resistance_pct is not None:
            assert result.resistance_pct > 0

    def test_support_pct_positive(self):
        result = analyze(_flat(60))
        assert result is not None
        if result.support_pct is not None:
            assert result.support_pct > 0


class TestAlligator:
    def test_alligator_lines_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.jaw is not None
        assert result.teeth is not None
        assert result.lips is not None

    def test_alligator_lines_positive(self):
        result = analyze(_flat(MIN_BARS, price=100.0))
        assert result is not None
        if result.jaw:
            assert result.jaw > 0
        if result.teeth:
            assert result.teeth > 0
        if result.lips:
            assert result.lips > 0

    def test_alligator_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.alligator_open, bool)
        assert isinstance(result.alligator_sleeping, bool)

    def test_not_both_bull_and_bear(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        if result.alligator_bullish is not None and result.alligator_bearish is not None:
            assert not (result.alligator_bullish and result.alligator_bearish)

    def test_uptrend_alligator_bullish(self):
        result = analyze(_uptrend(80))
        assert result is not None
        # In a sustained uptrend, price should be above Alligator
        if result.alligator_bullish is not None:
            assert result.alligator_bullish is True


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_bull", "bull", "neutral", "bear", "strong_bear"}
        for bars in [_flat(MIN_BARS), _uptrend(), _wavy(100)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_score_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _wavy(100)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="NVDA")
        assert result is not None
        assert result.symbol == "NVDA"

    def test_bars_used_correct(self):
        bars = _flat(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 60


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_fractal_or_alligator(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        text = result.verdict.lower()
        assert "fractal" in text or "alligator" in text
