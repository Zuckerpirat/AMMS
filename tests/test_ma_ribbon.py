"""Tests for amms.analysis.ma_ribbon."""

from __future__ import annotations

import pytest

from amms.analysis.ma_ribbon import MARibbonReport, analyze


class _Bar:
    def __init__(self, close, high=None, low=None):
        self.close = close
        self.high = high or close + 0.5
        self.low = low or close - 0.5


def _bullish(n: int = 80, start: float = 100.0, step: float = 0.3) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _bearish(n: int = 80, start: float = 130.0, step: float = 0.3) -> list[_Bar]:
    return [_Bar(start - i * step) for i in range(n)]


def _flat(n: int = 80, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + (0.5 if i % 2 == 0 else -0.5)) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(30)]
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, MARibbonReport)


class TestEMAs:
    def test_has_all_six_emas(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert len(result.emas) == 6

    def test_ema_periods(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert 5 in result.emas
        assert 55 in result.emas

    def test_custom_periods(self):
        bars = _bullish(80)
        result = analyze(bars, periods=[10, 20, 30])
        assert result is not None
        assert 10 in result.emas
        assert 20 in result.emas
        assert 30 in result.emas


class TestDirection:
    def test_uptrend_direction(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.direction == "up"

    def test_downtrend_direction(self):
        bars = _bearish(80)
        result = analyze(bars)
        assert result is not None
        assert result.direction == "down"

    def test_direction_valid(self):
        bars = _flat(80)
        result = analyze(bars)
        assert result is not None
        assert result.direction in ("up", "down", "tangled")

    def test_ordered_in_uptrend(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.ordered is True

    def test_direction_valid_for_flat(self):
        bars = _flat(80)
        result = analyze(bars)
        assert result is not None
        # Flat market may show any direction — just check it's valid
        assert result.direction in ("up", "down", "tangled")


class TestSpreadAndPosition:
    def test_spread_positive(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.ribbon_spread_pct >= 0

    def test_uptrend_price_above_all(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.price_position == "above_all"

    def test_downtrend_price_below_all(self):
        bars = _bearish(80)
        result = analyze(bars)
        assert result is not None
        assert result.price_position == "below_all"

    def test_price_position_valid(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.price_position in ("above_all", "below_all", "inside", "at_ribbon")


class TestMetadata:
    def test_symbol_stored(self):
        bars = _bullish(80)
        result = analyze(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_used_correct(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 80

    def test_current_price_correct(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_is_expanding_bool(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result.is_expanding, bool)


class TestVerdict:
    def test_verdict_present(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10
