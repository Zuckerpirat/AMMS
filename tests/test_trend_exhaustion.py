"""Tests for amms.analysis.trend_exhaustion."""

from __future__ import annotations

import pytest

from amms.analysis.trend_exhaustion import ExhaustionReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0, open_: float | None = None):
        self.close = close
        self.high = close + spread
        self.low = close - spread
        self.open = open_ if open_ is not None else close


def _uptrend(n: int = 70) -> list[_Bar]:
    price = 100.0
    bars = []
    for _ in range(n):
        bars.append(_Bar(price))
        price += 0.5
    return bars


def _extended_uptrend(n: int = 70) -> list[_Bar]:
    """Big initial gain then flat — shows extension and decay."""
    bars = []
    # First 50 bars: strong up
    price = 100.0
    for i in range(50):
        bars.append(_Bar(price, spread=1.5))
        price += 1.5
    # Last 20 bars: flat with wicks
    for _ in range(n - 50):
        bars.append(_Bar(price, spread=1.0, open_=price - 0.5))
    return bars


def _downtrend(n: int = 70) -> list[_Bar]:
    price = 200.0
    bars = []
    for _ in range(n):
        bars.append(_Bar(price))
        price -= 0.5
    return bars


def _flat(n: int = 70, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _flat(30)
        assert analyze(bars) is None

    def test_returns_result_enough_bars(self):
        bars = _flat(60)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, ExhaustionReport)


class TestScore:
    def test_score_in_range(self):
        for bars in [_uptrend(70), _downtrend(70), _flat(70)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.exhaustion_score <= 100.0

    def test_label_valid(self):
        for bars in [_uptrend(70), _downtrend(70), _flat(70)]:
            result = analyze(bars)
            if result:
                assert result.exhaustion_label in {"high", "moderate", "low", "none"}


class TestTrendDirection:
    def test_uptrend_detected(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert result.trend_direction == "up"

    def test_downtrend_detected(self):
        bars = _downtrend(70)
        result = analyze(bars)
        assert result is not None
        assert result.trend_direction == "down"

    def test_trend_direction_valid(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert result.trend_direction in {"up", "down"}


class TestComponents:
    def test_price_extension_in_range(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.price_extension <= 1.0

    def test_bar_rejection_in_range(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.bar_rejection <= 1.0

    def test_rsi_divergence_is_bool(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result.rsi_divergence, bool)

    def test_momentum_decay_is_bool(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result.momentum_decay, bool)

    def test_atr_contraction_is_bool(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result.atr_contraction, bool)


class TestIndicators:
    def test_rsi_in_range(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.rsi <= 100.0

    def test_sma20_positive(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        if result.sma20:
            assert result.sma20 > 0

    def test_current_price_positive(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert result.current_price > 0


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 70

    def test_symbol_stored(self):
        bars = _flat(70)
        result = analyze(bars, symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"


class TestVerdict:
    def test_verdict_present(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_exhaustion(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert "exhaust" in result.verdict.lower() or "trend" in result.verdict.lower()
