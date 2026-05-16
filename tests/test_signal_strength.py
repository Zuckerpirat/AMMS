"""Tests for amms.analysis.signal_strength."""

from __future__ import annotations

import pytest

from amms.analysis.signal_strength import SignalComponent, SignalStrengthReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0, volume: float = 100_000):
        self.close = close
        self.high = close + spread
        self.low = close - spread
        self.volume = volume


def _uptrend(n: int = 70, step: float = 0.5) -> list[_Bar]:
    price = 100.0
    bars = []
    for i in range(n):
        bars.append(_Bar(price, volume=120_000))
        price += step
    return bars


def _downtrend(n: int = 70, step: float = 0.5) -> list[_Bar]:
    price = 200.0
    bars = []
    for _ in range(n):
        bars.append(_Bar(price, volume=80_000))
        price -= step
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
        assert isinstance(result, SignalStrengthReport)


class TestScore:
    def test_score_in_range(self):
        for bars in [_uptrend(70), _downtrend(70), _flat(70)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.score <= 100.0

    def test_uptrend_high_score(self):
        bars = _uptrend(80, step=1.0)
        result = analyze(bars)
        assert result is not None
        assert result.score > 50.0

    def test_downtrend_low_score(self):
        bars = _downtrend(80, step=1.0)
        result = analyze(bars)
        assert result is not None
        assert result.score < 50.0


class TestGrade:
    def test_grade_valid(self):
        for bars in [_uptrend(70), _downtrend(70), _flat(70)]:
            result = analyze(bars)
            if result:
                assert result.grade in {"strong_bull", "bull", "neutral", "bear", "strong_bear"}

    def test_grade_consistent_with_score(self):
        bars = _flat(70)
        result = analyze(bars)
        if result:
            s = result.score
            if s >= 70:
                assert result.grade == "strong_bull"
            elif s >= 60:
                assert result.grade == "bull"
            elif s <= 30:
                assert result.grade == "strong_bear"
            elif s <= 40:
                assert result.grade == "bear"
            else:
                assert result.grade == "neutral"


class TestComponents:
    def test_six_components_returned(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert len(result.components) == 6

    def test_components_are_signal_component(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        for c in result.components:
            assert isinstance(c, SignalComponent)

    def test_component_vote_in_range(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        for c in result.components:
            assert -2 <= c.vote <= 2

    def test_component_names_present(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        names = {c.name for c in result.components}
        assert "Trend" in names
        assert "Momentum" in names
        assert "RSI" in names


class TestVotes:
    def test_total_vote_in_range(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert -result.max_possible <= result.total_vote <= result.max_possible

    def test_max_possible_positive(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert result.max_possible > 0


class TestMetadata:
    def test_current_price_positive(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert result.current_price > 0

    def test_bars_used_correct(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 70

    def test_symbol_stored(self):
        bars = _flat(70)
        result = analyze(bars, symbol="NVDA")
        assert result is not None
        assert result.symbol == "NVDA"


class TestVerdict:
    def test_verdict_present(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_signal(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert "signal" in result.verdict.lower() or "bull" in result.verdict.lower() or "bear" in result.verdict.lower()
