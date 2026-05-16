"""Tests for amms.analysis.price_action_score."""

from __future__ import annotations

import pytest

from amms.analysis.price_action_score import PAFactor, PAScoreReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0, bull: bool = True):
        self.close = close
        self.high = close + spread
        self.low = close - spread
        self.open = close - spread * 0.5 if bull else close + spread * 0.5


def _flat(n: int = 120, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = 120, start: float = 50.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price, bull=True))
        price += step
    return bars


def _downtrend(n: int = 120, start: float = 200.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price, bull=False))
        price -= step
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(80)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(110))
        assert result is not None
        assert isinstance(result, PAScoreReport)

    def test_returns_none_no_ohlc(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar()] * 110) is None


class TestComposite:
    def test_composite_in_range(self):
        for bars in [_flat(120), _uptrend(120), _downtrend(120)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.composite <= 100.0

    def test_uptrend_high_score(self):
        result = analyze(_uptrend(150, step=0.8))
        assert result is not None
        assert result.composite > 50.0

    def test_downtrend_low_score(self):
        result = analyze(_downtrend(150, step=0.8))
        assert result is not None
        assert result.composite < 50.0


class TestGrade:
    def test_grade_valid(self):
        for bars in [_flat(120), _uptrend(120), _downtrend(120)]:
            result = analyze(bars)
            if result:
                assert result.grade in {"strong_bull", "bull", "neutral", "bear", "strong_bear"}

    def test_grade_consistent_with_composite(self):
        result = analyze(_flat(120))
        assert result is not None
        s = result.composite
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


class TestFactors:
    def test_six_factors_returned(self):
        result = analyze(_flat(120))
        assert result is not None
        assert len(result.factors) == 6

    def test_factors_are_pafactor(self):
        result = analyze(_flat(120))
        assert result is not None
        for f in result.factors:
            assert isinstance(f, PAFactor)

    def test_factor_scores_in_range(self):
        result = analyze(_flat(120))
        assert result is not None
        for f in result.factors:
            assert 0.0 <= f.score <= 100.0

    def test_factor_weights_sum_to_one(self):
        result = analyze(_flat(120))
        assert result is not None
        total = sum(f.weight for f in result.factors)
        assert abs(total - 1.0) < 0.01

    def test_factor_names_present(self):
        result = analyze(_flat(120))
        assert result is not None
        names = {f.name for f in result.factors}
        assert "Trend Alignment" in names
        assert "Momentum" in names


class TestRawValues:
    def test_current_price_positive(self):
        result = analyze(_flat(120, price=150.0))
        assert result is not None
        assert result.current_price > 0

    def test_sma20_near_price_for_flat(self):
        result = analyze(_flat(120, price=100.0))
        assert result is not None
        assert result.sma20 is not None
        assert abs(result.sma20 - 100.0) < 1.0

    def test_atr_pct_non_negative(self):
        result = analyze(_flat(120))
        assert result is not None
        assert result.atr_pct >= 0

    def test_avg_body_ratio_in_range(self):
        result = analyze(_flat(120))
        assert result is not None
        assert 0.0 <= result.avg_body_ratio <= 1.0


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(120)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 120

    def test_symbol_stored(self):
        result = analyze(_flat(120), symbol="MSFT")
        assert result is not None
        assert result.symbol == "MSFT"


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(120))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_score(self):
        result = analyze(_flat(120))
        assert result is not None
        text = result.verdict.lower()
        assert "score" in text or "price action" in text
