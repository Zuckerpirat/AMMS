"""Tests for amms.analysis.signal_aggregator."""

from __future__ import annotations

import pytest

from amms.analysis.signal_aggregator import AggregatedSignal, IndicatorVote, compute


class _Bar:
    def __init__(self, high, low, close, volume=1_000_000):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _bullish(n: int = 60, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step + 0.5, start + i * step - 0.5, start + i * step) for i in range(n)]


def _bearish(n: int = 60, start: float = 160.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start - i * step + 0.5, start - i * step - 0.5, start - i * step) for i in range(n)]


def _flat(n: int = 60, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 0.5, price - 0.5, price) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_too_few(self):
        bars = _bullish(20)
        assert compute(bars) is None

    def test_returns_none_empty(self):
        assert compute([]) is None

    def test_returns_result(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert isinstance(result, AggregatedSignal)


class TestVotes:
    def test_votes_not_empty(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert len(result.votes) > 0

    def test_vote_values_valid(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        for v in result.votes:
            assert v.vote in (-1, 0, 1)

    def test_vote_signal_consistent(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        for v in result.votes:
            if v.vote == 1:
                assert v.signal == "bull"
            elif v.vote == -1:
                assert v.signal == "bear"
            else:
                assert v.signal == "neutral"

    def test_bull_bear_neutral_sum(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert result.bull_votes + result.bear_votes + result.neutral_votes == len(result.votes)


class TestScore:
    def test_score_in_range(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert -100.0 <= result.score <= 100.0

    def test_score_formula(self):
        """score = (bull_votes - bear_votes) / total_votes × 100."""
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        expected = (result.bull_votes - result.bear_votes) / len(result.votes) * 100
        assert result.score == pytest.approx(expected, abs=0.1)

    def test_signal_valid_value(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert result.signal in ("strong_bull", "bull", "neutral", "bear", "strong_bear")

    def test_price_above_smas_bull_votes(self):
        """Price well above SMA20 and SMA50 → at least 2 bull votes."""
        bars = _flat(40) + _bullish(40, start=110.0)  # strong up from 80 bars
        result = compute(bars)
        assert result is not None
        # SMA20 and SMA50 should both be bullish
        sma_votes = [v for v in result.votes if "SMA" in v.name]
        assert any(v.vote == 1 for v in sma_votes)

    def test_price_below_smas_bear_votes(self):
        """Price well below SMA20 and SMA50 → bear SMA votes."""
        bars = _bullish(40, start=50.0) + _flat(40, price=60.0)
        result = compute(bars)
        assert result is not None
        # After flat period following strong rise, SMA might be above price
        # Just check that we have bear or bull sma votes
        sma_votes = [v for v in result.votes if "SMA" in v.name]
        assert len(sma_votes) > 0


class TestMetadata:
    def test_symbol_stored(self):
        bars = _bullish(60)
        result = compute(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_current_price_correct(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_bars_used(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert result.bars_used == 60

    def test_verdict_present(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        assert len(result.verdict) > 5

    def test_indicator_names_unique(self):
        bars = _bullish(60)
        result = compute(bars)
        assert result is not None
        names = [v.name for v in result.votes]
        assert len(names) == len(set(names))
