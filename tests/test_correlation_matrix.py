"""Tests for amms.analysis.correlation_matrix."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.correlation_matrix import CorrelationMatrix, CorrPair, compute, _pearson


def _bar(sym: str, close: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close + 1, close - 1, close, 10_000)


def _bars(sym: str, prices: list[float]) -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


class _TwoBroker:
    def get_positions(self):
        class P:
            def __init__(self, sym): self.symbol = sym
        return [P("AAPL"), P("MSFT")]


class _OneBroker:
    def get_positions(self):
        class P:
            symbol = "AAPL"
        return [P()]


class _ErrorBroker:
    def get_positions(self):
        raise RuntimeError("broker error")


class _CorrelatedData:
    """AAPL and MSFT move identically → correlation = 1.0."""
    def get_bars(self, symbol, *, limit=35):
        prices = [100.0 + i * 0.5 for i in range(35)]
        return _bars(symbol, prices)


class _UncorrelatedData:
    """AAPL and MSFT have opposite daily return patterns → negative correlation."""
    def get_bars(self, symbol, *, limit=35):
        # Alternating returns: AAPL goes up/down, MSFT goes down/up
        prices = [100.0]
        for i in range(34):
            if symbol == "AAPL":
                delta = 1.0 if i % 2 == 0 else -0.9
            else:
                delta = -1.0 if i % 2 == 0 else 0.9
            prices.append(max(prices[-1] + delta, 0.1))
        return _bars(symbol, prices)


class _ShortData:
    def get_bars(self, symbol, *, limit=35):
        return _bars(symbol, [100.0, 101.0])


class TestComputeMatrix:
    def test_returns_none_one_position(self):
        assert compute(_OneBroker(), _CorrelatedData()) is None

    def test_returns_none_broker_error(self):
        assert compute(_ErrorBroker(), _CorrelatedData()) is None

    def test_returns_none_insufficient_data(self):
        assert compute(_TwoBroker(), _ShortData()) is None

    def test_returns_matrix_with_two_positions(self):
        result = compute(_TwoBroker(), _CorrelatedData())
        assert result is not None
        assert isinstance(result, CorrelationMatrix)

    def test_symbols_preserved(self):
        result = compute(_TwoBroker(), _CorrelatedData())
        assert result is not None
        assert set(result.symbols) == {"AAPL", "MSFT"}

    def test_one_pair_for_two_positions(self):
        result = compute(_TwoBroker(), _CorrelatedData())
        assert result is not None
        assert len(result.pairs) == 1

    def test_correlated_data_near_1(self):
        result = compute(_TwoBroker(), _CorrelatedData())
        assert result is not None
        assert result.pairs[0].correlation >= 0.9

    def test_uncorrelated_data_negative(self):
        result = compute(_TwoBroker(), _UncorrelatedData())
        assert result is not None
        assert result.pairs[0].correlation < 0.0

    def test_high_correlation_in_high_corr_pairs(self):
        result = compute(_TwoBroker(), _CorrelatedData())
        assert result is not None
        assert len(result.high_corr_pairs) >= 1

    def test_diversification_score_in_range(self):
        result = compute(_TwoBroker(), _CorrelatedData())
        assert result is not None
        assert 0.0 <= result.diversification_score <= 100.0

    def test_perfectly_correlated_low_diversity(self):
        result = compute(_TwoBroker(), _CorrelatedData())
        assert result is not None
        assert result.diversification_score < 20.0

    def test_negatively_correlated_high_diversity(self):
        result = compute(_TwoBroker(), _UncorrelatedData())
        assert result is not None
        assert result.diversification_score > 50.0


class TestPearson:
    def test_perfect_positive(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert _pearson(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _pearson(a, b) == pytest.approx(-1.0, abs=1e-9)

    def test_too_short_returns_none(self):
        assert _pearson([1.0, 2.0], [1.0, 2.0]) is None

    def test_constant_series_returns_none(self):
        a = [1.0, 1.0, 1.0, 1.0]
        b = [1.0, 2.0, 3.0, 4.0]
        assert _pearson(a, b) is None
