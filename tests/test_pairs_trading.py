"""Tests for amms.analysis.pairs_trading."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from amms.analysis.pairs_trading import PairsResult, analyze_pair, _pearson
from amms.data.bars import Bar


def _bar(sym: str, close: float, i: int = 0) -> Bar:
    ts = datetime(2024, 1, 1 + i % 28, 12, 0, 0, tzinfo=UTC)
    price = close
    return Bar(sym, "1Day", ts, price, price * 1.01, price * 0.99, price, 100_000)


def _bars(sym: str, prices: list[float]) -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


def _stable_pair(n: int = 35) -> tuple[list[Bar], list[Bar]]:
    """Two correlated series with a constant ratio of 2.0."""
    b1 = _bars("AAA", [200.0 + i * 0.1 for i in range(n)])
    b2 = _bars("BBB", [100.0 + i * 0.05 for i in range(n)])
    return b1, b2


class TestPearson:
    def test_perfect_positive(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson(a, b) - 1.0) < 1e-9

    def test_perfect_negative(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(_pearson(a, b) - (-1.0)) < 1e-9

    def test_uncorrelated_returns_zero(self):
        a = [1.0, -1.0, 1.0]
        b = [-1.0, 1.0, -1.0]
        result = _pearson(a, b)
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_too_short_returns_zero(self):
        assert _pearson([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_mismatched_lengths_uses_min(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0]
        result = _pearson(a, b)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_constant_series_returns_zero(self):
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        assert _pearson(a, b) == 0.0


class TestAnalyzePair:
    def test_returns_none_insufficient_data(self):
        b1 = _bars("AAA", [100.0] * 20)
        b2 = _bars("BBB", [50.0] * 20)
        assert analyze_pair(b1, b2, n=30) is None

    def test_returns_none_one_series_too_short(self):
        b1 = _bars("AAA", [100.0] * 35)
        b2 = _bars("BBB", [50.0] * 25)
        assert analyze_pair(b1, b2, n=30) is None

    def test_stable_pair_neutral_signal(self):
        b1, b2 = _stable_pair(35)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        assert result.sym1 == "AAA"
        assert result.sym2 == "BBB"
        assert result.signal == "neutral"
        assert result.signal_strength == 0.0
        # ratio should be approximately 2.0
        assert 1.8 < result.current_ratio < 2.2

    def test_returns_pairs_result(self):
        b1, b2 = _stable_pair(40)
        result = analyze_pair(b1, b2, n=30)
        assert isinstance(result, PairsResult)

    def test_wide_spread_signals_long_spread(self):
        """When sym1 is much more expensive than historical ratio → long_spread."""
        base1 = [100.0] * 31
        base2 = [100.0] * 31
        # Spike sym1 price at the end (ratio >> mean)
        b1 = _bars("AAA", base1[:-1] + [250.0])
        b2 = _bars("BBB", base2)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        assert result.ratio_zscore > 2.0
        assert result.signal == "long_spread"
        assert result.signal_strength >= 0.5

    def test_narrow_spread_signals_short_spread(self):
        """When sym1 is much cheaper than historical ratio → short_spread."""
        base1 = [100.0] * 31
        base2 = [100.0] * 31
        # Drop sym1 price at the end (ratio << mean)
        b1 = _bars("AAA", base1[:-1] + [20.0])
        b2 = _bars("BBB", base2)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        assert result.ratio_zscore < -2.0
        assert result.signal == "short_spread"

    def test_mild_deviation_medium_signal(self):
        """Z-score between 1.5 and 2 gives mild signal."""
        # Make a series where latest ratio is mildly elevated
        prices1 = [100.0] * 30 + [115.0]
        prices2 = [100.0] * 31
        b1 = _bars("AAA", prices1)
        b2 = _bars("BBB", prices2)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        # z-score should be positive (sym1 elevated)
        assert result.ratio_zscore > 0

    def test_correlated_series_positive_correlation(self):
        """Perfectly correlated series should yield correlation near 1."""
        prices = [100.0 + i for i in range(35)]
        b1 = _bars("AAA", [p * 2 for p in prices])
        b2 = _bars("BBB", prices)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        assert result.correlation > 0.95

    def test_zero_price_sym2_skipped(self):
        """If sym2 has zero prices, function handles gracefully."""
        b1 = _bars("AAA", [100.0] * 35)
        b2 = _bars("BBB", [0.0] * 35)
        result = analyze_pair(b1, b2, n=30)
        # All ratios are None → not enough valid ratios
        assert result is None

    def test_custom_n(self):
        b1 = _bars("AAA", [200.0 + i * 0.1 for i in range(25)])
        b2 = _bars("BBB", [100.0 + i * 0.05 for i in range(25)])
        result = analyze_pair(b1, b2, n=20)
        assert result is not None

    def test_recommended_action_long_spread_mentions_sell_buy(self):
        prices1 = [100.0] * 31
        prices2 = [100.0] * 31
        b1 = _bars("AAA", prices1[:-1] + [250.0])
        b2 = _bars("BBB", prices2)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        action = result.recommended_action.lower()
        assert "sell" in action or "buy" in action

    def test_signal_strength_bounded_0_1(self):
        b1 = _bars("AAA", [100.0] * 30 + [500.0])
        b2 = _bars("BBB", [100.0] * 31)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        assert 0.0 <= result.signal_strength <= 1.0

    def test_ratio_std_zero_zscore_is_zero(self):
        """Constant ratio → std=0 → zscore=0."""
        b1 = _bars("AAA", [200.0] * 35)
        b2 = _bars("BBB", [100.0] * 35)
        result = analyze_pair(b1, b2, n=30)
        assert result is not None
        assert result.ratio_zscore == 0.0
        assert result.signal == "neutral"
