"""Tests for Stochastic Oscillator."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.stochastic import StochasticResult, stochastic


def _bar(high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, 1000.0)


def _bars_up(n: int = 20) -> list[Bar]:
    return [_bar(100.0 + i + 1, 99.0 + i, 100.0 + i, i) for i in range(n)]


def _bars_down(n: int = 20) -> list[Bar]:
    return [_bar(200.0 - i + 1, 199.0 - i, 200.0 - i, i) for i in range(n)]


class TestStochastic:
    def test_returns_none_if_insufficient_bars(self) -> None:
        bars = [_bar(102.0, 98.0, 100.0, i) for i in range(10)]
        assert stochastic(bars, 14, 3) is None

    def test_returns_result_for_sufficient_bars(self) -> None:
        result = stochastic(_bars_up(20), 14, 3)
        assert result is not None
        assert isinstance(result, StochasticResult)

    def test_k_bounded_0_to_100(self) -> None:
        result = stochastic(_bars_up(20), 14, 3)
        assert result is not None
        assert 0.0 <= result.k <= 100.0

    def test_d_bounded_0_to_100(self) -> None:
        result = stochastic(_bars_up(20), 14, 3)
        assert result is not None
        assert 0.0 <= result.d <= 100.0

    def test_uptrend_overbought(self) -> None:
        """Strongly trending up should produce overbought %K."""
        result = stochastic(_bars_up(20), 14, 3)
        assert result is not None
        assert result.zone == "overbought"

    def test_downtrend_oversold(self) -> None:
        """Strongly trending down should produce oversold %K."""
        result = stochastic(_bars_down(20), 14, 3)
        assert result is not None
        assert result.zone == "oversold"

    def test_raises_for_invalid_k_period(self) -> None:
        with pytest.raises(ValueError):
            stochastic(_bars_up(20), k_period=0)

    def test_raises_for_invalid_d_period(self) -> None:
        with pytest.raises(ValueError):
            stochastic(_bars_up(20), d_period=0)

    def test_signal_is_valid_value(self) -> None:
        valid = {"none", "bullish_cross", "bearish_cross"}
        result = stochastic(_bars_up(20), 14, 3)
        assert result is not None
        assert result.signal in valid

    def test_zone_is_valid_value(self) -> None:
        valid = {"oversold", "overbought", "neutral"}
        bars = [_bar(101.0, 99.0, 100.0 + (i % 3) * 0.1, i) for i in range(25)]
        result = stochastic(bars, 14, 3)
        assert result is not None
        assert result.zone in valid

    def test_equal_range_returns_50(self) -> None:
        """When all prices equal, close is in the middle → %K = 50."""
        bars = [_bar(102.0, 98.0, 100.0, i) for i in range(20)]
        result = stochastic(bars, 14, 3)
        assert result is not None
        assert result.k == 50.0
