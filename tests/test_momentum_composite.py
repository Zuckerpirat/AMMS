"""Tests for amms.analysis.momentum_composite."""

from __future__ import annotations

import math

import pytest

from amms.data.bars import Bar
from amms.analysis.momentum_composite import MomentumComposite, compute


def _bar(sym: str, close: float, high: float, low: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, high, low, close, 10_000)


def _bars_flat(sym: str, n: int, price: float = 100.0) -> list[Bar]:
    return [_bar(sym, price, price + 1, price - 1, i) for i in range(n)]


def _bars_rising(sym: str, n: int, start: float = 100.0, step: float = 0.5) -> list[Bar]:
    prices = [start + step * i for i in range(n)]
    return [_bar(sym, p, p + 1, p - 1, i) for i, p in enumerate(prices)]


def _bars_falling(sym: str, n: int, start: float = 120.0, step: float = 0.5) -> list[Bar]:
    prices = [start - step * i for i in range(n)]
    return [_bar(sym, p, p + 1, p - 1, i) for i, p in enumerate(prices)]


class TestCompute:
    def test_returns_none_insufficient_bars(self):
        bars = _bars_rising("AAPL", 30)
        assert compute(bars) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars_rising("AAPL", 50)
        result = compute(bars)
        assert result is not None
        assert isinstance(result, MomentumComposite)

    def test_symbol_preserved(self):
        bars = _bars_rising("TSLA", 50)
        result = compute(bars)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_score_range(self):
        bars = _bars_rising("AAPL", 50)
        result = compute(bars)
        assert result is not None
        assert -100.0 <= result.score <= 100.0

    def test_rising_prices_bull_signal(self):
        """Strongly rising prices → bullish signal."""
        bars = _bars_rising("AAPL", 60, step=1.0)
        result = compute(bars)
        assert result is not None
        assert result.signal in ("bull", "strong_bull")
        assert result.score > 0

    def test_falling_prices_bear_signal(self):
        """Strongly falling prices → bearish signal."""
        bars = _bars_falling("AAPL", 60, step=1.0)
        result = compute(bars)
        assert result is not None
        assert result.signal in ("bear", "strong_bear")
        assert result.score < 0

    def test_n_components_at_least_one(self):
        bars = _bars_rising("AAPL", 50)
        result = compute(bars)
        assert result is not None
        assert result.n_components >= 1

    def test_components_present_with_enough_data(self):
        bars = _bars_rising("AAPL", 60)
        result = compute(bars)
        assert result is not None
        assert result.rsi_component is not None
        assert result.roc_component is not None
        assert result.macd_component is not None
        assert result.wr_component is not None

    def test_current_price_is_last_close(self):
        bars = _bars_rising("AAPL", 50, start=100.0, step=0.5)
        result = compute(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_bars_used_correct(self):
        bars = _bars_rising("AAPL", 50)
        result = compute(bars)
        assert result is not None
        assert result.bars_used == 50

    def test_strong_bull_signal(self):
        """Very steep rise → strong_bull."""
        bars = _bars_rising("AAPL", 60, step=2.0)
        result = compute(bars)
        assert result is not None
        assert result.signal in ("bull", "strong_bull")

    def test_signal_matches_score(self):
        """Signal should be consistent with score value."""
        bars = _bars_rising("AAPL", 50)
        result = compute(bars)
        assert result is not None
        if result.score > 60:
            assert result.signal == "strong_bull"
        elif result.score > 20:
            assert result.signal == "bull"
        elif result.score > -20:
            assert result.signal == "neutral"
        elif result.score > -60:
            assert result.signal == "bear"
        else:
            assert result.signal == "strong_bear"

    def test_rising_score_higher_than_falling(self):
        """Rising prices should score higher than falling prices."""
        r_rising = compute(_bars_rising("AAPL", 60, step=1.0))
        r_falling = compute(_bars_falling("AAPL", 60, step=1.0))
        assert r_rising is not None and r_falling is not None
        assert r_rising.score > r_falling.score
