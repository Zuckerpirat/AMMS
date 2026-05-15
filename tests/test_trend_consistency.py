"""Tests for amms.analysis.trend_consistency."""

from __future__ import annotations

import math
import pytest

from amms.data.bars import Bar
from amms.analysis.trend_consistency import TrendConsistency, score


def _bar(sym: str, close: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close * 1.01, close * 0.99, close, 100_000)


def _bars(sym: str, prices: list[float]) -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


def _linear(sym: str, n: int, start: float = 100.0, step: float = 1.0) -> list[Bar]:
    return _bars(sym, [start + step * i for i in range(n)])


def _zigzag(sym: str, n: int, amp: float = 5.0, base: float = 100.0) -> list[Bar]:
    return _bars(sym, [base + amp * math.sin(i * 1.5) for i in range(n)])


class TestScore:
    def test_returns_none_insufficient(self):
        bars = [_bar("AAPL", 100.0, i) for i in range(3)]
        assert score(bars, lookback=20) is None

    def test_returns_result(self):
        bars = _linear("AAPL", 25)
        result = score(bars)
        assert result is not None
        assert isinstance(result, TrendConsistency)

    def test_symbol_preserved(self):
        bars = _linear("TSLA", 25)
        result = score(bars)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_perfect_linear_high_score(self):
        """Perfect linear series → R²=1, high efficiency → high score."""
        bars = _linear("AAPL", 25, step=1.0)
        result = score(bars)
        assert result is not None
        assert result.r_squared >= 0.99
        assert result.score >= 70

    def test_zigzag_lower_score(self):
        """Choppy zigzag → lower score than linear."""
        linear = score(_linear("A", 25, step=1.0))
        zigzag = score(_zigzag("A", 25, amp=5.0))
        assert linear is not None and zigzag is not None
        assert linear.score > zigzag.score

    def test_r_squared_range(self):
        bars = _linear("AAPL", 25)
        result = score(bars)
        assert result is not None
        assert 0 <= result.r_squared <= 1

    def test_efficiency_range(self):
        bars = _linear("AAPL", 25)
        result = score(bars)
        assert result is not None
        assert 0 <= result.efficiency <= 1

    def test_efficiency_perfect_for_linear(self):
        """Linear bars → efficiency near 1.0."""
        bars = _linear("AAPL", 25, step=1.0)
        result = score(bars)
        assert result is not None
        assert result.efficiency >= 0.9

    def test_direction_up(self):
        bars = _linear("AAPL", 25, step=1.0)
        result = score(bars)
        assert result is not None
        assert result.direction == "up"

    def test_direction_down(self):
        bars = _linear("AAPL", 25, start=130.0, step=-1.0)
        result = score(bars)
        assert result is not None
        assert result.direction == "down"

    def test_direction_flat(self):
        bars = _bars("AAPL", [100.0] * 25)
        result = score(bars)
        assert result is not None
        assert result.direction == "flat"

    def test_score_range(self):
        bars = _zigzag("AAPL", 25)
        result = score(bars)
        assert result is not None
        assert 0 <= result.score <= 100

    def test_bars_used_correct(self):
        bars = _linear("AAPL", 30)
        result = score(bars, lookback=20)
        assert result is not None
        assert result.bars_used == 20

    def test_noise_pct_low_for_linear(self):
        """Linear series has near-zero noise."""
        bars = _linear("AAPL", 25, step=1.0)
        result = score(bars)
        assert result is not None
        assert result.noise_pct < 0.1

    def test_noise_pct_high_for_zigzag(self):
        """Choppy series has higher noise than linear."""
        linear = score(_linear("A", 25, step=0.5))
        zigzag = score(_zigzag("A", 25, amp=5.0))
        assert linear is not None and zigzag is not None
        assert zigzag.noise_pct > linear.noise_pct
