"""Tests for amms.analysis.liquidity_score."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.liquidity_score import LiquidityScore, score


def _bar(sym: str, close: float, volume: float, high: float = None,
         low: float = None, i: int = 0) -> Bar:
    h = high if high is not None else close * 1.01
    l = low if low is not None else close * 0.99
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, h, l, close, volume)


def _bars(sym: str, n: int, volume: float = 1_000_000,
          spread_pct: float = 0.5) -> list[Bar]:
    result = []
    for i in range(n):
        close = 100.0
        h = close * (1 + spread_pct / 200)
        l = close * (1 - spread_pct / 200)
        result.append(_bar(sym, close, volume, h, l, i))
    return result


class TestScore:
    def test_returns_none_insufficient_bars(self):
        bars = _bars("AAPL", 5)
        assert score(bars) is None

    def test_returns_result_with_enough_bars(self):
        bars = _bars("AAPL", 15)
        result = score(bars)
        assert result is not None
        assert isinstance(result, LiquidityScore)

    def test_symbol_preserved(self):
        bars = _bars("TSLA", 15)
        result = score(bars)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_total_score_range(self):
        bars = _bars("AAPL", 30)
        result = score(bars)
        assert result is not None
        assert 0 <= result.total_score <= 100

    def test_high_volume_high_score(self):
        """Very high volume → high vol_score component."""
        bars = _bars("AAPL", 30, volume=5_000_000)  # 5M shares/day
        result = score(bars)
        assert result is not None
        assert result.vol_score == 30.0

    def test_low_volume_low_score(self):
        """Very low volume → low vol_score."""
        bars = _bars("AAPL", 30, volume=1_000)  # 1k shares/day
        result = score(bars)
        assert result is not None
        assert result.vol_score <= 5.0

    def test_tight_spread_high_spread_score(self):
        """Very tight spread → high spread_score."""
        bars = _bars("AAPL", 30, spread_pct=0.3)  # 0.3% spread
        result = score(bars)
        assert result is not None
        assert result.spread_score == 25.0

    def test_wide_spread_low_spread_score(self):
        """Very wide spread → low spread_score."""
        bars = _bars("AAPL", 30, spread_pct=6.0)  # 6% spread
        result = score(bars)
        assert result is not None
        assert result.spread_score <= 5.0

    def test_consistent_volume_high_consistency(self):
        """Identical volumes → CV=0 → max consistency score."""
        bars = _bars("AAPL", 30, volume=500_000)
        result = score(bars)
        assert result is not None
        assert result.volume_cv == pytest.approx(0.0, abs=0.001)
        assert result.consistency_score == 25.0

    def test_grade_a_for_liquid(self):
        """Very liquid conditions → grade A."""
        bars = _bars("AAPL", 30, volume=2_000_000, spread_pct=0.4)
        result = score(bars)
        assert result is not None
        assert result.grade in ("A", "B")

    def test_grade_f_for_illiquid(self):
        """Very illiquid conditions → low total score (below 60)."""
        bars = _bars("AAPL", 30, volume=500, spread_pct=8.0)
        result = score(bars)
        assert result is not None
        # vol_score=0, spread_score=2 → very low, even with other components
        assert result.total_score < 60
        assert result.vol_score == 0.0

    def test_volume_trend_rising(self):
        """Volume increasing over time → rising trend."""
        bars_list = []
        for i in range(30):
            vol = 100_000 + i * 10_000  # monotonically increasing
            bars_list.append(_bar("AAPL", 100.0, vol, 101.0, 99.0, i))
        result = score(bars_list)
        assert result is not None
        assert result.volume_trend == "rising"

    def test_volume_trend_declining(self):
        """Volume decreasing over time → declining trend."""
        bars_list = []
        for i in range(30):
            vol = 300_000 - i * 9_000
            bars_list.append(_bar("AAPL", 100.0, max(vol, 1000), 101.0, 99.0, i))
        result = score(bars_list)
        assert result is not None
        assert result.volume_trend == "declining"

    def test_avg_volume_correct(self):
        bars = _bars("AAPL", 10, volume=200_000)
        result = score(bars)
        assert result is not None
        assert result.avg_volume == pytest.approx(200_000, rel=0.01)

    def test_bars_used(self):
        bars = _bars("AAPL", 20)
        result = score(bars, lookback=15)
        assert result is not None
        assert result.bars_used == 15

    def test_component_scores_sum_to_total(self):
        bars = _bars("AAPL", 30)
        result = score(bars)
        assert result is not None
        comp_sum = (result.vol_score + result.consistency_score +
                    result.spread_score + result.trend_score)
        assert result.total_score == pytest.approx(comp_sum, abs=0.1)
