"""Tests for amms.analysis.watchlist_scorer."""

from __future__ import annotations

import pytest

from amms.analysis.watchlist_scorer import (
    WatchlistScore,
    WatchlistScorerReport,
    score_symbol,
    score_many,
)


class _Bar:
    def __init__(self, open_, high, low, close, volume=100_000):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price, price + 0.4, price - 0.2, price + 0.3, 120_000))
        price += step
    return bars


def _trending_down(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price, price + 0.2, price - 0.4, price - 0.3, 80_000))
        price -= step
    return bars


def _flat_bars(n: int = 60, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price, price + 0.5, price - 0.5, price, 100_000) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert score_symbol([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _flat_bars(20)
        assert score_symbol(bars) is None

    def test_returns_score_with_enough_bars(self):
        bars = _flat_bars(30)
        result = score_symbol(bars)
        assert result is not None
        assert isinstance(result, WatchlistScore)

    def test_returns_none_zero_price(self):
        bars = [_Bar(0, 0, 0, 0)] * 30
        assert score_symbol(bars) is None


class TestScoreRange:
    def test_total_score_in_range(self):
        bars = _trending_up(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.total_score <= 100.0

    def test_subscore_momentum_in_range(self):
        bars = _trending_up(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.momentum_score <= 100.0

    def test_subscore_rsi_in_range(self):
        bars = _flat_bars(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.rsi_score <= 100.0

    def test_subscore_volume_in_range(self):
        bars = _flat_bars(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.volume_score <= 100.0

    def test_subscore_sma_in_range(self):
        bars = _flat_bars(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.sma_score <= 100.0

    def test_subscore_vol_in_range(self):
        bars = _flat_bars(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.vol_score <= 100.0

    def test_subscore_trend_in_range(self):
        bars = _flat_bars(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.trend_score <= 100.0


class TestGrades:
    def test_grade_is_letter(self):
        bars = _flat_bars(60)
        result = score_symbol(bars)
        assert result is not None
        assert result.grade in {"A", "B", "C", "D", "F"}

    def test_grade_a_requires_high_score(self):
        # Build bars with strong uptrend and rising volume
        bars = _trending_up(80, step=1.0)
        result = score_symbol(bars)
        assert result is not None
        if result.grade == "A":
            assert result.total_score >= 80.0

    def test_grade_consistent_with_score(self):
        for bars in [_trending_up(60), _trending_down(60), _flat_bars(60)]:
            result = score_symbol(bars)
            if result is None:
                continue
            s = result.total_score
            expected = (
                "A" if s >= 80 else
                "B" if s >= 65 else
                "C" if s >= 50 else
                "D" if s >= 35 else "F"
            )
            assert result.grade == expected


class TestSymbol:
    def test_symbol_stored(self):
        bars = _flat_bars(30)
        result = score_symbol(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_symbol_default_empty(self):
        bars = _flat_bars(30)
        result = score_symbol(bars)
        assert result is not None
        assert result.symbol == ""


class TestMetrics:
    def test_current_price_positive(self):
        bars = _trending_up(40)
        result = score_symbol(bars)
        assert result is not None
        assert result.current_price > 0

    def test_rsi_in_range(self):
        bars = _flat_bars(60)
        result = score_symbol(bars)
        assert result is not None
        assert 0.0 <= result.rsi <= 100.0

    def test_bars_used_matches_input(self):
        bars = _trending_up(50)
        result = score_symbol(bars)
        assert result is not None
        assert result.bars_used == 50

    def test_summary_nonempty(self):
        bars = _flat_bars(30)
        result = score_symbol(bars)
        assert result is not None
        assert len(result.summary) > 10


class TestScoreMany:
    def test_empty_dict_returns_report(self):
        report = score_many({})
        assert isinstance(report, WatchlistScorerReport)
        assert report.n_symbols == 0
        assert report.top_pick is None

    def test_single_symbol(self):
        bars = _trending_up(60)
        report = score_many({"AAPL": bars})
        assert report.n_symbols == 1
        assert report.n_graded == 1
        assert report.top_pick is not None
        assert report.top_pick.symbol == "AAPL"

    def test_multiple_symbols_sorted(self):
        bars_up = _trending_up(60, step=1.0)
        bars_down = _trending_down(60, step=1.0)
        bars_flat = _flat_bars(60)
        report = score_many({"UP": bars_up, "DOWN": bars_down, "FLAT": bars_flat})
        assert report.n_symbols == 3
        scores = [s.total_score for s in report.scores]
        assert scores == sorted(scores, reverse=True)

    def test_too_few_bars_excluded(self):
        bars_ok = _trending_up(60)
        bars_bad = _flat_bars(10)
        report = score_many({"OK": bars_ok, "BAD": bars_bad})
        assert report.n_graded == 1
        assert report.top_pick is not None
        assert report.top_pick.symbol == "OK"

    def test_top_pick_is_highest_score(self):
        bars_up = _trending_up(60, step=2.0)
        bars_flat = _flat_bars(60)
        report = score_many({"FAST": bars_up, "SLOW": bars_flat})
        assert report.top_pick is not None
        assert report.top_pick.total_score == max(s.total_score for s in report.scores)
