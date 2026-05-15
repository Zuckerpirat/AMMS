"""Tests for amms.analysis.relative_strength."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.relative_strength import RSRow, RSRanking, rank


def _bar(sym: str, close: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close * 1.01, close * 0.99, close, 100_000)


def _bars(sym: str, prices: list[float]) -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


def _flat(sym: str, n: int, p: float = 100.0) -> list[Bar]:
    return _bars(sym, [p] * n)


def _rising(sym: str, n: int, start: float = 100.0, step: float = 1.0) -> list[Bar]:
    return _bars(sym, [start + step * i for i in range(n)])


def _falling(sym: str, n: int, start: float = 120.0, step: float = 1.0) -> list[Bar]:
    return _bars(sym, [start - step * i for i in range(n)])


class TestRank:
    def test_returns_none_empty(self):
        assert rank({}) is None

    def test_returns_none_insufficient_bars(self):
        assert rank({"AAPL": [_bar("AAPL", 100.0)]}) is None

    def test_returns_result(self):
        result = rank({"AAPL": _rising("AAPL", 25)})
        assert result is not None
        assert isinstance(result, RSRanking)

    def test_rows_sorted_by_rs_score(self):
        bmap = {
            "A": _rising("A", 25, step=2.0),
            "B": _flat("B", 25),
            "C": _falling("C", 25, step=1.0),
        }
        result = rank(bmap)
        assert result is not None
        scores = [r.rs_score for r in result.rows]
        assert scores == sorted(scores, reverse=True)

    def test_leader_highest_score(self):
        bmap = {
            "A": _rising("A", 25, step=2.0),
            "B": _flat("B", 25),
        }
        result = rank(bmap)
        assert result is not None
        assert result.leader is not None
        assert result.leader.symbol == "A"

    def test_laggard_lowest_score(self):
        bmap = {
            "A": _rising("A", 25, step=2.0),
            "B": _falling("B", 25, step=1.0),
        }
        result = rank(bmap)
        assert result is not None
        assert result.laggard is not None
        assert result.laggard.symbol == "B"

    def test_rs_score_range(self):
        bmap = {"A": _rising("A", 25), "B": _flat("B", 25)}
        result = rank(bmap)
        assert result is not None
        for row in result.rows:
            assert 0 <= row.rs_score <= 100

    def test_outperforming_trend(self):
        """Symbol up much more than benchmark → outperforming."""
        bench = _flat("SPY", 25, p=100.0)
        bmap = {"A": _rising("A", 25, step=3.0)}
        result = rank(bmap, benchmark_bars=bench)
        assert result is not None
        assert result.rows[0].trend == "outperforming"

    def test_underperforming_trend(self):
        """Benchmark rising, symbol flat → underperforming."""
        bench = _rising("SPY", 25, step=3.0)
        bmap = {"A": _flat("A", 25)}
        result = rank(bmap, benchmark_bars=bench)
        assert result is not None
        assert result.rows[0].trend == "underperforming"

    def test_benchmark_return_used(self):
        bench = _rising("SPY", 25, step=1.0)  # ~24% over 25 bars from 100
        bmap = {"A": _flat("A", 25)}
        result = rank(bmap, benchmark_bars=bench)
        assert result is not None
        assert result.benchmark_return_pct != 0

    def test_portfolio_avg_benchmark_when_no_bench(self):
        """Without benchmark, bench_ret = avg of all symbol returns."""
        bmap = {
            "A": _rising("A", 25, step=2.0),
            "B": _falling("B", 25, step=2.0),
        }
        result = rank(bmap)
        assert result is not None
        # A and B move symmetrically → avg ~ 0
        assert abs(result.benchmark_return_pct) < 5.0

    def test_abs_return_correct(self):
        """Rising from 100 to 124 over 25 bars → ~24% abs return."""
        bmap = {"A": _rising("A", 25, start=100.0, step=1.0)}
        result = rank(bmap)
        assert result is not None
        assert result.rows[0].abs_return_pct == pytest.approx(18.1, abs=1.0)

    def test_lookback_respected(self):
        long_bars = _rising("A", 50, step=0.5)
        result = rank({"A": long_bars}, lookback=10)
        assert result is not None
        assert result.rows[0].bars_used == 10
        assert result.lookback == 10

    def test_single_symbol(self):
        """Single symbol → rs_score = 50 (only one peer)."""
        bmap = {"AAPL": _rising("AAPL", 25)}
        result = rank(bmap)
        assert result is not None
        assert result.rows[0].rs_score == pytest.approx(50.0, abs=0.1)
