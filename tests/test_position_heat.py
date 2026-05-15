"""Tests for amms.analysis.position_heat."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.position_heat import (
    PositionHeat, HeatReport, score_position, analyze,
    _pnl_score, _momentum_score_from_bars, _drawdown_score_from_bars,
    _liquidity_score_from_bars,
)


def _bar(sym: str, close: float, volume: float = 1_000_000, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close * 1.01, close * 0.99, close, volume)


def _bars(sym: str, n: int, start: float = 100.0, step: float = 0.0,
          volume: float = 1_000_000) -> list[Bar]:
    return [_bar(sym, start + step * i, volume, i) for i in range(n)]


class TestPnlScore:
    def test_big_gain(self):
        assert _pnl_score(12.0) == 30.0

    def test_moderate_gain(self):
        assert _pnl_score(7.0) == 25.0

    def test_small_gain(self):
        assert _pnl_score(3.0) == 20.0

    def test_breakeven(self):
        assert _pnl_score(0.5) == 15.0

    def test_small_loss(self):
        assert _pnl_score(-1.5) == 10.0

    def test_big_loss(self):
        assert _pnl_score(-15.0) == 0.0


class TestMomentumScore:
    def test_neutral_when_insufficient(self):
        bars = _bars("AAPL", 5)
        assert _momentum_score_from_bars(bars) == 12.5

    def test_high_score_rising(self):
        bars = _bars("AAPL", 15, step=1.0)
        score = _momentum_score_from_bars(bars)
        assert score >= 16.0

    def test_low_score_falling(self):
        bars = _bars("AAPL", 15, start=120.0, step=-1.0)
        score = _momentum_score_from_bars(bars)
        assert score <= 8.0


class TestDrawdownScore:
    def test_max_score_at_peak(self):
        bars = _bars("AAPL", 20, step=0.5)  # monotonically rising
        score = _drawdown_score_from_bars(bars)
        assert score == 25.0

    def test_low_score_big_drawdown(self):
        # Peak at bar 10, then big drop
        prices = [100.0] * 10 + [60.0] * 10
        bars = [_bar("AAPL", p, i=i) for i, p in enumerate(prices)]
        score = _drawdown_score_from_bars(bars)
        assert score <= 3.0

    def test_neutral_when_few_bars(self):
        bars = [_bar("AAPL", 100.0)]
        assert _drawdown_score_from_bars(bars) == 12.5


class TestLiquidityScore:
    def test_high_score_high_vol(self):
        bars = _bars("AAPL", 20, volume=5_000_000)
        assert _liquidity_score_from_bars(bars) == 20.0

    def test_low_score_low_vol(self):
        bars = _bars("AAPL", 20, volume=5_000)
        assert _liquidity_score_from_bars(bars) <= 4.0

    def test_neutral_when_empty(self):
        assert _liquidity_score_from_bars([]) == 10.0


class TestScorePosition:
    def test_returns_position_heat(self):
        bars = _bars("AAPL", 20, step=0.5)
        result = score_position("AAPL", 5.0, bars)
        assert isinstance(result, PositionHeat)

    def test_symbol_preserved(self):
        bars = _bars("TSLA", 20)
        result = score_position("TSLA", 0.0, bars)
        assert result.symbol == "TSLA"

    def test_pnl_pct_preserved(self):
        bars = _bars("AAPL", 20)
        result = score_position("AAPL", 7.5, bars)
        assert result.pnl_pct == pytest.approx(7.5, abs=0.01)

    def test_score_range(self):
        bars = _bars("AAPL", 20)
        result = score_position("AAPL", 0.0, bars)
        assert 0 <= result.score <= 100

    def test_hot_status_high_score(self):
        """Best-case: big gain + rising + at peak + liquid → hot."""
        bars = _bars("AAPL", 20, step=1.0, volume=5_000_000)
        result = score_position("AAPL", 15.0, bars)
        assert result.status in ("hot", "warm")
        assert result.score >= 60

    def test_cold_status_low_score(self):
        """Worst-case: big loss + falling + big drawdown + illiquid → cold."""
        prices = [100.0] * 10 + [60.0] * 10
        bars = [_bar("AAPL", p, volume=1000, i=i) for i, p in enumerate(prices)]
        result = score_position("AAPL", -20.0, bars)
        assert result.status in ("cold", "cool")
        assert result.score < 40

    def test_component_scores_sum_to_total(self):
        bars = _bars("AAPL", 20, step=0.5)
        result = score_position("AAPL", 5.0, bars)
        comp_sum = (result.pnl_score + result.momentum_score +
                    result.drawdown_score + result.liquidity_score)
        assert result.score == pytest.approx(comp_sum, abs=0.1)

    def test_bars_used_correct(self):
        bars = _bars("AAPL", 15)
        result = score_position("AAPL", 0.0, bars)
        assert result.bars_used == 15


class TestAnalyze:
    def test_returns_none_no_positions(self):
        class Broker:
            def get_positions(self):
                return []
        class Data:
            def get_bars(self, sym, *, limit=30):
                return _bars(sym, 20)
        assert analyze(Broker(), Data()) is None

    def test_returns_report_with_positions(self):
        class Pos:
            symbol = "AAPL"
            avg_entry_price = 100.0
            market_value = 1050.0
            qty = 10.0
        class Broker:
            def get_positions(self):
                return [Pos()]
        class Data:
            def get_bars(self, sym, *, limit=30):
                return _bars(sym, 20, step=0.5)
        result = analyze(Broker(), Data())
        assert result is not None
        assert isinstance(result, HeatReport)

    def test_hottest_coldest_ordering(self):
        class Pos:
            def __init__(self, sym, entry, mv, qty):
                self.symbol = sym
                self.avg_entry_price = entry
                self.market_value = mv
                self.qty = qty
        class Broker:
            def get_positions(self):
                return [
                    Pos("AAPL", 100.0, 1200.0, 10.0),  # +20%
                    Pos("TSLA", 100.0, 700.0, 10.0),   # -30%
                ]
        class Data:
            def get_bars(self, sym, *, limit=30):
                if sym == "AAPL":
                    return _bars(sym, 20, step=1.0, volume=5_000_000)
                return _bars(sym, 20, start=100.0, step=-1.0, volume=1_000)
        result = analyze(Broker(), Data())
        assert result is not None
        assert result.hottest is not None
        assert result.coldest is not None
        assert result.hottest.score >= result.coldest.score

    def test_positions_sorted_by_score(self):
        class Pos:
            def __init__(self, sym, pnl_mult):
                self.symbol = sym
                self.avg_entry_price = 100.0
                self.market_value = 100.0 * pnl_mult
                self.qty = 1.0
        class Broker:
            def get_positions(self):
                return [Pos("A", 1.15), Pos("B", 0.85), Pos("C", 1.05)]
        class Data:
            def get_bars(self, sym, *, limit=30):
                return _bars(sym, 20, volume=500_000)
        result = analyze(Broker(), Data())
        assert result is not None
        scores = [p.score for p in result.positions]
        assert scores == sorted(scores, reverse=True)

    def test_n_hot_and_cold_count(self):
        class Pos:
            def __init__(self, sym, pnl):
                self.symbol = sym
                self.avg_entry_price = 100.0
                self.market_value = 100.0 + pnl
                self.qty = 1.0
        class Broker:
            def get_positions(self):
                return [Pos("A", 20), Pos("B", -20)]
        class Data:
            def get_bars(self, sym, *, limit=30):
                return _bars(sym, 20, volume=2_000_000)
        result = analyze(Broker(), Data())
        assert result is not None
        assert result.n_hot + result.n_cold <= 2
