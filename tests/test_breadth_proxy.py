"""Tests for amms.analysis.breadth_proxy."""

from __future__ import annotations

import pytest

from amms.analysis.breadth_proxy import BreadthProxyReport, SymbolBreadth, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


def _uptrend(n: int = 60, start: float = 100.0) -> list[_Bar]:
    price = start
    bars = []
    for i in range(n):
        bars.append(_Bar(price))
        price += 0.5
    return bars


def _downtrend(n: int = 60, start: float = 100.0) -> list[_Bar]:
    price = start
    bars = []
    for _ in range(n):
        bars.append(_Bar(price))
        price -= 0.3
    return bars


def _flat(n: int = 60, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _bull_portfolio(n_symbols: int = 5) -> dict[str, list[_Bar]]:
    return {f"SYM{i}": _uptrend(60) for i in range(n_symbols)}


def _bear_portfolio(n_symbols: int = 5) -> dict[str, list[_Bar]]:
    return {f"SYM{i}": _downtrend(60) for i in range(n_symbols)}


def _mixed_portfolio() -> dict[str, list[_Bar]]:
    return {
        "BULL1": _uptrend(60),
        "BULL2": _uptrend(60),
        "BEAR1": _downtrend(60),
        "FLAT1": _flat(60),
    }


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze({}) is None

    def test_returns_none_single_symbol(self):
        result = analyze({"AAPL": _uptrend(60)})
        assert result is None

    def test_returns_none_too_few_bars(self):
        result = analyze({"A": _uptrend(10), "B": _uptrend(10)})
        assert result is None

    def test_returns_result_valid_portfolio(self):
        result = analyze(_bull_portfolio(3))
        assert result is not None
        assert isinstance(result, BreadthProxyReport)


class TestPercentages:
    def test_pct_above_sma50_in_range(self):
        result = analyze(_mixed_portfolio())
        assert result is not None
        assert 0.0 <= result.pct_above_sma50 <= 100.0

    def test_pct_positive_roc20_in_range(self):
        result = analyze(_mixed_portfolio())
        assert result is not None
        assert 0.0 <= result.pct_positive_roc20 <= 100.0

    def test_pct_rsi_above_50_in_range(self):
        result = analyze(_mixed_portfolio())
        assert result is not None
        assert 0.0 <= result.pct_rsi_above_50 <= 100.0

    def test_bull_portfolio_high_pct_sma(self):
        result = analyze(_bull_portfolio(5))
        assert result is not None
        assert result.pct_above_sma50 > 50.0

    def test_bear_portfolio_low_pct_sma(self):
        result = analyze(_bear_portfolio(5))
        assert result is not None
        assert result.pct_above_sma50 <= 50.0


class TestBreadthScore:
    def test_breadth_score_in_range(self):
        result = analyze(_mixed_portfolio())
        assert result is not None
        assert 0.0 <= result.breadth_score <= 100.0

    def test_bull_portfolio_high_score(self):
        result = analyze(_bull_portfolio(5))
        assert result is not None
        assert result.breadth_score > 50.0

    def test_breadth_label_valid(self):
        result = analyze(_mixed_portfolio())
        assert result is not None
        assert result.breadth_label in {"broad_bull", "neutral", "mixed", "broad_bear"}


class TestCounts:
    def test_advance_decline_sum_to_n_eval(self):
        result = analyze(_bull_portfolio(4))
        assert result is not None
        assert result.advance_count + result.decline_count == result.n_evaluated

    def test_n_symbols_correct(self):
        portfolio = _bull_portfolio(4)
        result = analyze(portfolio)
        assert result is not None
        assert result.n_symbols == 4


class TestSymbolDetail:
    def test_symbols_list_not_empty(self):
        result = analyze(_bull_portfolio(3))
        assert result is not None
        assert len(result.symbols) > 0

    def test_symbols_are_symbol_breadth(self):
        result = analyze(_bull_portfolio(3))
        assert result is not None
        for s in result.symbols:
            assert isinstance(s, SymbolBreadth)

    def test_symbol_price_positive(self):
        result = analyze(_bull_portfolio(3))
        assert result is not None
        for s in result.symbols:
            assert s.current_price > 0


class TestThrustDetection:
    def test_no_thrust_by_default(self):
        result = analyze(_bull_portfolio(4))
        assert result is not None
        assert result.breadth_thrust is False
        assert result.thrust_direction == "none"

    def test_thrust_detected_when_big_shift(self):
        # Old: all bearish. New: all bullish → big shift
        old_snap = _bear_portfolio(5)
        new_snap = _bull_portfolio(5)
        result = analyze(new_snap, history_snapshots=[old_snap])
        assert result is not None
        assert result.breadth_thrust is True
        assert result.thrust_direction == "bullish"


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_bull_portfolio(3))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_breadth(self):
        result = analyze(_bull_portfolio(3))
        assert result is not None
        assert "breadth" in result.verdict.lower()
