"""Tests for amms.analysis.market_internals."""

from __future__ import annotations

import pytest

from amms.analysis.market_internals import MarketInternalsReport, SymbolInternals, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0):
        self.close = close
        self.high = close + spread
        self.low = close - spread


def _uptrend_bars(n: int = 80, start: float = 50.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price += step
    return bars


def _downtrend_bars(n: int = 80, start: float = 200.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(max(price, 1.0)))
        price -= step
    return bars


def _flat_bars(n: int = 80, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _bull_basket() -> dict:
    return {
        "A": _uptrend_bars(80),
        "B": _uptrend_bars(80, step=0.8),
        "C": _uptrend_bars(80, step=0.6),
        "D": _uptrend_bars(80, step=0.4),
        "E": _uptrend_bars(80, step=0.3),
    }


def _bear_basket() -> dict:
    return {
        "A": _downtrend_bars(80),
        "B": _downtrend_bars(80, step=0.8),
        "C": _downtrend_bars(80, step=0.6),
        "D": _downtrend_bars(80, step=0.4),
        "E": _downtrend_bars(80, step=0.3),
    }


def _mixed_basket() -> dict:
    return {
        "UP1": _uptrend_bars(80),
        "UP2": _uptrend_bars(80, step=0.5),
        "DN1": _downtrend_bars(80),
        "FL1": _flat_bars(80),
    }


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze({}) is None

    def test_returns_none_insufficient_bars(self):
        result = analyze({"X": _flat_bars(30)})
        assert result is None

    def test_returns_result_enough_bars(self):
        result = analyze({"X": _flat_bars(60)})
        assert result is not None
        assert isinstance(result, MarketInternalsReport)


class TestComposite:
    def test_composite_in_range(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert 0.0 <= result.composite_score <= 100.0

    def test_bull_basket_high_score(self):
        result = analyze(_bull_basket())
        assert result is not None
        assert result.composite_score > 50.0

    def test_bear_basket_low_score(self):
        result = analyze(_bear_basket())
        assert result is not None
        assert result.composite_score < 50.0


class TestHealthLabel:
    def test_health_label_valid(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert result.health_label in {"strong_bull", "bull", "neutral", "bear", "strong_bear"}

    def test_bull_basket_bullish_label(self):
        result = analyze(_bull_basket())
        assert result is not None
        assert result.health_label in {"strong_bull", "bull"}

    def test_bear_basket_bearish_label(self):
        result = analyze(_bear_basket())
        assert result is not None
        assert result.health_label in {"strong_bear", "bear"}


class TestPercentages:
    def test_pct_in_range(self):
        result = analyze(_mixed_basket())
        assert result is not None
        for pct in [result.pct_above_sma20, result.pct_above_sma50, result.pct_above_sma200,
                    result.pct_new_highs, result.pct_new_lows]:
            assert 0.0 <= pct <= 100.0

    def test_bull_high_pct_above_sma50(self):
        result = analyze(_bull_basket())
        assert result is not None
        assert result.pct_above_sma50 > 50.0

    def test_bear_low_pct_above_sma50(self):
        result = analyze(_bear_basket())
        assert result is not None
        assert result.pct_above_sma50 < 50.0


class TestRatios:
    def test_ad_ratio_in_range(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert 0.0 <= result.advance_decline_ratio <= 1.0

    def test_nh_nl_ratio_in_range(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert 0.0 <= result.nh_nl_ratio <= 1.0

    def test_bull_high_ad_ratio(self):
        result = analyze(_bull_basket())
        assert result is not None
        assert result.advance_decline_ratio > 0.5


class TestBySymbol:
    def test_by_symbol_not_empty(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert len(result.by_symbol) > 0

    def test_by_symbol_are_symbol_internals(self):
        result = analyze(_mixed_basket())
        assert result is not None
        for s in result.by_symbol:
            assert isinstance(s, SymbolInternals)

    def test_symbol_scores_in_range(self):
        result = analyze(_mixed_basket())
        assert result is not None
        for s in result.by_symbol:
            assert 0.0 <= s.score <= 100.0

    def test_symbol_trend_valid(self):
        result = analyze(_mixed_basket())
        assert result is not None
        for s in result.by_symbol:
            assert s.trend in {"bull", "neutral", "bear"}

    def test_symbols_scored_count(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert result.symbols_scored == len(result.by_symbol)


class TestCounts:
    def test_bull_bear_neutral_sum(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert result.symbols_bull + result.symbols_bear + result.symbols_neutral == result.symbols_scored


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_mixed_basket())
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_internals(self):
        result = analyze(_mixed_basket())
        assert result is not None
        text = result.verdict.lower()
        assert "internals" in text or "sma" in text or "symbol" in text
