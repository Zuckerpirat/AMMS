"""Tests for amms.analysis.rolling_sharpe."""

from __future__ import annotations

import pytest

from amms.analysis.rolling_sharpe import RollingSharpeReport, SharpeSnapshot, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


def _flat(n: int = 50, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _trending_up(n: int = 60, start: float = 50.0, step: float = 0.3) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price += step
    return bars


def _volatile(n: int = 60, price: float = 100.0) -> list[_Bar]:
    """Large daily swings."""
    bars = []
    p = price
    for i in range(n):
        p += 5.0 * (1 if i % 2 == 0 else -1)
        bars.append(_Bar(max(p, 1.0)))
    return bars


def _declining(n: int = 60, start: float = 200.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(max(price, 1.0)))
        price -= step
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(20)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(40))
        assert result is not None
        assert isinstance(result, RollingSharpeReport)


class TestSharpeValues:
    def test_trending_up_positive_sharpe(self):
        result = analyze(_trending_up(60))
        assert result is not None
        assert result.current_sharpe > 0

    def test_declining_negative_sharpe(self):
        result = analyze(_declining(60))
        assert result is not None
        assert result.current_sharpe < 0

    def test_volatile_lower_sharpe_than_smooth(self):
        smooth = analyze(_trending_up(60))
        vol = analyze(_volatile(60))
        if smooth and vol:
            assert smooth.current_sharpe > vol.current_sharpe

    def test_sortino_non_negative_for_uptrend(self):
        result = analyze(_trending_up(60))
        assert result is not None
        assert result.current_sortino >= 0


class TestStats:
    def test_max_sharpe_ge_current(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.max_sharpe >= result.current_sharpe or abs(result.max_sharpe - result.current_sharpe) < 0.01

    def test_min_sharpe_le_current(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.min_sharpe <= result.current_sharpe or abs(result.min_sharpe - result.current_sharpe) < 0.01

    def test_sharpe_percentile_in_range(self):
        result = analyze(_flat(50))
        assert result is not None
        assert 0.0 <= result.sharpe_percentile <= 100.0

    def test_max_drawdown_non_negative(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.max_drawdown_pct >= 0

    def test_current_vol_non_negative(self):
        result = analyze(_trending_up(60))
        assert result is not None
        assert result.current_vol >= 0


class TestTrend:
    def test_sharpe_trend_valid(self):
        for bars in [_flat(50), _trending_up(60), _declining(60)]:
            result = analyze(bars)
            if result:
                assert result.sharpe_trend in {"improving", "worsening", "stable"}


class TestHistory:
    def test_history_not_empty(self):
        result = analyze(_flat(50))
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_snapshots(self):
        result = analyze(_flat(50))
        assert result is not None
        for s in result.history:
            assert isinstance(s, SharpeSnapshot)

    def test_history_vol_non_negative(self):
        result = analyze(_trending_up(60))
        assert result is not None
        for s in result.history:
            assert s.vol >= 0


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(50)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 50

    def test_window_stored(self):
        result = analyze(_flat(50), window=20)
        assert result is not None
        assert result.window == 20

    def test_symbol_stored(self):
        result = analyze(_flat(50), symbol="BRK")
        assert result is not None
        assert result.symbol == "BRK"

    def test_current_price_correct(self):
        result = analyze(_flat(50, price=200.0))
        assert result is not None
        assert abs(result.current_price - 200.0) < 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(40))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_sharpe(self):
        result = analyze(_flat(40))
        assert result is not None
        assert "sharpe" in result.verdict.lower()
