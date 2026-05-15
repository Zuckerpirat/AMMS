"""Tests for amms.analysis.market_breadth."""

from __future__ import annotations

import pytest

from amms.analysis.market_breadth import BreadthStats, analyze_breadth
from amms.data.bars import Bar


def _bar(sym: str, close: float, i: int = 0, high_offset: float = 1.0, low_offset: float = 1.0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close + high_offset, close - low_offset, close, 10_000)


def _bars(sym: str, prices: list[float]) -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


class _GoodBroker:
    def get_positions(self):
        class P:
            def __init__(self, sym):
                self.symbol = sym
        return [P("AAPL"), P("MSFT")]


class _EmptyBroker:
    def get_positions(self):
        return []


class _ErrorBroker:
    def get_positions(self):
        raise RuntimeError("broker down")


class _GoodDataClient:
    """Returns 25 bars trending upward (price above VWAP and SMA)."""
    def get_bars(self, symbol, *, limit=30):
        # Rising series: price at end is above SMA20 and VWAP
        prices = [100.0 + i * 0.5 for i in range(25)]
        return _bars(symbol, prices)


class _BadDataClient:
    """Returns 25 bars trending downward."""
    def get_bars(self, symbol, *, limit=30):
        prices = [150.0 - i * 0.5 for i in range(25)]
        return _bars(symbol, prices)


class _ShortDataClient:
    """Returns only 3 bars — insufficient."""
    def get_bars(self, symbol, *, limit=30):
        return _bars(symbol, [100.0, 101.0, 99.0])


class TestAnalyzeBreadth:
    def test_returns_none_no_positions(self):
        result = analyze_breadth(_EmptyBroker(), _GoodDataClient())
        assert result is None

    def test_returns_none_broker_error(self):
        result = analyze_breadth(_ErrorBroker(), _GoodDataClient())
        assert result is None

    def test_returns_none_all_data_too_short(self):
        result = analyze_breadth(_GoodBroker(), _ShortDataClient())
        assert result is None

    def test_returns_breadth_stats(self):
        result = analyze_breadth(_GoodBroker(), _GoodDataClient())
        assert result is not None
        assert isinstance(result, BreadthStats)

    def test_n_positions_correct(self):
        result = analyze_breadth(_GoodBroker(), _GoodDataClient())
        assert result is not None
        assert result.n_positions == 2

    def test_percentages_in_range(self):
        result = analyze_breadth(_GoodBroker(), _GoodDataClient())
        assert result is not None
        assert 0.0 <= result.pct_above_vwap <= 100.0
        assert 0.0 <= result.pct_rsi_above_50 <= 100.0
        assert 0.0 <= result.pct_above_sma20 <= 100.0
        assert 0.0 <= result.pct_obv_rising <= 100.0
        assert 0.0 <= result.overall_score <= 100.0

    def test_verdict_is_valid(self):
        result = analyze_breadth(_GoodBroker(), _GoodDataClient())
        assert result is not None
        assert result.verdict in {"strong", "moderate", "weak", "deteriorating"}

    def test_strong_verdict_for_uptrend(self):
        result = analyze_breadth(_GoodBroker(), _GoodDataClient())
        assert result is not None
        assert result.verdict in {"strong", "moderate"}  # uptrend data

    def test_weak_verdict_for_downtrend(self):
        result = analyze_breadth(_GoodBroker(), _BadDataClient())
        assert result is not None
        assert result.verdict in {"weak", "deteriorating", "moderate"}

    def test_detail_lines_populated(self):
        result = analyze_breadth(_GoodBroker(), _GoodDataClient())
        assert result is not None
        assert len(result.detail) == 2
        # Each detail line should mention the symbol
        assert any("AAPL" in line for line in result.detail)
        assert any("MSFT" in line for line in result.detail)

    def test_handles_data_error_gracefully(self):
        class _ErrorData:
            def get_bars(self, symbol, *, limit=30):
                raise RuntimeError("data down")
        result = analyze_breadth(_GoodBroker(), _ErrorData())
        # All positions fail → total=0 → None
        assert result is None
