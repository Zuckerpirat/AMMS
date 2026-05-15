"""Tests for performance attribution analysis."""

from __future__ import annotations

from amms.analysis.performance_attribution import AttributionReport, compute


class _Position:
    def __init__(self, sym: str, mv: float, pnl: float, qty: float, entry: float) -> None:
        self.symbol = sym
        self.market_value = mv
        self.unrealized_pl = pnl
        self.qty = qty
        self.avg_entry_price = entry


class _FakeBroker:
    def get_positions(self):
        return [
            _Position("AAPL", 5200.0, 200.0, 10.0, 500.0),
            _Position("MSFT", 4800.0, -100.0, 8.0, 612.5),
            _Position("NVDA", 3000.0, 500.0, 5.0, 500.0),
        ]


class _EmptyBroker:
    def get_positions(self):
        return []


class _ErrorBroker:
    def get_positions(self):
        raise RuntimeError("connection error")


class TestAttribution:
    def test_returns_attribution_report(self) -> None:
        result = compute(_FakeBroker())
        assert isinstance(result, AttributionReport)

    def test_row_count_matches_positions(self) -> None:
        result = compute(_FakeBroker())
        assert len(result.rows) == 3

    def test_weights_sum_to_100(self) -> None:
        result = compute(_FakeBroker())
        total_weight = sum(r.weight_pct for r in result.rows)
        assert abs(total_weight - 100.0) < 0.1

    def test_total_pnl_matches_sum(self) -> None:
        result = compute(_FakeBroker())
        expected = 200.0 + (-100.0) + 500.0
        assert abs(result.total_unrealized_pnl - expected) < 0.01

    def test_sorted_by_contribution_descending(self) -> None:
        result = compute(_FakeBroker())
        for i in range(len(result.rows) - 1):
            assert result.rows[i].contribution_pct >= result.rows[i + 1].contribution_pct

    def test_top_contributor_identified(self) -> None:
        result = compute(_FakeBroker())
        assert result.top_contributor is not None

    def test_top_detractor_identified(self) -> None:
        result = compute(_FakeBroker())
        assert result.top_detractor is not None

    def test_empty_positions(self) -> None:
        result = compute(_EmptyBroker())
        assert result.rows == []
        assert result.total_unrealized_pnl == 0.0
        assert result.total_market_value == 0.0

    def test_broker_error_returns_empty(self) -> None:
        result = compute(_ErrorBroker())
        assert result.rows == []

    def test_total_return_pct_is_float(self) -> None:
        result = compute(_FakeBroker())
        assert isinstance(result.total_return_pct, float)
