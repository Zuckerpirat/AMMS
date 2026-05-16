"""Tests for amms.analysis.sector_exposure."""

from __future__ import annotations

import pytest

from amms.analysis.sector_exposure import (
    SectorExposureReport, SectorWeight, analyze, SYMBOL_SECTOR,
)


def _make_broker(positions: list[tuple[str, float]]):
    class Pos:
        def __init__(self, sym, mv):
            self.symbol = sym
            self.market_value = mv

    class Broker:
        def get_positions(self):
            return [Pos(s, mv) for s, mv in positions]

    return Broker()


class TestEdgeCases:
    def test_returns_none_empty(self):
        broker = _make_broker([])
        assert analyze(broker) is None

    def test_returns_none_broker_error(self):
        class Bad:
            def get_positions(self):
                raise RuntimeError

        assert analyze(Bad()) is None

    def test_returns_result(self):
        broker = _make_broker([("AAPL", 10000), ("MSFT", 5000)])
        result = analyze(broker)
        assert result is not None
        assert isinstance(result, SectorExposureReport)


class TestSectorMapping:
    def test_known_symbols_mapped(self):
        broker = _make_broker([("AAPL", 5000), ("JPM", 5000)])
        result = analyze(broker)
        assert result is not None
        sectors = {s.sector for s in result.sectors}
        assert "Technology" in sectors
        assert "Financials" in sectors

    def test_unknown_symbol_goes_to_unknown(self):
        broker = _make_broker([("UNKN", 5000), ("AAPL", 5000)])
        result = analyze(broker)
        assert result is not None
        sectors = {s.sector for s in result.sectors}
        assert "Unknown" in sectors

    def test_weights_sum_to_100(self):
        broker = _make_broker([("AAPL", 3000), ("JPM", 4000), ("XOM", 3000)])
        result = analyze(broker)
        assert result is not None
        total = sum(s.weight_pct for s in result.sectors)
        assert total == pytest.approx(100.0, abs=0.5)

    def test_sorted_by_weight_desc(self):
        broker = _make_broker([("AAPL", 1000), ("JPM", 5000), ("XOM", 3000)])
        result = analyze(broker)
        assert result is not None
        weights = [s.weight_pct for s in result.sectors]
        assert weights == sorted(weights, reverse=True)


class TestWeights:
    def test_dominant_sector_correct(self):
        """Single large tech position → dominant = Technology."""
        broker = _make_broker([("AAPL", 9000), ("XOM", 1000)])
        result = analyze(broker)
        assert result is not None
        assert result.dominant_sector == "Technology"

    def test_n_positions_correct(self):
        broker = _make_broker([("AAPL", 5000), ("MSFT", 3000), ("JPM", 2000)])
        result = analyze(broker)
        assert result is not None
        assert result.n_positions == 3

    def test_symbols_grouped_in_sector(self):
        """Two tech stocks → Technology sector has both symbols."""
        broker = _make_broker([("AAPL", 5000), ("MSFT", 5000)])
        result = analyze(broker)
        assert result is not None
        tech = next(s for s in result.sectors if s.sector == "Technology")
        assert "AAPL" in tech.symbols
        assert "MSFT" in tech.symbols

    def test_n_positions_in_sector(self):
        broker = _make_broker([("AAPL", 3000), ("MSFT", 3000), ("JPM", 4000)])
        result = analyze(broker)
        assert result is not None
        tech = next(s for s in result.sectors if s.sector == "Technology")
        assert tech.n_positions == 2


class TestHHIAndRisk:
    def test_hhi_single_sector(self):
        """All in one sector → HHI ≈ 1."""
        broker = _make_broker([("AAPL", 5000), ("MSFT", 5000)])
        result = analyze(broker)
        assert result is not None
        assert result.portfolio_hhi > 0.8

    def test_hhi_equal_sectors(self):
        """Two equal sectors → HHI ≈ 0.5."""
        broker = _make_broker([("AAPL", 5000), ("JPM", 5000)])
        result = analyze(broker)
        assert result is not None
        assert result.portfolio_hhi == pytest.approx(0.5, abs=0.05)

    def test_risk_flag_overconcentrated(self):
        """80% in tech → overconcentration flag."""
        broker = _make_broker([("AAPL", 8000), ("XOM", 1000), ("JNJ", 1000)])
        result = analyze(broker)
        assert result is not None
        assert any("Technology" in f for f in result.risk_flags)

    def test_unknown_pct_correct(self):
        broker = _make_broker([("UNKN1", 5000), ("UNKN2", 5000)])
        result = analyze(broker)
        assert result is not None
        assert result.unknown_pct == pytest.approx(100.0, abs=1.0)

    def test_verdict_present(self):
        broker = _make_broker([("AAPL", 5000), ("JPM", 5000)])
        result = analyze(broker)
        assert result is not None
        assert len(result.verdict) > 5

    def test_benchmark_weight_in_sector(self):
        """Tech sector has non-zero benchmark weight."""
        broker = _make_broker([("AAPL", 10000)])
        result = analyze(broker)
        assert result is not None
        tech = next(s for s in result.sectors if s.sector == "Technology")
        assert tech.benchmark_weight_pct > 0

    def test_active_weight_computed(self):
        """Active weight = portfolio weight - benchmark weight."""
        broker = _make_broker([("AAPL", 10000)])
        result = analyze(broker)
        assert result is not None
        tech = next(s for s in result.sectors if s.sector == "Technology")
        expected = tech.weight_pct - tech.benchmark_weight_pct
        assert tech.active_weight_pct == pytest.approx(expected, abs=0.01)
