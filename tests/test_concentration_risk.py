"""Tests for amms.analysis.concentration_risk."""

from __future__ import annotations

import pytest

from amms.analysis.concentration_risk import (
    ConcentrationReport, PositionWeight, analyze,
)


def _make_broker(positions: list[tuple[str, float]]):
    """positions: list of (symbol, market_value)"""
    class Pos:
        def __init__(self, sym, mv):
            self.symbol = sym
            self.market_value = mv
    class Broker:
        def get_positions(self):
            return [Pos(s, mv) for s, mv in positions]
    return Broker()


class TestAnalyze:
    def test_returns_none_no_positions(self):
        broker = _make_broker([])
        assert analyze(broker) is None

    def test_returns_none_broker_error(self):
        class BadBroker:
            def get_positions(self):
                raise RuntimeError("down")
        assert analyze(BadBroker()) is None

    def test_returns_report(self):
        broker = _make_broker([("AAPL", 10000), ("TSLA", 5000)])
        result = analyze(broker)
        assert result is not None
        assert isinstance(result, ConcentrationReport)

    def test_positions_sorted_by_weight(self):
        broker = _make_broker([("TSLA", 5000), ("AAPL", 10000), ("NVDA", 3000)])
        result = analyze(broker)
        assert result is not None
        weights = [p.weight_pct for p in result.positions]
        assert weights == sorted(weights, reverse=True)

    def test_weights_sum_to_100(self):
        broker = _make_broker([("A", 1000), ("B", 2000), ("C", 3000)])
        result = analyze(broker)
        assert result is not None
        total = sum(p.weight_pct for p in result.positions)
        assert total == pytest.approx(100.0, abs=0.1)

    def test_hhi_range(self):
        broker = _make_broker([("A", 1000), ("B", 1000), ("C", 1000)])
        result = analyze(broker)
        assert result is not None
        assert 0 < result.hhi <= 1.0

    def test_hhi_equal_weights(self):
        """N equal positions → HHI = 1/N."""
        n = 5
        broker = _make_broker([(f"S{i}", 1000) for i in range(n)])
        result = analyze(broker)
        assert result is not None
        assert result.hhi == pytest.approx(1 / n, abs=0.001)

    def test_hhi_single_position(self):
        """Single position → HHI = 1."""
        broker = _make_broker([("AAPL", 10000)])
        result = analyze(broker)
        assert result is not None
        assert result.hhi == pytest.approx(1.0, abs=0.001)

    def test_effective_n_correct(self):
        """effective_n = 1/HHI."""
        broker = _make_broker([("A", 1000), ("B", 1000)])
        result = analyze(broker)
        assert result is not None
        assert result.effective_n == pytest.approx(1 / result.hhi, abs=0.1)

    def test_top1_pct_correct(self):
        broker = _make_broker([("AAPL", 7500), ("TSLA", 2500)])
        result = analyze(broker)
        assert result is not None
        assert result.top1_pct == pytest.approx(75.0, abs=0.1)

    def test_top3_pct_correct(self):
        broker = _make_broker([("A", 4000), ("B", 3000), ("C", 2000), ("D", 1000)])
        result = analyze(broker)
        assert result is not None
        assert result.top3_pct == pytest.approx(90.0, abs=0.1)

    def test_grade_a_well_diversified(self):
        """10 equal positions → HHI=0.1 → grade B (boundary)."""
        broker = _make_broker([(f"S{i}", 1000) for i in range(20)])
        result = analyze(broker)
        assert result is not None
        assert result.grade in ("A", "B")

    def test_grade_f_single_position(self):
        broker = _make_broker([("AAPL", 10000)])
        result = analyze(broker)
        assert result is not None
        assert result.grade == "F"

    def test_risk_flag_large_position(self):
        """Position > 30% triggers flag."""
        broker = _make_broker([("AAPL", 8000), ("TSLA", 1000), ("NVDA", 1000)])
        result = analyze(broker)
        assert result is not None
        assert any("AAPL" in f for f in result.risk_flags)

    def test_risk_flag_few_positions(self):
        """< 5 positions triggers flag."""
        broker = _make_broker([("A", 1000), ("B", 1000)])
        result = analyze(broker)
        assert result is not None
        assert any("positions" in f.lower() for f in result.risk_flags)

    def test_no_flags_well_diversified(self):
        """20 equal positions → no risk flags."""
        broker = _make_broker([(f"S{i}", 500) for i in range(20)])
        result = analyze(broker)
        assert result is not None
        assert result.risk_flags == []

    def test_total_value_correct(self):
        broker = _make_broker([("A", 3000), ("B", 7000)])
        result = analyze(broker)
        assert result is not None
        assert result.total_value == pytest.approx(10000.0, abs=1.0)

    def test_n_positions_correct(self):
        broker = _make_broker([("A", 1000), ("B", 2000), ("C", 3000)])
        result = analyze(broker)
        assert result is not None
        assert result.n_positions == 3

    def test_verdict_present(self):
        broker = _make_broker([("A", 1000)])
        result = analyze(broker)
        assert result is not None
        assert len(result.verdict) > 0
