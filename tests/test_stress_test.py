"""Tests for portfolio stress test."""

from __future__ import annotations

from amms.analysis.stress_test import SCENARIOS, StressResult, stress_test


class _Position:
    def __init__(self, sym, mv):
        self.symbol = sym
        self.market_value = mv
        self.unrealized_pl = 0.0
        self.qty = 10.0
        self.avg_entry_price = mv / 10.0


class _FakeBroker:
    def get_positions(self):
        return [_Position("AAPL", 5000.0), _Position("MSFT", 3000.0)]


class _EmptyBroker:
    def get_positions(self):
        return []


class _ErrorBroker:
    def get_positions(self):
        raise RuntimeError("error")


class TestStressTest:
    def test_returns_result_for_valid_scenario(self) -> None:
        result = stress_test(_FakeBroker(), "2008_crisis")
        assert result is not None
        assert isinstance(result, StressResult)

    def test_total_loss_negative_for_crash(self) -> None:
        result = stress_test(_FakeBroker(), "2008_crisis")
        assert result is not None
        assert result.total_loss < 0

    def test_stressed_mv_less_than_initial(self) -> None:
        result = stress_test(_FakeBroker(), "2020_covid")
        assert result is not None
        assert result.stressed_total_mv < result.initial_total_mv

    def test_position_count_matches(self) -> None:
        result = stress_test(_FakeBroker(), "mild_correction")
        assert result is not None
        assert len(result.positions) == 2

    def test_verdict_valid(self) -> None:
        result = stress_test(_FakeBroker(), "2008_crisis")
        assert result is not None
        assert result.verdict in ("low_risk", "moderate", "severe", "critical")

    def test_all_scenarios_work(self) -> None:
        for scenario in SCENARIOS:
            result = stress_test(_FakeBroker(), scenario)
            assert result is not None, f"scenario {scenario} failed"

    def test_custom_shock(self) -> None:
        result = stress_test(_FakeBroker(), "custom", custom_shock_pct=-0.15)
        assert result is not None
        assert abs(result.shock_pct - (-15.0)) < 0.01

    def test_unknown_scenario_returns_none(self) -> None:
        result = stress_test(_FakeBroker(), "nonexistent")
        assert result is None

    def test_empty_portfolio_returns_none(self) -> None:
        result = stress_test(_EmptyBroker(), "2008_crisis")
        assert result is None

    def test_scenarios_dict_has_entries(self) -> None:
        assert len(SCENARIOS) >= 5
