"""Tests for the daily risk watchdog."""

from __future__ import annotations

from amms.analysis.watchdog import WatchdogReport, WatchdogWarning, generate
from amms.data.bars import Bar


class _FakeAccount:
    equity = 100_000.0
    cash = 50_000.0
    buying_power = 50_000.0


class _FakePosition:
    def __init__(self, sym: str, pnl: float = 100.0, mv: float = 5000.0) -> None:
        self.symbol = sym
        self.unrealized_pl = pnl
        self.market_value = mv
        self.qty = 10.0
        self.avg_entry_price = 100.0


class _FakeBroker:
    def get_account(self):
        return _FakeAccount()

    def get_positions(self):
        return [_FakePosition("AAPL", 200.0, 5200.0), _FakePosition("MSFT", -50.0, 4800.0)]


class _EmptyBroker:
    def get_account(self):
        return _FakeAccount()

    def get_positions(self):
        return []


class _FakeData:
    def get_bars(self, symbol, *, limit=50):
        bars = []
        for i in range(limit):
            price = 100.0 + i * 0.2
            bars.append(Bar(symbol, "1D", f"2026-01-{1 + i % 28:02d}", price, price + 1.0, price - 1.0, price, 1000.0))
        return bars


class TestWatchdog:
    def test_returns_watchdog_report(self) -> None:
        report = generate(_FakeBroker())
        assert isinstance(report, WatchdogReport)

    def test_position_count_matches(self) -> None:
        report = generate(_FakeBroker())
        assert report.position_count == 2

    def test_total_pnl_is_sum(self) -> None:
        report = generate(_FakeBroker())
        assert abs(report.total_open_pnl - 150.0) < 0.01

    def test_no_positions_broker(self) -> None:
        report = generate(_EmptyBroker())
        assert report.position_count == 0
        assert report.total_open_pnl == 0.0

    def test_regime_defaults_unknown_without_data(self) -> None:
        report = generate(_FakeBroker())
        assert report.regime == "unknown"

    def test_circuit_closed_by_default(self) -> None:
        report = generate(_FakeBroker())
        assert report.circuit_open is False

    def test_summary_is_nonempty_string(self) -> None:
        report = generate(_FakeBroker())
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_warnings_is_list(self) -> None:
        report = generate(_FakeBroker())
        assert isinstance(report.warnings, list)

    def test_with_data_runs_technical_checks(self) -> None:
        report = generate(_FakeBroker(), data=_FakeData())
        assert isinstance(report, WatchdogReport)
        assert report.position_count == 2

    def test_top_rotating_in_is_list(self) -> None:
        report = generate(_FakeBroker())
        assert isinstance(report.top_rotating_in, list)

    def test_risk_multiplier_default(self) -> None:
        report = generate(_FakeBroker())
        assert report.regime_risk_multiplier in (0.5, 0.75, 1.0)
