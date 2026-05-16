"""Tests for amms.execution.scheduler.TraderScheduler."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from amms.execution.scheduler import SchedulerStatus, TraderScheduler


class _FakeAutoTrader:
    def __init__(self):
        self.calls = []

    def run_watchlist(self, symbols):
        self.calls.append(list(symbols))
        from amms.execution.auto_trader import AutoTradeDecision
        return [
            AutoTradeDecision(s, "skipped", 0.0, 0.0, 0.0, 100.0, reason="test")
            for s in symbols
        ]


class _FakeClock:
    def __init__(self, is_open: bool):
        self.is_open = is_open


@pytest.fixture
def fake_trader():
    return _FakeAutoTrader()


@pytest.fixture
def journal(tmp_path):
    return tmp_path / "sched.log"


class TestLifecycle:
    def test_initial_not_running(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], journal_path=journal)
        assert s.is_running() is False

    def test_start_then_stop(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], tick_seconds=10, journal_path=journal)
        assert s.start() is True
        assert s.is_running() is True
        assert s.stop(timeout=2.0) is True
        assert s.is_running() is False

    def test_double_start_returns_false(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], tick_seconds=10, journal_path=journal)
        s.start()
        try:
            assert s.start() is False
        finally:
            s.stop()

    def test_stop_when_not_running(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], journal_path=journal)
        assert s.stop() is False


class TestTick:
    def test_tick_calls_auto_trader(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL", "MSFT"], tick_seconds=10, journal_path=journal)
        s._tick_once()
        assert fake_trader.calls == [["AAPL", "MSFT"]]
        assert s.tick_count == 1

    def test_tick_updates_summary(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], journal_path=journal)
        s._tick_once()
        assert s.last_tick_summary

    def test_market_closed_skips(self, fake_trader, journal):
        s = TraderScheduler(
            fake_trader, ["AAPL"],
            market_hours_only=True,
            clock_fn=lambda: _FakeClock(is_open=False),
            journal_path=journal,
        )
        s._tick_once()
        assert "market closed" in s.last_tick_summary
        assert fake_trader.calls == []

    def test_market_open_runs(self, fake_trader, journal):
        s = TraderScheduler(
            fake_trader, ["AAPL"],
            market_hours_only=True,
            clock_fn=lambda: _FakeClock(is_open=True),
            journal_path=journal,
        )
        s._tick_once()
        assert fake_trader.calls == [["AAPL"]]


class TestJournal:
    def test_journal_written(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], journal_path=journal)
        s._tick_once()
        assert journal.exists()
        content = journal.read_text()
        assert "tick#1" in content


class TestSymbols:
    def test_set_symbols(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], journal_path=journal)
        s.set_symbols(["MSFT", "TSLA"])
        assert s.symbols == ["MSFT", "TSLA"]

    def test_empty_symbols_skips_tick(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, [], journal_path=journal)
        s._tick_once()
        assert fake_trader.calls == []
        assert "no symbols" in s.last_tick_summary


class TestStatus:
    def test_status_returns_dataclass(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], journal_path=journal)
        st = s.status()
        assert isinstance(st, SchedulerStatus)
        assert st.running is False

    def test_tick_seconds_clamped(self, fake_trader, journal):
        s = TraderScheduler(fake_trader, ["AAPL"], tick_seconds=2, journal_path=journal)
        # min is 10
        assert s.tick_seconds == 10
