"""Tests for the daily report generator."""

from __future__ import annotations

from pathlib import Path

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(symbol="SIM", n=220, step=0.3):
    from amms.data.bars import Bar
    bars = []
    for i in range(n):
        price = 100.0 + i * step
        bars.append(Bar(
            symbol=symbol,
            timeframe="1Day",
            ts=f"2025-{(i // 28 % 12) + 1:02d}-{(i % 28) + 1:02d}T10:00:00Z",
            open=price - 0.2,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=100_000,
        ))
    return bars


class _FakeData:
    def get_bars(self, symbol, *, limit=200, timeframe="1Day"):
        return _make_bars(symbol, n=max(limit, 220))

    def get_snapshots(self, symbols):
        return {s: {"change_pct": 0.0, "change_pct_week": 0.0} for s in symbols}


def _make_trader(tmp_path: Path, *, starting_cash: float = 10_000.0):
    from amms.execution.paper_trader import PaperTrader
    return PaperTrader(starting_cash=starting_cash)


def _make_meme(tmp_path: Path, main_trader):
    from amms.execution.meme_portfolio import MemePortfolio, MemeConfig
    cfg = MemeConfig(starting_cash=1_000.0, max_allocation_pct=1.0)
    return MemePortfolio(config=cfg, state_path=tmp_path / "meme.json", main_trader=main_trader)


# ── Core output tests ─────────────────────────────────────────────────────────

def test_report_returns_string(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader)
    assert isinstance(result, str)
    assert len(result) > 0


def test_report_contains_main_portfolio_section(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader)
    assert "Main Portfolio" in result
    assert "Value" in result
    assert "Return" in result


def test_report_contains_daily_date(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader)
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)


def test_report_with_meme_portfolio(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    mp = _make_meme(tmp_path, trader)
    result = generate_daily_report(trader, meme_portfolio=mp)
    assert "Meme Sandbox" in result


def test_report_with_data_and_symbols(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(
        trader,
        data=_FakeData(),
        symbols=["SIM"],
    )
    assert "DE Signal Scan" in result
    assert "SIM" in result


def test_report_de_scan_shows_score(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader, data=_FakeData(), symbols=["SIM"])
    assert "score" in result.lower()


def test_report_no_data_no_scan(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader, data=None, symbols=["SIM"])
    assert "DE Signal Scan" not in result


def test_report_macro_regime_section(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader, data=_FakeData())
    assert "Macro Regime" in result


def test_report_with_open_position(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path, starting_cash=10_000.0)
    trader.buy("AAPL", qty=5, price=100.0, reason="test")
    result = generate_daily_report(trader, data=_FakeData())
    assert "AAPL" in result


def test_report_mode_shown_in_scan_header(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader, data=_FakeData(), symbols=["SIM"], mode="meme")
    assert "meme" in result


def test_report_capped_at_10_symbols(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    # 15 symbols — only first 10 should be scanned
    syms = [f"S{i:02d}" for i in range(15)]
    result = generate_daily_report(trader, data=_FakeData(), symbols=syms)
    # Report should not contain symbols beyond index 10
    assert "S14" not in result


def test_report_no_crash_with_none_meme(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader, meme_portfolio=None)
    assert "Main Portfolio" in result


def test_report_no_crash_with_empty_symbols(tmp_path):
    from amms.reporting.daily_report import generate_daily_report
    trader = _make_trader(tmp_path)
    result = generate_daily_report(trader, data=_FakeData(), symbols=[])
    assert isinstance(result, str)


# ── Telegram command test ─────────────────────────────────────────────────────

def test_dailyreport_cmd_exists():
    from amms.notifier.inbound import build_command_handlers, PauseFlag

    class _Broker:
        def get_account(self):
            class _A:
                equity = 10_000.0
                buying_power = 10_000.0
                cash = 10_000.0
                status = "ACTIVE"
                daytrade_count = 0
            return _A()
        def get_positions(self):
            return []

    p = PauseFlag()
    h = build_command_handlers(broker=_Broker(), pause=p)
    assert "dailyreport" in h
    assert h["report"] is h["dailyreport"]
    assert h["nightly"] is h["dailyreport"]


def test_dailyreport_cmd_returns_string():
    from amms.notifier.inbound import build_command_handlers, PauseFlag

    class _Broker:
        def get_account(self):
            class _A:
                equity = 10_000.0
                buying_power = 10_000.0
                cash = 10_000.0
                status = "ACTIVE"
                daytrade_count = 0
            return _A()
        def get_positions(self):
            return []

    p = PauseFlag()
    h = build_command_handlers(broker=_Broker(), pause=p)
    out = h["dailyreport"]([])
    assert isinstance(out, str)
    assert "Main Portfolio" in out
