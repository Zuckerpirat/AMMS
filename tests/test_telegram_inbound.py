from __future__ import annotations

import sqlite3
from datetime import UTC, date, timedelta

import httpx
import pytest
import respx

from amms.notifier.inbound import PauseFlag, TelegramInbound, build_command_handlers


class _FakeAccount:
    equity = 100_000.0
    cash = 50_000.0
    buying_power = 50_000.0
    status = "ACTIVE"
    daytrade_count = 1


class _FakeBroker:
    def get_account(self):
        return _FakeAccount()

    def get_positions(self):
        class P:
            symbol = "AAPL"
            qty = 5.0
            avg_entry_price = 100.0
            market_value = 525.0
            unrealized_pl = 25.0

        return [P()]


def test_pause_flag_toggle() -> None:
    p = PauseFlag()
    assert p.paused is False
    p.set_paused(True)
    assert p.paused is True
    p.set_paused(False)
    assert p.paused is False


def test_status_handler_reports_account_state() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["status"]([])
    assert "$100,000" in out
    assert "open positions: 1" in out
    assert "paused: False" in out


def test_pause_resume_handlers_flip_flag() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "paused" in h["pause"]([]).lower()
    assert p.paused is True
    assert "resumed" in h["resume"]([]).lower()
    assert p.paused is False


def test_equity_handler() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["equity"]([]) == "$100,000.00"


def test_positions_handler() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["positions"]([])
    assert "AAPL" in out


def test_positions_handler_includes_isin_when_known() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["positions"]([])
    assert "ISIN US0378331005" in out  # AAPL is in the static table


def test_help_lists_scan_command() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["help"]([])
    assert "/scan" in out
    assert "/buylist" in out


def test_buylist_without_preview_explains_clearly() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["buylist"]([])
    assert "preview not wired" in out


def test_buylist_with_preview_lists_buy_signals() -> None:
    from amms.executor import TickResult
    from amms.strategy import Signal

    def _preview() -> TickResult:
        return TickResult(
            signals=[
                Signal(symbol="NVDA", kind="buy", reason="momentum 0.08",
                       price=487.30, score=0.62),
                Signal(symbol="PLTR", kind="buy", reason="composite hit",
                       price=24.10, score=0.41),
                Signal(symbol="TSLA", kind="sell", reason="reversal",
                       price=245.10, score=0.0),
            ],
            blocked=[("AAPL", "already long this symbol")],
        )

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, preview=_preview)
    out = h["buylist"]([])
    # Highest-score buy first
    assert out.index("NVDA") < out.index("PLTR")
    assert "TSLA" in out
    assert "SELL" in out
    assert "AAPL" in out
    assert "already long" in out


def test_preview_alias_routes_to_buylist() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["preview"] is h["buylist"]


def test_buylist_handles_empty_result_gracefully() -> None:
    from amms.executor import TickResult

    def _preview() -> TickResult:
        return TickResult()  # no signals, no blocks

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, preview=_preview)
    out = h["buylist"]([])
    assert "watching" in out


def test_buylist_returns_friendly_error_on_preview_failure() -> None:
    def _preview():
        raise RuntimeError("alpaca down")

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, preview=_preview)
    out = h["buylist"]([])
    assert out.startswith("preview failed")
    assert "alpaca down" in out


def test_scan_handler_returns_formatted_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch WSBScanner so the handler runs without hitting Reddit."""
    from amms.data import wsb_scanner as scanner_mod

    class _FakeScanner:
        def __init__(self, *a, **kw) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def scan(self, **_kw):
            return [
                scanner_mod.TrendingTicker(
                    symbol="NVDA",
                    mentions=42,
                    avg_sentiment=0.7,
                    bullish_posts=30,
                    bearish_posts=2,
                ),
                scanner_mod.TrendingTicker(
                    symbol="GME",
                    mentions=15,
                    avg_sentiment=-0.4,
                    bullish_posts=3,
                    bearish_posts=10,
                ),
            ]

    monkeypatch.setattr(scanner_mod, "WSBScanner", _FakeScanner)

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["scan"]([])
    assert "NVDA" in out
    assert "GME" in out
    assert "Trending" in out


def test_wsb_alias_routes_to_scan_handler() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["wsb"] is h["scan"]


def test_scan_handler_returns_friendly_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amms.data import wsb_scanner as scanner_mod

    class _BoomScanner:
        def __init__(self, *a, **kw) -> None:
            raise RuntimeError("reddit auth blew up")

    monkeypatch.setattr(scanner_mod, "WSBScanner", _BoomScanner)

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["scan"]([])
    assert out.startswith("WSB scan failed")
    assert "reddit auth blew up" in out


@respx.mock
def test_inbound_dispatches_status_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a Telegram update with /status text triggers the handler
    and posts the reply back to sendMessage."""
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    inbound = TelegramInbound(token="T", chat_id="42", handlers=h, timeout=1.0)

    # getUpdates returns one /status message from chat 42, then nothing.
    update = {
        "result": [
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 42},
                    "text": "/status",
                },
            }
        ]
    }
    respx.get("https://api.telegram.org/botT/getUpdates").mock(
        return_value=httpx.Response(200, json=update)
    )
    send_route = respx.post("https://api.telegram.org/botT/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    inbound._poll_once()
    assert send_route.called
    body = send_route.calls.last.request.read().decode()
    assert "equity" in body


@respx.mock
def test_inbound_ignores_messages_from_other_chats() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    inbound = TelegramInbound(token="T", chat_id="42", handlers=h, timeout=1.0)

    update = {
        "result": [
            {
                "update_id": 1,
                "message": {"chat": {"id": 999}, "text": "/status"},
            }
        ]
    }
    respx.get("https://api.telegram.org/botT/getUpdates").mock(
        return_value=httpx.Response(200, json=update)
    )
    send_route = respx.post("https://api.telegram.org/botT/sendMessage")
    inbound._poll_once()
    assert not send_route.called


@respx.mock
def test_inbound_replies_unknown_command() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    inbound = TelegramInbound(token="T", chat_id="42", handlers=h, timeout=1.0)

    update = {
        "result": [
            {"update_id": 1, "message": {"chat": {"id": 42}, "text": "/bogus"}}
        ]
    }
    respx.get("https://api.telegram.org/botT/getUpdates").mock(
        return_value=httpx.Response(200, json=update)
    )
    send_route = respx.post("https://api.telegram.org/botT/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    inbound._poll_once()
    assert send_route.called
    assert b"unknown" in send_route.calls.last.request.read()


def _make_conn_with_snapshots() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE equity_snapshots (ts TEXT PRIMARY KEY, equity REAL, cash REAL, buying_power REAL)"
    )
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    conn.executemany(
        "INSERT INTO equity_snapshots VALUES (?, ?, ?, ?)",
        [
            (f"{yesterday}T14:00:00+00:00", 99_000.0, 50_000.0, 50_000.0),
            (f"{yesterday}T20:00:00+00:00", 99_500.0, 50_000.0, 50_000.0),
            (f"{today}T14:00:00+00:00", 100_000.0, 50_000.0, 50_000.0),
            (f"{today}T20:00:00+00:00", 100_800.0, 50_000.0, 50_000.0),
        ],
    )
    conn.commit()
    return conn


def test_performance_handler_shows_pnl() -> None:
    conn = _make_conn_with_snapshots()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["performance"]([])
    assert "P&L" in out
    assert "+$800.00" in out or "+800" in out


def test_performance_handler_no_data_returns_message() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE equity_snapshots (ts TEXT PRIMARY KEY, equity REAL, cash REAL, buying_power REAL)"
    )
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["performance"]([])
    assert "No equity data" in out


def test_performance_handler_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["performance"]([]) == "DB not wired."


def test_performance_alias_perf() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "perf" in h


def test_today_handler_returns_snapshot() -> None:
    conn = _make_conn_with_snapshots()
    today_iso = date.today().isoformat()
    conn.execute(
        "CREATE TABLE orders ("
        "id TEXT, client_order_id TEXT, symbol TEXT, side TEXT, "
        "qty REAL, type TEXT, status TEXT, submitted_at TEXT, "
        "filled_at TEXT, filled_avg_price REAL)"
    )
    conn.execute(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("1", "c1", "NVDA", "buy", 5, "market", "filled",
         f"{today_iso}T15:00:00+00:00", None, None),
    )
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["today"]([])
    assert "Daily snapshot" in out
    assert "P&L today" in out
    assert "Trades today" in out
    assert "NVDA" in out
    assert "Equity:" in out


def test_today_handler_without_db_still_runs() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["today"]([])
    assert "Daily snapshot" in out
    assert "Open positions" in out


def test_help_lists_performance_command() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/performance" in h["help"]([])

