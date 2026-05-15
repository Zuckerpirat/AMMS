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
         f"{today_iso}T15:00:00+00:00", None, 487.30),
    )
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["today"]([])
    assert "Daily snapshot" in out
    assert "P&L today" in out
    assert "Trades today" in out
    assert "NVDA" in out
    assert "$487.30" in out
    assert "$2,436.50" in out  # 5 * 487.30
    assert "Equity:" in out


def test_set_show_unset_handlers() -> None:
    import sqlite3 as _sqlite
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    # /show with nothing
    assert "no runtime overrides" in h["show"]([])
    # /set
    out = h["set"](["stop_loss", "0.05"])
    assert "stop_loss = 0.05" in out
    # /show after set
    out = h["show"]([])
    assert "stop_loss" in out and "0.05" in out
    # /set with invalid value
    out = h["set"](["stop_loss", "1.5"])
    assert "rejected" in out
    # /unset
    out = h["unset"](["stop_loss"])
    assert "removed" in out
    # /set with bad key
    out = h["set"](["nope", "1"])
    assert "rejected" in out


def test_set_handler_usage_when_no_args() -> None:
    import sqlite3 as _sqlite
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["set"]([])
    assert "usage" in out
    assert "stop_loss" in out


def test_today_handler_without_db_still_runs() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["today"]([])
    assert "Daily snapshot" in out
    assert "Open positions" in out


def test_risk_handler_shows_defaults_when_no_overrides() -> None:
    import sqlite3 as _sqlite
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["risk"]([])
    assert "Active risk settings" in out
    assert "stop_loss: 0 (disabled)" in out


def test_risk_handler_shows_overrides_distinctly() -> None:
    import sqlite3 as _sqlite
    from amms.runtime_overrides import set_override
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    set_override(conn, "stop_loss", "0.05")
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["risk"]([])
    assert "stop_loss: 0.05  (override)" in out


def test_stops_handler_without_stop_loss() -> None:
    import sqlite3 as _sqlite
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["stops"]([])
    assert "Stop-loss not active" in out


def test_stops_handler_with_active_stop_loss() -> None:
    import sqlite3 as _sqlite
    from amms.runtime_overrides import set_override
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    set_override(conn, "stop_loss", "0.05")
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["stops"]([])
    assert "AAPL" in out
    assert "trigger $" in out
    assert "5.0%" in out


def test_backtest_handler_renders_stats() -> None:
    class _Stats:
        initial_equity = 100_000.0
        final_equity = 112_500.0
        total_return_pct = 12.5
        num_trades = 30
        num_buys = 16
        num_sells = 14
        closed_round_trips = 14
        win_rate = 0.64
        max_drawdown_pct = -0.08

    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(),
        pause=p,
        run_backtest=lambda days: _Stats(),
    )
    out = h["backtest"](["60"])
    assert "+12.50%" in out
    assert "Win rate: 64.0%" in out
    assert "Max drawdown: -8.00%" in out
    assert "60d" in out


def test_backtest_handler_without_wiring() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["backtest"]([])
    assert "not wired" in out


def test_backtest_handler_handles_failures() -> None:
    def _boom(_d):
        raise RuntimeError("no bars")

    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(),
        pause=p,
        run_backtest=_boom,
    )
    out = h["backtest"]([])
    assert "failed" in out
    assert "no bars" in out


def test_macro_handler_uses_provided_regime() -> None:
    from amms.data.macro import MacroRegime

    regime = MacroRegime(
        level="stressed",
        reason="VIXY 1d +6.5%",
        vixy_1d_pct=6.5,
        vixy_1w_pct=2.0,
    )
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(),
        pause=p,
        get_macro_regime=lambda: regime,
    )
    out = h["macro"]([])
    assert "STRESSED" in out
    assert "+6.50%" in out
    assert "VIXY" in out


def test_macro_handler_falls_back_to_data_when_no_state() -> None:
    class _FakeData:
        def get_snapshots(self, _symbols):
            return {
                "VIXY": {"change_pct": 0.5, "change_pct_week": 1.0}
            }

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeData())
    out = h["macro"]([])
    assert "CALM" in out


def test_explain_handler_returns_last_signal() -> None:
    import sqlite3 as _sqlite
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    conn.execute(
        "CREATE TABLE signals (ts TEXT, symbol TEXT, strategy TEXT, "
        "signal TEXT, reason TEXT, score REAL)"
    )
    conn.execute(
        "INSERT INTO signals VALUES (?,?,?,?,?,?)",
        ("2026-05-14T15:00:00+00:00", "NVDA", "composite", "buy",
         "momentum +5.2%, RSI 58", 1.42),
    )
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["explain"](["NVDA"])
    assert "Last decision for NVDA" in out
    assert "BUY" in out
    assert "composite" in out
    assert "+1.42" in out
    assert "momentum +5.2%" in out


def test_explain_handler_unknown_symbol() -> None:
    import sqlite3 as _sqlite
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    conn.execute(
        "CREATE TABLE signals (ts TEXT, symbol TEXT, strategy TEXT, "
        "signal TEXT, reason TEXT, score REAL)"
    )
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["explain"](["ZZZZ"])
    assert "no decision recorded" in out


def test_explain_handler_usage_when_no_args() -> None:
    import sqlite3 as _sqlite
    conn = _sqlite.connect(":memory:")
    conn.row_factory = _sqlite.Row
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["explain"]([])
    assert "usage" in out


def test_help_lists_performance_command() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/performance" in h["help"]([])



def _make_conn_with_overrides():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runtime_overrides "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def test_mode_handler_shows_current_mode() -> None:
    conn = _make_conn_with_overrides()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["mode"]([])
    assert "Active mode:" in out
    assert "swing" in out  # default mode


def test_mode_handler_switches_mode() -> None:
    conn = _make_conn_with_overrides()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["mode"](["conservative"])
    assert "conservative" in out.lower()


def test_mode_handler_rejects_invalid_mode() -> None:
    conn = _make_conn_with_overrides()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["mode"](["turbo"])
    assert "Unknown mode" in out


def test_mode_handler_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["mode"]([])
    assert "DB not wired" in out


def test_pnl_handler_shows_position_detail() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["pnl"]([])
    assert "AAPL" in out
    assert "cost" in out or "$" in out


def test_pnl_handler_specific_symbol() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["pnl"](["AAPL"])
    assert "AAPL" in out


def test_pnl_handler_unknown_symbol() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["pnl"](["ZZZZ"])
    assert "no open position" in out


def test_alert_handler_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["alert"]([])
    assert "DB not wired" in out


def test_alert_handler_add_and_list() -> None:
    conn = _make_conn_with_overrides()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["alert"](["AAPL", "200", "above"])
    assert "Alert set" in out
    assert "AAPL" in out
    out2 = h["alert"](["list"])
    assert "AAPL" in out2
    assert "200" in out2


def test_alert_handler_delete() -> None:
    conn = _make_conn_with_overrides()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    h["alert"](["AAPL", "200", "above"])
    out = h["alert"](["del", "1"])
    assert "deleted" in out.lower()


def test_alert_handler_empty_list() -> None:
    conn = _make_conn_with_overrides()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["alert"]([])
    assert "No active" in out


def test_alert_handler_invalid_direction() -> None:
    conn = _make_conn_with_overrides()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["alert"](["AAPL", "200", "sideways"])
    assert "must be" in out.lower() or "Error" in out


def test_help_lists_new_commands() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/pnl" in help_text
    assert "/mode" in help_text
    assert "/alert" in help_text


def _make_conn_with_orders():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runtime_overrides "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE orders (id TEXT, client_order_id TEXT, symbol TEXT, "
        "side TEXT, qty REAL, type TEXT, status TEXT, submitted_at TEXT, "
        "filled_at TEXT, filled_avg_price REAL)"
    )
    conn.execute(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("1", "c1", "AAPL", "buy", 5, "market", "filled",
         "2026-05-15T15:00:00+00:00", None, 180.0),
    )
    conn.commit()
    return conn


def test_summary_handler_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conn = _make_conn_with_orders()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["summary"]([])
    assert "Equity:" in out
    assert "AAPL" in out
    assert "LLM narration unavailable" in out


def test_summary_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["summary"]([]) == "DB not wired."


def test_help_includes_summary() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/summary" in h["help"]([])


def test_top_handler_shows_positions() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["top"]([])
    assert "AAPL" in out
    assert "Best:" in out


def test_top_handler_no_positions() -> None:
    class _EmptyBroker(_FakeBroker):
        def get_positions(self):
            return []

    p = PauseFlag()
    h = build_command_handlers(broker=_EmptyBroker(), pause=p)
    assert "no open positions" in h["top"]([])


def test_news_handler_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=None)
    out = h["news"](["AAPL"])
    assert "not wired" in out.lower()


def test_news_handler_with_data_returns_headlines() -> None:
    class _FakeData:
        def get_news(self, symbols, *, limit=5):
            return [
                {
                    "headline": "AAPL hits record high",
                    "created_at": "2026-05-15T10:00:00Z",
                    "url": "https://example.com/1",
                    "symbols": ["AAPL"],
                }
            ]

        def get_snapshots(self, symbols, *, feed="iex"):
            return {}

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeData())
    out = h["news"](["AAPL"])
    assert "AAPL" in out
    assert "record high" in out


def test_news_handler_no_results() -> None:
    class _EmptyData:
        def get_news(self, symbols, *, limit=5):
            return []

        def get_snapshots(self, symbols, *, feed="iex"):
            return {}

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_EmptyData())
    out = h["news"](["ZZZZ"])
    assert "No recent news" in out


def test_help_includes_top_and_news() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/top" in help_text
    assert "/news" in help_text


def _make_conn_with_roundtrip():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runtime_overrides "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE orders (id TEXT, client_order_id TEXT, symbol TEXT, "
        "side TEXT, qty REAL, type TEXT, status TEXT, submitted_at TEXT, "
        "filled_at TEXT, filled_avg_price REAL)"
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("1", "c1", "NVDA", "buy", 10, "market", "filled",
             "2026-05-10T15:00:00+00:00", "2026-05-10T15:01:00+00:00", 100.0),
            ("2", "c2", "NVDA", "sell", 10, "market", "filled",
             "2026-05-14T15:00:00+00:00", "2026-05-14T15:01:00+00:00", 115.0),
        ],
    )
    conn.commit()
    return conn


def test_journal_handler_shows_completed_trade() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["journal"]([])
    assert "NVDA" in out
    assert "$100.00" in out
    assert "$115.00" in out
    assert "150.00" in out  # 10 * 15 profit


def test_journal_handler_symbol_filter() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["journal"](["NVDA"])
    assert "NVDA" in out
    out2 = h["journal"](["AAPL"])
    assert "No completed round-trips" in out2


def test_journal_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["journal"]([]) == "DB not wired."


def test_help_includes_journal() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/journal" in h["help"]([])


def test_budget_handler_shows_slots() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["budget"]([])
    assert "Budget summary" in out
    assert "Cash:" in out
    assert "Equity:" in out
    assert "slots" in out.lower()


def test_corr_handler_needs_two_positions() -> None:
    import sqlite3 as _sqlite

    conn = _sqlite.connect(":memory:", check_same_thread=False)
    conn.row_factory = _sqlite.Row
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    # _FakeBroker returns only 1 position
    out = h["corr"]([])
    assert "at least 2" in out


def test_corr_handler_with_db_data() -> None:
    import sqlite3 as _sqlite

    conn = _sqlite.connect(":memory:", check_same_thread=False)
    conn.row_factory = _sqlite.Row
    conn.execute(
        "CREATE TABLE bars (symbol TEXT, timeframe TEXT, ts TEXT, "
        "open REAL, high REAL, low REAL, close REAL, volume REAL, "
        "PRIMARY KEY (symbol, timeframe, ts))"
    )
    # Insert 20 bars for AAPL and MSFT
    import math
    for i in range(20):
        ts = f"2026-04-{i+1:02d}T00:00:00Z"
        aapl_close = 100 + math.sin(i) * 5
        msft_close = 200 + math.sin(i) * 8
        conn.execute(
            "INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)",
            ("AAPL", "1Day", ts, aapl_close, aapl_close+1, aapl_close-1, aapl_close, 1_000_000),
        )
        conn.execute(
            "INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)",
            ("MSFT", "1Day", ts, msft_close, msft_close+1, msft_close-1, msft_close, 800_000),
        )
    conn.commit()

    class _TwoBroker:
        def get_account(self):
            return _FakeAccount()

        def get_positions(self):
            class P1:
                symbol = "AAPL"
                qty = 5.0
                avg_entry_price = 100.0
                market_value = 500.0
                unrealized_pl = 0.0

            class P2:
                symbol = "MSFT"
                qty = 3.0
                avg_entry_price = 200.0
                market_value = 600.0
                unrealized_pl = 0.0

            return [P1(), P2()]

    p = PauseFlag()
    h = build_command_handlers(broker=_TwoBroker(), pause=p, conn=conn)
    out = h["corr"]([])
    assert "AAPL" in out
    assert "MSFT" in out
    assert "correlation" in out.lower() or "↔" in out


def test_corr_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["corr"]([])
    assert "DB not wired" in out


def test_help_includes_budget_and_corr() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/budget" in help_text
    assert "/corr" in help_text
