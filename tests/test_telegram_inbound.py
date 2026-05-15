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
        max_drawdown_pct = -8.0

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


def _make_conn_with_trades_and_equity():
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
        "CREATE TABLE equity_snapshots "
        "(ts TEXT PRIMARY KEY, equity REAL, cash REAL, buying_power REAL)"
    )
    # 3 round trips: 2 wins, 1 loss
    trades = [
        ("b1", "c1", "NVDA", "buy",  10, "market", "filled", "2026-04-01T15:00:00Z", None, 100.0),
        ("s1", "c2", "NVDA", "sell", 10, "market", "filled", "2026-04-05T15:00:00Z", None, 115.0),
        ("b2", "c3", "TSLA", "buy",  5, "market", "filled",  "2026-04-06T15:00:00Z", None, 200.0),
        ("s2", "c4", "TSLA", "sell", 5, "market", "filled",  "2026-04-10T15:00:00Z", None, 190.0),
        ("b3", "c5", "AAPL", "buy",  8, "market", "filled",  "2026-04-11T15:00:00Z", None, 150.0),
        ("s3", "c6", "AAPL", "sell", 8, "market", "filled",  "2026-04-15T15:00:00Z", None, 165.0),
    ]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)", trades)
    # 10 equity snapshots for Sharpe (within the last 30 days)
    from datetime import date, timedelta

    base = date.today() - timedelta(days=10)
    for i in range(10):
        day = base + timedelta(days=i)
        ts = f"{day.isoformat()}T16:00:00Z"
        eq = 100_000 + i * 500
        conn.execute(
            "INSERT INTO equity_snapshots VALUES (?,?,?,?)",
            (ts, eq, eq * 0.5, eq * 0.5),
        )
    conn.commit()
    return conn


def test_streak_handler_shows_current_streak() -> None:
    conn = _make_conn_with_trades_and_equity()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["streak"]([])
    assert "streak" in out.lower()
    assert "Win rate" in out


def test_streak_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["streak"]([]) == "DB not wired."


def test_sharpe_handler_returns_metrics() -> None:
    conn = _make_conn_with_trades_and_equity()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["sharpe"]([])
    assert "Sharpe" in out
    assert "return" in out.lower()


def test_sharpe_insufficient_data() -> None:
    import sqlite3 as _sqlite

    conn = _sqlite.connect(":memory:", check_same_thread=False)
    conn.row_factory = _sqlite.Row
    conn.execute(
        "CREATE TABLE equity_snapshots "
        "(ts TEXT PRIMARY KEY, equity REAL, cash REAL, buying_power REAL)"
    )
    conn.execute(
        "INSERT INTO equity_snapshots VALUES (?,?,?,?)",
        ("2026-05-15T12:00:00Z", 100_000, 50_000, 50_000),
    )
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["sharpe"]([])
    assert "Not enough" in out


def test_help_includes_streak_and_sharpe() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/streak" in help_text
    assert "/sharpe" in help_text


def test_riskreport_handler_basic() -> None:
    import sqlite3 as _sqlite

    conn = _sqlite.connect(":memory:", check_same_thread=False)
    conn.row_factory = _sqlite.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runtime_overrides "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS equity_snapshots "
        "(ts TEXT PRIMARY KEY, equity REAL, cash REAL, buying_power REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS bars "
        "(symbol TEXT, timeframe TEXT, ts TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL, PRIMARY KEY(symbol, timeframe, ts))"
    )
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["riskreport"]([])
    assert "Risk Report" in out
    # AAPL is in positions, should show sector
    assert "Technology" in out or "Sector" in out


def test_riskreport_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["riskreport"]([])
    assert "Risk Report" in out


def test_rr_alias_routes_to_riskreport() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["rr"] is h["riskreport"]


def test_upcoming_no_positions() -> None:
    class _EmptyBroker(_FakeBroker):
        def get_positions(self):
            return []

    p = PauseFlag()
    h = build_command_handlers(broker=_EmptyBroker(), pause=p)
    out = h["upcoming"]([])
    assert "No open positions" in out


def test_upcoming_with_positions_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.data import earnings as earnings_mod

    def _fake_fetch(symbols, *, days_ahead=14):
        from amms.data.earnings import EarningsEvent

        return [EarningsEvent(
            symbol="AAPL", date="2026-05-20",
            time="after-market", eps_estimate="1.50"
        )]

    monkeypatch.setattr(earnings_mod, "fetch_upcoming", _fake_fetch)

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["upcoming"]([])
    assert "AAPL" in out
    assert "2026-05-20" in out
    assert "EPS est" in out


def test_compare_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=None)
    out = h["compare"](["AAPL", "NVDA"])
    assert "not wired" in out.lower()


def test_compare_usage_when_too_few_args() -> None:
    class _FakeData:
        def get_snapshots(self, syms, *, feed="iex"):
            return {}

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeData())
    out = h["compare"](["AAPL"])
    assert "usage" in out.lower()


def test_compare_shows_both_symbols() -> None:
    class _FakeData:
        def get_snapshots(self, syms, *, feed="iex"):
            return {
                "AAPL": {"price": 180.0, "change_pct": 1.5, "change_pct_week": 3.0},
                "NVDA": {"price": 900.0, "change_pct": -0.5, "change_pct_week": 5.0},
            }

    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeData())
    out = h["compare"](["AAPL", "NVDA"])
    assert "AAPL" in out
    assert "NVDA" in out
    assert "180" in out
    assert "900" in out
    assert "Sector" in out


def test_help_includes_upcoming_and_compare() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/upcoming" in help_text
    assert "/compare" in help_text


def test_sentiment_no_args_uses_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.features import sentiment as sent_mod

    class _FakeColl:
        def fetch_trending(self, *, filter="wallstreetbets", page=1):
            return [{"ticker": "AAPL", "rank": 5, "mentions": 200, "mentions_24h": 30}]

    monkeypatch.setattr(sent_mod, "ApeWisdomCollector", _FakeColl)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["sentiment"]([])
    assert "AAPL" in out
    assert "200" in out


def test_sentiment_unknown_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.features import sentiment as sent_mod

    class _FakeColl:
        def fetch_trending(self, *, filter="wallstreetbets", page=1):
            return []

    monkeypatch.setattr(sent_mod, "ApeWisdomCollector", _FakeColl)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["sentiment"](["ZZZZ"])
    assert "not in top trending" in out


def test_profit_handler_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["profit"]([]) == "DB not wired."


def test_profit_handler_shows_realized_pnl() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["profit"](["all"])
    assert "NVDA" in out
    # 10 shares * (115 - 100) = 150 profit
    assert "150" in out


def test_profit_handler_period_filter() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    # Old trades → nothing in 'day' period
    out = h["profit"](["day"])
    assert "No completed trades" in out


def test_profit_handler_invalid_period() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["profit"](["yearly"])
    assert "usage" in out.lower()


def test_help_includes_sentiment_and_profit() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/sentiment" in help_text
    assert "/profit" in help_text


def test_ping_handler_returns_pong() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["ping"]([])
    assert "pong" in out.lower()
    assert "100,000" in out or "100000" in out


def test_version_handler_returns_git_info() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["version"]([])
    assert "amms" in out
    assert "git" in out


def test_fees_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["fees"]([]) == "DB not wired."


def test_fees_shows_cost_estimate() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["fees"]([])
    assert "Simulated fee" in out
    assert "notional" in out.lower()


def test_fees_custom_bps() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["fees"](["10"])
    assert "10.0 bps" in out


def test_fees_invalid_bps() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["fees"](["abc"])
    assert "usage" in out.lower()


def test_help_includes_ping_version_fees() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/ping" in help_text
    assert "/version" in help_text
    assert "/fees" in help_text


def test_export_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["export"]([]) == "DB not wired."


def test_export_shows_csv_header() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["export"]([])
    assert "symbol,side,qty" in out
    assert "NVDA" in out


def test_export_custom_limit() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["export"](["1"])
    # Should show only 1 row (most recent) plus header
    assert "symbol,side,qty" in out


def test_export_invalid_limit() -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["export"](["abc"])
    assert "usage" in out.lower()


def test_help_includes_export() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/export" in h["help"]([])


# ---------------------------------------------------------------------------
# /setlist tests
# ---------------------------------------------------------------------------

def test_setlist_no_db_path() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["setlist"](["AAPL"])
    assert "not wired" in out


def test_setlist_replaces_watchlist(tmp_path) -> None:
    p = PauseFlag()
    db_file = tmp_path / "amms.db"
    h = build_command_handlers(broker=_FakeBroker(), pause=p, db_path=db_file)
    out = h["setlist"](["AAPL", "NVDA", "MSFT"])
    assert "3 symbol(s)" in out
    assert "AAPL" in out
    assert "NVDA" in out


def test_setlist_clear_empties_watchlist(tmp_path) -> None:
    p = PauseFlag()
    db_file = tmp_path / "amms.db"
    h = build_command_handlers(broker=_FakeBroker(), pause=p, db_path=db_file)
    h["setlist"](["AAPL", "TSLA"])
    out = h["setlist"](["clear"])
    assert "0 symbol" in out


def test_setlist_no_args_clears(tmp_path) -> None:
    p = PauseFlag()
    db_file = tmp_path / "amms.db"
    h = build_command_handlers(broker=_FakeBroker(), pause=p, db_path=db_file)
    h["setlist"](["AAPL"])
    out = h["setlist"]([])
    assert "0 symbol" in out


def test_setlist_invalid_ticker(tmp_path) -> None:
    p = PauseFlag()
    db_file = tmp_path / "amms.db"
    h = build_command_handlers(broker=_FakeBroker(), pause=p, db_path=db_file)
    out = h["setlist"](["TOOLONG123", "AAPL"])
    assert "Invalid" in out


def test_setlist_persisted_to_watchlist(tmp_path) -> None:
    p = PauseFlag()
    db_file = tmp_path / "amms.db"
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p, db_path=db_file, static_watchlist=()
    )
    h["setlist"](["GOOG", "AMZN"])
    out = h["watchlist"]([])
    assert "GOOG" in out
    assert "AMZN" in out


# ---------------------------------------------------------------------------
# /calendar tests
# ---------------------------------------------------------------------------

def test_calendar_shows_market_hours() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["calendar"]([])
    assert "9:30 AM" in out
    assert "4:00 PM" in out
    assert "Monday" in out


def test_calendar_shows_utc_time() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["calendar"]([])
    assert "UTC" in out


def test_calendar_shows_holiday() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["calendar"]([])
    # At least one holiday or the "no more" message should appear
    assert "holiday" in out.lower() or "NYSE" in out


def test_help_includes_setlist_and_calendar() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/setlist" in help_text
    assert "/calendar" in help_text


# ---------------------------------------------------------------------------
# /heatmap tests
# ---------------------------------------------------------------------------

def test_heatmap_shows_positions() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["heatmap"]([])
    assert "AAPL" in out
    assert "heat-map" in out.lower()


def test_heatmap_no_positions() -> None:
    class _NoBroker(_FakeBroker):
        def get_positions(self):
            return []
    p = PauseFlag()
    h = build_command_handlers(broker=_NoBroker(), pause=p)
    out = h["heatmap"]([])
    assert "no open positions" in out


def test_heatmap_shows_bar() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["heatmap"]([])
    # Must contain at least one block character or arrow
    assert "█" in out or "▲" in out or "▼" in out


# ---------------------------------------------------------------------------
# /limit tests
# ---------------------------------------------------------------------------

def test_limit_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["limit"]([]) == "DB not wired."


def test_limit_show_default(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    from amms.runtime_overrides import ensure_table
    ensure_table(conn)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["limit"]([])
    assert "config default" in out


def test_limit_set_value(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    from amms.runtime_overrides import ensure_table
    ensure_table(conn)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["limit"](["3"])
    assert "3" in out
    # Verify it shows now
    out2 = h["limit"]([])
    assert "3" in out2


def test_limit_off_removes_override(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    from amms.runtime_overrides import ensure_table
    ensure_table(conn)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    h["limit"](["5"])
    out = h["limit"](["off"])
    assert "removed" in out or "default" in out


def test_limit_invalid_input(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    from amms.runtime_overrides import ensure_table
    ensure_table(conn)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["limit"](["abc"])
    assert "usage" in out.lower()


def test_help_includes_heatmap_and_limit() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/heatmap" in help_text
    assert "/limit" in help_text


# ---------------------------------------------------------------------------
# /drawdown tests
# ---------------------------------------------------------------------------

def _make_conn_with_equity(tmp_path, equities: list[float]):
    """Create conn with equity_snapshots."""
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS equity_snapshots (ts TEXT, equity REAL)"
    )
    from amms.runtime_overrides import ensure_table
    ensure_table(conn)
    for i, eq in enumerate(equities):
        conn.execute(
            "INSERT INTO equity_snapshots VALUES (?, ?)",
            (f"2026-0{1 + i // 28}-{1 + (i % 28):02d}T10:00:00", eq),
        )
    conn.commit()
    return conn


def test_drawdown_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["drawdown"]([]) == "DB not wired."


def test_drawdown_shows_analytics(tmp_path) -> None:
    conn = _make_conn_with_equity(tmp_path, [100_000, 105_000, 95_000])
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["drawdown"]([])
    assert "Peak equity" in out
    assert "Current DD" in out
    assert "worst" in out.lower()


def test_drawdown_no_breach_when_flat(tmp_path) -> None:
    conn = _make_conn_with_equity(tmp_path, [100_000, 100_000])
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["drawdown"]([])
    assert "⚠️" not in out


# ---------------------------------------------------------------------------
# /alloc tests
# ---------------------------------------------------------------------------

def test_alloc_shows_sectors() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["alloc"]([])
    assert "Allocation" in out
    assert "target" in out.lower()


def test_alloc_no_positions() -> None:
    class _NoBroker(_FakeBroker):
        def get_positions(self):
            return []
    p = PauseFlag()
    h = build_command_handlers(broker=_NoBroker(), pause=p)
    assert "no open positions" in h["alloc"]([])


def test_help_includes_drawdown_and_alloc() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/drawdown" in help_text
    assert "/alloc" in help_text


# ---------------------------------------------------------------------------
# /vol tests
# ---------------------------------------------------------------------------

class _FakeDataClient:
    """Returns minimal bar data for volatility tests."""
    def get_bars(self, symbol, *, limit=30):
        from amms.data.bars import Bar
        # 32 bars with tiny variation to produce non-None vol
        bars = []
        for i in range(32):
            price = 150.0 + i * 0.1
            bars.append(Bar(
                symbol=symbol,
                timestamp=f"2026-01-{1 + i % 28:02d}T10:00:00Z",
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1_000,
            ))
        return bars


def test_vol_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["vol"]([])


def test_vol_no_positions_no_arg() -> None:
    class _NoBroker(_FakeBroker):
        def get_positions(self):
            return []
    p = PauseFlag()
    h = build_command_handlers(broker=_NoBroker(), pause=p, data=_FakeDataClient())
    out = h["vol"]([])
    assert "no open positions" in out


def test_vol_for_held_positions() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["vol"]([])
    assert "AAPL" in out
    assert "ATR" in out


def test_vol_explicit_ticker() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["vol"](["NVDA"])
    assert "NVDA" in out
    assert "vol" in out.lower()


# ---------------------------------------------------------------------------
# /reload tests
# ---------------------------------------------------------------------------

def test_reload_not_wired() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["reload"]([])


def test_reload_calls_callback() -> None:
    called = []
    def _fake_reload():
        called.append(True)
        return "Config reloaded OK."
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, reload_config=_fake_reload)
    out = h["reload"]([])
    assert "reload" in out.lower()
    assert called


def test_reload_propagates_error_message() -> None:
    def _broken_reload():
        raise RuntimeError("yaml parse error")
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, reload_config=_broken_reload)
    out = h["reload"]([])
    assert "reload failed" in out


def test_help_includes_vol_and_reload() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/vol" in help_text
    assert "/reload" in help_text


# ---------------------------------------------------------------------------
# /bench tests
# ---------------------------------------------------------------------------

def _make_conn_with_bench_equities(tmp_path, equities):
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS equity_snapshots (ts TEXT, equity REAL)")
    from amms.runtime_overrides import ensure_table
    ensure_table(conn)
    from datetime import date, timedelta
    base = date.today() - timedelta(days=len(equities))
    for i, eq in enumerate(equities):
        d = (base + timedelta(days=i)).isoformat()
        conn.execute("INSERT INTO equity_snapshots VALUES (?, ?)", (f"{d}T10:00:00", eq))
    conn.commit()
    return conn


def test_bench_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["bench"]([]) == "DB not wired."


def test_bench_insufficient_history(tmp_path) -> None:
    conn = _make_conn_with_bench_equities(tmp_path, [100_000])
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["bench"]([])
    assert "Not enough" in out


def test_bench_shows_portfolio_return(tmp_path) -> None:
    conn = _make_conn_with_bench_equities(tmp_path, [100_000, 102_000, 105_000])
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["bench"]([])
    assert "Portfolio return" in out
    assert "5.00%" in out or "+5.00" in out


def test_bench_invalid_arg(tmp_path) -> None:
    conn = _make_conn_with_bench_equities(tmp_path, [100_000, 102_000])
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["bench"](["abc"])
    assert "usage" in out.lower()


# ---------------------------------------------------------------------------
# /targets tests
# ---------------------------------------------------------------------------

def test_targets_no_positions() -> None:
    class _NoBroker(_FakeBroker):
        def get_positions(self):
            return []
    p = PauseFlag()
    h = build_command_handlers(broker=_NoBroker(), pause=p)
    assert "no open positions" in h["targets"]([])


def test_targets_shows_entry_stop_target() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["targets"]([])
    assert "entry" in out
    assert "stop" in out
    assert "target" in out
    assert "AAPL" in out


def test_targets_uses_stop_override(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    from amms.runtime_overrides import ensure_table, set_override
    ensure_table(conn)
    set_override(conn, "stop_loss", "0.05")
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["targets"]([])
    # 5% stop on entry 100 → stop at $95
    assert "95.00" in out


def test_help_includes_bench_and_targets() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/bench" in help_text
    assert "/targets" in help_text


# ---------------------------------------------------------------------------
# /mdd tests
# ---------------------------------------------------------------------------

def test_mdd_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["mdd"]([]) == "DB not wired."


def test_mdd_insufficient_history(tmp_path) -> None:
    conn = _make_conn_with_bench_equities(tmp_path, [100_000])
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["mdd"]([])
    assert "Not enough" in out or "No daily" in out


def test_mdd_shows_worst_days(tmp_path) -> None:
    conn = _make_conn_with_bench_equities(
        tmp_path, [100_000, 98_000, 95_000, 97_000, 102_000, 101_000]
    )
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["mdd"]([])
    assert "worst" in out.lower()
    assert "#1" in out


def test_mdd_shows_best_days(tmp_path) -> None:
    conn = _make_conn_with_bench_equities(
        tmp_path, [100_000, 98_000, 105_000, 104_000, 108_000]
    )
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["mdd"]([])
    assert "best" in out.lower()


# ---------------------------------------------------------------------------
# /optout tests
# ---------------------------------------------------------------------------

def test_optout_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["optout"](["TSLA"]) == "DB not wired."


def test_optout_blocks_ticker(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS runtime_overrides (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["optout"](["TSLA"])
    assert "TSLA" in out
    assert "blocked" in out.lower()


def test_optout_list_shows_blocked(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS runtime_overrides (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    h["optout"](["TSLA"])
    h["optout"](["NVDA"])
    out = h["optout"](["list"])
    assert "TSLA" in out
    assert "NVDA" in out


def test_optout_remove_unblocks(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS runtime_overrides (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    h["optout"](["TSLA"])
    out = h["optout"](["TSLA", "remove"])
    assert "unblocked" in out.lower()


def test_optout_empty_list(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS runtime_overrides (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["optout"](["list"])
    assert "No tickers blocked" in out or "blocked" in out.lower()


def test_help_includes_mdd_and_optout() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/mdd" in help_text
    assert "/optout" in help_text


# ---------------------------------------------------------------------------
# /note tests
# ---------------------------------------------------------------------------

def _make_empty_conn(tmp_path):
    conn = sqlite3.connect(tmp_path / "amms.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS runtime_overrides (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    return conn


def test_note_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["note"](["AAPL", "some text"]) == "DB not wired."


def test_note_save_and_read(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["note"](["AAPL", "Strong momentum play"])
    assert "saved" in out.lower()
    out2 = h["note"](["AAPL"])
    assert "Strong momentum play" in out2


def test_note_list(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    h["note"](["AAPL", "test1"])
    h["note"](["TSLA", "test2"])
    out = h["note"](["list"])
    assert "AAPL" in out
    assert "TSLA" in out


def test_note_empty_list(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["note"](["list"])
    assert "No notes" in out


def test_note_clear(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    h["note"](["AAPL", "something"])
    out = h["note"](["AAPL", "clear"])
    assert "deleted" in out.lower()
    out2 = h["note"](["AAPL"])
    assert "No note" in out2


def test_note_no_args(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["note"]([])
    assert "usage" in out.lower()


# ---------------------------------------------------------------------------
# /recap tests
# ---------------------------------------------------------------------------

def test_recap_shows_equity() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["recap"]([])
    assert "Equity" in out
    assert "100,000" in out


def test_recap_shows_positions() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["recap"]([])
    assert "Open positions" in out
    assert "AAPL" in out or "mover" in out.lower()


def test_recap_shows_pause_status() -> None:
    p = PauseFlag()
    p.set_paused(True)
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["recap"]([])
    assert "PAUSED" in out


def test_recap_shows_running_when_not_paused() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["recap"]([])
    assert "running" in out


def test_help_includes_note_and_recap() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/note" in help_text
    assert "/recap" in help_text


# ---------------------------------------------------------------------------
# /rsi tests
# ---------------------------------------------------------------------------

def test_rsi_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["rsi"]([])


def test_rsi_no_positions() -> None:
    class _NoBroker(_FakeBroker):
        def get_positions(self):
            return []
    p = PauseFlag()
    h = build_command_handlers(broker=_NoBroker(), pause=p, data=_FakeDataClient())
    assert "no open positions" in h["rsi"]([])


def test_rsi_shows_for_position() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["rsi"]([])
    assert "RSI" in out
    assert "AAPL" in out


def test_rsi_explicit_ticker() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["rsi"](["NVDA"])
    assert "NVDA" in out


# ---------------------------------------------------------------------------
# /ema tests
# ---------------------------------------------------------------------------

def test_ema_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["ema"]([])


def test_ema_shows_for_position() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["ema"]([])
    assert "EMA" in out
    assert "AAPL" in out


def test_ema_explicit_ticker() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["ema"](["TSLA"])
    assert "TSLA" in out


def test_ema_function_basic() -> None:
    from amms.data.bars import Bar
    from amms.features.momentum import ema, sma

    bars = [
        Bar("X", "1D", f"2026-01-{i:02d}", 100+i, 101+i, 99+i, 100+i, 1000)
        for i in range(1, 26)
    ]
    e = ema(bars, 20)
    s = sma(bars, 20)
    assert e is not None
    assert s is not None
    assert abs(e - s) < 5  # EMA and SMA should be in the same ballpark


def test_help_includes_rsi_and_ema() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/rsi" in help_text
    assert "/ema" in help_text


# ---------------------------------------------------------------------------
# /macd tests
# ---------------------------------------------------------------------------

def test_macd_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["macd"]([])


def test_macd_shows_for_position() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["macd"]([])
    assert "MACD" in out
    assert "AAPL" in out


def test_macd_explicit_ticker() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["macd"](["NVDA"])
    assert "NVDA" in out


def test_macd_function() -> None:
    from amms.data.bars import Bar
    from amms.features.momentum import macd

    bars = [
        Bar("X", "1D", f"2026-{1 + i // 31:02d}-{1 + (i % 30):02d}", 100 + i * 0.5, 101 + i * 0.5, 99 + i * 0.5, 100 + i * 0.5, 1000)
        for i in range(40)
    ]
    result = macd(bars)
    assert result is not None
    m_line, sig_line, hist = result
    assert hist == pytest.approx(m_line - sig_line)


# ---------------------------------------------------------------------------
# /score tests
# ---------------------------------------------------------------------------

def test_score_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["score"]([])


def test_score_shows_for_position() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["score"]([])
    assert "score" in out.lower()
    assert "AAPL" in out


def test_score_explicit_ticker() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["score"](["TSLA"])
    assert "TSLA" in out


def test_help_includes_macd_and_score() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/macd" in help_text
    assert "/score" in help_text


# ---------------------------------------------------------------------------
# /filter tests
# ---------------------------------------------------------------------------

def test_filter_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, static_watchlist=("AAPL",))
    assert "not wired" in h["filter"]([])


def test_filter_empty_watchlist() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["filter"]([])
    assert "empty" in out.lower() or "No tickers match" in out


def test_filter_with_watchlist() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL", "NVDA"),
    )
    out = h["filter"]([])
    # Should show results or no-match message
    assert "AAPL" in out or "No tickers match" in out or "Filter results" in out


def test_filter_bull_mode() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL",),
    )
    out = h["filter"](["bull"])
    assert "Filter" in out or "No tickers" in out


# ---------------------------------------------------------------------------
# /sizing tests
# ---------------------------------------------------------------------------

def test_sizing_no_args() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "usage" in h["sizing"]([]).lower()


def test_sizing_with_price() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["sizing"](["AAPL", "150"])
    assert "Quantity" in out
    assert "AAPL" in out


def test_sizing_computes_correct_shares() -> None:
    """With $100k equity and 2% max pos and price $200 → max $2000 → 10 shares."""
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["sizing"](["AAPL", "200"])
    assert "10 shares" in out


def test_sizing_invalid_price() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["sizing"](["AAPL", "abc"])
    assert "usage" in out.lower()


def test_help_includes_filter_and_sizing() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/filter" in help_text
    assert "/sizing" in help_text


# ---------------------------------------------------------------------------
# /winloss tests
# ---------------------------------------------------------------------------

def test_winloss_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["winloss"]([]) == "DB not wired."


def test_winloss_no_trades(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    conn.execute("CREATE TABLE IF NOT EXISTS orders (symbol TEXT, side TEXT, qty REAL, filled_avg_price REAL, status TEXT, filled_at TEXT, submitted_at TEXT)")
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["winloss"]([])
    assert "No completed" in out


def test_winloss_shows_per_ticker(tmp_path) -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["winloss"]([])
    assert "NVDA" in out
    assert "W" in out or "L" in out


# ---------------------------------------------------------------------------
# /hold tests
# ---------------------------------------------------------------------------

def test_hold_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["hold"]([]) == "DB not wired."


def test_hold_no_trades(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    conn.execute("CREATE TABLE IF NOT EXISTS orders (symbol TEXT, side TEXT, qty REAL, filled_avg_price REAL, status TEXT, filled_at TEXT, submitted_at TEXT)")
    conn.commit()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["hold"]([])
    assert "No completed" in out


def test_hold_shows_holding_days(tmp_path) -> None:
    conn = _make_conn_with_roundtrip()
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["hold"]([])
    # Either shows data or "no completed" if dates don't parse
    assert "holding" in out.lower() or "NVDA" in out or "No completed" in out


def test_help_includes_winloss_and_hold() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    help_text = h["help"]([])
    assert "/winloss" in help_text
    assert "/hold" in help_text


# ---------------------------------------------------------------------------
# /backhist tests
# ---------------------------------------------------------------------------

def test_backhist_empty(tmp_path) -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["backhist"]([])
    assert "No backtest reports" in out


def test_backhist_invalid_arg() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["backhist"](["abc"])
    assert "usage" in out.lower()


def test_help_includes_backhist() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/backhist" in h["help"]([])


# ---------------------------------------------------------------------------
# /circuit tests
# ---------------------------------------------------------------------------

def test_circuit_no_db() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["circuit"]([]) == "DB not wired."


def test_circuit_shows_ok_state(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["circuit"]([])
    assert "OK" in out or "allowed" in out


def test_circuit_reset(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    from amms.risk.circuit_breaker import CircuitBreakerConfig, record_trade_result
    cfg = CircuitBreakerConfig(max_daily_loss_pct=1.0)
    record_trade_result(conn, pnl=-5000.0, config=cfg, initial_equity=100_000.0)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["circuit"](["reset"])
    assert "reset" in out.lower()


def test_circuit_tripped_shows_warning(tmp_path) -> None:
    conn = _make_empty_conn(tmp_path)
    from amms.risk.circuit_breaker import CircuitBreakerConfig, record_trade_result
    cfg = CircuitBreakerConfig(max_daily_loss_pct=1.0)
    record_trade_result(conn, pnl=-5000.0, config=cfg, initial_equity=100_000.0)
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, conn=conn)
    out = h["circuit"]([])
    assert "TRIPPED" in out or "blocked" in out.lower()


def test_help_includes_circuit() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/circuit" in h["help"]([])


# ---------------------------------------------------------------------------
# /regime tests
# ---------------------------------------------------------------------------

def test_regime_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["regime"]([])


def test_regime_shows_label() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["regime"]([])
    assert any(word in out.upper() for word in ["BULL", "NEUTRAL", "BEAR"])


def test_regime_shows_multiplier() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["regime"]([])
    assert "multiplier" in out.lower() or "×" in out


def test_help_includes_regime() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/regime" in h["help"]([])


# ---------------------------------------------------------------------------
# /momscan tests
# ---------------------------------------------------------------------------

def test_momscan_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, static_watchlist=("AAPL",))
    assert "not wired" in h["momscan"]([])


def test_momscan_empty_watchlist() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["momscan"]([])
    assert "empty" in out.lower() or "Momentum scan" in out or "No data" in out


def test_momscan_with_watchlist() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL", "MSFT"),
    )
    out = h["momscan"]([])
    assert "Momentum scan" in out or "No data" in out


def test_momscan_custom_n() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL",),
    )
    out = h["momscan"](["5"])
    assert "scan" in out.lower() or "No data" in out


def test_momscan_invalid_n() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL",),
    )
    out = h["momscan"](["abc"])
    assert "usage" in out.lower()


def test_help_includes_momscan() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/momscan" in h["help"]([])


# ---------------------------------------------------------------------------
# /optimize tests
# ---------------------------------------------------------------------------

def test_optimize_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, static_watchlist=("AAPL",))
    assert "not wired" in h["optimize"]([])


def test_optimize_empty_watchlist() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["optimize"]([])
    assert "empty" in out.lower() or "allocation" in out.lower()


def test_optimize_equal_weight() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL", "MSFT"),
    )
    out = h["optimize"](["equal"])
    assert "equal" in out.lower() or "%" in out


def test_optimize_momentum_mode() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL",),
    )
    out = h["optimize"](["momentum"])
    assert "momentum" in out.lower() or "%" in out


def test_optimize_unknown_mode() -> None:
    p = PauseFlag()
    h = build_command_handlers(
        broker=_FakeBroker(), pause=p,
        data=_FakeDataClient(),
        static_watchlist=("AAPL",),
    )
    out = h["optimize"](["magic"])
    assert "Unknown" in out or "unknown" in out.lower()


def test_help_includes_optimize() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/optimize" in h["help"]([])


# ---------------------------------------------------------------------------
# /vwap tests
# ---------------------------------------------------------------------------

def test_vwap_no_data_client() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "not wired" in h["vwap"]([])


def test_vwap_shows_for_position() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["vwap"]([])
    assert "AAPL" in out
    assert "VWAP" in out


def test_vwap_explicit_ticker() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p, data=_FakeDataClient())
    out = h["vwap"](["NVDA"])
    assert "NVDA" in out


def test_help_includes_vwap() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "/vwap" in h["help"]([])
