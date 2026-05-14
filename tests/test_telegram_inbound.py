from __future__ import annotations

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
    out = h["status"]()
    assert "$100,000" in out
    assert "open positions: 1" in out
    assert "paused: False" in out


def test_pause_resume_handlers_flip_flag() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert "paused" in h["pause"]().lower()
    assert p.paused is True
    assert "resumed" in h["resume"]().lower()
    assert p.paused is False


def test_equity_handler() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    assert h["equity"]() == "$100,000.00"


def test_positions_handler() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["positions"]()
    assert "AAPL" in out


def test_help_lists_scan_command() -> None:
    p = PauseFlag()
    h = build_command_handlers(broker=_FakeBroker(), pause=p)
    out = h["help"]()
    assert "/scan" in out


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
    out = h["scan"]()
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
    out = h["scan"]()
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
