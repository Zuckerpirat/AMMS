from __future__ import annotations

import httpx
import pytest
import respx

from amms.notifier import NullNotifier, TelegramNotifier, build_notifier
from amms.notifier.telegram import TelegramNotifier as _T


def test_null_notifier_is_noop() -> None:
    NullNotifier().send("hello")  # should not raise


def test_build_notifier_returns_null_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert isinstance(build_notifier(), NullNotifier)


def test_build_notifier_returns_telegram_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    notifier = build_notifier()
    assert isinstance(notifier, _T)
    assert notifier.token == "abc"
    assert notifier.chat_id == "42"


@respx.mock
def test_telegram_notifier_posts_to_api() -> None:
    route = respx.post("https://api.telegram.org/botT/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    TelegramNotifier(token="T", chat_id="42").send("hi")
    assert route.called
    import json as _json
    body = _json.loads(route.calls.last.request.read())
    assert body == {"chat_id": "42", "text": "hi"}


@respx.mock
def test_telegram_notifier_swallows_errors() -> None:
    respx.post("https://api.telegram.org/botT/sendMessage").mock(
        return_value=httpx.Response(500, json={"ok": False})
    )
    # Must not raise — Telegram outages should never crash the trading loop.
    TelegramNotifier(token="T", chat_id="42").send("hi")
