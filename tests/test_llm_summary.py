from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from amms import db
from amms.notifier.llm_summary import augment_summary


def test_no_api_key_returns_plain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    out = augment_summary("plain", trades_today=[], conn=conn)
    assert out == "plain"
    conn.close()


@respx.mock
def test_appends_llm_paragraph_when_api_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "Solid day, no losses."}]},
        )
    )
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    out = augment_summary("plain", trades_today=[{"symbol": "AAPL"}], conn=conn)
    assert "Solid day" in out
    assert out.startswith("plain")
    conn.close()


@respx.mock
def test_fallback_on_api_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    out = augment_summary("plain", trades_today=[], conn=conn)
    assert out == "plain"
    conn.close()


@respx.mock
def test_cached_per_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "cached."}]},
        )
    )
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    augment_summary("plain", trades_today=[], conn=conn)
    augment_summary("plain", trades_today=[], conn=conn)
    augment_summary("plain", trades_today=[], conn=conn)
    assert route.call_count == 1  # Cached on subsequent calls
    conn.close()
