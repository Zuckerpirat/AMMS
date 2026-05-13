from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from amms import db
from amms.data.bars import Bar, MarketDataClient, upsert_bars

DATA_URL = "https://data.alpaca.markets"


@respx.mock
def test_get_bars_handles_pagination() -> None:
    page_1 = {
        "bars": {
            "AAPL": [
                {"t": "2025-01-02T05:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100},
            ]
        },
        "next_page_token": "tok",
    }
    page_2 = {
        "bars": {
            "AAPL": [
                {"t": "2025-01-03T05:00:00Z", "o": 1.5, "h": 2.5, "l": 1.0, "c": 2.0, "v": 200},
            ]
        },
        "next_page_token": None,
    }
    route = respx.get(f"{DATA_URL}/v2/stocks/bars").mock(
        side_effect=[httpx.Response(200, json=page_1), httpx.Response(200, json=page_2)]
    )
    with MarketDataClient("k", "s", DATA_URL) as client:
        bars = client.get_bars("aapl", "1Day", "2025-01-01", "2025-01-04")
    assert route.call_count == 2
    assert [b.ts for b in bars] == ["2025-01-02T05:00:00Z", "2025-01-03T05:00:00Z"]
    assert all(b.symbol == "AAPL" for b in bars)


def test_upsert_bars_writes_and_updates(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    first = [Bar("AAPL", "1Day", "2025-01-02T05:00:00Z", 1.0, 2.0, 0.5, 1.5, 100)]
    assert upsert_bars(conn, first) == 1
    updated = [Bar("AAPL", "1Day", "2025-01-02T05:00:00Z", 1.0, 2.0, 0.5, 1.9, 999)]
    assert upsert_bars(conn, updated) == 1
    row = conn.execute(
        "SELECT close, volume FROM bars WHERE symbol=? AND ts=?",
        ("AAPL", "2025-01-02T05:00:00Z"),
    ).fetchone()
    assert row["close"] == pytest.approx(1.9)
    assert row["volume"] == pytest.approx(999)
    conn.close()


def test_upsert_bars_handles_empty(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    assert upsert_bars(conn, []) == 0
    conn.close()
