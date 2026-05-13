from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class Bar:
    symbol: str
    timeframe: str
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataClient:
    """Alpaca market-data v2 client (IEX feed by default, which is free)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://data.alpaca.markets",
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
                "User-Agent": "amms/0.1",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> MarketDataClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        *,
        feed: str = "iex",
        limit: int = 1000,
    ) -> list[Bar]:
        symbol_u = symbol.upper()
        params: dict[str, Any] = {
            "symbols": symbol_u,
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "feed": feed,
            "limit": limit,
            "adjustment": "raw",
        }
        bars: list[Bar] = []
        next_token: str | None = None
        while True:
            if next_token is not None:
                params["page_token"] = next_token
            resp = self._client.get(f"{self._base_url}/v2/stocks/bars", params=params)
            resp.raise_for_status()
            data = resp.json()
            bar_map = data.get("bars") or {}
            for entry in bar_map.get(symbol_u, []):
                bars.append(
                    Bar(
                        symbol=symbol_u,
                        timeframe=timeframe,
                        ts=entry["t"],
                        open=float(entry["o"]),
                        high=float(entry["h"]),
                        low=float(entry["l"]),
                        close=float(entry["c"]),
                        volume=float(entry["v"]),
                    )
                )
            next_token = data.get("next_page_token")
            if not next_token:
                break
        return bars


def upsert_bars(conn: sqlite3.Connection, bars: list[Bar]) -> int:
    if not bars:
        return 0
    conn.executemany(
        """
        INSERT INTO bars(symbol, timeframe, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume
        """,
        [
            (b.symbol, b.timeframe, b.ts, b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ],
    )
    return len(bars)
