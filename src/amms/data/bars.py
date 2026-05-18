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

    def get_snapshots(
        self, symbols: list[str], *, feed: str = "iex"
    ) -> dict[str, dict[str, float]]:
        """Return latest price + daily and weekly change for each symbol.

        Output per symbol: {"price", "prev_close", "change_pct" (daily),
        "week_ago_close", "change_pct_week"}. Missing/failed symbols are
        omitted. The weekly close comes from the daily bar ~5 trading
        days ago, fetched via a single batched /v2/stocks/bars call.
        """
        if not symbols:
            return {}
        syms_csv = ",".join(s.upper() for s in symbols)

        # Step 1: latest price + previous day close via snapshots.
        try:
            resp = self._client.get(
                f"{self._base_url}/v2/stocks/snapshots",
                params={"symbols": syms_csv, "feed": feed},
            )
            resp.raise_for_status()
            snap_data = resp.json()
        except Exception:
            return {}

        # Step 2: weekly reference close via daily bars over the past
        # ~10 calendar days (covers weekends and short holidays).
        from datetime import UTC, datetime, timedelta

        end = datetime.now(UTC).date()
        start = end - timedelta(days=14)
        weekly_close: dict[str, float] = {}
        try:
            bars_resp = self._client.get(
                f"{self._base_url}/v2/stocks/bars",
                params={
                    "symbols": syms_csv,
                    "timeframe": "1Day",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "feed": feed,
                    "limit": 1000,
                    "adjustment": "raw",
                },
            )
            bars_resp.raise_for_status()
            bars_map = (bars_resp.json().get("bars") or {})
            for sym, bars in bars_map.items():
                if not bars:
                    continue
                # Reference = bar closest to "5 trading days ago" = 6th from end.
                ref = bars[-6] if len(bars) >= 6 else bars[0]
                close = ref.get("c")
                if close:
                    weekly_close[sym.upper()] = float(close)
        except Exception:
            weekly_close = {}

        out: dict[str, dict[str, float]] = {}
        snapshots = snap_data.get("snapshots") or snap_data
        for sym, snap in snapshots.items():
            if not isinstance(snap, dict):
                continue
            trade = snap.get("latestTrade") or {}
            prev = snap.get("prevDailyBar") or {}
            price = trade.get("p")
            if price is None:
                continue
            price = float(price)
            prev_close = prev.get("c")
            day_pct = (
                (price - float(prev_close)) / float(prev_close) * 100
                if prev_close
                else 0.0
            )
            wk_close = weekly_close.get(sym.upper(), 0.0)
            wk_pct = (price - wk_close) / wk_close * 100 if wk_close else 0.0
            out[sym.upper()] = {
                "price": price,
                "prev_close": float(prev_close or 0.0),
                "change_pct": day_pct,
                "week_ago_close": wk_close,
                "change_pct_week": wk_pct,
            }
        return out

    def get_news(
        self, symbols: list[str], *, limit: int = 5
    ) -> list[dict]:
        """Return recent news articles for one or more symbols.

        Each dict has keys: headline, summary, url, created_at, symbols.
        Returns up to ``limit`` articles sorted newest-first.
        Falls back to [] on any error (news is best-effort).
        """
        if not symbols:
            return []
        try:
            params: dict[str, Any] = {
                "symbols": ",".join(s.upper() for s in symbols),
                "limit": limit,
                "sort": "desc",
            }
            resp = self._client.get(
                f"{self._base_url}/v1beta1/news", params=params
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("news", [])
        except Exception:
            return []

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

    def get_assets(
        self,
        exchanges: list[str] | None = None,
        *,
        min_price: float = 0.5,
        max_price: float | None = None,
    ) -> list[dict]:
        """Return active tradeable US equity assets from Alpaca.

        Filters by exchange (default: NASDAQ + NYSE + ARCA) and optionally
        by price range. Returns list of dicts with 'symbol', 'name', 'exchange'.
        Cached in-memory for 6 hours.
        """
        import time as _time
        cache = getattr(self, "_assets_cache", None)
        if cache and _time.monotonic() - cache["ts"] < 21600:
            assets = cache["data"]
        else:
            exchanges = exchanges or ["NASDAQ", "NYSE", "ARCA", "BATS"]
            assets: list[dict] = []
            for exchange in exchanges:
                try:
                    resp = self._client.get(
                        "https://paper-api.alpaca.markets/v2/assets",
                        params={
                            "status": "active",
                            "asset_class": "us_equity",
                            "exchange": exchange,
                        },
                        timeout=15,
                    )
                    resp.raise_for_status()
                    for a in resp.json():
                        if a.get("tradable") and a.get("symbol"):
                            sym = a["symbol"].upper()
                            # Skip symbols with special characters (warrants, units)
                            if any(c in sym for c in "=.+/-"):
                                continue
                            assets.append({
                                "symbol": sym,
                                "name": a.get("name", ""),
                                "exchange": a.get("exchange", exchange),
                            })
                except Exception:
                    pass
            self._assets_cache = {"ts": _time.monotonic(), "data": assets}

        if min_price or max_price:
            return assets  # price filtering done later via snapshots
        return assets

    def get_broad_news(self, *, limit: int = 50) -> list[dict]:
        """Fetch recent market-wide news (no symbol filter).

        Returns articles sorted newest-first. Each article has:
        headline, summary, created_at, symbols (list of tickers mentioned).
        """
        try:
            resp = self._client.get(
                f"{self._base_url}/v1beta1/news",
                params={"limit": limit, "sort": "desc"},
            )
            resp.raise_for_status()
            return resp.json().get("news", [])
        except Exception:
            return []

    def get_snapshots_bulk(
        self, symbols: list[str], *, feed: str = "iex", batch_size: int = 200
    ) -> dict[str, dict]:
        """Efficient bulk snapshot fetch for large symbol lists.

        Batches requests to stay within API limits. Returns dict of
        symbol → raw snapshot data (dailyBar, latestTrade, etc.).
        """
        if not symbols:
            return {}
        results: dict[str, dict] = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                resp = self._client.get(
                    f"{self._base_url}/v2/stocks/snapshots",
                    params={"symbols": ",".join(batch), "feed": feed},
                    timeout=15,
                )
                resp.raise_for_status()
                results.update(resp.json())
            except Exception:
                pass
        return results


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
