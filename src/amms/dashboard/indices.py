"""International index quotes from Yahoo Finance.

Lightweight in-memory cache (default 60s TTL) keeps the dashboard from
hammering Yahoo on every poll. Falls back to stale data on transient
errors so the UI stays usable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

import httpx

log = logging.getLogger(__name__)

INDICES: dict[str, tuple[str, str]] = {
    "sp500": ("^GSPC", "S&P 500"),
    "nasdaq": ("^IXIC", "Nasdaq"),
    "dow": ("^DJI", "Dow Jones"),
    "dax": ("^GDAXI", "DAX"),
    "ftse": ("^FTSE", "FTSE 100"),
    "nikkei": ("^N225", "Nikkei 225"),
    "vix": ("^VIX", "VIX"),
}

CACHE_TTL_SEC = 60.0
_cache: dict[str, tuple[float, IndexQuote]] = {}


@dataclass(frozen=True)
class IndexQuote:
    key: str
    symbol: str
    name: str
    price: float
    prev_close: float
    day_change_abs: float
    day_change_pct: float
    history: list[float] = field(default_factory=list)
    sparkline_points: str = ""
    is_stale: bool = False
    error: str = ""


def _sparkline(values: list[float], width: int = 220, height: int = 50) -> str:
    if len(values) < 2:
        return ""
    lo = min(values)
    hi = max(values)
    span = hi - lo if hi > lo else 1.0
    n = len(values)
    return " ".join(
        f"{(i / (n - 1)) * width:.1f},{height - ((v - lo) / span) * height:.1f}"
        for i, v in enumerate(values)
    )


def _fetch_yahoo(symbol: str) -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = httpx.get(
        url,
        params={"interval": "5m", "range": "1d"},
        timeout=5.0,
        headers={"User-Agent": "Mozilla/5.0 (compatible; amms-dashboard)"},
    )
    r.raise_for_status()
    return r.json()


def _parse(key: str, symbol: str, name: str, data: dict) -> IndexQuote:
    result = data["chart"]["result"][0]
    meta = result["meta"]
    quote = result["indicators"]["quote"][0]
    closes = [float(c) for c in quote.get("close", []) if c is not None]
    price = float(meta["regularMarketPrice"])
    prev = float(meta.get("chartPreviousClose", meta.get("previousClose", price)))
    abs_change = price - prev
    pct = (abs_change / prev * 100.0) if prev else 0.0
    return IndexQuote(
        key=key,
        symbol=symbol,
        name=name,
        price=price,
        prev_close=prev,
        day_change_abs=abs_change,
        day_change_pct=pct,
        history=closes,
        sparkline_points=_sparkline(closes),
    )


def get_index(key: str, *, ttl: float = CACHE_TTL_SEC) -> IndexQuote:
    """Return a cached or freshly fetched quote for a known index key."""
    if key not in INDICES:
        return IndexQuote(
            key=key, symbol="?", name="Unbekannt",
            price=0.0, prev_close=0.0,
            day_change_abs=0.0, day_change_pct=0.0,
            error="unbekannter Index",
        )
    symbol, name = INDICES[key]
    cached = _cache.get(key)
    now = time.time()
    if cached and (now - cached[0]) < ttl:
        return cached[1]
    try:
        quote = _parse(key, symbol, name, _fetch_yahoo(symbol))
        _cache[key] = (now, quote)
        return quote
    except Exception as e:
        log.warning("Yahoo fetch failed for %s: %s", symbol, e)
        if cached:
            return replace(cached[1], is_stale=True)
        return IndexQuote(
            key=key, symbol=symbol, name=name,
            price=0.0, prev_close=0.0,
            day_change_abs=0.0, day_change_pct=0.0,
            error=str(e),
        )


def clear_cache() -> None:
    _cache.clear()
