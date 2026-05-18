"""International index quotes from Yahoo Finance.

Lightweight in-memory cache (default 60s TTL) keeps the dashboard from
hammering Yahoo on every poll. Falls back to stale data on transient
errors so the UI stays usable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from zoneinfo import ZoneInfo

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
DISPLAY_TZ = ZoneInfo("Europe/Berlin")
_cache: dict[str, tuple[float, IndexQuote]] = {}


@dataclass(frozen=True)
class IndexChart:
    width: int = 260
    height: int = 90
    pad_left: int = 40
    pad_right: int = 8
    pad_top: int = 8
    pad_bottom: int = 18
    polyline: str = ""
    area: str = ""
    y_ticks: list[tuple[float, str]] = field(default_factory=list)
    x_ticks: list[tuple[float, str]] = field(default_factory=list)
    has_data: bool = False

    @property
    def plot_w(self) -> int:
        return self.width - self.pad_left - self.pad_right

    @property
    def plot_h(self) -> int:
        return self.height - self.pad_top - self.pad_bottom

    @property
    def plot_right_x(self) -> int:
        return self.width - self.pad_right

    @property
    def plot_bottom_y(self) -> float:
        return self.pad_top + self.plot_h


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
    chart: IndexChart = field(default_factory=IndexChart)
    is_stale: bool = False
    error: str = ""


def _fmt_index_value(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.0f}".replace(",", ".")
    return f"{v:.1f}".replace(".", ",")


def _build_index_chart(timestamps: list[int], values: list[float]) -> IndexChart:
    chart = IndexChart()
    if len(values) < 2:
        return chart

    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + max(abs(lo) * 0.001, 0.1)
    span = hi - lo
    pad = span * 0.08
    lo, hi = lo - pad, hi + pad
    span = hi - lo

    n = len(values)
    points: list[tuple[float, float]] = []
    for i, v in enumerate(values):
        x = chart.pad_left + (i / (n - 1)) * chart.plot_w
        y = chart.pad_top + chart.plot_h - ((v - lo) / span) * chart.plot_h
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    bottom = chart.plot_bottom_y
    area_seg = " ".join(f"L {x:.1f},{y:.1f}" for x, y in points)
    area = (
        f"M {points[0][0]:.1f},{bottom:.1f} "
        f"L {points[0][0]:.1f},{points[0][1]:.1f} "
        f"{area_seg} "
        f"L {points[-1][0]:.1f},{bottom:.1f} Z"
    )

    y_ticks = [
        (chart.pad_top + 4, _fmt_index_value(hi - pad)),
        (chart.plot_bottom_y - 2, _fmt_index_value(lo + pad)),
    ]

    x_ticks: list[tuple[float, str]] = []
    if timestamps and len(timestamps) == n:
        idxs = [0, n // 2, n - 1] if n >= 3 else [0, n - 1]
        seen: set[int] = set()
        for i in idxs:
            if i in seen:
                continue
            seen.add(i)
            x = chart.pad_left + (i / (n - 1)) * chart.plot_w
            ts = timestamps[i]
            try:
                label = datetime.fromtimestamp(ts, tz=DISPLAY_TZ).strftime("%H:%M")
            except (ValueError, OSError):
                label = ""
            x_ticks.append((x, label))

    return IndexChart(
        polyline=polyline,
        area=area,
        y_ticks=y_ticks,
        x_ticks=x_ticks,
        has_data=True,
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
    timestamps_raw = result.get("timestamp", []) or []
    closes_raw = quote.get("close", []) or []
    pairs = [
        (int(t), float(c))
        for t, c in zip(timestamps_raw, closes_raw, strict=False)
        if c is not None
    ]
    timestamps = [t for t, _ in pairs]
    closes = [c for _, c in pairs]
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
        chart=_build_index_chart(timestamps, closes),
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
