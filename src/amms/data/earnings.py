"""Earnings calendar lookup using the free Nasdaq API.

No API key required. Returns upcoming earnings dates for a list of symbols.
Falls back to empty results on network errors — earnings dates are best-effort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.nasdaq.com/api/calendar/earnings"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; amms-bot/1.0)",
    "Accept": "application/json, text/plain, */*",
}


@dataclass(frozen=True)
class EarningsEvent:
    symbol: str
    date: str  # ISO date yyyy-mm-dd
    time: str  # "before-market" | "after-market" | "during-market" | ""
    eps_estimate: str


def fetch_upcoming(
    symbols: list[str],
    *,
    days_ahead: int = 14,
    timeout: float = 8.0,
) -> list[EarningsEvent]:
    """Return earnings events for ``symbols`` within the next ``days_ahead`` days.

    Returns an empty list on any error — callers treat this as best-effort.
    """
    if not symbols:
        return []
    symbols_upper = {s.upper() for s in symbols}
    events: list[EarningsEvent] = []

    today = date.today()
    for offset in range(days_ahead):
        day = today + timedelta(days=offset)
        day_events = _fetch_day(day, timeout=timeout)
        for ev in day_events:
            if ev.symbol in symbols_upper:
                events.append(ev)

    return events


def _fetch_day(day: date, *, timeout: float = 8.0) -> list[EarningsEvent]:
    """Fetch the earnings calendar for a specific date."""
    try:
        resp = httpx.get(
            _BASE,
            params={"date": day.isoformat()},
            headers=_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = (
            data.get("data", {})
            .get("rows", [])
        )
        events: list[EarningsEvent] = []
        for row in rows:
            sym = (row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            events.append(
                EarningsEvent(
                    symbol=sym,
                    date=day.isoformat(),
                    time=row.get("time", ""),
                    eps_estimate=str(row.get("epsForecast") or "N/A"),
                )
            )
        return events
    except Exception:
        logger.debug("earnings fetch failed for %s", day, exc_info=True)
        return []
