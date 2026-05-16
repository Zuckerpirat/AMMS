"""Earnings calendar storage and lookup.

Stores upcoming earnings dates in a local SQLite table so that:
  - Positions can be flagged before earnings events
  - Risk can be assessed for holding through earnings
  - The system can warn about imminent earnings

Table schema: earnings_calendar
  symbol TEXT
  report_date TEXT          (ISO date YYYY-MM-DD)
  time_of_day TEXT          "before_open" | "after_close" | "unknown"
  added_ts TEXT
  note TEXT

This is a pure data storage layer — no fetching from external APIs.
Dates are added manually or by a future data ingestion module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class EarningsEntry:
    symbol: str
    report_date: str          # YYYY-MM-DD
    time_of_day: str          # "before_open" | "after_close" | "unknown"
    days_until: int | None    # None if date is in the past
    added_ts: str
    note: str


def ensure_table(conn) -> None:
    """Create earnings_calendar table if it doesn't exist."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS earnings_calendar ("
        "symbol TEXT NOT NULL, "
        "report_date TEXT NOT NULL, "
        "time_of_day TEXT NOT NULL DEFAULT 'unknown', "
        "added_ts TEXT NOT NULL, "
        "note TEXT NOT NULL DEFAULT '', "
        "PRIMARY KEY (symbol, report_date)"
        ")"
    )
    conn.commit()


def add(
    conn,
    symbol: str,
    report_date: str,
    *,
    time_of_day: str = "unknown",
    note: str = "",
) -> bool:
    """Add or update an earnings entry. Returns True if inserted/updated."""
    from datetime import datetime, timezone

    symbol = symbol.upper().strip()
    ensure_table(conn)

    # Validate date format
    try:
        date.fromisoformat(report_date)
    except ValueError:
        return False

    valid_times = {"before_open", "after_close", "unknown"}
    if time_of_day not in valid_times:
        time_of_day = "unknown"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        conn.execute(
            "INSERT OR REPLACE INTO earnings_calendar "
            "(symbol, report_date, time_of_day, added_ts, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol, report_date, time_of_day, ts, note),
        )
        conn.commit()
        return True
    except Exception:
        return False


def remove(conn, symbol: str, report_date: str | None = None) -> int:
    """Remove earnings entries. Returns count deleted.

    If report_date is None, removes all entries for the symbol.
    """
    symbol = symbol.upper().strip()
    ensure_table(conn)
    try:
        if report_date is None:
            cursor = conn.execute(
                "DELETE FROM earnings_calendar WHERE symbol = ?", (symbol,)
            )
        else:
            cursor = conn.execute(
                "DELETE FROM earnings_calendar WHERE symbol = ? AND report_date = ?",
                (symbol, report_date),
            )
        conn.commit()
        return cursor.rowcount
    except Exception:
        return 0


def upcoming(
    conn,
    *,
    within_days: int = 30,
    symbols: list[str] | None = None,
    today: date | None = None,
) -> list[EarningsEntry]:
    """Return upcoming earnings within the next within_days days.

    symbols: if provided, only return entries for these symbols
    today: override today's date (for testing)
    """
    ensure_table(conn)
    if today is None:
        today = date.today()

    date_from = today.isoformat()
    date_to = (today + timedelta(days=within_days)).isoformat()

    try:
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            rows = conn.execute(
                f"SELECT symbol, report_date, time_of_day, added_ts, note "
                f"FROM earnings_calendar "
                f"WHERE report_date >= ? AND report_date <= ? "
                f"AND symbol IN ({placeholders}) "
                f"ORDER BY report_date, symbol",
                [date_from, date_to] + [s.upper() for s in symbols],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT symbol, report_date, time_of_day, added_ts, note "
                "FROM earnings_calendar "
                "WHERE report_date >= ? AND report_date <= ? "
                "ORDER BY report_date, symbol",
                (date_from, date_to),
            ).fetchall()
    except Exception:
        return []

    result = []
    for sym, rdate, tod, ts, note in rows:
        try:
            rd = date.fromisoformat(rdate)
            days = (rd - today).days
        except Exception:
            days = None
        result.append(EarningsEntry(
            symbol=sym,
            report_date=rdate,
            time_of_day=tod,
            days_until=days,
            added_ts=ts,
            note=note,
        ))
    return result


def check_positions(conn, symbols: list[str], *, within_days: int = 7, today: date | None = None) -> list[EarningsEntry]:
    """Check if any of the given symbols have earnings within within_days.

    Used by the risk layer to flag positions before earnings.
    Returns list of upcoming earnings for symbols in the portfolio.
    """
    return upcoming(conn, within_days=within_days, symbols=symbols, today=today)
