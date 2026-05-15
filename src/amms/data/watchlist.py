"""Watchlist persistence layer.

Manages a personal watchlist stored in SQLite.
Table: watchlist (symbol TEXT PRIMARY KEY, added_ts TEXT, note TEXT)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class WatchEntry:
    symbol: str
    added_ts: str
    note: str


def ensure_table(conn) -> None:
    """Create watchlist table if it doesn't exist."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS watchlist ("
        "symbol TEXT PRIMARY KEY, "
        "added_ts TEXT NOT NULL, "
        "note TEXT DEFAULT ''"
        ")"
    )
    conn.commit()


def add(conn, symbol: str, note: str = "") -> bool:
    """Add a symbol. Returns True if added, False if already present."""
    ensure_table(conn)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        conn.execute(
            "INSERT INTO watchlist (symbol, added_ts, note) VALUES (?, ?, ?)",
            (symbol.upper(), ts, note),
        )
        conn.commit()
        return True
    except Exception:
        return False


def remove(conn, symbol: str) -> bool:
    """Remove a symbol. Returns True if removed, False if not found."""
    ensure_table(conn)
    cursor = conn.execute(
        "DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),)
    )
    conn.commit()
    return cursor.rowcount > 0


def list_all(conn) -> list[WatchEntry]:
    """Return all watchlist entries sorted by added_ts."""
    ensure_table(conn)
    rows = conn.execute(
        "SELECT symbol, added_ts, note FROM watchlist ORDER BY added_ts ASC"
    ).fetchall()
    return [WatchEntry(symbol=r[0], added_ts=r[1], note=r[2] or "") for r in rows]


def contains(conn, symbol: str) -> bool:
    """Check if a symbol is on the watchlist."""
    ensure_table(conn)
    row = conn.execute(
        "SELECT 1 FROM watchlist WHERE symbol = ?", (symbol.upper(),)
    ).fetchone()
    return row is not None
