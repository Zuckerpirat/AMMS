"""Price alert management.

Alerts are stored in SQLite and checked each scheduler tick. When a price
crosses the configured threshold the alert fires once (marked triggered=1)
and a Telegram message is sent.

Schema (created on first use):
    price_alerts(id INTEGER PRIMARY KEY, symbol TEXT, price REAL,
                 direction TEXT,  -- 'above' | 'below'
                 created_at TEXT, triggered INTEGER)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class PriceAlert:
    id: int
    symbol: str
    price: float
    direction: str  # 'above' | 'below'
    created_at: str
    triggered: bool


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS price_alerts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "symbol TEXT NOT NULL, "
        "price REAL NOT NULL, "
        "direction TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "triggered INTEGER NOT NULL DEFAULT 0"
        ")"
    )
    conn.commit()


def add_alert(
    conn: sqlite3.Connection, symbol: str, price: float, direction: str
) -> PriceAlert:
    """Add a new price alert. direction must be 'above' or 'below'."""
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    if price <= 0:
        raise ValueError("price must be positive")
    ensure_table(conn)
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        "INSERT INTO price_alerts(symbol, price, direction, created_at) VALUES (?, ?, ?, ?)",
        (symbol.upper(), price, direction, now),
    )
    conn.commit()
    return PriceAlert(
        id=cur.lastrowid,
        symbol=symbol.upper(),
        price=price,
        direction=direction,
        created_at=now,
        triggered=False,
    )


def list_alerts(conn: sqlite3.Connection, *, include_triggered: bool = False) -> list[PriceAlert]:
    ensure_table(conn)
    if include_triggered:
        rows = conn.execute(
            "SELECT id, symbol, price, direction, created_at, triggered "
            "FROM price_alerts ORDER BY id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, symbol, price, direction, created_at, triggered "
            "FROM price_alerts WHERE triggered = 0 ORDER BY id"
        ).fetchall()
    return [
        PriceAlert(
            id=r[0], symbol=r[1], price=r[2], direction=r[3],
            created_at=r[4], triggered=bool(r[5])
        )
        for r in rows
    ]


def delete_alert(conn: sqlite3.Connection, alert_id: int) -> bool:
    """Delete an alert by id. Returns True if a row was deleted."""
    ensure_table(conn)
    cur = conn.execute("DELETE FROM price_alerts WHERE id = ?", (alert_id,))
    conn.commit()
    return cur.rowcount > 0


def mark_triggered(conn: sqlite3.Connection, alert_id: int) -> None:
    ensure_table(conn)
    conn.execute(
        "UPDATE price_alerts SET triggered = 1 WHERE id = ?", (alert_id,)
    )
    conn.commit()


def check_alerts(
    conn: sqlite3.Connection, prices: dict[str, float]
) -> list[PriceAlert]:
    """Return alerts that have just been triggered by the given price snapshot.

    Each returned alert is immediately marked triggered so it won't fire again.
    """
    active = list_alerts(conn)
    fired: list[PriceAlert] = []
    for alert in active:
        current = prices.get(alert.symbol)
        if current is None:
            continue
        crossed = (
            alert.direction == "above" and current >= alert.price
            or alert.direction == "below" and current <= alert.price
        )
        if crossed:
            mark_triggered(conn, alert.id)
            fired.append(alert)
    return fired
