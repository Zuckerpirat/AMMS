from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from amms.broker.alpaca import Account, Order

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(conn: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> int:
    """Apply any unapplied SQL migrations in alphabetical order. Returns count applied.

    Migrations are expected to be idempotent (use IF NOT EXISTS, etc.) because
    the connection runs in autocommit mode and partial scripts may persist on
    failure. Replaying is safe.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied: set[int] = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    count = 0
    for path in sorted(Path(migrations_dir).glob("*.sql")):
        version = int(path.name.split("_", 1)[0])
        if version in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (version, datetime.now(UTC).isoformat()),
        )
        count += 1
    return count


def upsert_order(conn: sqlite3.Connection, order: Order) -> None:
    conn.execute(
        """
        INSERT INTO orders(
            id, client_order_id, symbol, side, qty, type, status,
            submitted_at, filled_at, filled_avg_price, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status,
            filled_at = excluded.filled_at,
            filled_avg_price = excluded.filled_avg_price,
            raw_json = excluded.raw_json
        """,
        (
            order.id,
            order.client_order_id,
            order.symbol,
            order.side,
            order.qty,
            order.type,
            order.status,
            order.submitted_at,
            order.filled_at,
            order.filled_avg_price,
            json.dumps(order.raw),
        ),
    )


def upsert_features(
    conn: sqlite3.Connection,
    ts: str,
    symbol: str,
    features: dict[str, float],
) -> int:
    """Persist a per-symbol feature snapshot. Returns row count written."""
    if not features:
        return 0
    rows = [(ts, symbol, name, float(value)) for name, value in features.items()]
    conn.executemany(
        """
        INSERT INTO features(ts, symbol, name, value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ts, symbol, name) DO UPDATE SET value = excluded.value
        """,
        rows,
    )
    return len(rows)


def latest_buy_submitted_at(
    conn: sqlite3.Connection, symbol: str
) -> str | None:
    """ISO timestamp of the most recent BUY we recorded for ``symbol``, or None.

    Used to enforce min_hold_days: even if a sell signal fires, we refuse
    to close a position we just opened.
    """
    row = conn.execute(
        "SELECT max(submitted_at) FROM orders WHERE symbol = ? AND side = 'buy'",
        (symbol,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return row[0]


def insert_equity_snapshot(conn: sqlite3.Connection, account: Account) -> str:
    ts = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO equity_snapshots(ts, equity, cash, buying_power)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ts) DO UPDATE SET
            equity = excluded.equity,
            cash = excluded.cash,
            buying_power = excluded.buying_power
        """,
        (ts, account.equity, account.cash, account.buying_power),
    )
    return ts
