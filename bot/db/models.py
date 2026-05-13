import os
import sqlite3
from config import settings


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL,
                side        TEXT    NOT NULL,
                qty         INTEGER NOT NULL,
                price       REAL,
                order_id    TEXT,
                status      TEXT    DEFAULT 'pending',
                reason      TEXT,
                created_at  TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                equity           REAL    NOT NULL,
                cash             REAL    NOT NULL,
                positions_count  INTEGER NOT NULL,
                created_at       TEXT    DEFAULT (datetime('now'))
            );
        """)
