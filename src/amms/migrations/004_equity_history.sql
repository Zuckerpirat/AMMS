-- Migration 004: equity history snapshots
-- Stores periodic portfolio value snapshots for equity curve / Sharpe tracking.

CREATE TABLE IF NOT EXISTS equity_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT    NOT NULL,              -- ISO-8601 UTC timestamp
    equity    REAL    NOT NULL,              -- total portfolio value
    cash      REAL    NOT NULL,              -- cash component
    positions REAL    NOT NULL,             -- total market value of positions
    n_pos     INTEGER NOT NULL DEFAULT 0    -- number of open positions
);

CREATE INDEX IF NOT EXISTS idx_equity_history_ts ON equity_history(ts);
