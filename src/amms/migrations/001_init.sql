-- Phase 1 initial schema. All timestamps are ISO-8601 UTC strings.

CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    client_order_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    qty REAL NOT NULL CHECK (qty > 0),
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    filled_at TEXT,
    filled_avg_price REAL,
    raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TEXT NOT NULL PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    buying_power REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    signal TEXT NOT NULL CHECK (signal IN ('buy', 'sell', 'hold')),
    reason TEXT,
    PRIMARY KEY (ts, symbol, strategy)
);
