-- Phase 4b: persist per-symbol feature snapshots and signal scores.

CREATE TABLE IF NOT EXISTS features (
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (ts, symbol, name)
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_features_name_ts ON features(name, ts DESC);

ALTER TABLE signals ADD COLUMN score REAL;
