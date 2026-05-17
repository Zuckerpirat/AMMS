-- DE signal history: records each Decision Engine scan result for a symbol.
-- Used for signal evaluation, mode performance tracking, and audit trail.
CREATE TABLE IF NOT EXISTS de_signal_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,          -- ISO timestamp when signal was generated
    symbol      TEXT NOT NULL,
    mode        TEXT NOT NULL,          -- trading mode (swing/meme/conservative/event)
    action      TEXT NOT NULL,          -- strong_buy/buy/hold/sell/strong_sell
    score       REAL NOT NULL,          -- composite score -100..+100
    confidence  REAL NOT NULL,          -- 0..1
    horizon     TEXT,                   -- holding horizon estimate
    price       REAL,                   -- price at signal time
    macro_level TEXT                    -- macro regime level at signal time (calm/elevated/stressed)
);

CREATE INDEX IF NOT EXISTS idx_de_signal_history_ts     ON de_signal_history(ts);
CREATE INDEX IF NOT EXISTS idx_de_signal_history_symbol ON de_signal_history(symbol);
CREATE INDEX IF NOT EXISTS idx_de_signal_history_action ON de_signal_history(action);
