"""Circuit breaker for automated trading.

Tracks intraday loss and consecutive loss streaks, halting new buys when
thresholds are exceeded. Resets at the start of each trading day.

This lives in the risk layer (CLAUDE.md §4) and is evaluated by the
executor before any buy decision is made.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    trade_date TEXT PRIMARY KEY,
    daily_loss  REAL NOT NULL DEFAULT 0.0,
    consec_losses INTEGER NOT NULL DEFAULT 0,
    tripped     INTEGER NOT NULL DEFAULT 0
)
"""


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Thresholds for the circuit breaker."""

    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 5
    enabled: bool = True


@dataclass
class CircuitBreakerState:
    trade_date: str
    daily_loss: float = 0.0
    consec_losses: int = 0
    tripped: bool = False

    @property
    def is_open(self) -> bool:
        """True when the circuit is tripped (trading is blocked)."""
        return self.tripped


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_TABLE_DDL)
    conn.commit()


def _today() -> str:
    return date.today().isoformat()


def load_state(conn: sqlite3.Connection) -> CircuitBreakerState:
    """Load today's circuit-breaker state from DB; create fresh row if absent."""
    ensure_table(conn)
    today = _today()
    row = conn.execute(
        "SELECT daily_loss, consec_losses, tripped FROM circuit_breaker_state "
        "WHERE trade_date = ?",
        (today,),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT OR IGNORE INTO circuit_breaker_state "
            "(trade_date, daily_loss, consec_losses, tripped) VALUES (?, 0.0, 0, 0)",
            (today,),
        )
        conn.commit()
        return CircuitBreakerState(trade_date=today)
    return CircuitBreakerState(
        trade_date=today,
        daily_loss=float(row[0]),
        consec_losses=int(row[1]),
        tripped=bool(row[2]),
    )


def record_trade_result(
    conn: sqlite3.Connection,
    *,
    pnl: float,
    config: CircuitBreakerConfig,
    initial_equity: float,
) -> CircuitBreakerState:
    """Update circuit breaker state with a closed trade's P&L.

    Returns updated state. Trips the circuit if thresholds exceeded.
    """
    state = load_state(conn)
    if state.tripped:
        return state

    new_loss = state.daily_loss + (abs(pnl) if pnl < 0 else 0.0)
    new_consec = state.consec_losses + 1 if pnl <= 0 else 0

    tripped = False
    if config.enabled:
        loss_pct = (new_loss / initial_equity * 100) if initial_equity > 0 else 0.0
        if loss_pct >= config.max_daily_loss_pct:
            tripped = True
            logger.warning(
                "circuit breaker tripped: daily loss %.2f%% >= threshold %.2f%%",
                loss_pct, config.max_daily_loss_pct,
            )
        if new_consec >= config.max_consecutive_losses:
            tripped = True
            logger.warning(
                "circuit breaker tripped: %d consecutive losses >= threshold %d",
                new_consec, config.max_consecutive_losses,
            )

    conn.execute(
        "INSERT INTO circuit_breaker_state "
        "(trade_date, daily_loss, consec_losses, tripped) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(trade_date) DO UPDATE SET "
        "daily_loss=excluded.daily_loss, consec_losses=excluded.consec_losses, "
        "tripped=excluded.tripped",
        (_today(), new_loss, new_consec, int(tripped)),
    )
    conn.commit()
    return CircuitBreakerState(
        trade_date=_today(),
        daily_loss=new_loss,
        consec_losses=new_consec,
        tripped=tripped,
    )


def reset_circuit(conn: sqlite3.Connection) -> None:
    """Manually reset today's circuit (admin use only)."""
    today = _today()
    conn.execute(
        "INSERT INTO circuit_breaker_state "
        "(trade_date, daily_loss, consec_losses, tripped) VALUES (?, 0.0, 0, 0) "
        "ON CONFLICT(trade_date) DO UPDATE SET "
        "daily_loss=0.0, consec_losses=0, tripped=0",
        (today,),
    )
    conn.commit()
    logger.info("circuit breaker reset for %s", today)


def is_open(conn: sqlite3.Connection) -> bool:
    """Quick check: is the circuit currently tripped (blocking new buys)?"""
    return load_state(conn).is_open
