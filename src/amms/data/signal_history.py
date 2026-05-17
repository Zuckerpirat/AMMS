"""DE signal history storage and retrieval.

Persists every Decision Engine signal to the database for:
  - Historical review of what signals fired and when
  - Mode-specific accuracy tracking (did the signal lead to profit?)
  - Audit trail for all autonomous decisions

Table: de_signal_history (created by migration 003)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalRecord:
    id: int
    ts: str
    symbol: str
    mode: str
    action: str
    score: float
    confidence: float
    horizon: str
    price: float | None
    macro_level: str


def record_signal(
    conn,
    *,
    symbol: str,
    mode: str,
    action: str,
    score: float,
    confidence: float,
    horizon: str = "",
    price: float | None = None,
    macro_level: str = "calm",
) -> bool:
    """Insert one DE signal record into the database.

    Returns True on success, False on error (non-fatal — never raises).
    """
    if conn is None:
        return False
    try:
        conn.execute(
            """
            INSERT INTO de_signal_history
              (ts, symbol, mode, action, score, confidence, horizon, price, macro_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                symbol.upper(),
                mode,
                action,
                round(score, 2),
                round(confidence, 4),
                horizon or "",
                price,
                macro_level or "calm",
            ),
        )
        return True
    except Exception as exc:
        logger.debug("signal_history.record_signal failed: %s", exc)
        return False


def fetch_recent(conn, *, limit: int = 50, symbol: str | None = None,
                 mode: str | None = None, action: str | None = None) -> list[SignalRecord]:
    """Fetch recent signal records, optionally filtered."""
    if conn is None:
        return []
    try:
        where_clauses: list[str] = []
        params: list = []
        if symbol:
            where_clauses.append("symbol = ?")
            params.append(symbol.upper())
        if mode:
            where_clauses.append("mode = ?")
            params.append(mode)
        if action:
            where_clauses.append("action = ?")
            params.append(action)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(max(1, min(limit, 500)))
        rows = conn.execute(
            f"SELECT id, ts, symbol, mode, action, score, confidence, "
            f"horizon, price, macro_level "
            f"FROM de_signal_history {where} "
            f"ORDER BY ts DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            SignalRecord(
                id=r["id"], ts=r["ts"], symbol=r["symbol"], mode=r["mode"],
                action=r["action"], score=r["score"], confidence=r["confidence"],
                horizon=r["horizon"] or "", price=r["price"], macro_level=r["macro_level"] or "calm",
            )
            for r in rows
        ]
    except Exception as exc:
        logger.debug("signal_history.fetch_recent failed: %s", exc)
        return []


def signal_accuracy_by_mode(conn) -> dict[str, dict]:
    """Aggregate signal counts per mode for a quick accuracy overview.

    Returns dict: {mode: {action: count, ...}, ...}
    """
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT mode, action, COUNT(*) AS cnt "
            "FROM de_signal_history "
            "GROUP BY mode, action "
            "ORDER BY mode, cnt DESC"
        ).fetchall()
    except Exception:
        return {}
    result: dict[str, dict] = {}
    for r in rows:
        result.setdefault(r["mode"], {})[r["action"]] = r["cnt"]
    return result
