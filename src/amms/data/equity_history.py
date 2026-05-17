"""Equity history: snapshot portfolio value periodically for curve/metrics.

Records portfolio equity snapshots to the database (migration 004).
Provides:
  - record_snapshot()  — insert one snapshot
  - fetch_history()    — retrieve snapshots in a date range
  - compute_stats()    — Sharpe, max drawdown, CAGR from snapshot series
  - format_curve()     — ASCII sparkline + key stats for Telegram
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EquitySnapshot:
    id: int
    ts: str
    equity: float
    cash: float
    positions: float
    n_pos: int


@dataclass
class EquityStats:
    first_ts: str
    last_ts: str
    first_equity: float
    last_equity: float
    n_snapshots: int
    total_return_pct: float
    cagr_pct: float           # compound annual growth rate
    max_drawdown_pct: float   # worst peak-to-trough
    sharpe_ratio: float | None   # annualized, rf=0
    volatility_pct: float     # annualized daily return std
    best_day_pct: float
    worst_day_pct: float


def record_snapshot(conn, trader) -> bool:
    """Insert one equity snapshot. Returns True on success."""
    if conn is None or trader is None:
        return False
    try:
        snap = trader.snapshot()
        conn.execute(
            "INSERT INTO equity_history (ts, equity, cash, positions, n_pos) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                round(snap.portfolio_value, 4),
                round(snap.cash, 4),
                round(snap.total_market_value, 4),
                len(snap.positions),
            ),
        )
        return True
    except Exception as exc:
        logger.debug("equity_history.record_snapshot failed: %s", exc)
        return False


def fetch_history(conn, *, days: int = 30, limit: int = 1000) -> list[EquitySnapshot]:
    """Fetch equity snapshots from the last N days, oldest first."""
    if conn is None:
        return []
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT id, ts, equity, cash, positions, n_pos "
            "FROM equity_history "
            "WHERE ts >= ? ORDER BY ts ASC LIMIT ?",
            (since, max(1, min(limit, 10_000))),
        ).fetchall()
        return [
            EquitySnapshot(
                id=r["id"], ts=r["ts"], equity=r["equity"], cash=r["cash"],
                positions=r["positions"], n_pos=r["n_pos"],
            )
            for r in rows
        ]
    except Exception as exc:
        logger.debug("equity_history.fetch_history failed: %s", exc)
        return []


def compute_stats(snapshots: list[EquitySnapshot]) -> EquityStats | None:
    """Compute performance statistics from a list of snapshots."""
    if len(snapshots) < 2:
        return None

    equities = [s.equity for s in snapshots]
    first, last = equities[0], equities[-1]

    # Daily returns (consecutive differences)
    returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            r = (equities[i] / equities[i - 1]) - 1.0
            returns.append(r)

    total_return = (last / first - 1.0) * 100.0 if first > 0 else 0.0

    # CAGR: estimate days between first and last snapshot
    try:
        t0 = datetime.fromisoformat(snapshots[0].ts)
        t1 = datetime.fromisoformat(snapshots[-1].ts)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=timezone.utc)
        years = max((t1 - t0).total_seconds() / (365.25 * 86400), 1 / 365)
    except Exception:
        years = len(snapshots) / 252

    cagr = ((last / first) ** (1.0 / years) - 1.0) * 100.0 if first > 0 else 0.0

    # Max drawdown
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Annualized volatility and Sharpe (rf=0)
    vol_pct = 0.0
    sharpe: float | None = None
    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        vol_pct = std_r * math.sqrt(252) * 100.0
        if std_r > 0:
            sharpe = (mean_r * 252) / (std_r * math.sqrt(252))

    best_day = max(returns) * 100.0 if returns else 0.0
    worst_day = min(returns) * 100.0 if returns else 0.0

    return EquityStats(
        first_ts=snapshots[0].ts,
        last_ts=snapshots[-1].ts,
        first_equity=first,
        last_equity=last,
        n_snapshots=len(snapshots),
        total_return_pct=round(total_return, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 2) if sharpe is not None else None,
        volatility_pct=round(vol_pct, 2),
        best_day_pct=round(best_day, 2),
        worst_day_pct=round(worst_day, 2),
    )


def _sparkline(values: list[float], width: int = 30) -> str:
    """ASCII sparkline from equity values."""
    if len(values) < 2:
        return "─" * width
    lo, hi = min(values), max(values)
    if hi == lo:
        return "─" * width
    chars = " ▁▂▃▄▅▆▇█"
    # Downsample to `width` points
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    result = []
    for v in sampled:
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        result.append(chars[idx])
    return "".join(result)


def format_equity_curve(snapshots: list[EquitySnapshot], stats: EquityStats | None) -> str:
    if not snapshots:
        return "No equity history recorded yet."

    equities = [s.equity for s in snapshots]
    lines = ["── Equity Curve ──", ""]
    lines.append("  " + _sparkline(equities, width=40))
    lines.append(f"  ${equities[0]:>12,.2f}  →  ${equities[-1]:>12,.2f}")

    if stats:
        lines += [
            "",
            f"  Return:      {stats.total_return_pct:>+.2f}%  over {stats.n_snapshots} snapshots",
            f"  CAGR:        {stats.cagr_pct:>+.2f}%",
            f"  Max drawdown:{stats.max_drawdown_pct:>6.2f}%",
        ]
        if stats.sharpe_ratio is not None:
            lines.append(f"  Sharpe:      {stats.sharpe_ratio:>+.2f}")
        lines += [
            f"  Volatility:  {stats.volatility_pct:>6.2f}% ann.",
            f"  Best day:    {stats.best_day_pct:>+.2f}%",
            f"  Worst day:   {stats.worst_day_pct:>+.2f}%",
        ]
    return "\n".join(lines)
