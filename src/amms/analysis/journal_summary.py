"""Periodic trade journal summary (weekly / monthly).

Aggregates closed trades from trade_pairs by calendar period and
produces compact performance summaries for each period.

Metrics per period:
  - n_trades, n_wins, n_losses
  - win_rate
  - total_pnl, avg_pnl
  - best_trade, worst_trade
  - profit_factor

Also identifies the best and worst periods overall.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodStats:
    period: str          # e.g. "2026-W03" or "2026-05"
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float      # 0-100
    total_pnl: float
    avg_pnl: float
    best_trade: float
    worst_trade: float
    profit_factor: float | None


@dataclass(frozen=True)
class JournalSummary:
    periods: list[PeriodStats]    # chronological
    mode: str                     # "weekly" or "monthly"
    best_period: str | None
    worst_period: str | None
    overall_win_rate: float
    overall_pnl: float
    n_periods: int
    n_trades: int


def _period_key(sell_ts: str, mode: str) -> str | None:
    """Extract period key from timestamp string."""
    if not sell_ts:
        return None
    try:
        date_str = str(sell_ts)[:10]  # YYYY-MM-DD
        year, month, day = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
        if mode == "monthly":
            return f"{year:04d}-{month:02d}"
        else:
            # ISO week
            from datetime import date
            d = date(year, month, day)
            iso = d.isocalendar()
            return f"{iso[0]:04d}-W{iso[1]:02d}"
    except Exception:
        return None


def compute(conn, *, mode: str = "monthly", limit: int = 500) -> JournalSummary | None:
    """Compute periodic journal summary from trade_pairs.

    conn: SQLite connection
    mode: "monthly" or "weekly"
    limit: max trades to analyze

    Returns None if fewer than 3 trades.
    """
    if mode not in ("monthly", "weekly"):
        mode = "monthly"

    try:
        rows = conn.execute(
            "SELECT sell_ts, pnl FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND sell_price IS NOT NULL "
            "ORDER BY sell_ts ASC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 3:
        return None

    # Group by period
    grouped: dict[str, list[float]] = {}
    for sell_ts, pnl in rows:
        key = _period_key(str(sell_ts) if sell_ts else "", mode)
        if key is None:
            continue
        grouped.setdefault(key, []).append(float(pnl))

    if not grouped:
        return None

    period_stats: list[PeriodStats] = []
    for period in sorted(grouped.keys()):
        pnls = grouped[period]
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        win_rate = len(wins) / n * 100 if n > 0 else 0.0
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / n if n > 0 else 0.0
        total_wins = sum(wins)
        total_losses = sum(losses)
        pf = total_wins / total_losses if total_losses > 0 else None

        period_stats.append(PeriodStats(
            period=period,
            n_trades=n,
            n_wins=len(wins),
            n_losses=len(losses),
            win_rate=round(win_rate, 1),
            total_pnl=round(total_pnl, 2),
            avg_pnl=round(avg_pnl, 2),
            best_trade=round(max(pnls), 2),
            worst_trade=round(min(pnls), 2),
            profit_factor=round(pf, 2) if pf is not None else None,
        ))

    if not period_stats:
        return None

    best = max(period_stats, key=lambda p: p.total_pnl)
    worst = min(period_stats, key=lambda p: p.total_pnl)

    all_pnls = [p for pnls in grouped.values() for p in pnls]
    overall_wr = sum(1 for p in all_pnls if p > 0) / len(all_pnls) * 100 if all_pnls else 0.0

    return JournalSummary(
        periods=period_stats,
        mode=mode,
        best_period=best.period,
        worst_period=worst.period,
        overall_win_rate=round(overall_wr, 1),
        overall_pnl=round(sum(all_pnls), 2),
        n_periods=len(period_stats),
        n_trades=len(all_pnls),
    )
