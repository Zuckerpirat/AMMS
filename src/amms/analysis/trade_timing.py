"""Trade timing analysis.

Analyzes when (day of week, hour of day) trades tend to perform best.
Aggregates win rate and average PnL per time bucket to surface patterns
like "Tuesday entries outperform" or "opening-hour buys underperform".

Reads from trade_pairs table (buy_ts, pnl, buy_price, qty).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass(frozen=True)
class TimeBucket:
    label: str
    n_trades: int
    win_rate: float       # 0-100
    avg_pnl_pct: float
    total_pnl: float


@dataclass(frozen=True)
class TradeTimingReport:
    by_weekday: list[TimeBucket]
    by_hour: list[TimeBucket]
    best_weekday: str | None
    worst_weekday: str | None
    best_hour: str | None
    worst_hour: str | None
    n_trades: int
    verdict: str


def _make_buckets(groups: dict[str, list[float]]) -> list[TimeBucket]:
    buckets = []
    for label in sorted(groups):
        pnls = groups[label]
        if not pnls:
            continue
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        buckets.append(TimeBucket(
            label=label,
            n_trades=n,
            win_rate=round(wins / n * 100, 1),
            avg_pnl_pct=round(sum(pnls) / n, 3),
            total_pnl=round(sum(pnls), 2),
        ))
    return buckets


def compute(conn, *, limit: int = 500) -> TradeTimingReport | None:
    """Analyze trade timing patterns from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 10 usable trades.
    """
    try:
        rows = conn.execute(
            "SELECT buy_ts, pnl, buy_price, qty "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND buy_ts IS NOT NULL "
            "AND buy_price IS NOT NULL AND qty IS NOT NULL "
            "ORDER BY buy_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 10:
        return None

    weekday_groups: dict[str, list[float]] = {}
    hour_groups: dict[str, list[float]] = {}

    for buy_ts, pnl, buy_price, qty in rows:
        try:
            dt = datetime.fromisoformat(str(buy_ts)[:19])
            pnl_f = float(pnl)
            bp = float(buy_price)
            qty_f = float(qty)
            entry_value = bp * qty_f
            pnl_pct = pnl_f / entry_value * 100 if entry_value > 0 else 0.0

            day_name = DAYS[dt.weekday()] if dt.weekday() < 5 else f"Day{dt.weekday()}"
            hour_label = f"{dt.hour:02d}:00"

            weekday_groups.setdefault(day_name, []).append(pnl_pct)
            hour_groups.setdefault(hour_label, []).append(pnl_pct)
        except Exception:
            continue

    if len(weekday_groups) == 0:
        return None

    by_weekday = _make_buckets(weekday_groups)
    by_hour = _make_buckets(hour_groups)

    if not by_weekday:
        return None

    n_total = sum(b.n_trades for b in by_weekday)

    best_wd = max(by_weekday, key=lambda b: b.avg_pnl_pct) if by_weekday else None
    worst_wd = min(by_weekday, key=lambda b: b.avg_pnl_pct) if by_weekday else None
    best_hr = max(by_hour, key=lambda b: b.avg_pnl_pct) if by_hour else None
    worst_hr = min(by_hour, key=lambda b: b.avg_pnl_pct) if by_hour else None

    parts = []
    if best_wd and worst_wd and best_wd.label != worst_wd.label:
        parts.append(
            f"Best entry day: {best_wd.label} "
            f"({best_wd.avg_pnl_pct:+.2f}% avg, {best_wd.win_rate:.0f}% win rate)"
        )
        parts.append(
            f"Weakest day: {worst_wd.label} "
            f"({worst_wd.avg_pnl_pct:+.2f}% avg)"
        )
    if best_hr:
        parts.append(
            f"Best entry hour: {best_hr.label} "
            f"({best_hr.avg_pnl_pct:+.2f}% avg)"
        )

    verdict = " | ".join(parts) if parts else "Insufficient timing patterns detected."

    return TradeTimingReport(
        by_weekday=by_weekday,
        by_hour=by_hour,
        best_weekday=best_wd.label if best_wd else None,
        worst_weekday=worst_wd.label if worst_wd else None,
        best_hour=best_hr.label if best_hr else None,
        worst_hour=worst_hr.label if worst_hr else None,
        n_trades=n_total,
        verdict=verdict,
    )
