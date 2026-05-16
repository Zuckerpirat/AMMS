"""Daily profit calendar.

Aggregates closed-trade PnL by calendar date to show which days were
profitable and which were not. Produces a compact text calendar view
and summary statistics per month.

Reads from trade_pairs (sell_ts, pnl).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class DayStats:
    date: date
    n_trades: int
    total_pnl: float
    win_rate: float   # 0-100


@dataclass(frozen=True)
class MonthStats:
    year: int
    month: int
    days: list[DayStats]
    n_trading_days: int
    n_profitable_days: int
    total_pnl: float
    best_day_pnl: float
    worst_day_pnl: float
    avg_daily_pnl: float


@dataclass(frozen=True)
class ProfitCalendarReport:
    months: list[MonthStats]
    overall_profitable_days: int
    overall_losing_days: int
    overall_pnl: float
    best_day: date | None
    worst_day: date | None
    n_months: int
    verdict: str


def _month_name(m: int) -> str:
    return calendar.month_abbr[m]


def compute(conn, *, limit: int = 500) -> ProfitCalendarReport | None:
    """Build a daily profit calendar from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 5 trades with valid sell timestamps.
    """
    try:
        rows = conn.execute(
            "SELECT sell_ts, pnl "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND sell_ts IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 5:
        return None

    day_groups: dict[date, list[float]] = {}
    for sell_ts, pnl in rows:
        try:
            dt = datetime.fromisoformat(str(sell_ts)[:19])
            d = dt.date()
            day_groups.setdefault(d, []).append(float(pnl))
        except Exception:
            continue

    if len(day_groups) < 3:
        return None

    # Group days by (year, month)
    month_groups: dict[tuple[int, int], list[DayStats]] = {}
    for d, pnls in day_groups.items():
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        ds = DayStats(
            date=d,
            n_trades=n,
            total_pnl=round(sum(pnls), 2),
            win_rate=round(wins / n * 100, 1),
        )
        key = (d.year, d.month)
        month_groups.setdefault(key, []).append(ds)

    months: list[MonthStats] = []
    for (year, month), days in sorted(month_groups.items()):
        days_sorted = sorted(days, key=lambda d: d.date)
        n_prof = sum(1 for d in days if d.total_pnl > 0)
        total = sum(d.total_pnl for d in days)
        best = max(d.total_pnl for d in days)
        worst = min(d.total_pnl for d in days)
        avg = total / len(days)
        months.append(MonthStats(
            year=year,
            month=month,
            days=days_sorted,
            n_trading_days=len(days),
            n_profitable_days=n_prof,
            total_pnl=round(total, 2),
            best_day_pnl=round(best, 2),
            worst_day_pnl=round(worst, 2),
            avg_daily_pnl=round(avg, 2),
        ))

    if not months:
        return None

    all_days = [d for m in months for d in m.days]
    prof_days = sum(1 for d in all_days if d.total_pnl > 0)
    loss_days = sum(1 for d in all_days if d.total_pnl <= 0)
    overall_pnl = sum(d.total_pnl for d in all_days)
    best_day = max(all_days, key=lambda d: d.total_pnl)
    worst_day = min(all_days, key=lambda d: d.total_pnl)

    win_pct = prof_days / (prof_days + loss_days) * 100 if (prof_days + loss_days) > 0 else 0
    verdict = (
        f"{prof_days} profitable days vs {loss_days} losing days "
        f"({win_pct:.0f}% daily win rate). "
        f"Total PnL over {len(months)} month(s): {overall_pnl:+.2f}. "
        f"Best day: {best_day.date} ({best_day.total_pnl:+.2f}), "
        f"worst: {worst_day.date} ({worst_day.total_pnl:+.2f})."
    )

    return ProfitCalendarReport(
        months=months,
        overall_profitable_days=prof_days,
        overall_losing_days=loss_days,
        overall_pnl=round(overall_pnl, 2),
        best_day=best_day.date,
        worst_day=worst_day.date,
        n_months=len(months),
        verdict=verdict,
    )
