"""Monthly performance trend analysis.

Fits a linear regression over monthly PnL totals to determine whether
trading performance is improving, declining, or flat over time.

Also detects:
  - Acceleration (2nd-half better than 1st-half)
  - Best and worst calendar months
  - Consistency score (% of months profitable)

Reads from trade_pairs (sell_ts, pnl).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MonthlyPnL:
    year: int
    month: int
    label: str           # "2026-01"
    total_pnl: float
    n_trades: int
    win_rate: float


@dataclass(frozen=True)
class PerformanceTrendReport:
    monthly: list[MonthlyPnL]
    slope: float                 # PnL change per month (linear trend)
    trend_direction: str         # "improving" / "declining" / "flat"
    r_squared: float             # 0-1, goodness of fit
    consistency_pct: float       # % of months profitable
    acceleration: str            # "accelerating" / "decelerating" / "stable"
    best_month: str | None
    worst_month: str | None
    n_months: int
    verdict: str


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    ss_xx = sum((x - mx) ** 2 for x in xs)
    if ss_xx == 0:
        return 0.0, my, 0.0
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    y_pred = [slope * x + intercept for x in xs]
    ss_res = sum((ys[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, max(0.0, r2)


def compute(conn, *, limit: int = 500) -> PerformanceTrendReport | None:
    """Analyze monthly PnL trend from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 3 calendar months of data.
    """
    try:
        rows = conn.execute(
            "SELECT sell_ts, pnl "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND sell_ts IS NOT NULL "
            "ORDER BY sell_ts ASC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 5:
        return None

    month_groups: dict[tuple[int, int], list[float]] = {}
    for sell_ts, pnl in rows:
        try:
            dt = datetime.fromisoformat(str(sell_ts)[:19])
            key = (dt.year, dt.month)
            month_groups.setdefault(key, []).append(float(pnl))
        except Exception:
            continue

    if len(month_groups) < 3:
        return None

    monthly: list[MonthlyPnL] = []
    for (year, month), pnls in sorted(month_groups.items()):
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        monthly.append(MonthlyPnL(
            year=year,
            month=month,
            label=f"{year}-{month:02d}",
            total_pnl=round(sum(pnls), 2),
            n_trades=n,
            win_rate=round(wins / n * 100, 1),
        ))

    xs = list(range(len(monthly)))
    ys = [m.total_pnl for m in monthly]
    slope, _, r2 = _ols(xs, ys)

    if abs(slope) < 10:
        direction = "flat"
    elif slope > 0:
        direction = "improving"
    else:
        direction = "declining"

    consistency = sum(1 for m in monthly if m.total_pnl > 0) / len(monthly) * 100

    # Acceleration: compare avg PnL of first half vs second half
    mid = len(monthly) // 2
    first_half_avg = sum(ys[:mid]) / mid if mid > 0 else 0.0
    second_half_avg = sum(ys[mid:]) / (len(monthly) - mid) if len(monthly) > mid else 0.0
    if second_half_avg > first_half_avg + abs(first_half_avg) * 0.1:
        acceleration = "accelerating"
    elif second_half_avg < first_half_avg - abs(first_half_avg) * 0.1:
        acceleration = "decelerating"
    else:
        acceleration = "stable"

    best = max(monthly, key=lambda m: m.total_pnl)
    worst = min(monthly, key=lambda m: m.total_pnl)

    trend_icon = {"improving": "↗", "declining": "↘", "flat": "→"}.get(direction, "")
    verdict = (
        f"Performance trend: {direction} {trend_icon} "
        f"(slope {slope:+.0f}/month, R²={r2:.2f}). "
        f"{consistency:.0f}% of months profitable. "
        f"Trend {acceleration}. "
        f"Best month: {best.label} ({best.total_pnl:+.0f}), "
        f"worst: {worst.label} ({worst.total_pnl:+.0f})."
    )

    return PerformanceTrendReport(
        monthly=monthly,
        slope=round(slope, 2),
        trend_direction=direction,
        r_squared=round(r2, 3),
        consistency_pct=round(consistency, 1),
        acceleration=acceleration,
        best_month=best.label,
        worst_month=worst.label,
        n_months=len(monthly),
        verdict=verdict,
    )
