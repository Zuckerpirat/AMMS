"""Trade frequency vs performance analysis.

Analyzes whether trading more frequently correlates with better or worse
outcomes. Groups trading days by number of trades and computes PnL and
win rate per frequency bucket.

Also computes:
  - Average trades per day / per week
  - Best and worst frequency buckets
  - Correlation: n_trades_per_day vs daily_pnl_pct

Reads from trade_pairs (sell_ts, pnl, buy_price, qty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FrequencyBucket:
    label: str           # "1 trade/day", "2-3 trades/day", etc.
    min_trades: int
    max_trades: int
    n_days: int
    avg_trades_per_day: float
    avg_daily_pnl_pct: float
    win_days_pct: float  # % of days with positive PnL


@dataclass(frozen=True)
class TradeFrequencyReport:
    buckets: list[FrequencyBucket]
    avg_trades_per_day: float
    avg_trades_per_week: float
    most_active_day_count: int    # max trades in a single day
    correlation: float | None     # corr(n_trades_day, daily_pnl_pct)
    correlation_label: str        # "positive" / "negative" / "none"
    best_bucket: str | None
    worst_bucket: str | None
    n_trading_days: int
    n_trades: int
    verdict: str


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def compute(conn, *, limit: int = 500) -> TradeFrequencyReport | None:
    """Analyze trade frequency vs daily performance from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 10 usable trades across 5+ days.
    """
    try:
        rows = conn.execute(
            "SELECT sell_ts, pnl, buy_price, qty "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND sell_ts IS NOT NULL "
            "AND buy_price IS NOT NULL AND qty IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 10:
        return None

    # Group by trading day
    day_trades: dict[str, list[float]] = {}  # date_str -> [pnl_pct, ...]
    for sell_ts, pnl, buy_price, qty in rows:
        try:
            dt = datetime.fromisoformat(str(sell_ts)[:19])
            d = dt.strftime("%Y-%m-%d")
            bp = float(buy_price)
            qty_f = float(qty)
            entry_value = bp * qty_f
            pnl_f = float(pnl)
            pnl_pct = pnl_f / entry_value * 100 if entry_value > 0 else 0.0
            day_trades.setdefault(d, []).append(pnl_pct)
        except Exception:
            continue

    if len(day_trades) < 5:
        return None

    # Build daily stats: (n_trades, daily_pnl_pct)
    daily: list[tuple[int, float]] = []
    for d, pnls in day_trades.items():
        n = len(pnls)
        daily_pnl = sum(pnls) / n  # avg trade pnl% for that day
        daily.append((n, daily_pnl))

    daily.sort(key=lambda x: x[0])
    n_days = len(daily)
    total_trades = sum(d[0] for d in daily)
    avg_per_day = total_trades / n_days
    avg_per_week = avg_per_day * 5  # approximate 5 trading days/week
    most_active = max(d[0] for d in daily)

    # Correlation
    xs = [float(d[0]) for d in daily]
    ys = [d[1] for d in daily]
    corr = _pearson(xs, ys)

    if corr is None or abs(corr) < 0.1:
        corr_label = "none"
    elif corr > 0:
        corr_label = "positive"
    else:
        corr_label = "negative"

    # Create frequency buckets
    max_trades = most_active
    if max_trades <= 3:
        thresholds = [(1, 1), (2, 2), (3, max_trades)]
    elif max_trades <= 6:
        thresholds = [(1, 1), (2, 3), (4, 6)]
    else:
        thresholds = [(1, 1), (2, 3), (4, 6), (7, max_trades)]

    buckets: list[FrequencyBucket] = []
    for lo, hi in thresholds:
        days_in_bucket = [(n, p) for n, p in daily if lo <= n <= hi]
        if not days_in_bucket:
            continue
        n_bucket_days = len(days_in_bucket)
        avg_n = sum(d[0] for d in days_in_bucket) / n_bucket_days
        avg_pnl = sum(d[1] for d in days_in_bucket) / n_bucket_days
        win_days = sum(1 for d in days_in_bucket if d[1] > 0)
        if lo == hi:
            label = f"{lo} trade/day"
        else:
            label = f"{lo}-{hi} trades/day"
        buckets.append(FrequencyBucket(
            label=label,
            min_trades=lo,
            max_trades=hi,
            n_days=n_bucket_days,
            avg_trades_per_day=round(avg_n, 1),
            avg_daily_pnl_pct=round(avg_pnl, 3),
            win_days_pct=round(win_days / n_bucket_days * 100, 1),
        ))

    if not buckets:
        return None

    best = max(buckets, key=lambda b: b.avg_daily_pnl_pct)
    worst = min(buckets, key=lambda b: b.avg_daily_pnl_pct)

    corr_str = f"{corr:.2f}" if corr is not None else "n/a"
    if corr_label == "positive":
        insight = "More trading days tend to produce better returns"
    elif corr_label == "negative":
        insight = "Fewer trades per day tend to produce better returns — less is more"
    else:
        insight = "Trade frequency has little impact on daily returns"

    verdict = (
        f"{insight} (corr={corr_str}). "
        f"Avg {avg_per_day:.1f} trades/day ({avg_per_week:.0f}/week). "
        f"Best frequency: {best.label} ({best.avg_daily_pnl_pct:+.2f}% avg)."
    )

    return TradeFrequencyReport(
        buckets=buckets,
        avg_trades_per_day=round(avg_per_day, 2),
        avg_trades_per_week=round(avg_per_week, 1),
        most_active_day_count=most_active,
        correlation=round(corr, 3) if corr is not None else None,
        correlation_label=corr_label,
        best_bucket=best.label,
        worst_bucket=worst.label,
        n_trading_days=n_days,
        n_trades=total_trades,
        verdict=verdict,
    )
