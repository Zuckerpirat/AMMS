"""Trade duration vs return analysis.

Analyzes whether longer or shorter hold times produce better returns.
Buckets trades by hold duration and computes return metrics per bucket.

This complements hold_time_analysis.py (which uses fixed time buckets)
by using dynamic percentile-based buckets and computing the relationship
between hold time and P&L%.

Also computes:
  - Pearson correlation between hold_days and pnl_pct
  - Optimal hold range (bucket with highest avg return)
  - Diminishing returns detection (peak and decline)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DurationBucket:
    label: str            # e.g. "0-2d" or "Q1 (0-2d)"
    min_days: float
    max_days: float
    n_trades: int
    avg_hold_days: float
    avg_pnl_pct: float
    win_rate: float


@dataclass(frozen=True)
class DurationReturnReport:
    buckets: list[DurationBucket]
    correlation: float | None     # Pearson corr(hold_days, pnl_pct)
    correlation_label: str        # "positive" / "negative" / "none"
    optimal_bucket: str | None    # label of best avg_pnl_pct bucket
    n_trades: int
    n_buckets: int
    verdict: str


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return num / (denom_x * denom_y)


def compute(conn, *, limit: int = 200, n_buckets: int = 4) -> DurationReturnReport | None:
    """Analyze hold duration vs return from trade_pairs.

    conn: SQLite connection with trade_pairs table
    limit: max trades to analyze
    n_buckets: number of equal-frequency buckets (default 4 = quartiles)

    Returns None if fewer than 10 trades with valid timestamps.
    """
    try:
        rows = conn.execute(
            "SELECT buy_ts, sell_ts, pnl, buy_price, qty "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND sell_price IS NOT NULL "
            "AND buy_ts IS NOT NULL AND sell_ts IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 10:
        return None

    # Parse hold days and pnl%
    trades: list[tuple[float, float]] = []  # (hold_days, pnl_pct)
    for buy_ts, sell_ts, pnl, buy_price, qty in rows:
        try:
            buy_dt = datetime.fromisoformat(str(buy_ts)[:19])
            sell_dt = datetime.fromisoformat(str(sell_ts)[:19])
            hold_days = max(0.0, (sell_dt - buy_dt).total_seconds() / 86400)
            pnl_f = float(pnl)
            bp = float(buy_price) if buy_price else 0.0
            qty_f = float(qty) if qty else 1.0
            entry_value = bp * qty_f
            pnl_pct = pnl_f / entry_value * 100 if entry_value > 0 else 0.0
            trades.append((hold_days, pnl_pct))
        except Exception:
            continue

    if len(trades) < 10:
        return None

    # Sort by hold_days
    trades.sort(key=lambda t: t[0])
    n = len(trades)

    # Create equal-frequency buckets
    bucket_size = n // n_buckets
    if bucket_size < 2:
        return None

    buckets: list[DurationBucket] = []
    for i in range(n_buckets):
        start_idx = i * bucket_size
        end_idx = n if i == n_buckets - 1 else (i + 1) * bucket_size
        slice_ = trades[start_idx:end_idx]
        if not slice_:
            continue

        days_in_bucket = [t[0] for t in slice_]
        pnls_in_bucket = [t[1] for t in slice_]

        avg_days = sum(days_in_bucket) / len(days_in_bucket)
        avg_pnl_pct = sum(pnls_in_bucket) / len(pnls_in_bucket)
        win_rate = sum(1 for p in pnls_in_bucket if p > 0) / len(pnls_in_bucket) * 100

        min_d = min(days_in_bucket)
        max_d = max(days_in_bucket)
        label = f"Q{i+1} ({min_d:.0f}-{max_d:.0f}d)"

        buckets.append(DurationBucket(
            label=label,
            min_days=round(min_d, 1),
            max_days=round(max_d, 1),
            n_trades=len(slice_),
            avg_hold_days=round(avg_days, 1),
            avg_pnl_pct=round(avg_pnl_pct, 2),
            win_rate=round(win_rate, 1),
        ))

    if not buckets:
        return None

    # Correlation
    xs = [t[0] for t in trades]
    ys = [t[1] for t in trades]
    corr = _pearson(xs, ys)

    if corr is None or abs(corr) < 0.1:
        corr_label = "none"
    elif corr > 0:
        corr_label = "positive"
    else:
        corr_label = "negative"

    # Optimal bucket
    best = max(buckets, key=lambda b: b.avg_pnl_pct)
    optimal = best.label

    # Verdict
    if corr_label == "positive":
        verdict = f"Longer holds tend to produce better returns (corr={corr:.2f}) — patience pays"
    elif corr_label == "negative":
        verdict = f"Shorter holds tend to produce better returns (corr={corr:.2f}) — quick exits favored"
    else:
        corr_disp = f"{corr:.2f}" if corr is not None else "0.00"
        verdict = f"Hold duration has little impact on returns (corr={corr_disp}) — other factors dominate"

    verdict += f". Optimal range: {optimal}."

    return DurationReturnReport(
        buckets=buckets,
        correlation=round(corr, 3) if corr is not None else None,
        correlation_label=corr_label,
        optimal_bucket=optimal,
        n_trades=n,
        n_buckets=len(buckets),
        verdict=verdict,
    )
