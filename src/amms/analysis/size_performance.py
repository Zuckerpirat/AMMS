"""Trade size vs return analysis.

Tests whether larger position sizes correlate with better or worse
returns. Buckets trades by entry value (buy_price × qty) and computes
return metrics per size tier.

Also computes Pearson correlation between position value and PnL%.

Reads from trade_pairs (buy_price, qty, pnl).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizeTier:
    label: str            # "Small", "Medium", "Large", "XLarge"
    min_value: float
    max_value: float
    n_trades: int
    avg_position_value: float
    avg_pnl_pct: float
    win_rate: float       # 0-100
    total_pnl: float


@dataclass(frozen=True)
class SizePerformanceReport:
    tiers: list[SizeTier]
    correlation: float | None    # Pearson corr(position_value, pnl_pct)
    correlation_label: str       # "positive" / "negative" / "none"
    best_tier: str | None        # tier with highest avg_pnl_pct
    worst_tier: str | None
    n_trades: int
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


TIER_LABELS = ["Small", "Medium", "Large", "XLarge"]


def compute(conn, *, limit: int = 500, n_tiers: int = 4) -> SizePerformanceReport | None:
    """Analyze position size vs return from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 10 usable trades.
    """
    try:
        rows = conn.execute(
            "SELECT buy_price, qty, pnl "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND buy_price IS NOT NULL AND qty IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 10:
        return None

    trades: list[tuple[float, float]] = []  # (position_value, pnl_pct)
    for buy_price, qty, pnl in rows:
        try:
            bp = float(buy_price)
            qty_f = float(qty)
            pnl_f = float(pnl)
            entry_value = bp * qty_f
            if entry_value <= 0:
                continue
            pnl_pct = pnl_f / entry_value * 100
            trades.append((entry_value, pnl_pct))
        except Exception:
            continue

    if len(trades) < 10:
        return None

    trades.sort(key=lambda t: t[0])
    n = len(trades)
    tier_size = n // n_tiers

    if tier_size < 2:
        return None

    tiers: list[SizeTier] = []
    for i in range(n_tiers):
        start = i * tier_size
        end = n if i == n_tiers - 1 else (i + 1) * tier_size
        slice_ = trades[start:end]
        if not slice_:
            continue
        values = [t[0] for t in slice_]
        pnls = [t[1] for t in slice_]
        wins = sum(1 for p in pnls if p > 0)
        label = TIER_LABELS[i] if i < len(TIER_LABELS) else f"Tier{i+1}"
        tiers.append(SizeTier(
            label=label,
            min_value=round(min(values), 2),
            max_value=round(max(values), 2),
            n_trades=len(slice_),
            avg_position_value=round(sum(values) / len(values), 2),
            avg_pnl_pct=round(sum(pnls) / len(pnls), 3),
            win_rate=round(wins / len(pnls) * 100, 1),
            total_pnl=round(sum(pnls), 2),
        ))

    if not tiers:
        return None

    xs = [t[0] for t in trades]
    ys = [t[1] for t in trades]
    corr = _pearson(xs, ys)

    if corr is None or abs(corr) < 0.1:
        corr_label = "none"
    elif corr > 0:
        corr_label = "positive"
    else:
        corr_label = "negative"

    best = max(tiers, key=lambda t: t.avg_pnl_pct)
    worst = min(tiers, key=lambda t: t.avg_pnl_pct)

    corr_str = f"{corr:.2f}" if corr is not None else "n/a"
    if corr_label == "positive":
        insight = "Larger positions tend to outperform — sizing up in conviction is rewarded"
    elif corr_label == "negative":
        insight = "Larger positions tend to underperform — oversizing may hurt returns"
    else:
        insight = "Position size has little impact on returns"

    verdict = (
        f"{insight} (corr={corr_str}). "
        f"Best tier: {best.label} ({best.avg_pnl_pct:+.2f}% avg). "
        f"Worst tier: {worst.label} ({worst.avg_pnl_pct:+.2f}% avg)."
    )

    return SizePerformanceReport(
        tiers=tiers,
        correlation=round(corr, 3) if corr is not None else None,
        correlation_label=corr_label,
        best_tier=best.label,
        worst_tier=worst.label,
        n_trades=n,
        verdict=verdict,
    )
