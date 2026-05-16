"""Trade payoff distribution analysis.

Classifies trades by return % into buckets to visualize the shape of
the return distribution. Identifies whether the system is:
  - Skewed right (many small wins, few big wins)
  - Skewed left (many small losses, few big losses)
  - Fat-tailed (outlier wins or losses dominate)
  - Normal (balanced distribution)

Also computes:
  - Skewness and kurtosis of the return distribution
  - % of PnL from top/bottom 10% of trades (outlier dependence)

Reads from trade_pairs (pnl, buy_price, qty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReturnBucket:
    label: str
    lower_pct: float     # lower bound (%)
    upper_pct: float     # upper bound (%)
    n_trades: int
    pct_of_total: float  # % of all trades
    total_pnl_pct: float # sum of PnL% in this bucket


@dataclass(frozen=True)
class PayoffDistributionReport:
    buckets: list[ReturnBucket]
    skewness: float             # positive = right-skewed
    kurtosis: float             # >3 = fat tails vs normal
    top10_pct_contribution: float   # % of total PnL from top 10% wins
    bottom10_pct_drag: float        # % of total PnL lost to bottom 10% losses
    distribution_shape: str         # "right_skewed" / "left_skewed" / "normal" / "fat_tails"
    mean_return: float
    median_return: float
    n_trades: int
    verdict: str


def _skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mu = sum(values) / n
    sigma = math.sqrt(sum((v - mu) ** 2 for v in values) / n)
    if sigma == 0:
        return 0.0
    return sum(((v - mu) / sigma) ** 3 for v in values) / n


def _kurtosis(values: list[float]) -> float:
    """Excess kurtosis (normal = 0)."""
    n = len(values)
    if n < 4:
        return 0.0
    mu = sum(values) / n
    sigma = math.sqrt(sum((v - mu) ** 2 for v in values) / n)
    if sigma == 0:
        return 0.0
    return sum(((v - mu) / sigma) ** 4 for v in values) / n - 3


BUCKET_THRESHOLDS = [
    ("Big Loss (< -5%)", None, -5.0),
    ("Loss (-5% to -2%)", -5.0, -2.0),
    ("Small Loss (-2% to 0%)", -2.0, 0.0),
    ("Small Win (0% to 2%)", 0.0, 2.0),
    ("Win (2% to 5%)", 2.0, 5.0),
    ("Big Win (> 5%)", 5.0, None),
]


def compute(conn, *, limit: int = 500) -> PayoffDistributionReport | None:
    """Analyze return distribution from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 15 usable trades.
    """
    try:
        rows = conn.execute(
            "SELECT pnl, buy_price, qty "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND buy_price IS NOT NULL AND qty IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 15:
        return None

    pnl_pcts: list[float] = []
    for pnl, buy_price, qty in rows:
        try:
            bp = float(buy_price)
            qty_f = float(qty)
            pnl_f = float(pnl)
            entry_value = bp * qty_f
            if entry_value > 0:
                pnl_pcts.append(pnl_f / entry_value * 100)
        except Exception:
            continue

    if len(pnl_pcts) < 15:
        return None

    n = len(pnl_pcts)
    total_pnl = sum(pnl_pcts)

    # Build buckets
    buckets: list[ReturnBucket] = []
    for label, lo, hi in BUCKET_THRESHOLDS:
        in_bucket = [
            p for p in pnl_pcts
            if (lo is None or p >= lo) and (hi is None or p < hi)
        ]
        if not in_bucket:
            count = 0
            bucket_pnl = 0.0
        else:
            count = len(in_bucket)
            bucket_pnl = sum(in_bucket)
        buckets.append(ReturnBucket(
            label=label,
            lower_pct=lo if lo is not None else -999.0,
            upper_pct=hi if hi is not None else 999.0,
            n_trades=count,
            pct_of_total=round(count / n * 100, 1),
            total_pnl_pct=round(bucket_pnl, 2),
        ))

    # Stats
    skew = _skewness(pnl_pcts)
    kurt = _kurtosis(pnl_pcts)
    mean_r = sum(pnl_pcts) / n
    sorted_pcts = sorted(pnl_pcts)
    median_r = sorted_pcts[n // 2]

    # Outlier dependence
    top10_n = max(1, n // 10)
    top10 = sorted(pnl_pcts, reverse=True)[:top10_n]
    bottom10 = sorted(pnl_pcts)[:top10_n]
    top10_contrib = sum(top10) / total_pnl * 100 if total_pnl != 0 else 0.0
    bottom10_drag = sum(bottom10) / total_pnl * 100 if total_pnl != 0 else 0.0

    # Classify shape
    if skew > 1.0:
        shape = "right_skewed"
        shape_desc = "right-skewed (long right tail — occasional big wins drive results)"
    elif skew < -1.0:
        shape = "left_skewed"
        shape_desc = "left-skewed (large losses drag results — risk of catastrophic trades)"
    elif abs(kurt) > 2.0:
        shape = "fat_tails"
        shape_desc = "fat tails (outlier trades dominate — results driven by extremes)"
    else:
        shape = "normal"
        shape_desc = "near-normal (balanced, consistent distribution)"

    verdict = (
        f"Distribution is {shape_desc}. "
        f"Skewness={skew:.2f}, excess kurtosis={kurt:.2f}. "
        f"Top 10% trades contribute {top10_contrib:.0f}% of total PnL. "
        f"Bottom 10% account for {bottom10_drag:.0f}% of total PnL impact."
    )

    return PayoffDistributionReport(
        buckets=buckets,
        skewness=round(skew, 3),
        kurtosis=round(kurt, 3),
        top10_pct_contribution=round(top10_contrib, 1),
        bottom10_pct_drag=round(bottom10_drag, 1),
        distribution_shape=shape,
        mean_return=round(mean_r, 3),
        median_return=round(median_r, 3),
        n_trades=n,
        verdict=verdict,
    )
