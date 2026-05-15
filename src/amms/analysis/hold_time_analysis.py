"""Win rate and P&L analysis by holding period.

Groups closed trades from trade_pairs into hold-time buckets and
computes win rate, average P&L, and profit factor per bucket.

Hold-time buckets:
  day:    0 - 2 days
  swing:  2 - 14 days
  medium: 14 - 90 days
  long:   90+ days

Metrics per bucket:
  - n_trades
  - win_rate (%)
  - avg_pnl ($)
  - avg_pnl_pct (%)
  - profit_factor: total_wins / |total_losses|  (None if no losses)
  - best_trade_pnl, worst_trade_pnl
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


BUCKETS = ["day", "swing", "medium", "long"]

BUCKET_RANGES = {
    "day":    (0,   2),
    "swing":  (2,   14),
    "medium": (14,  90),
    "long":   (90,  float("inf")),
}


@dataclass(frozen=True)
class BucketStats:
    bucket: str
    n_trades: int
    win_rate: float            # 0-100
    avg_pnl: float
    avg_pnl_pct: float
    profit_factor: float | None  # None if no losses
    best_pnl: float
    worst_pnl: float
    label: str                  # e.g. "0-2d"


@dataclass(frozen=True)
class HoldTimeReport:
    buckets: list[BucketStats]   # only populated buckets
    best_bucket: str | None      # bucket with highest win_rate
    total_trades: int
    overall_win_rate: float


def _classify(hold_days: float) -> str:
    for bucket, (lo, hi) in BUCKET_RANGES.items():
        if lo <= hold_days < hi:
            return bucket
    return "long"


BUCKET_LABELS = {
    "day":    "0-2d",
    "swing":  "2-14d",
    "medium": "14-90d",
    "long":   "90d+",
}


def compute(conn, limit: int = 200) -> HoldTimeReport | None:
    """Compute hold-time analysis from trade_pairs table.

    conn: SQLite connection
    limit: max number of recent closed trades to analyze
    Returns None if no data or table missing.
    """
    try:
        rows = conn.execute(
            "SELECT buy_ts, sell_ts, pnl, buy_price, qty "
            "FROM trade_pairs "
            "WHERE sell_price IS NOT NULL AND pnl IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    # Group into buckets
    grouped: dict[str, list[tuple[float, float]]] = {b: [] for b in BUCKETS}
    all_pnls: list[float] = []

    for row in rows:
        buy_ts, sell_ts, pnl, buy_price, qty = row
        pnl = float(pnl) if pnl else 0.0
        buy_price = float(buy_price) if buy_price else 0.0
        qty = float(qty) if qty else 1.0
        entry_value = buy_price * qty

        pnl_pct = pnl / entry_value * 100 if entry_value > 0 else 0.0
        all_pnls.append(pnl)

        # Compute hold days
        hold_days = None
        if buy_ts and sell_ts:
            try:
                buy_dt = datetime.fromisoformat(str(buy_ts)[:19])
                sell_dt = datetime.fromisoformat(str(sell_ts)[:19])
                hold_days = (sell_dt - buy_dt).total_seconds() / 86400
            except Exception:
                pass

        if hold_days is None:
            continue  # skip trades without timestamps

        bucket = _classify(max(0, hold_days))
        grouped[bucket].append((pnl, pnl_pct))

    # Compute stats per bucket
    bucket_stats: list[BucketStats] = []
    for bucket in BUCKETS:
        trades = grouped[bucket]
        if not trades:
            continue

        pnls = [t[0] for t in trades]
        pcts = [t[1] for t in trades]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / n * 100

        total_wins = sum(p for p in pnls if p > 0)
        total_losses = sum(abs(p) for p in pnls if p < 0)
        pf = total_wins / total_losses if total_losses > 0 else None

        bucket_stats.append(BucketStats(
            bucket=bucket,
            n_trades=n,
            win_rate=round(win_rate, 1),
            avg_pnl=round(sum(pnls) / n, 2),
            avg_pnl_pct=round(sum(pcts) / n, 2),
            profit_factor=round(pf, 2) if pf is not None else None,
            best_pnl=round(max(pnls), 2),
            worst_pnl=round(min(pnls), 2),
            label=BUCKET_LABELS[bucket],
        ))

    if not bucket_stats:
        return None

    best = max(bucket_stats, key=lambda s: s.win_rate)

    total = len(all_pnls)
    overall_wins = sum(1 for p in all_pnls if p > 0)
    overall_wr = overall_wins / total * 100 if total > 0 else 0.0

    return HoldTimeReport(
        buckets=bucket_stats,
        best_bucket=best.bucket if bucket_stats else None,
        total_trades=total,
        overall_win_rate=round(overall_wr, 1),
    )
