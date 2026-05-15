"""Drawdown heatmap: per-symbol drawdown from recent peak.

For each symbol in the portfolio this module computes:
  - Peak price over the lookback window
  - Current drawdown from that peak (%)
  - Max drawdown in the window (worst intra-period dip from a rolling peak)
  - Drawdown duration (bars since peak)
  - Recovery score: momentum of recent bars vs drawdown depth
  - Status: "new_high" | "recovering" | "stalling" | "deepening"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownRow:
    symbol: str
    current_price: float
    peak_price: float
    drawdown_pct: float       # current distance from peak (negative = drawdown)
    max_drawdown_pct: float   # worst point in window
    bars_since_peak: int      # how long since the peak bar
    recovery_pct: float       # % recovered from max drawdown toward peak
    status: str               # "new_high"|"recovering"|"stalling"|"deepening"
    bars_used: int


@dataclass(frozen=True)
class DrawdownHeatmap:
    rows: list[DrawdownRow]
    avg_drawdown_pct: float
    worst_symbol: str | None
    best_symbol: str | None    # least drawdown / at new high
    n_at_new_high: int
    n_deepening: int


def analyze(bars_map: dict[str, list], *, lookback: int = 60) -> DrawdownHeatmap | None:
    """Compute drawdown metrics for each symbol.

    bars_map: dict mapping symbol → list[Bar]
    lookback: number of recent bars to analyze
    Returns None if bars_map is empty.
    """
    if not bars_map:
        return None

    rows: list[DrawdownRow] = []

    for sym, bars in bars_map.items():
        if not bars or len(bars) < 2:
            continue

        window = bars[-lookback:] if len(bars) > lookback else bars
        closes = [b.close for b in window]
        n = len(closes)

        # Rolling peak and max drawdown
        running_peak = closes[0]
        max_dd = 0.0
        max_dd_close = closes[0]

        for c in closes:
            if c > running_peak:
                running_peak = c
            dd = (c - running_peak) / running_peak * 100
            if dd < max_dd:
                max_dd = dd
                max_dd_close = c

        current_price = closes[-1]

        # Peak in window
        peak_price = max(closes)
        peak_idx = closes.index(peak_price)
        bars_since_peak = n - 1 - peak_idx

        current_dd = (current_price - peak_price) / peak_price * 100

        # Recovery: how much of max_dd has been recovered
        if max_dd < -0.001:
            # recovery_pct: 0 = still at worst, 100 = back at peak
            recovery_pct = (current_price - max_dd_close) / (peak_price - max_dd_close) * 100
            recovery_pct = max(0.0, min(100.0, recovery_pct))
        else:
            recovery_pct = 100.0

        # Status classification
        recent_n = min(5, n)
        recent_slope = (closes[-1] - closes[-recent_n]) / closes[-recent_n] * 100

        if current_dd > -0.5:
            status = "new_high"
        elif recent_slope > 1.0 and recovery_pct > 30:
            status = "recovering"
        elif recent_slope < -1.0:
            status = "deepening"
        else:
            status = "stalling"

        rows.append(DrawdownRow(
            symbol=sym,
            current_price=round(current_price, 4),
            peak_price=round(peak_price, 4),
            drawdown_pct=round(current_dd, 2),
            max_drawdown_pct=round(max_dd, 2),
            bars_since_peak=bars_since_peak,
            recovery_pct=round(recovery_pct, 1),
            status=status,
            bars_used=n,
        ))

    if not rows:
        return None

    avg_dd = sum(r.drawdown_pct for r in rows) / len(rows)
    worst = min(rows, key=lambda r: r.drawdown_pct)
    best = max(rows, key=lambda r: r.drawdown_pct)  # closest to 0 / new high
    n_new_high = sum(1 for r in rows if r.status == "new_high")
    n_deepening = sum(1 for r in rows if r.status == "deepening")

    return DrawdownHeatmap(
        rows=sorted(rows, key=lambda r: r.drawdown_pct),
        avg_drawdown_pct=round(avg_dd, 2),
        worst_symbol=worst.symbol,
        best_symbol=best.symbol,
        n_at_new_high=n_new_high,
        n_deepening=n_deepening,
    )
