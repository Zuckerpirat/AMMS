"""Drawdown Curve Analyser.

Reconstructs the cumulative equity curve from the closed trade journal
and identifies all drawdown periods: peak, trough, depth, duration, and
recovery time.

Metrics returned:
  - Full equity curve (cumulative PnL%)
  - All individual drawdown episodes
  - Maximum drawdown depth and duration
  - Average drawdown depth and duration
  - Longest recovery period
  - Current drawdown status (in drawdown / recovered)
  - Ulcer Index (RMS of drawdown depths)
  - Recovery factor (total gain / max drawdown)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownEpisode:
    peak_idx: int           # index in equity curve at last all-time-high
    trough_idx: int         # index at deepest point
    recovery_idx: int | None  # index when curve returns to peak (None if not yet)
    peak_value: float       # equity at peak
    trough_value: float     # equity at trough
    depth_pct: float        # drawdown depth as % (negative)
    duration_bars: int      # bars from peak to trough
    recovery_bars: int | None  # bars from trough to recovery (None if ongoing)
    is_recovered: bool


@dataclass(frozen=True)
class DrawdownCurveReport:
    equity_curve: list[float]       # cumulative PnL% after each trade (starts at 0)
    drawdown_curve: list[float]     # drawdown at each point (0 or negative)
    episodes: list[DrawdownEpisode]
    max_drawdown_pct: float         # worst drawdown depth (negative or 0)
    max_drawdown_duration: int      # bars at peak-to-trough for worst episode
    avg_drawdown_pct: float
    avg_drawdown_duration: float
    longest_recovery_bars: int | None
    current_drawdown_pct: float     # 0 if at all-time-high
    in_drawdown: bool
    ulcer_index: float              # RMS of drawdown curve
    recovery_factor: float          # total_gain / abs(max_drawdown); 0 if no drawdown
    n_trades: int
    n_episodes: int
    verdict: str


def compute(conn, *, limit: int = 1000) -> DrawdownCurveReport | None:
    """Analyse drawdown curve from closed trade history.

    conn: SQLite connection with a trades table.
    limit: maximum trades to load (most recent first).
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct, closed_at
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
            ORDER BY closed_at ASC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 5:
        return None

    pnls = [float(r[0]) for r in rows]

    # Build cumulative equity curve (starts at 0)
    equity: list[float] = []
    cum = 0.0
    for p in pnls:
        cum += p
        equity.append(round(cum, 4))

    n = len(equity)

    # Build drawdown curve
    drawdown: list[float] = []
    peak = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        dd = v - peak  # 0 or negative
        drawdown.append(round(dd, 4))

    # Identify drawdown episodes
    episodes: list[DrawdownEpisode] = []
    i = 0
    while i < n:
        if drawdown[i] < 0:
            # Find where the drawdown started (last point where drawdown was 0)
            peak_idx = i - 1
            while peak_idx > 0 and drawdown[peak_idx] < 0:
                peak_idx -= 1
            # Find trough
            trough_val = drawdown[i]
            trough_idx = i
            j = i
            while j < n and drawdown[j] < 0:
                if drawdown[j] < trough_val:
                    trough_val = drawdown[j]
                    trough_idx = j
                j += 1
            # j is now either past the end or back to 0 (recovery)
            recovery_idx = j if j < n else None
            is_recovered = recovery_idx is not None

            peak_val = equity[peak_idx]
            trough_equity = equity[trough_idx]
            depth = trough_equity - peak_val  # negative
            depth_pct = (depth / abs(peak_val) * 100.0) if peak_val != 0 else depth

            episodes.append(DrawdownEpisode(
                peak_idx=peak_idx,
                trough_idx=trough_idx,
                recovery_idx=recovery_idx,
                peak_value=round(peak_val, 4),
                trough_value=round(trough_equity, 4),
                depth_pct=round(depth_pct, 3),
                duration_bars=trough_idx - peak_idx,
                recovery_bars=(recovery_idx - trough_idx) if recovery_idx is not None else None,
                is_recovered=is_recovered,
            ))
            i = j if j < n else n
        else:
            i += 1

    max_dd = min(drawdown) if drawdown else 0.0
    max_ep = min(episodes, key=lambda e: e.depth_pct) if episodes else None
    max_dd_duration = max_ep.duration_bars if max_ep else 0

    avg_dd = (sum(e.depth_pct for e in episodes) / len(episodes)) if episodes else 0.0
    avg_dd_dur = (sum(e.duration_bars for e in episodes) / len(episodes)) if episodes else 0.0

    recovery_bars_list = [e.recovery_bars for e in episodes if e.recovery_bars is not None]
    longest_recovery = max(recovery_bars_list) if recovery_bars_list else None

    current_dd = drawdown[-1]
    in_dd = current_dd < 0

    # Ulcer Index: RMS of drawdown percentages (using drawdown curve values)
    ulcer = math.sqrt(sum(d * d for d in drawdown) / n) if n > 0 else 0.0

    total_gain = equity[-1] - equity[0] + pnls[0]  # full range
    recovery_factor = abs(total_gain / max_dd) if max_dd < 0 else 0.0

    # Verdict
    parts = []
    if max_dd < 0:
        parts.append(f"max drawdown {max_dd:.2f}% over {max_dd_duration} trades")
    if in_dd:
        parts.append(f"currently in drawdown ({current_dd:.2f}%)")
    else:
        parts.append("currently at or near equity high")
    if ulcer > 3:
        parts.append(f"high ulcer index {ulcer:.1f} (choppy equity curve)")
    elif ulcer > 1:
        parts.append(f"moderate ulcer index {ulcer:.1f}")
    if recovery_factor > 0:
        parts.append(f"recovery factor {recovery_factor:.2f}")

    verdict = "Drawdown analysis: " + ("; ".join(parts) if parts else "no drawdowns detected") + "."

    return DrawdownCurveReport(
        equity_curve=equity,
        drawdown_curve=drawdown,
        episodes=episodes,
        max_drawdown_pct=round(max_dd, 3),
        max_drawdown_duration=max_dd_duration,
        avg_drawdown_pct=round(avg_dd, 3),
        avg_drawdown_duration=round(avg_dd_dur, 1),
        longest_recovery_bars=longest_recovery,
        current_drawdown_pct=round(current_dd, 3),
        in_drawdown=in_dd,
        ulcer_index=round(ulcer, 4),
        recovery_factor=round(recovery_factor, 3),
        n_trades=n,
        n_episodes=len(episodes),
        verdict=verdict,
    )
