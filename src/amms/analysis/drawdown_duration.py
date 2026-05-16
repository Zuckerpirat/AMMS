"""Drawdown duration analysis.

Measures the time spent in drawdown and recovery periods from the
equity curve (equity_snapshots table).

Tracks:
  - Current drawdown depth and duration (days since peak)
  - Maximum historical drawdown depth and its duration
  - Average recovery time from drawdowns > threshold
  - Underwater status (currently in drawdown vs at new high)
  - Pain index: average drawdown across all periods

Useful for understanding the psychological and financial burden
of drawdowns on a trading account.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownPeriod:
    peak_equity: float
    trough_equity: float
    drawdown_pct: float       # negative value
    duration_periods: int     # from peak to trough
    recovery_periods: int | None   # periods to recover (None if not recovered)
    recovered: bool


@dataclass(frozen=True)
class DrawdownDurationReport:
    current_drawdown_pct: float      # 0 if at new high
    current_drawdown_periods: int    # 0 if at new high
    is_underwater: bool
    max_drawdown_pct: float          # worst ever (negative)
    max_drawdown_duration: int       # periods for max drawdown
    avg_recovery_periods: float | None   # avg recovery time (completed DDs only)
    pain_index: float                # avg drawdown across all periods (negative)
    n_drawdown_periods: int          # number of distinct drawdown episodes
    n_recovered: int                 # episodes that recovered
    longest_underwater: int          # longest stretch below a peak
    equity_high: float
    current_equity: float
    n_periods: int
    verdict: str


def compute(conn, *, limit: int = 252, dd_threshold_pct: float = 2.0) -> DrawdownDurationReport | None:
    """Analyze drawdown durations from equity_snapshots.

    conn: SQLite connection with equity_snapshots table
    limit: max snapshots to analyze
    dd_threshold_pct: minimum drawdown % to count as a distinct episode

    Returns None if fewer than 10 rows.
    """
    try:
        rows = conn.execute(
            "SELECT ts, equity FROM equity_snapshots "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 10:
        return None

    # Chronological order
    rows = list(reversed(rows))
    equities = [float(r[1]) for r in rows]
    n = len(equities)

    # Running peak
    peak = equities[0]
    equity_high = equities[0]
    dd_series: list[float] = []     # drawdown at each period

    for eq in equities:
        if eq > peak:
            peak = eq
        if eq > equity_high:
            equity_high = eq
        dd = (eq - peak) / peak
        dd_series.append(dd)

    # Pain index = average drawdown
    pain_index = sum(dd_series) / n

    # Current drawdown
    current_peak = equities[0]
    for eq in equities:
        if eq > current_peak:
            current_peak = eq
    current_dd = (equities[-1] - current_peak) / current_peak
    current_dd_pct = current_dd * 100

    # Periods since last new high (current drawdown duration)
    current_dd_periods = 0
    for eq in reversed(equities):
        if eq >= current_peak * 0.9999:
            break
        current_dd_periods += 1
    is_underwater = current_dd_pct < -0.01

    # Identify drawdown episodes (peak-to-trough-to-recovery)
    episodes: list[DrawdownPeriod] = []
    in_dd = False
    ep_peak = equities[0]
    ep_peak_idx = 0
    ep_trough = equities[0]
    ep_trough_idx = 0
    running_peak = equities[0]
    longest_underwater = 0
    current_uw_len = 0

    for i, eq in enumerate(equities):
        if eq >= running_peak:
            # New high
            if in_dd:
                dd_depth = (ep_trough - ep_peak) / ep_peak * 100
                if abs(dd_depth) >= dd_threshold_pct:
                    recovery_periods = i - ep_trough_idx
                    episodes.append(DrawdownPeriod(
                        peak_equity=round(ep_peak, 2),
                        trough_equity=round(ep_trough, 2),
                        drawdown_pct=round(dd_depth, 2),
                        duration_periods=ep_trough_idx - ep_peak_idx,
                        recovery_periods=recovery_periods,
                        recovered=True,
                    ))
                in_dd = False
                current_uw_len = 0
            running_peak = eq
            ep_peak = eq
            ep_peak_idx = i
        else:
            in_dd = True
            current_uw_len += 1
            longest_underwater = max(longest_underwater, current_uw_len)
            if eq < ep_trough:
                ep_trough = eq
                ep_trough_idx = i

    # Handle open drawdown at end
    if in_dd:
        dd_depth = (ep_trough - ep_peak) / ep_peak * 100
        if abs(dd_depth) >= dd_threshold_pct:
            episodes.append(DrawdownPeriod(
                peak_equity=round(ep_peak, 2),
                trough_equity=round(ep_trough, 2),
                drawdown_pct=round(dd_depth, 2),
                duration_periods=ep_trough_idx - ep_peak_idx,
                recovery_periods=None,
                recovered=False,
            ))

    # Stats
    max_dd_pct = min((ep.drawdown_pct for ep in episodes), default=0.0)
    max_dd_ep = next((ep for ep in episodes if ep.drawdown_pct == max_dd_pct), None)
    max_dd_dur = max_dd_ep.duration_periods if max_dd_ep else 0

    recovered = [ep for ep in episodes if ep.recovered and ep.recovery_periods is not None]
    avg_recovery = (
        sum(ep.recovery_periods for ep in recovered) / len(recovered)
        if recovered else None
    )

    # Verdict
    if not is_underwater:
        verdict = "At all-time high — no current drawdown"
    elif current_dd_pct > -5:
        verdict = f"Minor drawdown ({current_dd_pct:.1f}%%) for {current_dd_periods} periods"
    elif current_dd_pct > -15:
        verdict = f"Moderate drawdown ({current_dd_pct:.1f}%%) — monitor closely"
    else:
        verdict = f"Significant drawdown ({current_dd_pct:.1f}%%) — {current_dd_periods} periods underwater"

    return DrawdownDurationReport(
        current_drawdown_pct=round(current_dd_pct, 2),
        current_drawdown_periods=current_dd_periods,
        is_underwater=is_underwater,
        max_drawdown_pct=round(max_dd_pct, 2),
        max_drawdown_duration=max_dd_dur,
        avg_recovery_periods=round(avg_recovery, 1) if avg_recovery is not None else None,
        pain_index=round(pain_index * 100, 3),
        n_drawdown_periods=len(episodes),
        n_recovered=len(recovered),
        longest_underwater=longest_underwater,
        equity_high=round(equity_high, 2),
        current_equity=round(equities[-1], 2),
        n_periods=n,
        verdict=verdict,
    )
