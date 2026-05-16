"""Win rate stability analysis.

Measures how consistent the win rate is over rolling windows of trades.
A stable win rate suggests a reproducible edge; high variability may
indicate luck or regime-dependent performance.

Computes:
  - Rolling win rate over N-trade windows
  - Coefficient of variation (CV = std / mean) of win rates
  - Stability grade: Excellent / Good / Moderate / Unstable
  - Z-score of current win rate vs historical distribution
  - Confidence interval for the true win rate (Wilson interval)

Reads from trade_pairs (pnl) ordered by sell_ts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WinRateWindow:
    window_number: int
    start_trade: int
    end_trade: int
    n_trades: int
    win_rate: float      # 0-100


@dataclass(frozen=True)
class WinRateStabilityReport:
    windows: list[WinRateWindow]
    overall_win_rate: float
    win_rate_std: float           # std dev of rolling win rates
    win_rate_cv: float            # coefficient of variation
    win_rate_min: float
    win_rate_max: float
    win_rate_range: float         # max - min
    stability_grade: str          # "Excellent" / "Good" / "Moderate" / "Unstable"
    ci_lower: float               # 95% Wilson CI lower bound
    ci_upper: float               # 95% Wilson CI upper bound
    current_window_wr: float      # most recent window win rate
    z_score: float                # how unusual is current window vs history
    n_trades: int
    window_size: int
    verdict: str


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 100.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100)


def compute(conn, *, limit: int = 300, window_size: int = 20) -> WinRateStabilityReport | None:
    """Analyze win rate stability from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    limit: max recent trades.
    window_size: rolling window size (default 20 trades).
    Returns None if fewer than 2 full windows.
    """
    try:
        rows = conn.execute(
            "SELECT pnl "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL "
            "ORDER BY sell_ts ASC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    outcomes = []
    for (pnl,) in rows:
        try:
            outcomes.append(1 if float(pnl) > 0 else 0)
        except Exception:
            continue

    n = len(outcomes)
    if n < window_size * 2:
        return None

    # Rolling windows (non-overlapping)
    windows: list[WinRateWindow] = []
    w = 0
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        slice_ = outcomes[start:end]
        wr = sum(slice_) / len(slice_) * 100
        w += 1
        windows.append(WinRateWindow(
            window_number=w,
            start_trade=start + 1,
            end_trade=end,
            n_trades=len(slice_),
            win_rate=round(wr, 1),
        ))

    if len(windows) < 2:
        return None

    overall_wr = sum(outcomes) / n * 100
    wrs = [wnd.win_rate for wnd in windows]
    mean_wr = sum(wrs) / len(wrs)
    std_wr = math.sqrt(sum((w - mean_wr) ** 2 for w in wrs) / len(wrs))
    cv = std_wr / mean_wr * 100 if mean_wr > 0 else 0.0
    min_wr = min(wrs)
    max_wr = max(wrs)
    wr_range = max_wr - min_wr

    # Stability grade by CV
    if cv < 10:
        grade = "Excellent"
    elif cv < 20:
        grade = "Good"
    elif cv < 35:
        grade = "Moderate"
    else:
        grade = "Unstable"

    # Wilson CI for overall win rate
    wins_total = sum(outcomes)
    ci_lo, ci_hi = _wilson_ci(wins_total, n)

    # Z-score of current window vs historical
    current_wr = windows[-1].win_rate
    z_score = (current_wr - mean_wr) / std_wr if std_wr > 0 else 0.0

    verdict = (
        f"Win rate stability: {grade} (CV={cv:.1f}%). "
        f"Overall {overall_wr:.1f}% ± {std_wr:.1f}pp across {len(windows)} windows. "
        f"Range: {min_wr:.0f}%-{max_wr:.0f}%. "
        f"95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]. "
        f"Current window: {current_wr:.0f}% (z={z_score:+.1f})."
    )

    return WinRateStabilityReport(
        windows=windows,
        overall_win_rate=round(overall_wr, 1),
        win_rate_std=round(std_wr, 2),
        win_rate_cv=round(cv, 1),
        win_rate_min=round(min_wr, 1),
        win_rate_max=round(max_wr, 1),
        win_rate_range=round(wr_range, 1),
        stability_grade=grade,
        ci_lower=round(ci_lo, 1),
        ci_upper=round(ci_hi, 1),
        current_window_wr=round(current_wr, 1),
        z_score=round(z_score, 2),
        n_trades=n,
        window_size=window_size,
        verdict=verdict,
    )
