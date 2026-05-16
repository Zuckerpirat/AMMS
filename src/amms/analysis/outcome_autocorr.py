"""Trade outcome autocorrelation.

Tests whether trade outcomes (win/loss) are independent or show
serial correlation — meaning: does a win today predict a win tomorrow?

Positive autocorrelation (hot hand): wins cluster together
Negative autocorrelation (mean-reversion): alternating wins and losses
Near-zero: outcomes appear random/independent

Uses lag-1 autocorrelation of the binary win/loss series.
Also tests against a simple null hypothesis using the runs test.

Reads from trade_pairs (pnl), ordered chronologically by sell_ts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OutcomeAutocorrReport:
    lag1_autocorr: float       # -1 to +1
    lag2_autocorr: float
    lag3_autocorr: float
    interpretation: str        # "hot_hand" / "mean_reversion" / "random"
    runs_count: int            # actual number of runs (W-L sequences)
    expected_runs: float       # expected under independence
    runs_z_score: float        # z=(runs-expected)/std; |z|>1.96 = significant
    runs_significant: bool     # True if runs differ from random at 95%
    win_rate: float
    n_trades: int
    verdict: str


def _lag_autocorr(series: list[float], lag: int) -> float:
    n = len(series)
    if n <= lag:
        return 0.0
    mu = sum(series) / n
    var = sum((x - mu) ** 2 for x in series)
    if var == 0:
        return 0.0
    cov = sum((series[i] - mu) * (series[i - lag] - mu) for i in range(lag, n))
    return cov / var


def _runs_test(series: list[int]) -> tuple[int, float, float]:
    """Returns (n_runs, expected_runs, z_score)."""
    n = len(series)
    if n < 2:
        return 1, 1.0, 0.0
    n1 = sum(series)   # wins
    n2 = n - n1         # losses
    if n1 == 0 or n2 == 0:
        return 1, 1.0, 0.0

    runs = 1
    for i in range(1, n):
        if series[i] != series[i - 1]:
            runs += 1

    expected = 2 * n1 * n2 / n + 1
    variance = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1)) if n > 1 else 1.0
    std = math.sqrt(variance) if variance > 0 else 1.0
    z = (runs - expected) / std
    return runs, expected, z


def compute(conn, *, limit: int = 200) -> OutcomeAutocorrReport | None:
    """Analyze trade outcome autocorrelation from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 20 trades.
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

    if not rows or len(rows) < 20:
        return None

    outcomes = []
    for (pnl,) in rows:
        try:
            outcomes.append(1 if float(pnl) > 0 else 0)
        except Exception:
            continue

    if len(outcomes) < 20:
        return None

    n = len(outcomes)
    win_rate = sum(outcomes) / n * 100

    series_f = [float(o) for o in outcomes]
    ac1 = _lag_autocorr(series_f, 1)
    ac2 = _lag_autocorr(series_f, 2)
    ac3 = _lag_autocorr(series_f, 3)

    runs, expected, z = _runs_test(outcomes)
    significant = abs(z) > 1.96

    if ac1 > 0.15:
        interpretation = "hot_hand"
    elif ac1 < -0.15:
        interpretation = "mean_reversion"
    else:
        interpretation = "random"

    interp_desc = {
        "hot_hand": "hot-hand effect (wins cluster → momentum in personal trading)",
        "mean_reversion": "mean-reversion (alternating wins/losses → avoid revenge trading)",
        "random": "independent outcomes (no significant serial correlation)",
    }.get(interpretation, "")

    sig_note = "significant (p<0.05)" if significant else "not significant"
    verdict = (
        f"Outcomes show {interp_desc}. "
        f"Lag-1 autocorrelation: {ac1:+.3f}. "
        f"Runs test: {runs} runs vs {expected:.0f} expected (z={z:+.2f}, {sig_note})."
    )

    return OutcomeAutocorrReport(
        lag1_autocorr=round(ac1, 4),
        lag2_autocorr=round(ac2, 4),
        lag3_autocorr=round(ac3, 4),
        interpretation=interpretation,
        runs_count=runs,
        expected_runs=round(expected, 1),
        runs_z_score=round(z, 3),
        runs_significant=significant,
        win_rate=round(win_rate, 1),
        n_trades=n,
        verdict=verdict,
    )
