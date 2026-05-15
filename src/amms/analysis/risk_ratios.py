"""Risk-adjusted performance ratios.

Computes Sharpe, Sortino, and Calmar ratios from equity snapshot history.
Uses equity_snapshots table (ts TEXT, equity REAL).

Sharpe  = (annualized_return - risk_free) / annualized_volatility
Sortino = (annualized_return - risk_free) / annualized_downside_vol
Calmar  = annualized_return / |max_drawdown|

Interpretation guidelines:
  Sharpe:  < 0 bad | 0-1 ok | 1-2 good | >2 excellent
  Sortino: generally 2× Sharpe for symmetric returns
  Calmar:  < 0.5 poor | 0.5-1 ok | > 1 good | > 3 excellent
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRatios:
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    annualized_return_pct: float
    annualized_vol_pct: float
    annualized_downside_vol_pct: float
    max_drawdown_pct: float        # negative value
    n_periods: int
    start_equity: float
    end_equity: float
    sharpe_grade: str              # "excellent"/"good"/"ok"/"poor"
    verdict: str


def _annualize(daily_val: float, periods_per_year: float = 252) -> float:
    return daily_val * (periods_per_year ** 0.5)


def compute(conn, *, limit: int = 252, risk_free_annual_pct: float = 4.5) -> RiskRatios | None:
    """Compute risk-adjusted ratios from equity_snapshots.

    conn: SQLite connection with equity_snapshots table (ts TEXT, equity REAL)
    limit: max snapshots to use (most recent)
    risk_free_annual_pct: annual risk-free rate in percent (default 4.5%)

    Returns None if insufficient data (< 10 rows) or table missing.
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

    # Reverse to chronological order
    rows = list(reversed(rows))
    equities = [float(r[1]) for r in rows]

    # Daily returns
    returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
    ]

    if not returns:
        return None

    n = len(returns)
    mean_ret = sum(returns) / n

    # Variance
    variance = sum((r - mean_ret) ** 2 for r in returns) / n
    vol = math.sqrt(variance) if variance > 0 else 0.0

    # Downside volatility (semi-deviation below 0)
    downside_sq = [r ** 2 for r in returns if r < 0]
    downside_vol = math.sqrt(sum(downside_sq) / n) if downside_sq else 0.0

    # Annualize (assuming daily snapshots, 252 trading days)
    ann_return = (equities[-1] / equities[0]) ** (252 / n) - 1 if n >= 2 else mean_ret * 252
    ann_return_pct = ann_return * 100
    ann_vol = vol * math.sqrt(252) * 100
    ann_down_vol = downside_vol * math.sqrt(252) * 100

    rf_daily = risk_free_annual_pct / 100 / 252
    excess_mean = mean_ret - rf_daily
    rf_annual = risk_free_annual_pct / 100

    # Sharpe
    sharpe = None
    if vol > 0:
        sharpe = round((ann_return - rf_annual) / (vol * math.sqrt(252)), 3)

    # Sortino
    sortino = None
    if downside_vol > 0:
        sortino = round((ann_return - rf_annual) / (downside_vol * math.sqrt(252)), 3)

    # Max drawdown
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd:
            max_dd = dd
    max_dd_pct = max_dd * 100

    # Calmar
    calmar = None
    if max_dd_pct < 0:
        calmar = round(ann_return_pct / abs(max_dd_pct), 3)

    # Grade
    sh = sharpe if sharpe is not None else -999
    if sh >= 2.0:
        grade = "excellent"
    elif sh >= 1.0:
        grade = "good"
    elif sh >= 0.0:
        grade = "ok"
    else:
        grade = "poor"

    # Verdict
    if sharpe is not None and sharpe >= 1.5:
        verdict = "Strong risk-adjusted returns"
    elif sharpe is not None and sharpe >= 0.5:
        verdict = "Acceptable risk-adjusted returns"
    elif sharpe is not None and sharpe >= 0:
        verdict = "Low edge — returns barely cover risk"
    else:
        verdict = "Negative edge — returns below risk-free rate"

    return RiskRatios(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        annualized_return_pct=round(ann_return_pct, 2),
        annualized_vol_pct=round(ann_vol, 2),
        annualized_downside_vol_pct=round(ann_down_vol, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        n_periods=n + 1,
        start_equity=round(equities[0], 2),
        end_equity=round(equities[-1], 2),
        sharpe_grade=grade,
        verdict=verdict,
    )
