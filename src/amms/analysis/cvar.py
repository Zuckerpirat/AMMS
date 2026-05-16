"""Conditional Value at Risk (CVaR / Expected Shortfall) Calculator.

CVaR (also called Expected Shortfall, ES) answers:
  "Given that we're in the worst X% of outcomes, what is the average loss?"

This is more informative than VaR (which only says "we lose at least $X
with X% probability") because CVaR captures the severity of tail events.

Two modes:
  1. Historical CVaR from closed trade PnL distribution
  2. Parametric CVaR from bars-based return distribution

Computed at multiple confidence levels: 90%, 95%, 99%.

Also computes:
  - Historical VaR (percentile of loss distribution)
  - Tail Conditional Expectation (TCE): same as CVaR
  - Tail Risk Score: how bad the expected tail loss is relative to avg loss
  - Maximum Historical Loss: worst single trade/return

Risk levels:
  < 3%  per trade → low tail risk
  3-6%  per trade → moderate tail risk
  > 6%  per trade → high tail risk
  > 10% per trade → extreme tail risk
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CVaRLevel:
    confidence: float   # e.g. 0.95
    var: float          # Value at Risk (loss, positive = loss)
    cvar: float         # Expected Shortfall (avg of worse outcomes)
    n_tail_obs: int     # observations in the tail


@dataclass(frozen=True)
class CVaRReport:
    levels: list[CVaRLevel]     # at 90%, 95%, 99%
    cvar_95: float              # main headline figure
    var_95: float
    max_loss: float             # worst single observation
    avg_loss: float             # average of all losses
    tail_risk_score: float      # cvar_95 / avg_loss ratio
    tail_risk_label: str        # "low" / "moderate" / "high" / "extreme"
    n_observations: int
    n_losses: int
    source: str                 # "historical_trades" / "bar_returns"
    verdict: str


def _percentile(values: list[float], p: float) -> float:
    """p-th percentile (0-100) of sorted values (ascending)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _compute_cvar(losses: list[float], confidence: float) -> CVaRLevel:
    """Compute VaR and CVaR for losses (positive = loss) at given confidence level."""
    s = sorted(losses)
    n = len(s)
    tail_start = int(math.ceil(n * confidence))
    tail_start = max(0, min(tail_start, n - 1))
    tail = s[tail_start:]
    var_val = s[tail_start] if tail_start < n else s[-1]
    cvar_val = sum(tail) / len(tail) if tail else var_val
    return CVaRLevel(
        confidence=confidence,
        var=round(var_val, 4),
        cvar=round(cvar_val, 4),
        n_tail_obs=len(tail),
    )


def from_trades(conn, *, limit: int = 500) -> CVaRReport | None:
    """Compute CVaR from historical closed trade PnL% distribution.

    conn: SQLite connection.
    limit: max trades to use.
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct FROM trades
            WHERE status = 'closed' AND pnl_pct IS NOT NULL
            ORDER BY closed_at DESC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 10:
        return None

    pnls = [float(r[0]) for r in rows]
    return _compute_report(pnls, source="historical_trades")


def from_bars(bars: list, *, symbol: str = "") -> CVaRReport | None:
    """Compute CVaR from bar log-return distribution.

    bars: list[Bar] with .close — at least 30 bars.
    """
    if not bars or len(bars) < 30:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            try:
                returns.append(math.log(closes[i] / closes[i - 1]) * 100.0)
            except Exception:
                pass

    if len(returns) < 10:
        return None

    return _compute_report(returns, source=f"bar_returns({symbol})")


def _compute_report(observations: list[float], *, source: str) -> CVaRReport | None:
    if not observations:
        return None

    # Convert to losses: positive = bad
    # We analyse the left tail: observations sorted ascending
    losses_all = sorted(observations)  # ascending (most negative first)

    # Flip: treat negative returns as positive losses
    losses = [-v for v in observations if v < 0]
    if not losses:
        return None

    avg_loss = sum(losses) / len(losses)
    max_loss = max(losses)

    confidences = [0.90, 0.95, 0.99]
    levels = [_compute_cvar(losses, c) for c in confidences]

    cvar_95 = levels[1].cvar
    var_95 = levels[1].var

    # Tail risk score: cvar_95 relative to avg_loss
    tail_score = cvar_95 / avg_loss if avg_loss > 0 else 1.0

    if cvar_95 >= 10.0:
        label = "extreme"
    elif cvar_95 >= 6.0:
        label = "high"
    elif cvar_95 >= 3.0:
        label = "moderate"
    else:
        label = "low"

    verdict = (
        f"CVaR 95%: {cvar_95:.2f}% avg loss in worst 5% of {'trades' if 'trade' in source else 'sessions'}. "
        f"VaR 95%: {var_95:.2f}%. Max single loss: {max_loss:.2f}%. "
        f"Tail risk: {label.upper()}. "
        f"Tail/avg ratio: {tail_score:.2f}×."
    )

    return CVaRReport(
        levels=levels,
        cvar_95=round(cvar_95, 3),
        var_95=round(var_95, 3),
        max_loss=round(max_loss, 3),
        avg_loss=round(avg_loss, 3),
        tail_risk_score=round(tail_score, 3),
        tail_risk_label=label,
        n_observations=len(observations),
        n_losses=len(losses),
        source=source,
        verdict=verdict,
    )
