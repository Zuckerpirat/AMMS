"""Position sizing strategies.

Three approaches:
  1. Fixed fraction: risk a fixed % of equity per trade
  2. Kelly criterion: optimal fraction based on win rate and payoff ratio
  3. ATR-based: risk a fixed $ amount, sized by ATR (volatility-adjusted)

All methods output a recommended share count and dollar allocation.
Respects max_position_pct cap (default 20% of equity).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizingResult:
    strategy: str           # "fixed_fraction" | "kelly" | "atr_based"
    shares: int             # recommended share count (floor)
    dollar_amount: float    # shares × price
    pct_of_equity: float    # dollar_amount / equity × 100
    risk_amount: float      # expected $ at risk (shares × stop_distance)
    risk_pct_equity: float  # risk_amount / equity × 100
    notes: str


def fixed_fraction(
    equity: float,
    price: float,
    stop_loss_pct: float,
    *,
    risk_pct: float = 1.0,
    max_position_pct: float = 20.0,
) -> SizingResult:
    """Size position by risking a fixed % of equity.

    risk_pct: % of equity to risk per trade (default 1%)
    stop_loss_pct: distance from entry to stop as % of price
    """
    if price <= 0 or equity <= 0:
        return _empty("fixed_fraction", "Invalid price or equity.")
    if stop_loss_pct <= 0:
        return _empty("fixed_fraction", "Stop loss % must be > 0.")

    risk_amount = equity * risk_pct / 100
    stop_distance = price * stop_loss_pct / 100
    shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0

    # Cap at max_position_pct
    max_shares = int(equity * max_position_pct / 100 / price)
    shares = min(shares, max_shares)
    shares = max(shares, 0)

    dollar_amount = shares * price
    actual_risk = shares * stop_distance

    return SizingResult(
        strategy="fixed_fraction",
        shares=shares,
        dollar_amount=round(dollar_amount, 2),
        pct_of_equity=round(dollar_amount / equity * 100, 2),
        risk_amount=round(actual_risk, 2),
        risk_pct_equity=round(actual_risk / equity * 100, 3),
        notes=f"Risking {risk_pct:.1f}% equity = ${risk_amount:.2f} per trade.",
    )


def kelly_criterion(
    equity: float,
    price: float,
    stop_loss_pct: float,
    *,
    win_rate: float = 0.5,
    avg_win_pct: float = 3.0,
    avg_loss_pct: float = 1.5,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 20.0,
) -> SizingResult:
    """Size position using fractional Kelly criterion.

    kelly_fraction: fraction of full Kelly to use (0.25 = quarter Kelly, safer)
    win_rate: historical win rate (0..1)
    avg_win_pct: average winning trade return %
    avg_loss_pct: average losing trade loss %
    """
    if price <= 0 or equity <= 0:
        return _empty("kelly", "Invalid price or equity.")
    if stop_loss_pct <= 0:
        return _empty("kelly", "Stop loss % must be > 0.")
    if not (0 < win_rate < 1):
        return _empty("kelly", "Win rate must be between 0 and 1.")

    b = avg_win_pct / avg_loss_pct  # payoff ratio
    p = win_rate
    q = 1 - p
    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return SizingResult(
            strategy="kelly",
            shares=0,
            dollar_amount=0.0,
            pct_of_equity=0.0,
            risk_amount=0.0,
            risk_pct_equity=0.0,
            notes=f"Kelly = {full_kelly:.3f} (negative edge — skip trade).",
        )

    frac_kelly = full_kelly * kelly_fraction
    dollar_alloc = equity * frac_kelly
    max_alloc = equity * max_position_pct / 100
    dollar_alloc = min(dollar_alloc, max_alloc)

    shares = int(dollar_alloc / price)
    shares = max(shares, 0)
    dollar_amount = shares * price
    stop_distance = price * stop_loss_pct / 100
    risk_amount = shares * stop_distance

    return SizingResult(
        strategy="kelly",
        shares=shares,
        dollar_amount=round(dollar_amount, 2),
        pct_of_equity=round(dollar_amount / equity * 100, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct_equity=round(risk_amount / equity * 100, 3),
        notes=(
            f"Full Kelly {full_kelly:.3f} × {kelly_fraction:.2f} = {frac_kelly:.3f} "
            f"(b={b:.2f}, win={win_rate:.0%})."
        ),
    )


def atr_based(
    equity: float,
    price: float,
    atr: float,
    *,
    atr_multiplier: float = 2.0,
    risk_pct: float = 1.0,
    max_position_pct: float = 20.0,
) -> SizingResult:
    """Size position by risking a fixed % of equity, stop at ATR × multiplier.

    atr_multiplier: stop = price - (atr × multiplier) for longs
    """
    if price <= 0 or equity <= 0 or atr <= 0:
        return _empty("atr_based", "Invalid price, equity, or ATR.")

    stop_distance = atr * atr_multiplier
    risk_amount = equity * risk_pct / 100
    shares = int(risk_amount / stop_distance)

    max_shares = int(equity * max_position_pct / 100 / price)
    shares = min(shares, max_shares)
    shares = max(shares, 0)

    dollar_amount = shares * price
    actual_risk = shares * stop_distance
    stop_price = price - stop_distance

    return SizingResult(
        strategy="atr_based",
        shares=shares,
        dollar_amount=round(dollar_amount, 2),
        pct_of_equity=round(dollar_amount / equity * 100, 2),
        risk_amount=round(actual_risk, 2),
        risk_pct_equity=round(actual_risk / equity * 100, 3),
        notes=(
            f"ATR stop: ${price:.2f} - {atr_multiplier}×ATR({atr:.2f}) = ${stop_price:.2f}. "
            f"Risking {risk_pct:.1f}% equity."
        ),
    )


def _empty(strategy: str, reason: str) -> SizingResult:
    return SizingResult(
        strategy=strategy, shares=0, dollar_amount=0.0,
        pct_of_equity=0.0, risk_amount=0.0, risk_pct_equity=0.0, notes=reason,
    )
