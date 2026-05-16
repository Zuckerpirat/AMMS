"""Fixed-fractional position sizing.

Computes the correct number of shares to buy so that if the stop-loss
is triggered, you lose exactly risk_pct% of total capital.

Formula:
  risk_amount = capital × risk_pct / 100
  risk_per_share = entry_price - stop_price
  shares = risk_amount / risk_per_share

Also computes:
  - Position value (shares × entry_price)
  - Position weight % of portfolio
  - Risk-reward ratio targets (1R, 2R, 3R prices)
  - Maximum loss in $

Supports multiple risk levels (0.5%, 1%, 2%) for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizeLevel:
    risk_pct: float        # % of capital risked
    shares: int
    position_value: float  # shares × entry_price
    position_weight_pct: float  # % of capital
    max_loss: float        # = risk_amount in $
    target_1r: float       # break-even return = 1R
    target_2r: float
    target_3r: float
    risk_reward_needed: float   # risk_per_share in $


@dataclass(frozen=True)
class PositionSizeResult:
    symbol: str
    entry_price: float
    stop_price: float
    capital: float
    risk_per_share: float          # entry - stop
    risk_per_share_pct: float      # risk_per_share / entry × 100
    levels: list[SizeLevel]        # one per risk level
    note: str


def compute(
    symbol: str,
    entry_price: float,
    stop_price: float,
    capital: float,
    *,
    risk_levels_pct: list[float] | None = None,
) -> PositionSizeResult | None:
    """Compute position sizes for multiple risk levels.

    symbol: ticker symbol
    entry_price: planned entry price per share
    stop_price: stop-loss price per share
    capital: total account capital in $
    risk_levels_pct: list of risk % levels to compute (default [0.5, 1.0, 2.0])

    Returns None if inputs are invalid (stop >= entry, capital <= 0, etc.)
    """
    if risk_levels_pct is None:
        risk_levels_pct = [0.5, 1.0, 2.0]

    if entry_price <= 0 or capital <= 0:
        return None
    if stop_price >= entry_price:
        return None

    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return None

    risk_per_share_pct = risk_per_share / entry_price * 100

    levels: list[SizeLevel] = []
    for rp in risk_levels_pct:
        if rp <= 0:
            continue
        risk_amount = capital * rp / 100
        shares = max(0, int(risk_amount / risk_per_share))
        position_value = shares * entry_price
        position_weight = position_value / capital * 100

        # R targets
        t1r = entry_price + risk_per_share        # 1:1 risk-reward
        t2r = entry_price + 2 * risk_per_share
        t3r = entry_price + 3 * risk_per_share

        levels.append(SizeLevel(
            risk_pct=rp,
            shares=shares,
            position_value=round(position_value, 2),
            position_weight_pct=round(position_weight, 2),
            max_loss=round(risk_amount, 2),
            target_1r=round(t1r, 2),
            target_2r=round(t2r, 2),
            target_3r=round(t3r, 2),
            risk_reward_needed=round(risk_per_share, 2),
        ))

    if not levels:
        return None

    # Note
    wide_stop = risk_per_share_pct > 5
    tight_stop = risk_per_share_pct < 1
    if wide_stop:
        note = f"Wide stop ({risk_per_share_pct:.1f}%% risk/share) — position size will be small"
    elif tight_stop:
        note = f"Tight stop ({risk_per_share_pct:.1f}%% risk/share) — position size will be large, check liquidity"
    else:
        note = f"Stop distance {risk_per_share_pct:.1f}%% — standard range"

    return PositionSizeResult(
        symbol=symbol,
        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
        capital=round(capital, 2),
        risk_per_share=round(risk_per_share, 2),
        risk_per_share_pct=round(risk_per_share_pct, 2),
        levels=levels,
        note=note,
    )
