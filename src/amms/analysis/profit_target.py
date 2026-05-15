"""Profit target tracker.

For each open position calculates progress toward configurable
profit targets and flags positions that are close to target.

Targets are computed as multiples of the initial risk (R):
  - R = entry - stop (1x ATR as proxy for stop distance)
  - 1R target: entry + 1 * R
  - 2R target: entry + 2 * R  (default "standard" target)
  - 3R target: entry + 3 * R

Also supports a simple % target (e.g. 10% above entry).

Progress is reported as:
  - pct_to_target: how far current price is toward the target (0-100%)
  - exceeded: True if current price already passed the standard target
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetProgress:
    symbol: str
    current_price: float
    entry_price: float
    atr: float
    stop_1atr: float        # 1x ATR stop level
    target_1r: float        # 1R above entry
    target_2r: float        # 2R above entry (standard)
    target_3r: float        # 3R above entry
    pnl_pct: float          # current unrealized P&L %
    pct_to_2r: float        # progress toward 2R target (0-100, can exceed 100)
    r_multiple: float       # current P&L expressed in R multiples
    exceeded_2r: bool
    bars_used: int


def _atr14(bars: list) -> float | None:
    if len(bars) < 15:
        return None
    window = bars[-15:]
    trs = []
    for i in range(1, len(window)):
        h = window[i].high
        l = window[i].low
        pc = window[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs) if trs else None


def compute(
    symbol: str,
    entry_price: float,
    bars: list,
) -> TargetProgress | None:
    """Compute profit target progress for a position.

    symbol: ticker
    entry_price: average entry price (> 0)
    bars: list[Bar] — needs at least 15 bars for ATR
    Returns None if insufficient data or invalid entry.
    """
    if not bars or entry_price <= 0:
        return None

    atr = _atr14(bars)
    if atr is None or atr <= 0:
        return None

    current_price = bars[-1].close
    pnl_pct = (current_price - entry_price) / entry_price * 100

    stop_1atr = entry_price - atr
    risk = atr  # = entry - stop

    target_1r = entry_price + 1 * risk
    target_2r = entry_price + 2 * risk
    target_3r = entry_price + 3 * risk

    # Progress toward 2R target
    if risk > 0:
        pct_to_2r = (current_price - entry_price) / (2 * risk) * 100
        r_multiple = (current_price - entry_price) / risk
    else:
        pct_to_2r = 0.0
        r_multiple = 0.0

    exceeded_2r = current_price >= target_2r

    return TargetProgress(
        symbol=symbol,
        current_price=round(current_price, 4),
        entry_price=round(entry_price, 4),
        atr=round(atr, 4),
        stop_1atr=round(stop_1atr, 4),
        target_1r=round(target_1r, 4),
        target_2r=round(target_2r, 4),
        target_3r=round(target_3r, 4),
        pnl_pct=round(pnl_pct, 2),
        pct_to_2r=round(pct_to_2r, 1),
        r_multiple=round(r_multiple, 2),
        exceeded_2r=exceeded_2r,
        bars_used=len(bars),
    )
