"""ATR-based stop loss optimizer.

For each open position, suggests stop-loss levels at:
  - Conservative: 1x ATR below entry
  - Standard:     1.5x ATR below entry
  - Wide:         2x ATR below entry

Also computes:
  - Current risk: (entry - current_stop) / entry * 100
  - R-multiple target: suggested take-profit at 2x initial risk
  - Whether the current price has already violated the standard stop

Uses ATR-14 (Average True Range, 14-period).
If the position has a known entry price, stops are anchored to entry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopSuggestion:
    symbol: str
    current_price: float
    entry_price: float
    atr: float                # 14-period ATR
    atr_pct: float            # ATR as % of current price
    stop_conservative: float  # 1x ATR from entry
    stop_standard: float      # 1.5x ATR from entry
    stop_wide: float          # 2x ATR from entry
    target_2r: float          # take-profit at 2x standard risk
    risk_standard_pct: float  # % risk at standard stop
    stop_violated: bool       # current price < stop_standard
    bars_used: int


def _atr(bars: list, period: int = 14) -> float | None:
    """Compute ATR-14 from bars."""
    if len(bars) < period + 1:
        return None
    window = bars[-(period + 1):]
    trs: list[float] = []
    for i in range(1, len(window)):
        high = window[i].high
        low = window[i].low
        prev_close = window[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


def suggest_stops(
    symbol: str,
    entry_price: float,
    bars: list,
    *,
    atr_period: int = 14,
) -> StopSuggestion | None:
    """Compute ATR-based stop suggestions for a single position.

    symbol: ticker
    entry_price: position entry price
    bars: recent bar data (list[Bar]) needs at least atr_period + 1 bars
    atr_period: ATR lookback (default 14)
    Returns None if insufficient data or entry_price <= 0.
    """
    if not bars or entry_price <= 0:
        return None

    atr_val = _atr(bars, period=atr_period)
    if atr_val is None or atr_val <= 0:
        return None

    current_price = bars[-1].close
    atr_pct = atr_val / current_price * 100 if current_price > 0 else 0.0

    stop_cons = entry_price - 1.0 * atr_val
    stop_std = entry_price - 1.5 * atr_val
    stop_wide = entry_price - 2.0 * atr_val

    # Ensure stops don't go negative
    stop_cons = max(stop_cons, current_price * 0.01)
    stop_std = max(stop_std, current_price * 0.01)
    stop_wide = max(stop_wide, current_price * 0.01)

    risk_std_pct = (entry_price - stop_std) / entry_price * 100

    # Target at 2R (2x the standard risk from entry)
    target_2r = entry_price + 2 * (entry_price - stop_std)

    stop_violated = current_price < stop_std

    return StopSuggestion(
        symbol=symbol,
        current_price=round(current_price, 4),
        entry_price=round(entry_price, 4),
        atr=round(atr_val, 4),
        atr_pct=round(atr_pct, 2),
        stop_conservative=round(stop_cons, 4),
        stop_standard=round(stop_std, 4),
        stop_wide=round(stop_wide, 4),
        target_2r=round(target_2r, 4),
        risk_standard_pct=round(risk_std_pct, 2),
        stop_violated=stop_violated,
        bars_used=len(bars),
    )
