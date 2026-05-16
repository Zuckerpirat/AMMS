"""Historical Volatility Cone.

Computes realized (historical) volatility over multiple lookback windows
and shows where the current short-term volatility sits within its
historical distribution (percentile rank).

The "cone" shows the range of historical volatility outcomes across
multiple measurement windows (5, 10, 20, 30, 60 days), helping
identify:
  - Whether current vol is high or low vs history
  - Whether short-term vol is elevated vs long-term (term structure)
  - Whether the stock is likely cheap or expensive to options traders

Realized vol formula:
  HV_N = std(log_returns[-N:]) × sqrt(252) × 100  (annualized %)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VolWindow:
    window: int           # lookback in bars
    hv: float             # current HV for this window (%)
    hv_min: float         # historical min HV for this window
    hv_max: float         # historical max HV for this window
    hv_25th: float        # 25th percentile
    hv_75th: float        # 75th percentile
    hv_median: float      # median
    percentile: float     # where current HV sits in history (0-100)
    regime: str           # "low" / "normal" / "elevated" / "extreme"


@dataclass(frozen=True)
class VolCone:
    symbol: str
    windows: list[VolWindow]   # sorted by window size
    term_structure: str        # "normal" (short<long) / "inverted" (short>long) / "flat"
    short_term_regime: str     # regime of shortest window
    bars_used: int
    verdict: str


def _percentile_rank(value: float, series: list[float]) -> float:
    """Percentile rank using interpolation."""
    n = len(series)
    if n == 0:
        return 50.0
    n_below = sum(1 for x in series if x < value)
    n_equal = sum(1 for x in series if x == value)
    return (n_below + 0.5 * n_equal) / n * 100


def _hv(log_returns: list[float]) -> float:
    """Annualized HV in percent."""
    n = len(log_returns)
    if n < 2:
        return 0.0
    mean = sum(log_returns) / n
    var = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    return math.sqrt(var) * math.sqrt(252) * 100


def _percentile(series: list[float], pct: float) -> float:
    """Linear interpolation percentile."""
    if not series:
        return 0.0
    s = sorted(series)
    n = len(s)
    idx = pct / 100 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def compute(
    bars: list,
    *,
    symbol: str = "",
    windows: list[int] | None = None,
    history_window: int = 252,
) -> VolCone | None:
    """Compute HV cone for multiple lookback windows.

    bars: list of bars with .close
    symbol: ticker symbol for display
    windows: list of lookback periods (default [5, 10, 20, 30, 60])
    history_window: how many bars of history to compute distributions from

    Returns None if fewer than max(windows) + 20 bars available.
    """
    if windows is None:
        windows = [5, 10, 20, 30, 60]

    if not bars or len(bars) < max(windows) + 20:
        return None

    closes = [float(b.close) for b in bars]
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
    ]

    if len(log_returns) < max(windows) + 10:
        return None

    # Limit to history_window recent returns
    log_returns = log_returns[-history_window:]
    n_returns = len(log_returns)

    vol_windows: list[VolWindow] = []

    for w in sorted(windows):
        if w >= n_returns:
            continue

        # Current HV: most recent w returns
        current_hv = _hv(log_returns[-w:])

        # Rolling HV distribution: compute HV for every window starting point
        rolling_hvs: list[float] = []
        for start in range(0, n_returns - w + 1):
            window_returns = log_returns[start: start + w]
            rolling_hvs.append(_hv(window_returns))

        if len(rolling_hvs) < 5:
            continue

        pct_rank = _percentile_rank(current_hv, rolling_hvs)

        if pct_rank < 25:
            regime = "low"
        elif pct_rank < 60:
            regime = "normal"
        elif pct_rank < 85:
            regime = "elevated"
        else:
            regime = "extreme"

        vol_windows.append(VolWindow(
            window=w,
            hv=round(current_hv, 2),
            hv_min=round(min(rolling_hvs), 2),
            hv_max=round(max(rolling_hvs), 2),
            hv_25th=round(_percentile(rolling_hvs, 25), 2),
            hv_75th=round(_percentile(rolling_hvs, 75), 2),
            hv_median=round(_percentile(rolling_hvs, 50), 2),
            percentile=round(pct_rank, 1),
            regime=regime,
        ))

    if not vol_windows:
        return None

    # Term structure: compare shortest vs longest window HV
    short_hv = vol_windows[0].hv
    long_hv = vol_windows[-1].hv
    ratio = short_hv / long_hv if long_hv > 0 else 1.0

    if ratio > 1.15:
        term_structure = "inverted"   # short-term > long-term (stress)
    elif ratio < 0.85:
        term_structure = "normal"     # short-term < long-term (calm)
    else:
        term_structure = "flat"

    short_regime = vol_windows[0].regime

    # Verdict
    if term_structure == "inverted" and short_regime in ("elevated", "extreme"):
        verdict = "Short-term vol spike — options expensive, consider reducing risk"
    elif term_structure == "inverted":
        verdict = "Inverted term structure — recent vol above long-term average"
    elif short_regime == "low":
        verdict = "Low vol environment — potential squeeze setup, watch for breakout"
    elif short_regime == "extreme":
        verdict = "Extreme vol — wide spreads, position sizing caution"
    else:
        verdict = f"Vol in {short_regime} regime — {term_structure} term structure"

    return VolCone(
        symbol=symbol,
        windows=vol_windows,
        term_structure=term_structure,
        short_term_regime=short_regime,
        bars_used=len(bars),
        verdict=verdict,
    )
