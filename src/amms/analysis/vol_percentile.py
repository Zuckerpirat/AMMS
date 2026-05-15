"""Volatility percentile rank.

Computes where the current realized volatility sits relative to its
own history.  Useful for:
  - Identifying unusually high/low vol regimes
  - Sizing positions inversely to vol
  - Detecting vol expansions before breakouts

Metrics:
  - realized_vol: current N-day HV (annualized %)
  - percentile: rank vs historical window (0-100)
  - regime: "low" | "normal" | "elevated" | "extreme"
  - vol_of_vol: volatility of the rolling vol series (stability measure)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VolPercentileResult:
    symbol: str
    realized_vol: float          # current HV annualized %
    percentile: float            # 0-100: where current vol sits in history
    regime: str                  # "low"|"normal"|"elevated"|"extreme"
    vol_of_vol: float            # std of rolling HV series (annualized)
    current_vol_window: int      # window used for current HV
    history_window: int          # total bars used for percentile ranking
    mean_vol: float              # mean historical vol
    median_vol: float            # median historical vol


def compute(
    bars: list,
    *,
    vol_window: int = 20,
    history_window: int = 252,
) -> VolPercentileResult | None:
    """Compute volatility percentile for the given bars.

    bars: list[Bar] — needs at least history_window + 1 bars
    vol_window: rolling window for HV computation (default 20 days)
    history_window: how many HV samples to rank against (default 252)
    """
    needed = history_window + vol_window
    if len(bars) < needed or vol_window < 3:
        return None

    symbol = bars[0].symbol
    closes = [b.close for b in bars]

    if any(c <= 0 for c in closes):
        return None

    # Compute log returns
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

    # Build rolling HV series
    hv_series: list[float] = []
    for i in range(vol_window - 1, len(log_rets)):
        window_rets = log_rets[i - vol_window + 1: i + 1]
        mean = sum(window_rets) / vol_window
        variance = sum((r - mean) ** 2 for r in window_rets) / vol_window
        hv_series.append(math.sqrt(variance) * math.sqrt(252) * 100)

    if len(hv_series) < history_window:
        return None

    # Use only the last history_window samples
    hist = hv_series[-history_window:]
    current_hv = hist[-1]

    # Percentile rank with mid-point formula
    n = len(hist)
    n_below = sum(1 for v in hist if v < current_hv)
    n_equal = sum(1 for v in hist if v == current_hv)
    percentile = (n_below + 0.5 * n_equal) / n * 100

    # Regime classification
    if percentile <= 25:
        regime = "low"
    elif percentile <= 60:
        regime = "normal"
    elif percentile <= 85:
        regime = "elevated"
    else:
        regime = "extreme"

    # Vol of vol (std of the hv series itself)
    mean_hv = sum(hist) / n
    vol_of_vol = math.sqrt(sum((v - mean_hv) ** 2 for v in hist) / n)

    sorted_hist = sorted(hist)
    if n % 2 == 0:
        median_hv = (sorted_hist[n // 2 - 1] + sorted_hist[n // 2]) / 2
    else:
        median_hv = sorted_hist[n // 2]

    return VolPercentileResult(
        symbol=symbol,
        realized_vol=round(current_hv, 2),
        percentile=round(percentile, 1),
        regime=regime,
        vol_of_vol=round(vol_of_vol, 2),
        current_vol_window=vol_window,
        history_window=n,
        mean_vol=round(mean_hv, 2),
        median_vol=round(median_hv, 2),
    )
