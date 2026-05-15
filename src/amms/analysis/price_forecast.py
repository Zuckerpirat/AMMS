"""Statistical price forecast using historical volatility.

Projects a confidence interval for price over the next N trading days
using a random walk assumption with historical daily return volatility.

The forecast is based on:
  - Historical daily log returns (last n_hist bars)
  - Geometric Brownian Motion approximation
  - Confidence intervals: 68%, 95% (±1σ, ±2σ)

This is NOT a prediction of future price — it's a statistical baseline
showing the range of price outcomes expected given historical volatility.

Interpretation:
  - Wide bands = high historical volatility (uncertain price range)
  - Narrow bands = low volatility (price likely to stay near current)
  - Expected path = drift-adjusted median forecast
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class PriceForecast:
    symbol: str
    current_price: float
    horizon_days: int
    expected: float          # drift-adjusted expected price
    p68_low: float           # 68% confidence interval (±1σ)
    p68_high: float
    p95_low: float           # 95% confidence interval (±2σ)
    p95_high: float
    daily_vol_pct: float     # annualized daily volatility %
    annualized_vol_pct: float  # annualized volatility (×√252)
    drift_pct: float         # estimated daily drift %
    bars_used: int


def forecast(
    bars: list[Bar],
    horizon_days: int = 10,
    *,
    n_hist: int = 30,
) -> PriceForecast | None:
    """Compute a statistical price forecast for the given horizon.

    bars: historical bar series (needs at least n_hist + 1 bars)
    horizon_days: how many trading days to project forward
    n_hist: lookback period for volatility estimation

    Returns None if insufficient data.
    """
    if len(bars) < n_hist + 1 or horizon_days < 1:
        return None

    symbol = bars[0].symbol
    window = bars[-(n_hist + 1):]
    closes = [b.close for b in window]

    if any(c <= 0 for c in closes):
        return None

    # Compute log returns
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(log_rets)

    if n < 3:
        return None

    mean_ret = sum(log_rets) / n
    variance = sum((r - mean_ret) ** 2 for r in log_rets) / n
    daily_std = math.sqrt(variance)

    if daily_std <= 0:
        daily_std = 1e-6

    current_price = closes[-1]
    t = horizon_days  # days forward

    # Geometric Brownian Motion:
    # E[S_t] = S_0 × exp(mean + 0.5 × variance) ^ t — risk-neutral path
    # Confidence intervals use: S_0 × exp(mean × t ± z × std × √t)

    drift = mean_ret  # daily drift (log)

    # Expected price (drift-adjusted)
    expected = current_price * math.exp((drift + 0.5 * variance) * t)

    sigma_t = daily_std * math.sqrt(t)

    # 68% CI (±1σ)
    p68_low = current_price * math.exp(drift * t - sigma_t)
    p68_high = current_price * math.exp(drift * t + sigma_t)

    # 95% CI (±2σ)
    p95_low = current_price * math.exp(drift * t - 2 * sigma_t)
    p95_high = current_price * math.exp(drift * t + 2 * sigma_t)

    daily_vol_pct = daily_std * 100
    annualized_vol_pct = daily_std * math.sqrt(252) * 100
    drift_pct = drift * 100

    return PriceForecast(
        symbol=symbol,
        current_price=round(current_price, 4),
        horizon_days=horizon_days,
        expected=round(expected, 4),
        p68_low=round(p68_low, 4),
        p68_high=round(p68_high, 4),
        p95_low=round(p95_low, 4),
        p95_high=round(p95_high, 4),
        daily_vol_pct=round(daily_vol_pct, 3),
        annualized_vol_pct=round(annualized_vol_pct, 2),
        drift_pct=round(drift_pct, 4),
        bars_used=n,
    )
