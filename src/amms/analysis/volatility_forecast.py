"""EWMA Volatility Forecast.

Estimates the next-period volatility using the RiskMetrics EWMA model
(Exponentially Weighted Moving Average of squared log-returns).

Model: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
  λ = decay factor (default 0.94 for daily, as per J.P. Morgan RiskMetrics)
  r = log return

Outputs:
  - Current EWMA variance (annualised vol %)
  - Forecast for next 1, 5, 10 business days
  - Vol percentile vs the recent history
  - Vol trend: is volatility rising or falling?
  - VaR estimate at 95%: -1.645 × σ_daily
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VolForecastReport:
    symbol: str
    ewma_vol_daily: float    # daily vol % (1 std dev)
    ewma_vol_annual: float   # annualised vol %
    vol_1d: float            # forecast for tomorrow (daily %)
    vol_5d: float            # 5-day forward forecast (daily %)
    vol_10d: float           # 10-day forward forecast (daily %)
    var_95_1d: float         # 1-day 95% VaR (as % loss; negative)
    vol_percentile: float    # current vol vs 90-bar history (0=calmest, 100=wildest)
    vol_trend: str           # "rising", "falling", "stable"
    lambda_: float           # decay factor used
    bars_used: int
    n_returns: int
    verdict: str


def _log_returns(closes: list[float]) -> list[float]:
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            returns.append(math.log(closes[i] / closes[i - 1]))
        else:
            returns.append(0.0)
    return returns


def _ewma_variance_series(returns: list[float], lambda_: float) -> list[float]:
    """Compute EWMA variance for all returns."""
    if not returns:
        return []
    # Initialise with unconditional variance from first 10 returns
    init_n = min(10, len(returns))
    sigma2 = sum(r ** 2 for r in returns[:init_n]) / init_n
    series = [sigma2]
    for r in returns[1:]:
        sigma2 = lambda_ * sigma2 + (1 - lambda_) * r ** 2
        series.append(sigma2)
    return series


def analyze(
    bars: list,
    *,
    symbol: str = "",
    lambda_: float = 0.94,
    annualize: float = 252.0,
    history_pct_window: int = 90,
) -> VolForecastReport | None:
    """Compute EWMA volatility forecast from bar history.

    bars: bar objects with .close attribute.
    lambda_: EWMA decay factor (0.94 = RiskMetrics daily default).
    annualize: trading days per year for annualisation.
    history_pct_window: bars to use for percentile ranking.
    """
    if not bars or len(bars) < 15:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    returns = _log_returns(closes)
    if len(returns) < 10:
        return None

    variance_series = _ewma_variance_series(returns, lambda_)
    current_var = variance_series[-1]
    current_vol_daily = math.sqrt(current_var) * 100.0  # as %
    current_vol_annual = current_vol_daily * math.sqrt(annualize)

    # Forward forecast: EWMA mean-reverts to long-run vol (unconditional var)
    # For simplicity, assume vol is persistent: 1-period forecast = current
    # Multi-step: EWMA variance decays toward long-run variance
    long_run_var = sum(r ** 2 for r in returns) / len(returns)
    # k-step ahead variance: σ²_{t+k} ≈ λ^k · σ²_t + (1-λ^k) · σ²_lr
    def fwd_vol(k: int) -> float:
        fwd_var = lambda_ ** k * current_var + (1 - lambda_ ** k) * long_run_var
        return math.sqrt(fwd_var) * 100.0

    vol_1d = fwd_vol(1)
    vol_5d = fwd_vol(5)
    vol_10d = fwd_vol(10)

    # 1-day 95% VaR: -1.645 × σ_daily
    var_95 = -1.645 * current_vol_daily

    # Percentile of current vol vs history
    lookback = min(len(variance_series), history_pct_window)
    recent_vols = [math.sqrt(v) * 100.0 for v in variance_series[-lookback:]]
    below = sum(1 for v in recent_vols if v <= current_vol_daily)
    pct = below / len(recent_vols) * 100.0

    # Trend: compare recent 5-bar avg vol to older 5-bar avg
    if len(variance_series) >= 10:
        recent_avg = sum(math.sqrt(v) for v in variance_series[-5:]) / 5
        older_avg = sum(math.sqrt(v) for v in variance_series[-10:-5]) / 5
        if recent_avg > older_avg * 1.1:
            vol_trend = "rising"
        elif recent_avg < older_avg * 0.9:
            vol_trend = "falling"
        else:
            vol_trend = "stable"
    else:
        vol_trend = "stable"

    # Verdict
    parts = [f"daily vol {current_vol_daily:.2f}% ({current_vol_annual:.1f}% annualised)"]
    parts.append(f"vol percentile {pct:.0f}% (vs last {lookback} bars)")
    if vol_trend != "stable":
        parts.append(f"vol {vol_trend}")
    parts.append(f"1-day 95% VaR {var_95:.2f}%")

    verdict = "EWMA vol forecast: " + "; ".join(parts) + "."

    return VolForecastReport(
        symbol=symbol,
        ewma_vol_daily=round(current_vol_daily, 4),
        ewma_vol_annual=round(current_vol_annual, 2),
        vol_1d=round(vol_1d, 4),
        vol_5d=round(vol_5d, 4),
        vol_10d=round(vol_10d, 4),
        var_95_1d=round(var_95, 4),
        vol_percentile=round(pct, 1),
        vol_trend=vol_trend,
        lambda_=lambda_,
        bars_used=len(bars),
        n_returns=len(returns),
        verdict=verdict,
    )
