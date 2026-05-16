"""Mean Reversion Band Analyser.

Computes adaptive mean-reversion bands using a z-score approach.
Measures how many standard deviations the current price is from its
rolling mean, and flags extreme deviations as reversion opportunities.

Metrics:
  - Z-score: (price - mean) / std  over configurable periods
  - Mean (SMA), +1σ, +2σ, -1σ, -2σ bands
  - Distance from mean in % and sigma units
  - Half-life of mean reversion (AR(1) estimate)
  - Reversion signal: oversold/overbought with strength
  - Historical z-score distribution (what % of time is price this extreme)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MRBand:
    sigma: float      # -2, -1, 0 (mean), +1, +2
    price: float


@dataclass(frozen=True)
class MRBReport:
    symbol: str
    period: int          # lookback for mean/std
    current_price: float
    mean: float
    std: float
    z_score: float       # (price - mean) / std
    pct_from_mean: float # (price - mean) / mean * 100

    bands: list[MRBand]  # -2σ to +2σ

    # Signal
    signal: str          # "oversold_extreme", "oversold", "neutral", "overbought", "overbought_extreme"
    signal_strength: float  # abs(z_score) / 2 capped at 1

    # Z-score percentile vs history
    z_percentile: float  # 0=lowest ever, 100=highest ever

    # Half-life of mean reversion (estimated from AR1 autocorrelation)
    half_life: float | None   # in bars; smaller = faster reversion

    bars_used: int
    verdict: str


def _ar1_halflife(returns: list[float]) -> float | None:
    """Estimate mean-reversion half-life from AR(1) lag-1 autocorrelation."""
    if len(returns) < 10:
        return None
    n = len(returns)
    mean = sum(returns) / n
    # Compute lag-1 autocorrelation
    num = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n))
    den = sum((r - mean) ** 2 for r in returns)
    if den < 1e-12:
        return None
    rho = num / den
    rho = max(-0.999, min(0.999, rho))
    if rho >= 0:
        return None  # trending, not mean-reverting
    # Half life: -ln(2) / ln(rho)
    try:
        hl = -math.log(2) / math.log(abs(rho))
        return round(hl, 1)
    except (ValueError, ZeroDivisionError):
        return None


def analyze(bars: list, *, symbol: str = "", period: int = 20) -> MRBReport | None:
    """Compute mean-reversion bands from bar history.

    bars: bar objects with .close attribute.
    period: lookback for rolling mean and std.
    """
    if not bars or len(bars) < period + 10:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    current = closes[-1]
    if current <= 0:
        return None

    subset = closes[-period:]
    mean = sum(subset) / period
    variance = sum((v - mean) ** 2 for v in subset) / period
    std = math.sqrt(variance) if variance > 0 else 1e-9

    z = (current - mean) / std
    pct_from_mean = (current - mean) / mean * 100.0 if mean > 0 else 0.0

    bands = [
        MRBand(sigma=-2.0, price=round(mean - 2 * std, 4)),
        MRBand(sigma=-1.0, price=round(mean - std, 4)),
        MRBand(sigma=0.0, price=round(mean, 4)),
        MRBand(sigma=1.0, price=round(mean + std, 4)),
        MRBand(sigma=2.0, price=round(mean + 2 * std, 4)),
    ]

    # Signal
    if z > 2.0:
        signal = "overbought_extreme"
    elif z > 1.0:
        signal = "overbought"
    elif z < -2.0:
        signal = "oversold_extreme"
    elif z < -1.0:
        signal = "oversold"
    else:
        signal = "neutral"
    strength = min(1.0, abs(z) / 2.0)

    # Z-score percentile: compute rolling z-scores for history
    z_history: list[float] = []
    for i in range(period, len(closes)):
        sub = closes[i - period:i]
        m = sum(sub) / period
        var = sum((v - m) ** 2 for v in sub) / period
        sd = math.sqrt(var) if var > 0 else 1e-9
        z_history.append((closes[i] - m) / sd)

    if z_history:
        below = sum(1 for zh in z_history if zh <= z)
        z_pct = below / len(z_history) * 100.0
    else:
        z_pct = 50.0

    # Half-life from log price returns
    if len(closes) >= 20:
        log_returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]
        hl = _ar1_halflife(log_returns)
    else:
        hl = None

    # Verdict
    parts = [f"z-score {z:+.2f} ({signal})"]
    parts.append(f"{pct_from_mean:+.1f}% from mean ({mean:.2f})")
    if abs(z) > 1.5:
        parts.append(f"{'reversion opportunity' if abs(z) > 2 else 'stretched'}")
    if hl is not None and hl < 10:
        parts.append(f"half-life {hl:.0f} bars (fast reversion)")
    verdict = "Mean reversion: " + "; ".join(parts) + "."

    return MRBReport(
        symbol=symbol,
        period=period,
        current_price=round(current, 4),
        mean=round(mean, 4),
        std=round(std, 4),
        z_score=round(z, 4),
        pct_from_mean=round(pct_from_mean, 3),
        bands=bands,
        signal=signal,
        signal_strength=round(strength, 3),
        z_percentile=round(z_pct, 1),
        half_life=hl,
        bars_used=len(bars),
        verdict=verdict,
    )
