"""Realized Volatility Skew analysis.

Splits log returns into upside (positive) and downside (negative) components
and measures their realized volatility separately.

  RV_up   = std(r | r > 0) × √252
  RV_down = std(r | r < 0) × √252
  Skew    = (RV_up - RV_down) / ((RV_up + RV_down) / 2)

Skew interpretation:
  > +0.15  → Positive skew: upside moves larger than downside (good for longs)
  < -0.15  → Negative skew: downside moves larger than upside (crash risk)
  ≈ 0      → Symmetric volatility

Also computes:
  - Tail ratio: 95th percentile upside return / 5th percentile downside return
  - Gain-to-pain ratio: sum of gains / |sum of losses|
  - Semi-deviation (downside deviation, used in Sortino ratio)
  - Up/down day ratio and vol asymmetry
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VolSkewReport:
    symbol: str
    rv_total: float          # annualized total realized vol (%)
    rv_up: float             # annualized upside realized vol (%)
    rv_down: float           # annualized downside realized vol (%)
    skew: float              # (rv_up - rv_down) / avg_rv; positive = pos skew
    skew_label: str          # "positive" / "negative" / "symmetric"
    tail_ratio: float        # p95 up return / |p5 down return|; >1 = fat upside
    gain_to_pain: float      # sum(gains) / |sum(losses)|
    semi_deviation: float    # annualized downside std dev (%)
    up_days_pct: float       # % of days with positive return
    avg_up_return: float     # mean positive daily return (%)
    avg_down_return: float   # mean negative daily return (%)
    n_up: int
    n_down: int
    bars_used: int
    verdict: str


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mu = sum(values) / n
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def _percentile(values: list[float], p: float) -> float:
    """Return p-th percentile of sorted values (linear interpolation)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def analyze(
    bars: list,
    *,
    symbol: str = "",
    annualize: float = 252.0,
    skew_threshold: float = 0.15,
) -> VolSkewReport | None:
    """Compute realized volatility skew from bars.

    bars: list[Bar] with .close — at least 10 bars.
    symbol: ticker for display.
    annualize: trading days per year for annualizing vol (default 252).
    skew_threshold: |skew| above this is considered meaningful (default 0.15).
    """
    if not bars or len(bars) < 10:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    # Log returns
    log_rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            continue
        try:
            log_rets.append(math.log(closes[i] / closes[i - 1]) * 100.0)
        except Exception:
            continue

    if len(log_rets) < 5:
        return None

    up_rets = [r for r in log_rets if r > 0]
    down_rets = [r for r in log_rets if r < 0]

    if not up_rets or not down_rets:
        return None

    scale = math.sqrt(annualize)

    rv_total = _std(log_rets) * scale
    rv_up = _std(up_rets) * scale if len(up_rets) >= 2 else (sum(up_rets) / len(up_rets) * scale)
    rv_down = _std(down_rets) * scale if len(down_rets) >= 2 else (abs(sum(down_rets) / len(down_rets)) * scale)

    avg_rv = (rv_up + rv_down) / 2.0
    skew = (rv_up - rv_down) / avg_rv if avg_rv > 0 else 0.0

    if skew > skew_threshold:
        skew_label = "positive"
    elif skew < -skew_threshold:
        skew_label = "negative"
    else:
        skew_label = "symmetric"

    # Tail ratio: p95 upside / |p5 downside|
    p95_up = _percentile(up_rets, 95)
    p5_down = abs(_percentile(down_rets, 5))
    tail_ratio = p95_up / p5_down if p5_down > 0 else 1.0

    # Gain-to-pain
    total_gains = sum(up_rets)
    total_losses = abs(sum(down_rets))
    gain_to_pain = total_gains / total_losses if total_losses > 0 else float("inf")

    # Semi-deviation (downside only, against 0 as target)
    semi_dev = math.sqrt(sum(r ** 2 for r in down_rets) / len(down_rets)) * scale

    avg_up = sum(up_rets) / len(up_rets)
    avg_down = sum(down_rets) / len(down_rets)
    up_pct = len(up_rets) / len(log_rets) * 100.0

    # Verdict
    skew_desc = {
        "positive": f"Positive vol skew ({skew:+.2f}): upside moves larger than downside — breakout bias",
        "negative": f"Negative vol skew ({skew:+.2f}): downside moves larger than upside — crash risk",
        "symmetric": f"Symmetric vol ({skew:+.2f}): balanced upside/downside volatility",
    }.get(skew_label, "")

    tail_desc = (
        f"Tail ratio {tail_ratio:.2f} (>1 = fat upside, <1 = fat downside). "
        if tail_ratio != 1.0 else ""
    )
    verdict = (
        f"{skew_desc}. "
        f"RV: {rv_total:.1f}% (up {rv_up:.1f}% / down {rv_down:.1f}%). "
        f"{tail_desc}"
        f"Gain-to-pain: {gain_to_pain:.2f}. Semi-dev: {semi_dev:.1f}%."
    )

    return VolSkewReport(
        symbol=symbol,
        rv_total=round(rv_total, 2),
        rv_up=round(rv_up, 2),
        rv_down=round(rv_down, 2),
        skew=round(skew, 4),
        skew_label=skew_label,
        tail_ratio=round(tail_ratio, 3),
        gain_to_pain=round(gain_to_pain, 3),
        semi_deviation=round(semi_dev, 2),
        up_days_pct=round(up_pct, 1),
        avg_up_return=round(avg_up, 4),
        avg_down_return=round(avg_down, 4),
        n_up=len(up_rets),
        n_down=len(down_rets),
        bars_used=len(bars),
        verdict=verdict,
    )
