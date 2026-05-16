"""Chaikin Volatility (CV) Analyser.

Measures the rate of change in the EMA of the high-low price range,
indicating whether volatility is expanding or contracting.

Algorithm:
  1. HL  = High - Low  (bar range)
  2. EMA_HL = EMA(HL, ema_period)
  3. CV  = (EMA_HL[t] - EMA_HL[t - roc_period]) / EMA_HL[t - roc_period] × 100

Interpretation:
  CV rising  → volatility expanding (trend or breakout likely)
  CV falling → volatility contracting (consolidation / coiling)
  Sharp CV spike from low base → breakout alert
  CV decelerating after high level → volatility normalizing
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CVReport:
    symbol: str

    cv: float               # current Chaikin Volatility (%)
    cv_prev: float          # value `roc_period` bars ago
    ema_hl: float           # current EMA of HL range
    ema_hl_prev: float      # EMA of HL `roc_period` bars ago

    expanding: bool         # volatility increasing
    contracting: bool       # volatility decreasing
    spike: bool             # CV jumped sharply from low base (breakout alert)

    cv_percentile: float    # CV's rank within history (0-100)
    avg_cv: float           # mean CV over history window
    cv_std: float           # std dev of CV over history

    score: float            # -100..+100 (expansion = positive)
    signal: str             # "expanding", "contracting", "neutral", "spike", "squeeze"

    history: list[float]    # recent CV values
    bars_used: int
    verdict: str


def _ema(values: list[float], period: int) -> list[float]:
    if not values or period < 1:
        return []
    seed = sum(values[:period]) / period
    out = [seed]
    k = 2.0 / (period + 1)
    for v in values[period:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def analyze(
    bars: list,
    *,
    symbol: str = "",
    ema_period: int = 10,
    roc_period: int = 10,
    history: int = 30,
    spike_threshold: float = 2.0,
) -> CVReport | None:
    """Compute Chaikin Volatility from OHLC bars.

    bars: bar objects with .high and .low attributes.
    ema_period: EMA period for smoothing HL range.
    roc_period: lookback for rate-of-change computation.
    history: number of recent CV values to retain.
    spike_threshold: z-score above which a CV value is flagged as a spike.
    """
    min_bars = ema_period + roc_period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs = [float(b.high) for b in bars]
        lows  = [float(b.low)  for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    hl = [h - l for h, l in zip(highs, lows)]
    if any(v < 0 for v in hl):
        return None

    ema_hl = _ema(hl, ema_period)
    if len(ema_hl) < roc_period + history + 2:
        return None

    # CV series: ROC of EMA_HL
    cv_series: list[float] = []
    for i in range(roc_period, len(ema_hl)):
        base = ema_hl[i - roc_period]
        if base < 1e-9:
            cv_series.append(0.0)
        else:
            cv_series.append((ema_hl[i] - base) / base * 100.0)

    if len(cv_series) < history + 2:
        return None

    cv_val   = cv_series[-1]
    cv_prev  = cv_series[-2]

    ema_hl_curr = ema_hl[-1]
    ema_hl_prev = ema_hl[-roc_period - 1] if len(ema_hl) > roc_period else ema_hl[0]

    recent_cv = cv_series[-history:]
    avg = _mean(recent_cv)
    std = _std(recent_cv)
    pctile = sum(1 for v in recent_cv[:-1] if v < cv_val) / max(len(recent_cv) - 1, 1) * 100.0

    expanding   = cv_val > 0
    contracting = cv_val < 0
    z_score = (cv_val - avg) / std if std > 1e-9 else 0.0
    spike = z_score > spike_threshold and cv_val > 0

    # Score: positive = expanding (may signal breakout or trend), negative = squeeze
    score = max(-100.0, min(100.0, cv_val * 2.0))

    if spike:
        signal = "spike"
    elif cv_val < -10 or (cv_val < 0 and pctile < 20):
        signal = "squeeze"
    elif expanding and cv_val > 5:
        signal = "expanding"
    elif contracting and cv_val < -2:
        signal = "contracting"
    else:
        signal = "neutral"

    spike_str = " ⚡ Spike (breakout alert)." if spike else ""
    verdict = (
        f"Chaikin Volatility ({symbol}): CV={cv_val:+.1f}% "
        f"({'expanding' if expanding else 'contracting'}). "
        f"EMA(HL)={ema_hl_curr:.4f}. Pctile={pctile:.0f}%.{spike_str}"
    )

    return CVReport(
        symbol=symbol,
        cv=round(cv_val, 3),
        cv_prev=round(cv_prev, 3),
        ema_hl=round(ema_hl_curr, 6),
        ema_hl_prev=round(ema_hl_prev, 6),
        expanding=expanding,
        contracting=contracting,
        spike=spike,
        cv_percentile=round(pctile, 1),
        avg_cv=round(avg, 3),
        cv_std=round(std, 3),
        score=round(score, 2),
        signal=signal,
        history=[round(v, 3) for v in recent_cv],
        bars_used=len(bars),
        verdict=verdict,
    )
