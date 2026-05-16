"""Regime-Conditioned Position Sizer.

Combines a volatility/market regime estimate with Kelly Criterion to
produce a regime-appropriate position size. The idea: risk more when
the environment is calm and edge is high; risk less when volatility
spikes or the market is in a downtrend.

Regime multipliers applied to the base Kelly fraction:
  calm_bull    (low vol, uptrend)   → 1.00× (full)
  calm_neutral (low vol, sideways)  → 0.75×
  calm_bear    (low vol, downtrend) → 0.50×
  hot_bull     (high vol, uptrend)  → 0.60×
  hot_neutral  (high vol, sideways) → 0.40×
  hot_bear     (high vol, downtrend)→ 0.25×
  extreme_vol  (VIX-like spike)     → 0.10× (near-shutdown)

Base Kelly is computed from win_rate and payoff_ratio from trade history.
Half-Kelly (×0.5) is used as the practical base to reduce variance.

The module is self-contained: regime is derived from the bars directly
(ATR percentile for vol, EMA slope for trend) — no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeSizerReport:
    win_rate: float             # historical win rate (%)
    payoff_ratio: float         # avg_win / avg_loss
    kelly_pct: float            # raw full Kelly (%)
    half_kelly_pct: float       # 0.5 × Kelly (base before regime adj)
    regime: str                 # e.g. "calm_bull", "hot_bear"
    regime_multiplier: float    # 0.10 – 1.00
    adjusted_pct: float         # half_kelly × regime_multiplier
    max_loss_pct: float         # estimated worst-case per trade (%)
    suggested_shares: int | None
    portfolio_value: float | None
    current_price: float | None
    atr_pct: float              # ATR as % of price (vol proxy)
    trend_direction: str        # "up" / "down" / "flat"
    bars_used: int
    verdict: str


_REGIME_MULT: dict[str, float] = {
    "calm_bull":    1.00,
    "calm_neutral": 0.75,
    "calm_bear":    0.50,
    "hot_bull":     0.60,
    "hot_neutral":  0.40,
    "hot_bear":     0.25,
    "extreme_vol":  0.10,
}


def _compute_atr_pct(bars: list, period: int = 14) -> float:
    """ATR as % of last close."""
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, min(period + 1, len(bars))):
        h = float(bars[-i].high)
        l = float(bars[-i].low)
        pc = float(bars[-i - 1].close)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs) / len(trs) if trs else 0.0
    last_close = float(bars[-1].close)
    return atr / last_close * 100.0 if last_close > 0 else 0.0


def _compute_trend(bars: list, ema_period: int = 20) -> str:
    """Classify trend as up/down/flat using EMA slope."""
    if len(bars) < ema_period + 5:
        return "flat"
    closes = [float(b.close) for b in bars]
    k = 2.0 / (ema_period + 1)
    ema = sum(closes[:ema_period]) / ema_period
    emas = [ema]
    for c in closes[ema_period:]:
        ema = c * k + ema * (1 - k)
        emas.append(ema)
    if len(emas) < 5:
        return "flat"
    slope = (emas[-1] - emas[-5]) / emas[-5] * 100.0 if emas[-5] > 0 else 0.0
    if slope > 0.3:
        return "up"
    elif slope < -0.3:
        return "down"
    return "flat"


def _detect_regime(atr_pct: float, trend: str, extreme_threshold: float = 4.0, high_threshold: float = 2.0) -> str:
    """Map (atr_pct, trend) to a regime label."""
    if atr_pct >= extreme_threshold:
        return "extreme_vol"
    vol_label = "hot" if atr_pct >= high_threshold else "calm"
    trend_label = {"up": "bull", "down": "bear", "flat": "neutral"}.get(trend, "neutral")
    return f"{vol_label}_{trend_label}"


def analyze(
    bars: list,
    *,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    portfolio_value: float | None = None,
    current_price: float | None = None,
    kelly_cap: float = 25.0,
    extreme_atr: float = 4.0,
    high_atr: float = 2.0,
) -> RegimeSizerReport | None:
    """Compute regime-conditioned position size.

    bars: list[Bar] with .high .low .close — at least 15 bars for ATR/trend.
    win_rate: historical win rate 0–100.
    avg_win_pct: average winning return as % of position.
    avg_loss_pct: average losing return as % of position (positive number).
    portfolio_value: total capital ($) — needed for share count.
    current_price: stock price — needed for share count.
    kelly_cap: maximum Kelly fraction (default 25%).
    extreme_atr: ATR% >= this = extreme vol regime (default 4%).
    high_atr: ATR% >= this = high vol regime (default 2%).
    """
    if not bars or len(bars) < 5:
        return None

    if win_rate <= 0 or win_rate >= 100:
        return None
    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        return None

    w = win_rate / 100.0
    r = avg_win_pct / avg_loss_pct  # payoff ratio

    # Full Kelly
    kelly = (w * r - (1 - w)) / r
    kelly_pct = max(0.0, min(kelly * 100.0, kelly_cap))
    half_kelly = kelly_pct * 0.5

    # Regime detection
    try:
        atr_pct = _compute_atr_pct(bars)
        trend = _compute_trend(bars)
    except Exception:
        atr_pct = 0.0
        trend = "flat"

    regime = _detect_regime(atr_pct, trend, extreme_threshold=extreme_atr, high_threshold=high_atr)
    mult = _REGIME_MULT.get(regime, 0.5)
    adjusted_pct = half_kelly * mult

    # Worst-case per trade estimate: adjusted_pct × avg_loss_pct
    max_loss_pct = adjusted_pct / 100.0 * avg_loss_pct

    # Suggested shares
    suggested_shares = None
    if portfolio_value and current_price and current_price > 0 and portfolio_value > 0:
        position_value = portfolio_value * adjusted_pct / 100.0
        suggested_shares = max(0, int(position_value / current_price))

    try:
        bars_used = len(bars)
        last_price = float(bars[-1].close) if current_price is None else current_price
    except Exception:
        bars_used = len(bars)
        last_price = current_price

    # Verdict
    regime_desc = {
        "calm_bull": "calm bull market — full allocation",
        "calm_neutral": "calm sideways — reduced allocation",
        "calm_bear": "calm downtrend — defensive sizing",
        "hot_bull": "high vol uptrend — caution despite direction",
        "hot_neutral": "high vol sideways — significant reduction",
        "hot_bear": "high vol downtrend — minimal exposure",
        "extreme_vol": "extreme volatility — near-shutdown sizing",
    }.get(regime, regime)

    shares_str = f"  Suggested: {suggested_shares} shares." if suggested_shares is not None else ""
    verdict = (
        f"Regime: {regime_desc}. "
        f"Half-Kelly {half_kelly:.1f}% × {mult:.2f} = {adjusted_pct:.2f}% of capital. "
        f"Max loss/trade: {max_loss_pct:.2f}%.{shares_str}"
    )

    return RegimeSizerReport(
        win_rate=round(win_rate, 1),
        payoff_ratio=round(r, 3),
        kelly_pct=round(kelly_pct, 2),
        half_kelly_pct=round(half_kelly, 2),
        regime=regime,
        regime_multiplier=mult,
        adjusted_pct=round(adjusted_pct, 2),
        max_loss_pct=round(max_loss_pct, 3),
        suggested_shares=suggested_shares,
        portfolio_value=portfolio_value,
        current_price=last_price,
        atr_pct=round(atr_pct, 3),
        trend_direction=trend,
        bars_used=bars_used,
        verdict=verdict,
    )
