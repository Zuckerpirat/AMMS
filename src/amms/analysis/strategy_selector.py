"""Regime-based strategy selector.

Given a market regime and current conditions, recommends which registered
strategy is best suited for current conditions.

Mapping logic:
  bull  + low vol  → composite or sma_cross (momentum, trend following)
  bull  + high vol → rsi_reversal (pullback buys in uptrend)
  neutral          → mean_reversion or vwap (range-bound)
  bear             → cash / defensive (nothing, or very small mean_reversion)

Also considers: ADX trend strength, RSI zone, and sector rotation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyRecommendation:
    primary: str            # recommended strategy name
    secondary: str | None   # backup/alternative
    regime: str             # current regime label
    reasoning: list[str]    # explanation bullets
    risk_multiplier: float  # suggested position size scaling
    avoid: list[str]        # strategies to avoid in current conditions


def recommend(
    regime: str,
    *,
    vix_proxy: float | None = None,      # e.g. VIXY price
    adx: float | None = None,            # current ADX value
    spy_rsi: float | None = None,        # SPY RSI
    top_sector: str | None = None,       # top rotating-in sector ETF
) -> StrategyRecommendation:
    """Recommend a strategy based on regime and optional indicators."""
    reasoning: list[str] = [f"Market regime: {regime}"]
    avoid: list[str] = []

    high_vol = vix_proxy is not None and vix_proxy > 25.0
    strong_trend = adx is not None and adx > 25.0
    oversold = spy_rsi is not None and spy_rsi < 40.0
    overbought = spy_rsi is not None and spy_rsi > 65.0

    if high_vol:
        reasoning.append(f"High volatility (VIXY ~{vix_proxy:.1f}) → wider stops needed")
    if strong_trend:
        reasoning.append(f"Strong trend (ADX {adx:.1f}) → trend-following favored")

    if regime == "bull":
        if strong_trend and not high_vol:
            primary = "sma_cross"
            secondary = "composite"
            reasoning.append("Strong uptrend with low vol → SMA cross momentum strategy")
            avoid = ["mean_reversion", "rsi_reversal"]
        elif high_vol or oversold:
            primary = "rsi_reversal"
            secondary = "composite"
            reasoning.append("Bull regime but elevated vol → buy RSI dips/pullbacks")
            avoid = ["breakout"]
        else:
            primary = "composite"
            secondary = "sma_cross"
            reasoning.append("Steady bull market → composite multi-signal strategy")
            avoid = ["mean_reversion"]
        risk_multiplier = 1.0

    elif regime == "neutral":
        if strong_trend:
            primary = "breakout"
            secondary = "composite"
            reasoning.append("Neutral regime but strong trend → breakout setups")
        else:
            primary = "mean_reversion"
            secondary = "vwap"
            reasoning.append("Neutral/sideways regime → mean reversion and VWAP strategies")
            avoid = ["sma_cross", "breakout"]
        risk_multiplier = 0.75

    elif regime == "bear":
        primary = "rsi_reversal"
        secondary = "mean_reversion"
        reasoning.append("Bear regime → only high-confidence oversold bounces, small size")
        reasoning.append("Consider reducing exposure or staying in cash")
        avoid = ["sma_cross", "breakout", "composite"]
        risk_multiplier = 0.5

    else:  # unknown
        primary = "composite"
        secondary = None
        reasoning.append("Unknown regime → conservative composite with reduced size")
        risk_multiplier = 0.5

    if overbought:
        reasoning.append(f"SPY overbought (RSI {spy_rsi:.1f}) → tighten stops, avoid new longs")
    if oversold and regime == "bear":
        reasoning.append(f"SPY oversold (RSI {spy_rsi:.1f}) in bear → potential bounce candidate")

    if top_sector:
        reasoning.append(f"Sector rotation: {top_sector} leading → favor {top_sector}-heavy names")

    return StrategyRecommendation(
        primary=primary,
        secondary=secondary,
        regime=regime,
        reasoning=reasoning,
        risk_multiplier=risk_multiplier,
        avoid=avoid,
    )
