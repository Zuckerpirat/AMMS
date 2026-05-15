"""Market regime detector.

Classifies the current market environment as bull / neutral / bear using
price-action signals on SPY (trend proxy) and VIXY (volatility proxy).

Regime is determined by a simple rule hierarchy:
  1. If VIXY is stressed (>5% 1d move) → bear
  2. If SPY is above its 50-day SMA → bull
  3. If SPY is below its 200-day SMA → bear
  4. Otherwise → neutral

This lives in the analysis layer (CLAUDE.md §2) and has no side effects.
The scheduler/decision engine reads the regime to adjust position sizing
and strategy mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketRegime:
    label: str           # "bull" | "neutral" | "bear"
    confidence: float    # 0..1 — higher = more signals agree
    reason: str
    spy_vs_sma50: float | None = None   # % above/below SMA-50
    spy_vs_sma200: float | None = None  # % above/below SMA-200
    vixy_1d_pct: float | None = None    # VIXY 1-day % change

    @property
    def is_bull(self) -> bool:
        return self.label == "bull"

    @property
    def is_bear(self) -> bool:
        return self.label == "bear"

    @property
    def risk_multiplier(self) -> float:
        """Position-size multiplier: bull=1.0, neutral=0.75, bear=0.5."""
        return {"bull": 1.0, "neutral": 0.75, "bear": 0.5}.get(self.label, 0.75)


_FALLBACK = MarketRegime(
    label="neutral",
    confidence=0.0,
    reason="No market data available (fallback to neutral)",
)


def detect_regime(data, *, spy_symbol: str = "SPY", vixy_symbol: str = "VIXY") -> MarketRegime:
    """Detect regime using SPY SMA crossover and VIXY stress.

    ``data`` must expose ``get_bars(symbol, limit=N)`` returning a list of Bar.
    Returns ``_FALLBACK`` on any error.
    """
    from amms.features.momentum import sma as compute_sma

    # Fetch SPY bars
    try:
        spy_bars = data.get_bars(spy_symbol, limit=210)
    except Exception as e:
        logger.warning("regime: could not fetch SPY bars: %s", e)
        return _FALLBACK

    if not spy_bars:
        return _FALLBACK

    spy_price = spy_bars[-1].close
    sma50 = compute_sma(spy_bars, 50)
    sma200 = compute_sma(spy_bars, 200)

    spy_vs_50 = ((spy_price - sma50) / sma50 * 100) if sma50 else None
    spy_vs_200 = ((spy_price - sma200) / sma200 * 100) if sma200 else None

    # Fetch VIXY for stress check
    vixy_1d = None
    try:
        vixy_bars = data.get_bars(vixy_symbol, limit=5)
        if len(vixy_bars) >= 2:
            vixy_1d = (vixy_bars[-1].close - vixy_bars[-2].close) / vixy_bars[-2].close * 100
    except Exception:
        pass

    # Rule hierarchy
    signals: list[str] = []
    bull_count = 0
    bear_count = 0

    if vixy_1d is not None and vixy_1d > 5.0:
        bear_count += 2
        signals.append(f"VIXY +{vixy_1d:.1f}% (stressed)")
    elif vixy_1d is not None and vixy_1d > 2.5:
        bear_count += 1
        signals.append(f"VIXY +{vixy_1d:.1f}% (elevated)")

    if spy_vs_50 is not None:
        if spy_vs_50 > 0:
            bull_count += 1
            signals.append(f"SPY {spy_vs_50:+.1f}% above SMA-50")
        else:
            bear_count += 1
            signals.append(f"SPY {spy_vs_50:+.1f}% below SMA-50")

    if spy_vs_200 is not None:
        if spy_vs_200 > 0:
            bull_count += 1
            signals.append(f"SPY {spy_vs_200:+.1f}% above SMA-200")
        else:
            bear_count += 2
            signals.append(f"SPY {spy_vs_200:+.1f}% below SMA-200")

    total = bull_count + bear_count or 1
    if bull_count > bear_count:
        label = "bull"
        confidence = bull_count / total
    elif bear_count > bull_count:
        label = "bear"
        confidence = bear_count / total
    else:
        label = "neutral"
        confidence = 0.5

    reason = "; ".join(signals) if signals else "insufficient data"

    return MarketRegime(
        label=label,
        confidence=round(confidence, 2),
        reason=reason,
        spy_vs_sma50=spy_vs_50,
        spy_vs_sma200=spy_vs_200,
        vixy_1d_pct=vixy_1d,
    )
