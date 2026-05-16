"""Market Breadth Proxy.

Estimates market breadth from a portfolio of bar histories:
  - % of symbols above SMA-50
  - % of symbols with positive 20-day ROC
  - % of symbols with RSI > 50 (bullish momentum)
  - New highs vs new lows (within lookback)
  - Breadth thrust: rapid shift from widespread selling to buying

The breadth composite score (0-100) summarises overall market health.
Higher = broader bull market participation. Lower = narrowing / divergence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolBreadth:
    symbol: str
    above_sma50: bool
    roc20_positive: bool
    rsi_above_50: bool
    near_52w_high: bool    # within 5% of lookback high
    near_52w_low: bool     # within 5% of lookback low
    roc_20: float
    rsi: float
    current_price: float


@dataclass(frozen=True)
class BreadthProxyReport:
    n_symbols: int
    n_evaluated: int

    pct_above_sma50: float
    pct_positive_roc20: float
    pct_rsi_above_50: float
    pct_near_highs: float
    pct_near_lows: float

    breadth_score: float         # 0-100 composite
    breadth_label: str           # "broad_bull" / "neutral" / "mixed" / "broad_bear"

    advance_count: int
    decline_count: int

    symbols: list[SymbolBreadth]

    breadth_thrust: bool
    thrust_direction: str        # "bullish" / "bearish" / "none"

    verdict: str


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    last = changes[-period:]
    gains = [max(0.0, c) for c in last]
    losses = [abs(min(0.0, c)) for c in last]
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def _roc(closes: list[float], period: int) -> float:
    if len(closes) <= period or closes[-(period + 1)] <= 0:
        return 0.0
    return (closes[-1] / closes[-(period + 1)] - 1.0) * 100.0


def _symbol_breadth(symbol: str, bars: list) -> SymbolBreadth | None:
    if not bars or len(bars) < 25:
        return None
    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None
    current = closes[-1]
    if current <= 0:
        return None

    sma50 = _sma(closes, 50)
    above_sma = sma50 is not None and current > sma50

    roc20 = _roc(closes, 20)
    rsi_val = _rsi(closes)

    lookback_high = max(closes)
    lookback_low = min(closes)
    near_high = current >= lookback_high * 0.95
    near_low = current <= lookback_low * 1.05

    return SymbolBreadth(
        symbol=symbol,
        above_sma50=above_sma,
        roc20_positive=roc20 > 0,
        rsi_above_50=rsi_val > 50,
        near_52w_high=near_high,
        near_52w_low=near_low,
        roc_20=round(roc20, 2),
        rsi=round(rsi_val, 1),
        current_price=round(current, 2),
    )


def analyze(
    bars_by_symbol: dict[str, list],
    *,
    history_snapshots: list[dict[str, list]] | None = None,
) -> BreadthProxyReport | None:
    """Compute market breadth from multiple symbol bar histories.

    bars_by_symbol: dict of symbol → bar list.
    history_snapshots: optional older snapshots for thrust detection.
    """
    if not bars_by_symbol:
        return None

    results: list[SymbolBreadth] = []
    for sym, bars in bars_by_symbol.items():
        sb = _symbol_breadth(sym, bars)
        if sb is not None:
            results.append(sb)

    n_eval = len(results)
    if n_eval < 2:
        return None

    above_sma = sum(1 for s in results if s.above_sma50)
    pos_roc = sum(1 for s in results if s.roc20_positive)
    rsi_bull = sum(1 for s in results if s.rsi_above_50)
    near_hi = sum(1 for s in results if s.near_52w_high)
    near_lo = sum(1 for s in results if s.near_52w_low)

    pct_sma = above_sma / n_eval * 100.0
    pct_roc = pos_roc / n_eval * 100.0
    pct_rsi = rsi_bull / n_eval * 100.0
    pct_hi = near_hi / n_eval * 100.0

    breadth_score = (pct_sma + pct_roc + pct_rsi + pct_hi) / 4.0

    if breadth_score >= 65:
        label = "broad_bull"
    elif breadth_score >= 45:
        label = "neutral"
    elif breadth_score >= 30:
        label = "mixed"
    else:
        label = "broad_bear"

    # Thrust detection
    thrust = False
    thrust_dir = "none"
    if history_snapshots:
        old_snap = history_snapshots[-1]
        old_valid = [_symbol_breadth(sym, ob) for sym, ob in old_snap.items()]
        old_valid = [r for r in old_valid if r is not None]
        if old_valid:
            old_pct = sum(1 for r in old_valid if r.above_sma50) / len(old_valid) * 100.0
            delta = pct_sma - old_pct
            if abs(delta) > 20:
                thrust = True
                thrust_dir = "bullish" if delta > 0 else "bearish"

    parts = [f"{breadth_score:.0f}/100 ({label})"]
    parts.append(f"{pct_sma:.0f}% above SMA-50")
    parts.append(f"{pct_roc:.0f}% positive ROC-20")
    if near_hi > 0:
        parts.append(f"{near_hi} near highs")
    if near_lo > 0:
        parts.append(f"{near_lo} near lows")
    if thrust:
        parts.append(f"breadth thrust {thrust_dir}")
    verdict = "Market breadth: " + "; ".join(parts) + "."

    return BreadthProxyReport(
        n_symbols=len(bars_by_symbol),
        n_evaluated=n_eval,
        pct_above_sma50=round(pct_sma, 1),
        pct_positive_roc20=round(pct_roc, 1),
        pct_rsi_above_50=round(pct_rsi, 1),
        pct_near_highs=round(pct_hi, 1),
        pct_near_lows=round(near_lo / n_eval * 100.0, 1),
        breadth_score=round(breadth_score, 1),
        breadth_label=label,
        advance_count=above_sma,
        decline_count=n_eval - above_sma,
        symbols=results,
        breadth_thrust=thrust,
        thrust_direction=thrust_dir,
        verdict=verdict,
    )
