"""Position heat score: composite rating for open positions.

Aggregates multiple dimensions into a single 0-100 score:
  1. P&L performance (0-30 pts): unrealized gain vs entry
  2. Momentum (0-25 pts): recent price direction and strength
  3. Drawdown health (0-25 pts): how far from peak (inverted drawdown)
  4. Liquidity (0-20 pts): simplified volume score

Score interpretation:
  80-100: HOT — position is working well, maintain/add
  60-79:  WARM — decent position, monitor
  40-59:  NEUTRAL — mixed signals, review
  20-39:  COOL — underperforming, watch closely
  0-19:   COLD — position in trouble, consider cutting

Status labels: "hot" | "warm" | "neutral" | "cool" | "cold"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionHeat:
    symbol: str
    score: float              # 0-100
    status: str               # "hot"|"warm"|"neutral"|"cool"|"cold"
    pnl_pct: float            # unrealized P&L %
    pnl_score: float          # 0-30
    momentum_score: float     # 0-25
    drawdown_score: float     # 0-25
    liquidity_score: float    # 0-20
    bars_used: int


@dataclass(frozen=True)
class HeatReport:
    positions: list[PositionHeat]
    avg_score: float
    hottest: PositionHeat | None
    coldest: PositionHeat | None
    n_hot: int     # score >= 60
    n_cold: int    # score < 40


def _pnl_score(pnl_pct: float) -> float:
    """0-30 points based on unrealized P&L %."""
    if pnl_pct >= 10:
        return 30.0
    elif pnl_pct >= 5:
        return 25.0
    elif pnl_pct >= 2:
        return 20.0
    elif pnl_pct >= 0:
        return 15.0
    elif pnl_pct >= -2:
        return 10.0
    elif pnl_pct >= -5:
        return 5.0
    elif pnl_pct >= -10:
        return 2.0
    else:
        return 0.0


def _momentum_score_from_bars(bars: list) -> float:
    """0-25 points based on recent price momentum (last 5 vs 10 bars)."""
    if len(bars) < 11:
        return 12.5  # neutral
    closes = [b.close for b in bars]
    recent = closes[-1]
    ref5 = closes[-6]
    ref10 = closes[-11]
    mom5 = (recent - ref5) / ref5 * 100 if ref5 > 0 else 0.0
    mom10 = (recent - ref10) / ref10 * 100 if ref10 > 0 else 0.0
    combined = 0.6 * mom5 + 0.4 * mom10

    if combined >= 5:
        return 25.0
    elif combined >= 2:
        return 20.0
    elif combined >= 0.5:
        return 16.0
    elif combined >= -0.5:
        return 12.0
    elif combined >= -2:
        return 8.0
    elif combined >= -5:
        return 4.0
    else:
        return 0.0


def _drawdown_score_from_bars(bars: list, lookback: int = 20) -> float:
    """0-25 points based on distance from recent peak (inverted drawdown)."""
    if len(bars) < 3:
        return 12.5
    window = bars[-lookback:] if len(bars) > lookback else bars
    closes = [b.close for b in window]
    peak = max(closes)
    current = closes[-1]
    dd_pct = (current - peak) / peak * 100 if peak > 0 else 0.0

    if dd_pct >= -1:
        return 25.0
    elif dd_pct >= -3:
        return 20.0
    elif dd_pct >= -7:
        return 14.0
    elif dd_pct >= -15:
        return 7.0
    elif dd_pct >= -25:
        return 3.0
    else:
        return 0.0


def _liquidity_score_from_bars(bars: list) -> float:
    """0-20 points based on average volume."""
    if not bars:
        return 10.0
    avg_vol = sum(b.volume for b in bars[-20:]) / min(20, len(bars))
    if avg_vol >= 2_000_000:
        return 20.0
    elif avg_vol >= 500_000:
        return 16.0
    elif avg_vol >= 100_000:
        return 12.0
    elif avg_vol >= 50_000:
        return 8.0
    elif avg_vol >= 10_000:
        return 4.0
    else:
        return 1.0


def score_position(
    symbol: str,
    pnl_pct: float,
    bars: list,
) -> PositionHeat:
    """Score a single open position.

    symbol: ticker
    pnl_pct: unrealized P&L percentage
    bars: recent bar data (list[Bar])
    """
    ps = _pnl_score(pnl_pct)
    ms = _momentum_score_from_bars(bars)
    ds = _drawdown_score_from_bars(bars)
    ls = _liquidity_score_from_bars(bars)

    total = ps + ms + ds + ls

    if total >= 80:
        status = "hot"
    elif total >= 60:
        status = "warm"
    elif total >= 40:
        status = "neutral"
    elif total >= 20:
        status = "cool"
    else:
        status = "cold"

    return PositionHeat(
        symbol=symbol,
        score=round(total, 1),
        status=status,
        pnl_pct=round(pnl_pct, 2),
        pnl_score=round(ps, 1),
        momentum_score=round(ms, 1),
        drawdown_score=round(ds, 1),
        liquidity_score=round(ls, 1),
        bars_used=len(bars),
    )


def analyze(broker, data, *, lookback: int = 20) -> HeatReport | None:
    """Compute heat scores for all open positions.

    broker: broker interface with get_positions()
    data: data client with get_bars()
    Returns None if no positions.
    """
    try:
        positions = broker.get_positions()
    except Exception:
        return None

    if not positions:
        return None

    heats: list[PositionHeat] = []
    for pos in positions:
        sym = pos.symbol
        try:
            entry = float(pos.avg_entry_price)
            current_mv = float(pos.market_value)
            qty = float(pos.qty)
            if entry > 0 and qty > 0:
                current_price = current_mv / qty
                pnl_pct = (current_price - entry) / entry * 100
            else:
                pnl_pct = 0.0
        except Exception:
            pnl_pct = 0.0

        try:
            bars = data.get_bars(sym, limit=lookback + 5)
        except Exception:
            bars = []

        heats.append(score_position(sym, pnl_pct, bars))

    if not heats:
        return None

    avg = sum(h.score for h in heats) / len(heats)
    hottest = max(heats, key=lambda h: h.score)
    coldest = min(heats, key=lambda h: h.score)
    n_hot = sum(1 for h in heats if h.score >= 60)
    n_cold = sum(1 for h in heats if h.score < 40)

    return HeatReport(
        positions=sorted(heats, key=lambda h: -h.score),
        avg_score=round(avg, 1),
        hottest=hottest,
        coldest=coldest,
        n_hot=n_hot,
        n_cold=n_cold,
    )
