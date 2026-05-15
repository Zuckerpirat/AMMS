"""Portfolio concentration risk analysis.

Measures how concentrated a portfolio is using:
  1. Herfindahl-Hirschman Index (HHI): sum of squared weights
     - < 0.10: well diversified
     - 0.10-0.25: moderate concentration
     - > 0.25: highly concentrated
  2. Top-N weight: how much of the portfolio the top 1/3/5 positions represent
  3. Effective N: 1 / HHI — the "effective number of equal positions"
  4. Max single position %: largest single holding
  5. Concentration grade: A-F

All weights based on market value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionWeight:
    symbol: str
    market_value: float
    weight_pct: float     # % of total portfolio


@dataclass(frozen=True)
class ConcentrationReport:
    positions: list[PositionWeight]    # sorted by weight desc
    total_value: float
    hhi: float                          # 0..1 (lower = more diversified)
    effective_n: float                  # 1/HHI (higher = more diversified)
    top1_pct: float                     # weight of largest position
    top3_pct: float                     # weight of top 3
    top5_pct: float                     # weight of top 5
    max_position_pct: float             # same as top1_pct
    n_positions: int
    grade: str                          # A-F
    verdict: str                        # plain-language assessment
    risk_flags: list[str]               # specific warnings


def analyze(broker) -> ConcentrationReport | None:
    """Compute concentration risk for the current portfolio.

    broker: broker interface with get_positions() returning objects
            with .symbol, .market_value attributes

    Returns None if no positions or total_value is 0.
    """
    try:
        positions = broker.get_positions()
    except Exception:
        return None

    if not positions:
        return None

    # Gather market values
    raw: list[tuple[str, float]] = []
    for pos in positions:
        try:
            mv = float(pos.market_value)
            if mv > 0:
                raw.append((pos.symbol, mv))
        except Exception:
            pass

    if not raw:
        return None

    total = sum(mv for _, mv in raw)
    if total <= 0:
        return None

    # Sort by market value descending
    raw.sort(key=lambda x: -x[1])

    pw = [
        PositionWeight(
            symbol=sym,
            market_value=round(mv, 2),
            weight_pct=round(mv / total * 100, 2),
        )
        for sym, mv in raw
    ]

    n = len(pw)
    weights = [p.weight_pct / 100 for p in pw]

    # HHI = sum of squared weights
    hhi = sum(w ** 2 for w in weights)
    effective_n = 1 / hhi if hhi > 0 else float("inf")

    top1 = pw[0].weight_pct if n >= 1 else 0.0
    top3 = sum(p.weight_pct for p in pw[:3])
    top5 = sum(p.weight_pct for p in pw[:5])

    # Grade based on HHI
    if hhi < 0.05:
        grade = "A"
    elif hhi < 0.10:
        grade = "B"
    elif hhi < 0.18:
        grade = "C"
    elif hhi < 0.30:
        grade = "D"
    else:
        grade = "F"

    # Verdict
    if hhi < 0.05:
        verdict = "Well diversified"
    elif hhi < 0.10:
        verdict = "Moderately diversified"
    elif hhi < 0.18:
        verdict = "Moderate concentration"
    elif hhi < 0.30:
        verdict = "High concentration"
    else:
        verdict = "Extreme concentration — significant single-name risk"

    # Risk flags
    flags: list[str] = []
    if top1 > 30:
        flags.append(f"Largest position {pw[0].symbol} is {top1:.1f}%% of portfolio")
    if top3 > 60:
        flags.append(f"Top 3 positions account for {top3:.1f}%% of portfolio")
    if n < 5:
        flags.append(f"Only {n} positions — very concentrated by count")
    if hhi > 0.25:
        flags.append(f"HHI {hhi:.3f} indicates significant concentration risk")

    return ConcentrationReport(
        positions=pw,
        total_value=round(total, 2),
        hhi=round(hhi, 4),
        effective_n=round(effective_n, 1),
        top1_pct=round(top1, 2),
        top3_pct=round(top3, 2),
        top5_pct=round(top5, 2),
        max_position_pct=round(top1, 2),
        n_positions=n,
        grade=grade,
        verdict=verdict,
        risk_flags=flags,
    )
