"""Trade quality scoring.

Evaluates historical trades from the trade_pairs table and assigns
a quality score to each based on multiple dimensions:

1. Outcome score (0-30 pts):
   - Win: +30 (profitable)
   - Loss ≤ 2% (small, controlled loss): +15
   - Loss 2-5%: +5
   - Loss > 5%: 0

2. Size discipline (0-20 pts):
   - Position < 5% of equity at entry: +20
   - 5-10%: +15
   - 10-20%: +10
   - >20% (oversized): 0

3. Hold time alignment (0-20 pts):
   - Hold < 1 day (very short): +5
   - 1-5 days (swing): +20
   - 5-30 days (medium): +15
   - >30 days: +10

4. Risk-reward realized (0-30 pts):
   - Win/loss ratio vs expected: compares actual P&L to initial risk
   - RR >= 2:1: +30
   - RR 1:1 to 2:1: +20
   - RR 0.5:1 to 1:1: +10
   - RR < 0.5:1 or loss: scales

Overall score: 0-100 (higher = better trade quality)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TradeQualityScore:
    trade_id: int | None
    symbol: str
    pnl: float
    pnl_pct: float
    hold_days: float | None
    outcome_score: float      # 0-30
    hold_score: float         # 0-20
    rr_score: float           # 0-30
    total_score: float        # 0-80 (no size scoring without equity data)
    grade: str                # "A" | "B" | "C" | "D" | "F"
    notes: list[str]


@dataclass(frozen=True)
class TradeQualityReport:
    n_trades: int
    avg_score: float
    grade_distribution: dict[str, int]
    best_trade: TradeQualityScore | None
    worst_quality_trade: TradeQualityScore | None
    scores: list[TradeQualityScore]


def compute_quality(conn, limit: int = 50) -> TradeQualityReport | None:
    """Compute trade quality scores from the trade_pairs table.

    conn: SQLite connection
    limit: max number of recent trades to analyze
    Returns None if no trades found.
    """
    try:
        rows = conn.execute(
            "SELECT id, symbol, buy_ts, sell_ts, buy_price, sell_price, qty, pnl "
            "FROM trade_pairs "
            "WHERE sell_price IS NOT NULL AND pnl IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    scores: list[TradeQualityScore] = []
    for row in rows:
        trade_id, sym, buy_ts, sell_ts, buy_price, sell_price, qty, pnl = row
        trade_id = int(trade_id) if trade_id is not None else None
        buy_price = float(buy_price) if buy_price else 0.0
        sell_price = float(sell_price) if sell_price else 0.0
        qty = float(qty) if qty else 1.0
        pnl = float(pnl) if pnl else 0.0

        # PnL %
        entry_value = buy_price * qty
        pnl_pct = pnl / entry_value * 100 if entry_value > 0 else 0.0

        # Hold time
        hold_days = None
        if buy_ts and sell_ts:
            try:
                buy_dt = datetime.fromisoformat(str(buy_ts)[:19])
                sell_dt = datetime.fromisoformat(str(sell_ts)[:19])
                delta = sell_dt - buy_dt
                hold_days = delta.total_seconds() / 86400
            except Exception:
                pass

        notes: list[str] = []

        # 1. Outcome score (0-30)
        if pnl > 0:
            outcome = 30.0
            notes.append(f"Win: +${pnl:.2f}")
        elif abs(pnl_pct) <= 2.0:
            outcome = 15.0
            notes.append(f"Small controlled loss: {pnl_pct:.1f}%%")
        elif abs(pnl_pct) <= 5.0:
            outcome = 5.0
            notes.append(f"Moderate loss: {pnl_pct:.1f}%%")
        else:
            outcome = 0.0
            notes.append(f"Large loss: {pnl_pct:.1f}%% — review stop")

        # 2. Hold time score (0-20)
        if hold_days is None:
            hold_score = 10.0  # neutral when unknown
        elif hold_days < 1:
            hold_score = 5.0
            notes.append("Very short hold (<1 day)")
        elif hold_days <= 5:
            hold_score = 20.0
        elif hold_days <= 30:
            hold_score = 15.0
        else:
            hold_score = 10.0
            notes.append(f"Long hold: {hold_days:.0f} days")

        # 3. Risk-reward realized (0-30)
        # Use the move as a ratio: gain / initial_risk_assumed (1% default)
        assumed_risk = entry_value * 0.01  # assume 1% initial stop
        if assumed_risk > 0:
            rr = pnl / assumed_risk
            if rr >= 2.0:
                rr_score = 30.0
            elif rr >= 1.0:
                rr_score = 20.0
            elif rr >= 0.5:
                rr_score = 10.0
            elif rr >= 0:
                rr_score = 5.0
            else:
                rr_score = max(0.0, 5.0 + rr * 5.0)  # penalty for losses
        else:
            rr_score = 10.0

        total = outcome + hold_score + rr_score

        if total >= 70:
            grade = "A"
        elif total >= 55:
            grade = "B"
        elif total >= 40:
            grade = "C"
        elif total >= 25:
            grade = "D"
        else:
            grade = "F"

        scores.append(TradeQualityScore(
            trade_id=trade_id,
            symbol=sym,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            hold_days=round(hold_days, 1) if hold_days is not None else None,
            outcome_score=round(outcome, 1),
            hold_score=round(hold_score, 1),
            rr_score=round(rr_score, 1),
            total_score=round(total, 1),
            grade=grade,
            notes=notes,
        ))

    if not scores:
        return None

    avg_score = sum(s.total_score for s in scores) / len(scores)
    grade_dist: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for s in scores:
        grade_dist[s.grade] = grade_dist.get(s.grade, 0) + 1

    best = max(scores, key=lambda s: s.total_score)
    worst = min(scores, key=lambda s: s.total_score)

    return TradeQualityReport(
        n_trades=len(scores),
        avg_score=round(avg_score, 1),
        grade_distribution=grade_dist,
        best_trade=best,
        worst_quality_trade=worst,
        scores=scores,
    )
