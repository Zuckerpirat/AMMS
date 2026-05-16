"""Trading score card.

Aggregates key performance metrics from trade_pairs into a single
weighted score (0-100) and letter grade (A-F).

Metrics evaluated:
  1. Win rate           (target ≥55% = full score)
  2. Profit factor      (target ≥1.5 = full score)
  3. Avg trade return   (target ≥0.5% = full score)
  4. Consistency        (% profitable months, target ≥60% = full score)
  5. Risk-adjusted      (Sharpe-like: avg_pnl_pct / vol of returns)
  6. Expectancy         (positive = full score)
  7. Losing streak      (current_streak ≤ 3 = full score)

Each metric is scored 0-100, weighted, then combined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MetricScore:
    name: str
    value: str          # human-readable value
    score: float        # 0-100
    weight: float       # contribution weight
    grade: str          # A/B/C/D/F for this metric


@dataclass(frozen=True)
class ScorecardReport:
    overall_score: float         # 0-100 weighted
    grade: str                   # A/B/C/D/F
    metrics: list[MetricScore]
    n_trades: int
    verdict: str


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _score_win_rate(wr: float) -> float:
    """55% → 100, 50% → 70, 40% → 30, <35% → 0."""
    if wr >= 55:
        return 100.0
    if wr >= 50:
        return 70 + (wr - 50) / 5 * 30
    if wr >= 40:
        return 30 + (wr - 40) / 10 * 40
    if wr >= 35:
        return (wr - 35) / 5 * 30
    return 0.0


def _score_profit_factor(pf: float | None) -> float:
    if pf is None:
        return 50.0
    if pf >= 1.5:
        return 100.0
    if pf >= 1.2:
        return 70 + (pf - 1.2) / 0.3 * 30
    if pf >= 1.0:
        return 40 + (pf - 1.0) / 0.2 * 30
    return max(0.0, pf / 1.0 * 40)


def _score_avg_return(avg_pct: float) -> float:
    if avg_pct >= 0.5:
        return 100.0
    if avg_pct >= 0.2:
        return 60 + (avg_pct - 0.2) / 0.3 * 40
    if avg_pct >= 0:
        return 30 + avg_pct / 0.2 * 30
    return max(0.0, 30 + avg_pct * 10)


def _score_consistency(pct: float) -> float:
    if pct >= 60:
        return 100.0
    if pct >= 50:
        return 70 + (pct - 50) / 10 * 30
    if pct >= 40:
        return 40 + (pct - 40) / 10 * 30
    return max(0.0, pct / 40 * 40)


def _score_sharpe_like(avg: float, vol: float) -> float:
    if vol == 0:
        return 50.0
    ratio = avg / vol
    if ratio >= 0.3:
        return 100.0
    if ratio >= 0.1:
        return 60 + (ratio - 0.1) / 0.2 * 40
    if ratio >= 0:
        return 30 + ratio / 0.1 * 30
    return max(0.0, 30 + ratio * 50)


def _score_expectancy(exp: float) -> float:
    if exp >= 0.5:
        return 100.0
    if exp >= 0.2:
        return 70 + (exp - 0.2) / 0.3 * 30
    if exp >= 0:
        return 40 + exp / 0.2 * 30
    return max(0.0, 40 + exp * 20)


def _score_streak(current_loss_streak: int) -> float:
    if current_loss_streak <= 1:
        return 100.0
    if current_loss_streak <= 3:
        return 80 - (current_loss_streak - 1) * 15
    if current_loss_streak <= 6:
        return 40 - (current_loss_streak - 3) * 10
    return 0.0


def compute(conn, *, limit: int = 500) -> ScorecardReport | None:
    """Build trading score card from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 10 usable trades.
    """
    try:
        rows = conn.execute(
            "SELECT sell_ts, pnl, buy_price, qty "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND buy_price IS NOT NULL AND qty IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 10:
        return None

    pnl_pcts: list[float] = []
    pnls_abs: list[float] = []
    sell_dates: list[str] = []

    for sell_ts, pnl, buy_price, qty in rows:
        try:
            bp = float(buy_price)
            qty_f = float(qty)
            pnl_f = float(pnl)
            entry_value = bp * qty_f
            pnl_pct = pnl_f / entry_value * 100 if entry_value > 0 else 0.0
            pnl_pcts.append(pnl_pct)
            pnls_abs.append(pnl_f)
            sell_dates.append(str(sell_ts)[:7])  # "YYYY-MM"
        except Exception:
            continue

    if len(pnl_pcts) < 10:
        return None

    n = len(pnl_pcts)
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    win_rate = len(wins) / n * 100
    avg_return = sum(pnl_pcts) / n
    vol = math.sqrt(sum((p - avg_return) ** 2 for p in pnl_pcts) / n) if n > 1 else 0.0

    # Profit factor (on absolute PnL)
    wins_abs = [p for p in pnls_abs if p > 0]
    losses_abs = [p for p in pnls_abs if p < 0]
    pf = sum(wins_abs) / abs(sum(losses_abs)) if losses_abs and sum(losses_abs) != 0 else None

    # Expectancy
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

    # Consistency (% profitable months)
    month_pnls: dict[str, float] = {}
    for i, ym in enumerate(sell_dates):
        month_pnls[ym] = month_pnls.get(ym, 0.0) + pnl_pcts[i]
    consistent = sum(1 for v in month_pnls.values() if v > 0)
    consistency = consistent / len(month_pnls) * 100 if month_pnls else 50.0

    # Current losing streak
    current_loss_streak = 0
    for p in pnl_pcts:
        if p <= 0:
            current_loss_streak += 1
        else:
            break

    pf_str = f"{pf:.2f}" if pf is not None else "n/a"

    WEIGHTS = [
        ("Win Rate", f"{win_rate:.1f}%", _score_win_rate(win_rate), 0.20),
        ("Profit Factor", pf_str, _score_profit_factor(pf), 0.20),
        ("Avg Return %", f"{avg_return:+.2f}%", _score_avg_return(avg_return), 0.15),
        ("Consistency", f"{consistency:.0f}% months +", _score_consistency(consistency), 0.15),
        ("Risk-Adj Return", f"ratio {avg_return / vol:.2f}" if vol > 0 else "n/a", _score_sharpe_like(avg_return, vol), 0.15),
        ("Expectancy", f"{expectancy:+.3f}%/trade", _score_expectancy(expectancy), 0.10),
        ("Streak Risk", f"{current_loss_streak} current losses", _score_streak(current_loss_streak), 0.05),
    ]

    metrics = [
        MetricScore(name=name, value=val, score=round(sc, 1), weight=w, grade=_grade(sc))
        for name, val, sc, w in WEIGHTS
    ]

    overall = sum(m.score * m.weight for m in metrics)
    grade = _grade(overall)

    grade_text = {
        "A": "Excellent — strong, consistent, profitable trading",
        "B": "Good — above average with room to improve",
        "C": "Average — profitable but inconsistent or low edge",
        "D": "Below average — needs improvement in key areas",
        "F": "Poor — significant issues require immediate attention",
    }.get(grade, "")

    verdict = f"Grade {grade} ({overall:.0f}/100). {grade_text}."

    return ScorecardReport(
        overall_score=round(overall, 1),
        grade=grade,
        metrics=metrics,
        n_trades=n,
        verdict=verdict,
    )
