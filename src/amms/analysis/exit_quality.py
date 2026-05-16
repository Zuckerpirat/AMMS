"""Trade Exit Quality Analyzer.

Measures how well trade exits were timed by comparing the actual exit PnL%
with what was achievable:

  Entry Efficiency = (exit_pnl - max_possible_loss) / (max_possible_gain - max_possible_loss)
    = how far along the full theoretical range did the trader exit?
    1.0 = perfect exit (sold at the exact top for longs)
    0.5 = middle of the range
    0.0 = exited at the theoretical floor (worst possible)

Since we typically don't have intraday post-exit data from just trade records,
we approximate using the pnl_pct distribution:

  Exit Quality = pnl_pct / (avg_win × 2)  for winners
               = 1 - |pnl_pct| / (avg_loss × 2)  for losers

This gives a relative quality score where:
  - Consistent high exit quality: disciplined profit-taking
  - Variable exit quality: inconsistent; some exits too early, some too late
  - Negative exit quality: losing more than average, exits too late

Also classifies each trade as:
  "optimal"  : pnl > avg_win × 0.8 (winner) or |pnl| < avg_loss × 0.5 (small loser)
  "too_early": winner closed below 50% of avg win (left money on table)
  "late_exit": loser held beyond avg loss level (let losses run)
  "normal"   : neither extreme

Reports aggregate statistics: % optimal, % too_early, % late_exit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitGrade:
    pnl_pct: float
    classification: str    # "optimal" / "too_early" / "late_exit" / "normal"
    quality_score: float   # 0-1 (higher = better exit)


@dataclass(frozen=True)
class ExitQualityReport:
    avg_exit_quality: float           # mean quality score 0-1
    optimal_pct: float                # % classified as optimal
    too_early_pct: float              # % winners closed below avg win
    late_exit_pct: float              # % losers held past avg loss
    normal_pct: float
    avg_winner_pct: float             # avg winning trade PnL%
    avg_loser_pct: float              # avg losing trade PnL% (negative)
    exit_consistency: float           # 1 - (std(quality) / mean(quality)); higher = consistent
    best_exit_pnl: float
    worst_exit_pnl: float
    n_winners: int
    n_losers: int
    n_trades: int
    verdict: str


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mu = sum(values) / n
    return (sum((v - mu) ** 2 for v in values) / (n - 1)) ** 0.5


def compute(conn, *, limit: int = 500) -> ExitQualityReport | None:
    """Analyse exit quality from closed trade history.

    conn: SQLite connection.
    limit: max trades to analyse.
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct
            FROM trades
            WHERE status = 'closed' AND pnl_pct IS NOT NULL
            ORDER BY closed_at DESC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 5:
        return None

    pnls = [float(r[0]) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    if not wins or not losses:
        return None

    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)  # negative

    grades: list[ExitGrade] = []
    for pnl in pnls:
        if pnl > 0:
            # Winner quality: how close to 2× avg win?
            quality = min(1.0, pnl / (avg_win * 2)) if avg_win > 0 else 0.5
            if pnl >= avg_win * 0.8:
                cls = "optimal"
            elif pnl < avg_win * 0.5:
                cls = "too_early"
            else:
                cls = "normal"
        else:
            # Loser quality: how small relative to avg loss?
            quality = max(0.0, 1.0 + pnl / (avg_loss * 2)) if avg_loss < 0 else 0.5
            if abs(pnl) < abs(avg_loss) * 0.5:
                cls = "optimal"
            elif abs(pnl) > abs(avg_loss) * 1.5:
                cls = "late_exit"
            else:
                cls = "normal"

        grades.append(ExitGrade(pnl_pct=pnl, classification=cls, quality_score=round(quality, 4)))

    n = len(grades)
    qualities = [g.quality_score for g in grades]
    avg_q = sum(qualities) / n

    # Consistency: 1 - CV (coefficient of variation)
    q_std = _std(qualities)
    consistency = max(0.0, 1.0 - (q_std / avg_q)) if avg_q > 0 else 0.0

    counts = {c: sum(1 for g in grades if g.classification == c) for c in ("optimal", "too_early", "late_exit", "normal")}

    opt_pct = counts["optimal"] / n * 100
    early_pct = counts["too_early"] / n * 100
    late_pct = counts["late_exit"] / n * 100
    norm_pct = counts["normal"] / n * 100

    # Verdict
    issues = []
    if early_pct > 30:
        issues.append(f"cutting winners too early ({early_pct:.0f}% of trades below avg win)")
    if late_pct > 20:
        issues.append(f"letting losers run ({late_pct:.0f}% exceeded avg loss)")
    if consistency < 0.5:
        issues.append("inconsistent exits (high variance in exit quality)")

    if issues:
        verdict = f"Exit quality {avg_q:.2f}/1.0 — issues: " + "; ".join(issues) + "."
    else:
        verdict = (
            f"Exit quality {avg_q:.2f}/1.0 — {opt_pct:.0f}% optimal exits. "
            f"Consistency: {consistency:.2f}."
        )

    return ExitQualityReport(
        avg_exit_quality=round(avg_q, 3),
        optimal_pct=round(opt_pct, 1),
        too_early_pct=round(early_pct, 1),
        late_exit_pct=round(late_pct, 1),
        normal_pct=round(norm_pct, 1),
        avg_winner_pct=round(avg_win, 3),
        avg_loser_pct=round(avg_loss, 3),
        exit_consistency=round(consistency, 3),
        best_exit_pnl=round(max(pnls), 3),
        worst_exit_pnl=round(min(pnls), 3),
        n_winners=len(wins),
        n_losers=len(losses),
        n_trades=n,
        verdict=verdict,
    )
