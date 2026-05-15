"""Trade streak analysis.

Tracks current and historical win/loss streaks in closed trades.
Identifies hot/cold streaks and momentum of recent form.

Metrics:
  - current_streak: current consecutive wins (+) or losses (-)
  - longest_win_streak / longest_loss_streak: all-time bests
  - recent_form: win rate over last N trades
  - momentum: trend of win rate (improving/declining/stable)
  - hot_hand: True if currently on a significant win streak
  - tilt_risk: True if on a significant loss streak (emotional risk)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreakResult:
    current_streak: int           # positive = win streak, negative = loss streak
    current_streak_label: str     # "3W" or "5L"
    longest_win_streak: int
    longest_loss_streak: int
    recent_form: float            # win rate over last 10 trades (0-100)
    recent_form_label: str        # "hot"/"warm"/"cold"/"icy"
    momentum: str                 # "improving"/"declining"/"stable"
    hot_hand: bool                # on a notable win streak (≥3)
    tilt_risk: bool               # on a notable loss streak (≥3)
    n_trades: int
    overall_win_rate: float
    verdict: str


def compute(conn, *, limit: int = 100) -> StreakResult | None:
    """Analyze trade streaks from trade_pairs.

    conn: SQLite connection with trade_pairs table
    limit: max trades to analyze (most recent first)

    Returns None if < 5 trades.
    """
    try:
        rows = conn.execute(
            "SELECT pnl FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND sell_price IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 5:
        return None

    # Chronological order (oldest first)
    outcomes = [1 if float(r[0]) > 0 else 0 for r in reversed(rows)]
    n = len(outcomes)

    # Current streak (from most recent)
    recent_outcomes = list(reversed(outcomes))
    current = 0
    first = recent_outcomes[0]
    for o in recent_outcomes:
        if o == first:
            current += 1
        else:
            break
    current_streak = current if first == 1 else -current
    current_streak_label = f"{current}W" if first == 1 else f"{current}L"

    # Longest win/loss streaks
    longest_win = 0
    longest_loss = 0
    run = 1
    for i in range(1, n):
        if outcomes[i] == outcomes[i - 1]:
            run += 1
        else:
            if outcomes[i - 1] == 1:
                longest_win = max(longest_win, run)
            else:
                longest_loss = max(longest_loss, run)
            run = 1
    # Last run
    if outcomes[-1] == 1:
        longest_win = max(longest_win, run)
    else:
        longest_loss = max(longest_loss, run)

    # Recent form (last 10 trades)
    recent_window = min(10, n)
    recent = outcomes[-recent_window:]
    recent_form = sum(recent) / recent_window * 100

    if recent_form >= 80:
        recent_form_label = "hot"
    elif recent_form >= 60:
        recent_form_label = "warm"
    elif recent_form >= 40:
        recent_form_label = "cool"
    else:
        recent_form_label = "icy"

    # Momentum: compare first half vs second half of recent window
    if recent_window >= 6:
        half = recent_window // 2
        early_wr = sum(outcomes[-(recent_window):-half]) / half
        late_wr = sum(outcomes[-half:]) / half
        diff = late_wr - early_wr
        if diff > 0.15:
            momentum = "improving"
        elif diff < -0.15:
            momentum = "declining"
        else:
            momentum = "stable"
    else:
        momentum = "stable"

    hot_hand = current_streak >= 3
    tilt_risk = current_streak <= -3

    overall_wr = sum(outcomes) / n * 100

    # Verdict
    if tilt_risk:
        verdict = f"Loss streak of {abs(current_streak)} — consider pausing and reviewing"
    elif hot_hand:
        verdict = f"Win streak of {current_streak} — stay disciplined, avoid overtrading"
    elif momentum == "improving":
        verdict = "Recent form improving — strategy gaining traction"
    elif momentum == "declining":
        verdict = "Recent form declining — review recent trades for patterns"
    else:
        verdict = "Consistent form — no unusual streak behavior"

    return StreakResult(
        current_streak=current_streak,
        current_streak_label=current_streak_label,
        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,
        recent_form=round(recent_form, 1),
        recent_form_label=recent_form_label,
        momentum=momentum,
        hot_hand=hot_hand,
        tilt_risk=tilt_risk,
        n_trades=n,
        overall_win_rate=round(overall_wr, 1),
        verdict=verdict,
    )
