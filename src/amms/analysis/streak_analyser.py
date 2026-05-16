"""Streak Analyser.

Analyses consecutive win/loss streaks in the closed trade journal.

Metrics:
  - Longest winning streak and its total PnL%
  - Longest losing streak and its total PnL%
  - Current streak (type + length)
  - Streak length distribution (how often streaks of length N occur)
  - Average streak lengths (win / loss separately)
  - Post-streak performance: what happens after a streak ends?
  - Hot-hand probability: P(win | prev win) vs baseline win rate
  - Cold-hand probability: P(loss | prev loss) vs baseline loss rate
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreakRun:
    kind: str           # "win" or "loss"
    length: int
    total_pnl: float
    avg_pnl: float
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class StreakAnalyserReport:
    longest_win_streak: StreakRun | None
    longest_loss_streak: StreakRun | None
    current_streak: StreakRun | None
    all_streaks: list[StreakRun]

    n_win_streaks: int
    n_loss_streaks: int
    avg_win_streak_len: float
    avg_loss_streak_len: float

    # Conditional probabilities
    p_win_after_win: float | None     # P(win | prev win)
    p_loss_after_loss: float | None   # P(loss | prev loss)
    baseline_win_rate: float          # overall P(win)

    hot_hand_effect: float | None     # p_win_after_win - baseline; positive = hot hand
    cold_hand_effect: float | None    # p_loss_after_loss - (1 - baseline); positive = cold hand

    # Post-streak: avg PnL of the trade immediately after a streak ends
    avg_pnl_after_win_streak: float | None
    avg_pnl_after_loss_streak: float | None

    n_trades: int
    verdict: str


def _build_streaks(outcomes: list[tuple[str, float]]) -> list[StreakRun]:
    """Build list of StreakRun from (kind, pnl) sequence."""
    if not outcomes:
        return []
    streaks: list[StreakRun] = []
    cur_kind, cur_pnl = outcomes[0]
    cur_start = 0
    cur_pnls = [cur_pnl]

    for i, (kind, pnl) in enumerate(outcomes[1:], start=1):
        if kind == cur_kind:
            cur_pnls.append(pnl)
        else:
            streaks.append(StreakRun(
                kind=cur_kind,
                length=len(cur_pnls),
                total_pnl=round(sum(cur_pnls), 4),
                avg_pnl=round(sum(cur_pnls) / len(cur_pnls), 4),
                start_idx=cur_start,
                end_idx=i - 1,
            ))
            cur_kind = kind
            cur_pnl = pnl
            cur_start = i
            cur_pnls = [pnl]

    streaks.append(StreakRun(
        kind=cur_kind,
        length=len(cur_pnls),
        total_pnl=round(sum(cur_pnls), 4),
        avg_pnl=round(sum(cur_pnls) / len(cur_pnls), 4),
        start_idx=cur_start,
        end_idx=len(outcomes) - 1,
    ))
    return streaks


def compute(conn, *, limit: int = 500) -> StreakAnalyserReport | None:
    """Analyse win/loss streaks from closed trade history.

    conn: SQLite connection.
    limit: max trades to analyse (most recent first).
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct, closed_at
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
            ORDER BY closed_at ASC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 10:
        return None

    pnls = [float(r[0]) for r in rows]
    outcomes: list[tuple[str, float]] = [("win" if p > 0 else "loss", p) for p in pnls]

    n = len(outcomes)
    n_wins = sum(1 for k, _ in outcomes if k == "win")
    baseline_wr = n_wins / n

    streaks = _build_streaks(outcomes)
    win_streaks = [s for s in streaks if s.kind == "win"]
    loss_streaks = [s for s in streaks if s.kind == "loss"]

    longest_win = max(win_streaks, key=lambda s: s.length) if win_streaks else None
    longest_loss = max(loss_streaks, key=lambda s: s.length) if loss_streaks else None
    current = streaks[-1] if streaks else None

    avg_win_len = (sum(s.length for s in win_streaks) / len(win_streaks)) if win_streaks else 0.0
    avg_loss_len = (sum(s.length for s in loss_streaks) / len(loss_streaks)) if loss_streaks else 0.0

    # Conditional probabilities
    win_after_win = sum(
        1 for i in range(1, n)
        if outcomes[i - 1][0] == "win" and outcomes[i][0] == "win"
    )
    win_after_win_total = sum(1 for i in range(1, n) if outcomes[i - 1][0] == "win")
    p_waw = (win_after_win / win_after_win_total) if win_after_win_total > 0 else None

    loss_after_loss = sum(
        1 for i in range(1, n)
        if outcomes[i - 1][0] == "loss" and outcomes[i][0] == "loss"
    )
    loss_after_loss_total = sum(1 for i in range(1, n) if outcomes[i - 1][0] == "loss")
    p_lal = (loss_after_loss / loss_after_loss_total) if loss_after_loss_total > 0 else None

    hot_hand = (p_waw - baseline_wr) if p_waw is not None else None
    cold_hand = (p_lal - (1.0 - baseline_wr)) if p_lal is not None else None

    # Post-streak PnL: trade immediately after a streak ends
    post_win_pnls: list[float] = []
    post_loss_pnls: list[float] = []
    for s in streaks[:-1]:  # all but last (last streak may be ongoing)
        next_idx = s.end_idx + 1
        if next_idx < n:
            if s.kind == "win":
                post_win_pnls.append(outcomes[next_idx][1])
            else:
                post_loss_pnls.append(outcomes[next_idx][1])

    avg_post_win = (sum(post_win_pnls) / len(post_win_pnls)) if post_win_pnls else None
    avg_post_loss = (sum(post_loss_pnls) / len(post_loss_pnls)) if post_loss_pnls else None

    # Verdict
    parts = []
    if longest_win:
        parts.append(f"longest win streak {longest_win.length} ({longest_win.total_pnl:+.1f}%)")
    if longest_loss:
        parts.append(f"longest loss streak {longest_loss.length} ({longest_loss.total_pnl:+.1f}%)")
    if hot_hand is not None and abs(hot_hand) > 0.05:
        direction = "hot-hand bias" if hot_hand > 0 else "mean-reversion bias"
        parts.append(f"{direction} {hot_hand:+.2f} vs baseline")
    if current:
        parts.append(f"currently on {current.length}-{current.kind} streak")

    verdict = "Streak analysis: " + ("; ".join(parts) if parts else "no notable streak patterns") + "."

    return StreakAnalyserReport(
        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,
        current_streak=current,
        all_streaks=streaks,
        n_win_streaks=len(win_streaks),
        n_loss_streaks=len(loss_streaks),
        avg_win_streak_len=round(avg_win_len, 2),
        avg_loss_streak_len=round(avg_loss_len, 2),
        p_win_after_win=round(p_waw, 4) if p_waw is not None else None,
        p_loss_after_loss=round(p_lal, 4) if p_lal is not None else None,
        baseline_win_rate=round(baseline_wr, 4),
        hot_hand_effect=round(hot_hand, 4) if hot_hand is not None else None,
        cold_hand_effect=round(cold_hand, 4) if cold_hand is not None else None,
        avg_pnl_after_win_streak=round(avg_post_win, 4) if avg_post_win is not None else None,
        avg_pnl_after_loss_streak=round(avg_post_loss, 4) if avg_post_loss is not None else None,
        n_trades=n,
        verdict=verdict,
    )
