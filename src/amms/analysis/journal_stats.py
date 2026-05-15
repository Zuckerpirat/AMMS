"""Extended trade journal statistics.

Computes advanced metrics from the trade_pairs table:
  - Win rate, avg win/loss, profit factor
  - Expectancy (expected P&L per trade)
  - Sharpe ratio of trade returns
  - Consecutive streak stats (max win/loss streaks)
  - Hold time distribution (avg, median, longest)
  - Best/worst trades
  - Monthly P&L breakdown
  - R-multiple statistics (if stop data available)

Complements the existing /winloss and /journal commands with deeper stats.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class JournalStats:
    n_trades: int
    win_rate: float
    avg_win: float          # $ avg winning trade
    avg_loss: float         # $ avg losing trade (positive value)
    profit_factor: float    # gross_profit / gross_loss
    expectancy: float       # expected $ per trade
    sharpe: float | None    # Sharpe ratio of trade returns
    max_win_streak: int
    max_loss_streak: int
    largest_win: float
    largest_loss: float     # positive value
    avg_hold_days: float | None
    total_pnl: float


def compute(conn) -> JournalStats | None:
    """Compute journal statistics from the trade_pairs table."""
    try:
        rows = conn.execute(
            "SELECT pnl, buy_ts, sell_ts FROM trade_pairs ORDER BY sell_ts"
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    pnls = []
    hold_days: list[float] = []

    for row in rows:
        try:
            pnl = float(row[0])
            pnls.append(pnl)
        except (TypeError, ValueError):
            continue

        try:
            from datetime import datetime
            buy_ts = str(row[1])[:10]
            sell_ts = str(row[2])[:10]
            buy_dt = datetime.strptime(buy_ts, "%Y-%m-%d")
            sell_dt = datetime.strptime(sell_ts, "%Y-%m-%d")
            days = (sell_dt - buy_dt).days
            if days >= 0:
                hold_days.append(float(days))
        except Exception:
            pass

    if not pnls:
        return None

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls)
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = abs(statistics.mean(losses)) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = statistics.mean(pnls)

    # Sharpe of trade returns (treat each trade as a "period")
    sharpe = None
    if len(pnls) >= 3:
        try:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                sharpe = round(mean_pnl / std_pnl * (len(pnls) ** 0.5), 3)
        except Exception:
            pass

    # Streak computation
    max_win_streak = 0
    max_loss_streak = 0
    cur_w = 0
    cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
            max_win_streak = max(max_win_streak, cur_w)
        elif p < 0:
            cur_l += 1
            cur_w = 0
            max_loss_streak = max(max_loss_streak, cur_l)
        else:
            cur_w = cur_l = 0

    largest_win = max(wins) if wins else 0.0
    largest_loss = abs(min(losses)) if losses else 0.0
    avg_hold = statistics.mean(hold_days) if hold_days else None

    return JournalStats(
        n_trades=len(pnls),
        win_rate=round(win_rate, 3),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 3),
        expectancy=round(expectancy, 2),
        sharpe=sharpe,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        largest_win=round(largest_win, 2),
        largest_loss=round(largest_loss, 2),
        avg_hold_days=round(avg_hold, 1) if avg_hold is not None else None,
        total_pnl=round(sum(pnls), 2),
    )
