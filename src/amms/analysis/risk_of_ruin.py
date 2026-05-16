"""Risk of ruin calculator.

Estimates the probability that a trading system wipes out a given
percentage of capital using empirical bootstrapping from trade history.

Approach:
  1. Sample trade PnL% from historical trades (with replacement)
  2. Simulate N equity paths of T trades
  3. Count how many paths cross the ruin threshold
  4. P(ruin) = ruined_paths / total_paths

Also computes:
  - Expected drawdown distribution (median, 95th percentile)
  - Time to ruin (median trades until breach, for ruined paths)

Reads from trade_pairs (pnl, buy_price, qty).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RuinReport:
    ruin_probability: float        # 0-1
    ruin_pct_threshold: float      # what counts as "ruin" (e.g., 30%)
    median_max_drawdown: float     # median of max drawdowns across simulations
    p95_max_drawdown: float        # 95th percentile max drawdown
    expected_trades_to_ruin: int | None  # median trades until ruin (when it occurs)
    n_simulations: int
    n_trades_per_sim: int
    win_rate: float                # empirical from history
    avg_win_pct: float
    avg_loss_pct: float
    n_historical_trades: int
    verdict: str


def compute(
    conn,
    *,
    limit: int = 200,
    ruin_threshold_pct: float = 30.0,
    n_simulations: int = 1000,
    n_trades_per_sim: int = 200,
    seed: int | None = 42,
) -> RuinReport | None:
    """Estimate risk of ruin via bootstrapped Monte Carlo.

    conn: SQLite connection with trade_pairs table.
    ruin_threshold_pct: drawdown that counts as ruin (default 30%).
    n_simulations: number of equity path simulations.
    n_trades_per_sim: trades per simulation path.
    seed: random seed for reproducibility.
    Returns None if fewer than 10 historical trades.
    """
    try:
        rows = conn.execute(
            "SELECT pnl, buy_price, qty "
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
    for pnl, buy_price, qty in rows:
        try:
            bp = float(buy_price)
            qty_f = float(qty)
            pnl_f = float(pnl)
            entry_value = bp * qty_f
            if entry_value > 0:
                pnl_pcts.append(pnl_f / entry_value * 100)
        except Exception:
            continue

    if len(pnl_pcts) < 10:
        return None

    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    win_rate = len(wins) / len(pnl_pcts) * 100
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0  # negative

    rng = random.Random(seed)

    ruined = 0
    trades_to_ruin: list[int] = []
    max_drawdowns: list[float] = []

    threshold_factor = 1.0 - ruin_threshold_pct / 100

    for _ in range(n_simulations):
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        ruin_trade = None

        for t in range(n_trades_per_sim):
            pnl_pct = rng.choice(pnl_pcts)
            equity *= (1 + pnl_pct / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
            if equity <= threshold_factor and ruin_trade is None:
                ruin_trade = t + 1

        max_drawdowns.append(max_dd * 100)
        if ruin_trade is not None:
            ruined += 1
            trades_to_ruin.append(ruin_trade)

    ruin_prob = ruined / n_simulations

    max_drawdowns.sort()
    median_dd = max_drawdowns[len(max_drawdowns) // 2]
    p95_dd = max_drawdowns[int(len(max_drawdowns) * 0.95)]

    expected_ttr: int | None = None
    if trades_to_ruin:
        trades_to_ruin.sort()
        expected_ttr = trades_to_ruin[len(trades_to_ruin) // 2]

    risk_level = (
        "CRITICAL" if ruin_prob >= 0.20 else
        "HIGH" if ruin_prob >= 0.10 else
        "MODERATE" if ruin_prob >= 0.05 else
        "LOW"
    )

    verdict = (
        f"Risk of ruin ({ruin_threshold_pct:.0f}% drawdown): "
        f"{ruin_prob * 100:.1f}%  [{risk_level}]. "
        f"Median max drawdown: {median_dd:.1f}%, "
        f"95th pct: {p95_dd:.1f}%. "
    )
    if expected_ttr:
        verdict += f"Median trades to ruin (when it occurs): {expected_ttr}."

    return RuinReport(
        ruin_probability=round(ruin_prob, 4),
        ruin_pct_threshold=ruin_threshold_pct,
        median_max_drawdown=round(median_dd, 2),
        p95_max_drawdown=round(p95_dd, 2),
        expected_trades_to_ruin=expected_ttr,
        n_simulations=n_simulations,
        n_trades_per_sim=n_trades_per_sim,
        win_rate=round(win_rate, 1),
        avg_win_pct=round(avg_win, 3),
        avg_loss_pct=round(avg_loss, 3),
        n_historical_trades=len(pnl_pcts),
        verdict=verdict,
    )
