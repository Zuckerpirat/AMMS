"""Kelly Criterion position sizing.

Computes optimal position size based on historical win rate and
average payoff ratio from closed trades.

Kelly formula:
  f* = (W * R - (1 - W)) / R
  where W = win_rate, R = avg_win / avg_loss

Fractional Kelly (0.25× and 0.5×) are also returned to reduce
variance for real-world trading.

The output includes:
  - kelly_pct: full Kelly fraction (% of capital)
  - half_kelly_pct: 0.5× Kelly (recommended for live trading)
  - quarter_kelly_pct: 0.25× Kelly (conservative)
  - suggested_shares: given portfolio_value and current_price
  - win_rate, avg_win, avg_loss, payoff_ratio, n_trades used
  - edge: expected value per dollar risked (Kelly numerator / 1)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KellyResult:
    win_rate: float           # 0-100
    avg_win: float            # average winning PnL ($)
    avg_loss: float           # average losing PnL (positive value, $)
    payoff_ratio: float       # avg_win / avg_loss
    edge: float               # expected value per unit risked
    kelly_pct: float          # full Kelly % of capital (capped at 25%)
    half_kelly_pct: float     # 0.5 × kelly_pct
    quarter_kelly_pct: float  # 0.25 × kelly_pct
    n_trades: int
    n_wins: int
    n_losses: int
    suggested_shares: int | None    # None if price not provided
    suggested_value: float | None   # dollar amount to allocate
    grade: str                      # A/B/C/D/F by edge quality
    note: str


def compute(
    conn,
    *,
    limit: int = 100,
    portfolio_value: float | None = None,
    current_price: float | None = None,
) -> KellyResult | None:
    """Compute Kelly position size from trade_pairs history.

    conn: SQLite connection with trade_pairs table
    limit: max trades to analyze (most recent)
    portfolio_value: total portfolio value in $ (for share calc)
    current_price: price per share (for share calc)

    Returns None if insufficient data (< 5 trades or no losses).
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

    pnls = [float(r[0]) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    if not wins or not losses:
        return None

    n = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = n_wins / n          # fraction 0-1
    avg_win = sum(wins) / n_wins
    avg_loss = sum(losses) / n_losses
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Kelly formula: f* = (W*R - (1-W)) / R
    kelly_raw = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio if payoff_ratio > 0 else 0.0
    kelly_raw = max(0.0, kelly_raw)

    # Cap at 25% to protect against over-sizing
    kelly_pct = min(kelly_raw * 100, 25.0)
    half_kelly_pct = kelly_pct * 0.5
    quarter_kelly_pct = kelly_pct * 0.25

    # Edge = expected value per dollar risked
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Suggested allocation
    suggested_shares = None
    suggested_value = None
    if portfolio_value and portfolio_value > 0 and kelly_pct > 0:
        suggested_value = portfolio_value * (half_kelly_pct / 100)
        if current_price and current_price > 0:
            suggested_shares = max(0, int(suggested_value / current_price))

    # Grade by edge quality relative to avg_loss
    edge_pct = edge / avg_loss * 100 if avg_loss > 0 else 0.0
    if edge_pct >= 30:
        grade = "A"
    elif edge_pct >= 15:
        grade = "B"
    elif edge_pct >= 5:
        grade = "C"
    elif edge_pct >= 0:
        grade = "D"
    else:
        grade = "F"

    # Note
    if kelly_pct == 0:
        note = "Negative edge — no position recommended"
    elif kelly_pct < 2:
        note = "Marginal edge — very small position only"
    elif kelly_pct < 8:
        note = "Moderate edge — use half-Kelly"
    elif kelly_pct < 15:
        note = "Good edge — half-Kelly appropriate"
    else:
        note = "Strong edge — quarter-Kelly for safety"

    return KellyResult(
        win_rate=round(win_rate * 100, 1),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        payoff_ratio=round(payoff_ratio, 2),
        edge=round(edge, 2),
        kelly_pct=round(kelly_pct, 2),
        half_kelly_pct=round(half_kelly_pct, 2),
        quarter_kelly_pct=round(quarter_kelly_pct, 2),
        n_trades=n,
        n_wins=n_wins,
        n_losses=n_losses,
        suggested_shares=suggested_shares,
        suggested_value=round(suggested_value, 2) if suggested_value is not None else None,
        grade=grade,
        note=note,
    )
