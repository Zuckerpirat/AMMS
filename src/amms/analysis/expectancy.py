"""Trade expectancy analysis.

Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

Shows how much on average each trade is expected to return.
Broken down per symbol and overall.

Also computes R-Multiple expectancy when stop distances are available.
Without stop info, uses avg_loss as 1R proxy.

Interpretation:
  Expectancy > 0: positive edge system
  Expectancy per trade / avg_loss: R-multiple expectancy
  e.g. expectancy=$50, avg_loss=$100 → 0.5R per trade
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolExpectancy:
    symbol: str
    n_trades: int
    win_rate: float          # 0-100
    avg_win: float           # $
    avg_loss: float          # positive $
    expectancy: float        # $ per trade
    r_expectancy: float      # R-multiple per trade
    grade: str               # A/B/C/D/F


@dataclass(frozen=True)
class ExpectancyReport:
    overall: SymbolExpectancy             # aggregated all trades
    by_symbol: list[SymbolExpectancy]     # per-symbol breakdown
    best_symbol: str | None
    worst_symbol: str | None
    n_symbols: int
    n_trades: int


def _grade(r_exp: float) -> str:
    if r_exp >= 0.5:
        return "A"
    elif r_exp >= 0.25:
        return "B"
    elif r_exp >= 0.0:
        return "C"
    elif r_exp >= -0.25:
        return "D"
    else:
        return "F"


def _compute_for_trades(trades: list[float], symbol: str) -> SymbolExpectancy | None:
    if not trades or len(trades) < 3:
        return None

    wins = [p for p in trades if p > 0]
    losses = [abs(p) for p in trades if p < 0]
    n = len(trades)

    if not wins and not losses:
        return None

    win_rate = len(wins) / n * 100
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    win_r = win_rate / 100
    loss_r = 1.0 - win_r

    expectancy = win_r * avg_win - loss_r * avg_loss
    r_expectancy = expectancy / avg_loss if avg_loss > 0 else 0.0

    return SymbolExpectancy(
        symbol=symbol,
        n_trades=n,
        win_rate=round(win_rate, 1),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        expectancy=round(expectancy, 2),
        r_expectancy=round(r_expectancy, 3),
        grade=_grade(r_expectancy),
    )


def compute(conn, *, limit: int = 200, min_trades: int = 3) -> ExpectancyReport | None:
    """Compute trade expectancy from trade_pairs.

    conn: SQLite connection with trade_pairs table
    limit: max trades to analyze (most recent)
    min_trades: minimum trades per symbol to include in breakdown

    Returns None if fewer than 5 trades total.
    """
    try:
        rows = conn.execute(
            "SELECT symbol, pnl FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND sell_price IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 5:
        return None

    # Group by symbol
    by_symbol: dict[str, list[float]] = {}
    all_pnls: list[float] = []
    for sym, pnl in rows:
        pnl_f = float(pnl)
        all_pnls.append(pnl_f)
        by_symbol.setdefault(sym, []).append(pnl_f)

    # Overall
    overall = _compute_for_trades(all_pnls, "OVERALL")
    if overall is None:
        return None

    # Per-symbol
    sym_results: list[SymbolExpectancy] = []
    for sym, pnls in by_symbol.items():
        if len(pnls) < min_trades:
            continue
        se = _compute_for_trades(pnls, sym)
        if se is not None:
            sym_results.append(se)

    sym_results.sort(key=lambda s: -s.expectancy)

    best = sym_results[0].symbol if sym_results else None
    worst = sym_results[-1].symbol if sym_results else None

    return ExpectancyReport(
        overall=overall,
        by_symbol=sym_results,
        best_symbol=best,
        worst_symbol=worst,
        n_symbols=len(sym_results),
        n_trades=len(all_pnls),
    )
