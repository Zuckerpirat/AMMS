"""Closed-trade performance by symbol.

Aggregates trade_pairs history to rank symbols by:
  - Total PnL
  - Win rate
  - Profit factor
  - Number of trades
  - Average PnL per trade

Helps identify which symbols are consistent winners vs drags.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolStats:
    symbol: str
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float          # 0-100
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float | None
    rank: int                # 1 = best by total PnL


@dataclass(frozen=True)
class SymbolPerformanceReport:
    symbols: list[SymbolStats]   # sorted best to worst by total_pnl
    best_symbol: str | None
    worst_symbol: str | None
    most_traded: str | None
    n_symbols: int
    n_trades: int
    total_pnl: float
    verdict: str


def compute(conn, *, limit: int = 500, min_trades: int = 2) -> SymbolPerformanceReport | None:
    """Aggregate closed-trade PnL by symbol.

    conn: SQLite connection with trade_pairs table.
    limit: max recent trades to include.
    min_trades: minimum trades per symbol to include in ranking.
    Returns None if fewer than 5 total trades.
    """
    try:
        rows = conn.execute(
            "SELECT symbol, pnl "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND symbol IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 5:
        return None

    groups: dict[str, list[float]] = {}
    for symbol, pnl in rows:
        try:
            groups.setdefault(str(symbol), []).append(float(pnl))
        except Exception:
            continue

    if not groups:
        return None

    stats_list: list[SymbolStats] = []
    for sym, pnls in groups.items():
        if len(pnls) < min_trades:
            continue
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = sum(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
        stats_list.append(SymbolStats(
            symbol=sym,
            n_trades=n,
            n_wins=len(wins),
            n_losses=len(losses),
            win_rate=round(len(wins) / n * 100, 1),
            total_pnl=round(total, 2),
            avg_pnl=round(total / n, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(pf, 2) if pf is not None else None,
            rank=0,
        ))

    if not stats_list:
        return None

    # Rank by total PnL
    stats_list.sort(key=lambda s: s.total_pnl, reverse=True)
    ranked = [
        SymbolStats(
            symbol=s.symbol, n_trades=s.n_trades, n_wins=s.n_wins, n_losses=s.n_losses,
            win_rate=s.win_rate, total_pnl=s.total_pnl, avg_pnl=s.avg_pnl,
            avg_win=s.avg_win, avg_loss=s.avg_loss, profit_factor=s.profit_factor,
            rank=i + 1,
        )
        for i, s in enumerate(stats_list)
    ]

    best = ranked[0].symbol if ranked else None
    worst = ranked[-1].symbol if ranked else None
    most_traded = max(ranked, key=lambda s: s.n_trades).symbol if ranked else None
    n_trades = sum(s.n_trades for s in ranked)
    total_pnl = sum(s.total_pnl for s in ranked)

    if best and worst and best != worst:
        verdict = (
            f"Top performer: {best} ({ranked[0].total_pnl:+.2f}, "
            f"{ranked[0].win_rate:.0f}% WR). "
            f"Weakest: {worst} ({ranked[-1].total_pnl:+.2f}). "
            f"{len(ranked)} symbols analyzed, total PnL: {total_pnl:+.2f}."
        )
    else:
        verdict = f"{len(ranked)} symbol(s), total PnL: {total_pnl:+.2f}."

    return SymbolPerformanceReport(
        symbols=ranked,
        best_symbol=best,
        worst_symbol=worst,
        most_traded=most_traded,
        n_symbols=len(ranked),
        n_trades=n_trades,
        total_pnl=round(total_pnl, 2),
        verdict=verdict,
    )
