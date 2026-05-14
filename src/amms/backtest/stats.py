from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from amms.backtest.engine import BacktestResult, Trade


@dataclass(frozen=True)
class BacktestStats:
    initial_equity: float
    final_equity: float
    total_return_pct: float
    num_trades: int
    num_buys: int
    num_sells: int
    closed_round_trips: int
    win_rate: float
    max_drawdown_pct: float


def compute_stats(result: BacktestResult) -> BacktestStats:
    initial = result.config.initial_equity
    final = result.equity_curve[-1][1] if result.equity_curve else initial

    peak = initial
    max_dd = 0.0
    for _, eq in result.equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (eq - peak) / peak
            max_dd = min(max_dd, dd)

    buys_by_sym: dict[str, list[Trade]] = {}
    wins = 0
    closed = 0
    for t in result.trades:
        if t.side == "buy":
            buys_by_sym.setdefault(t.symbol, []).append(t)
            continue
        queue = buys_by_sym.get(t.symbol, [])
        if not queue:
            continue
        entry = queue.pop(0)
        closed += 1
        if t.price > entry.price:
            wins += 1

    total_return_pct = ((final / initial - 1.0) * 100) if initial > 0 else 0.0
    win_rate = (wins / closed) if closed else 0.0

    return BacktestStats(
        initial_equity=initial,
        final_equity=final,
        total_return_pct=total_return_pct,
        num_trades=len(result.trades),
        num_buys=sum(1 for t in result.trades if t.side == "buy"),
        num_sells=sum(1 for t in result.trades if t.side == "sell"),
        closed_round_trips=closed,
        win_rate=win_rate,
        max_drawdown_pct=max_dd * 100,
    )


def write_trades_csv(trades: list[Trade], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "symbol", "side", "qty", "price", "reason"])
        for t in trades:
            writer.writerow([t.date, t.symbol, t.side, t.qty, f"{t.price:.4f}", t.reason])
    return len(trades)


def write_equity_curve_csv(
    equity_curve: list[tuple[str, float]], path: Path
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "equity"])
        for d, eq in equity_curve:
            writer.writerow([d, f"{eq:.2f}"])
    return len(equity_curve)
