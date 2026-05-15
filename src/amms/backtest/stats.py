from __future__ import annotations

import csv
import math
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
    sharpe: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


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
    losses = 0
    closed = 0
    gross_profit = 0.0
    gross_loss = 0.0
    win_amounts: list[float] = []
    loss_amounts: list[float] = []
    for t in result.trades:
        if t.side == "buy":
            buys_by_sym.setdefault(t.symbol, []).append(t)
            continue
        queue = buys_by_sym.get(t.symbol, [])
        if not queue:
            continue
        entry = queue.pop(0)
        closed += 1
        pnl = (t.price - entry.price) * t.qty
        if pnl > 0:
            wins += 1
            gross_profit += pnl
            win_amounts.append(pnl)
        else:
            losses += 1
            gross_loss += abs(pnl)
            loss_amounts.append(abs(pnl))

    total_return_pct = ((final / initial - 1.0) * 100) if initial > 0 else 0.0
    win_rate = (wins / closed) if closed else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    avg_win = sum(win_amounts) / len(win_amounts) if win_amounts else 0.0
    avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 0.0

    # Annualised Sharpe from daily equity returns
    sharpe = 0.0
    if len(result.equity_curve) >= 3:
        equities = [eq for _, eq in result.equity_curve]
        rets = [(equities[i] - equities[i - 1]) / equities[i - 1] for i in range(1, len(equities))]
        n = len(rets)
        mean = sum(rets) / n
        variance = sum((r - mean) ** 2 for r in rets) / n
        std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0

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
        sharpe=sharpe,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
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
