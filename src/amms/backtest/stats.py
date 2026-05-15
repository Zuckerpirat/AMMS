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


@dataclass(frozen=True)
class ExtendedStats:
    """Additional backtest metrics beyond the basics."""
    calmar_ratio: float          # annualized return / max drawdown (higher = better)
    sortino_ratio: float         # annualized return / downside deviation
    recovery_factor: float       # total return / max drawdown (abs)
    max_consec_wins: int
    max_consec_losses: int
    expectancy: float            # avg(win_pct * win_size - loss_pct * loss_size)
    payoff_ratio: float          # avg_win / avg_loss (>1 = wins > losses in size)
    tail_ratio: float            # 95th pctile return / abs(5th pctile return)


def compute_extended_stats(result: BacktestResult, base: BacktestStats) -> ExtendedStats:
    """Compute extended backtest metrics from a result and its base stats."""
    initial = result.config.initial_equity

    # Annualised return estimate (assume 252 trading days)
    equities = [eq for _, eq in result.equity_curve] if result.equity_curve else [initial]
    n_days = max(len(equities) - 1, 1)
    annual_factor = 252.0 / n_days
    annualized_return = ((base.final_equity / initial) ** annual_factor - 1.0) * 100 if initial > 0 else 0.0

    max_dd_abs = abs(base.max_drawdown_pct)

    # Calmar ratio
    calmar = annualized_return / max_dd_abs if max_dd_abs > 0 else 0.0

    # Sortino ratio (downside deviation only)
    sortino = 0.0
    if len(equities) >= 3:
        rets = [(equities[i] - equities[i - 1]) / equities[i - 1] for i in range(1, len(equities))]
        mean = sum(rets) / len(rets)
        neg_rets = [r for r in rets if r < 0]
        if neg_rets:
            downside_var = sum(r ** 2 for r in neg_rets) / len(neg_rets)
            downside_dev = math.sqrt(downside_var)
            sortino = mean / downside_dev * math.sqrt(252) if downside_dev > 0 else 0.0

    # Recovery factor
    recovery = base.total_return_pct / max_dd_abs if max_dd_abs > 0 else 0.0

    # Consecutive wins/losses
    streak_wins = 0
    streak_losses = 0
    max_w = 0
    max_l = 0
    buys: dict[str, list[Trade]] = {}
    for t in result.trades:
        if t.side == "buy":
            buys.setdefault(t.symbol, []).append(t)
            continue
        queue = buys.get(t.symbol, [])
        if not queue:
            continue
        entry = queue.pop(0)
        pnl = (t.price - entry.price) * t.qty
        if pnl > 0:
            streak_wins += 1
            streak_losses = 0
        else:
            streak_losses += 1
            streak_wins = 0
        max_w = max(max_w, streak_wins)
        max_l = max(max_l, streak_losses)

    # Expectancy
    win_rate = base.win_rate
    loss_rate = 1.0 - win_rate
    expectancy = (win_rate * base.avg_win - loss_rate * base.avg_loss) if base.closed_round_trips > 0 else 0.0

    # Payoff ratio
    payoff = base.avg_win / base.avg_loss if base.avg_loss > 0 else 0.0

    # Tail ratio from equity curve returns
    tail = 0.0
    if len(equities) >= 10:
        rets = sorted((equities[i] - equities[i - 1]) / equities[i - 1] for i in range(1, len(equities)))
        p95_idx = int(len(rets) * 0.95)
        p05_idx = int(len(rets) * 0.05)
        p95 = rets[p95_idx] if p95_idx < len(rets) else rets[-1]
        p05 = rets[p05_idx] if p05_idx < len(rets) else rets[0]
        tail = abs(p95 / p05) if p05 != 0 else 0.0

    return ExtendedStats(
        calmar_ratio=round(calmar, 3),
        sortino_ratio=round(sortino, 3),
        recovery_factor=round(recovery, 3),
        max_consec_wins=max_w,
        max_consec_losses=max_l,
        expectancy=round(expectancy, 2),
        payoff_ratio=round(payoff, 3),
        tail_ratio=round(tail, 3),
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
