"""In-memory Decision Engine backtest.

Simulates trading the Decision Engine strategy on a single symbol's bar
history without requiring a pre-populated SQLite database. Bars are
fetched on demand via any object that exposes `get_bars(symbol, limit=N)`.

Algorithm
---------
Expanding-window walk-forward: for each bar index i starting at
`warmup_bars`, the engine sees only bars[0:i] (no look-ahead). The
signal at bar i determines the action taken at bar i+1's open price
(signals at close, fills at next open — same convention as the main
backtest engine).

This is intentionally simple (one symbol, market orders, fractional
shares, no slippage beyond open-price fill) so results are easy to
reason about and the code stays short.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# Minimum bars before the engine tries to signal anything meaningful.
_WARMUP_BARS = 200


@dataclass
class DEBacktestConfig:
    starting_cash: float = 100_000.0
    position_pct: float = 0.10       # fraction of portfolio per trade
    commission_pct: float = 0.001    # 0.1 % per trade (Alpaca is zero, this is conservative)
    min_confidence: float = 0.60
    min_score: float = 35.0
    allow_strong_only: bool = False
    cooldown_bars: int = 3           # bars to wait after a trade before the next


@dataclass
class _Trade:
    bar_index: int
    side: str       # "buy" | "sell"
    price: float
    qty: float
    commission: float
    pnl: float      # realized P&L at sell (0 for buys)


@dataclass
class DEBacktestResult:
    symbol: str
    config: DEBacktestConfig
    bars_total: int
    bars_simulated: int   # bars where the engine could run (>= warmup)

    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    num_trades: int
    num_round_trips: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consec_wins: int
    max_consec_losses: int
    expectancy: float

    equity_curve: list[float] = field(default_factory=list)
    trades: list[_Trade] = field(default_factory=list)

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        lines = [
            f"DE Backtest — {self.symbol} ({self.bars_total} bars, "
            f"{self.bars_simulated} simulated)",
            f"Return: {self.total_return_pct:+.2f}%  "
            f"Ann: {self.annualized_return_pct:+.2f}%  "
            f"MaxDD: {self.max_drawdown_pct:.2f}%",
            f"Sharpe: {self.sharpe_ratio:.2f}  "
            f"Sortino: {self.sortino_ratio:.2f}  "
            f"Calmar: {self.calmar_ratio:.2f}",
            f"Trades: {self.num_trades}  "
            f"RoundTrips: {self.num_round_trips}  "
            f"WinRate: {self.win_rate:.0%}",
            f"ProfitFactor: {self.profit_factor:.2f}  "
            f"Expectancy: ${self.expectancy:+.2f}",
            f"AvgWin: ${self.avg_win:.2f}  AvgLoss: ${self.avg_loss:.2f}",
            f"MaxConsecWins: {self.max_consec_wins}  "
            f"MaxConsecLoss: {self.max_consec_losses}",
        ]
        return "\n".join(lines)


def run_de_backtest(
    bars: list[Any],
    symbol: str = "",
    config: DEBacktestConfig | None = None,
) -> DEBacktestResult:
    """Simulate the Decision Engine on `bars` and return performance metrics.

    `bars` must be a list of objects with at least `.open` and `.close`
    attributes (same Bar type used everywhere in AMMS).

    The simulation uses signal-at-close / fill-at-next-open convention
    with fractional share quantities.
    """
    cfg = config or DEBacktestConfig()
    n = len(bars)

    cash = cfg.starting_cash
    held_qty: float = 0.0
    held_avg_cost: float = 0.0
    trades: list[_Trade] = []
    equity_curve: list[float] = []
    last_trade_bar: int = -999

    from amms.engine.decision import analyze

    bars_simulated = 0
    pending_side: str | None = None   # signal at bar i, filled at bar i+1

    for i in range(n):
        bar = bars[i]
        cur_price = float(bar.close)
        equity = cash + held_qty * cur_price

        # Fill yesterday's pending signal at today's open
        if pending_side is not None and i > 0:
            fill_price = float(bars[i].open)
            if pending_side == "buy" and held_qty == 0.0:
                dollar_amount = equity * cfg.position_pct
                commission = dollar_amount * cfg.commission_pct
                qty = (dollar_amount - commission) / fill_price if fill_price > 0 else 0.0
                if qty > 0 and dollar_amount <= cash:
                    cash -= dollar_amount
                    held_qty = qty
                    held_avg_cost = fill_price
                    trades.append(_Trade(i, "buy", fill_price, qty, commission, pnl=0.0))
                    last_trade_bar = i
            elif pending_side == "sell" and held_qty > 0.0:
                proceeds = held_qty * fill_price
                commission = proceeds * cfg.commission_pct
                pnl = (fill_price - held_avg_cost) * held_qty - commission
                cash += proceeds - commission
                trades.append(_Trade(i, "sell", fill_price, held_qty, commission, pnl=pnl))
                held_qty = 0.0
                held_avg_cost = 0.0
                last_trade_bar = i
            pending_side = None

        # Recalculate equity after fill
        equity = cash + held_qty * cur_price
        equity_curve.append(equity)

        # Only signal when we have enough bars for the engine
        if i < _WARMUP_BARS:
            continue
        bars_simulated += 1

        # Cooldown: don't signal too soon after a trade
        if i - last_trade_bar < cfg.cooldown_bars:
            continue

        report = analyze(
            bars[: i + 1],
            symbol=symbol,
            min_confidence=cfg.min_confidence,
        )
        if report is None or report.risk_blocked:
            continue
        if abs(report.composite_score) < cfg.min_score:
            continue
        if cfg.allow_strong_only and report.action not in {"strong_buy", "strong_sell"}:
            continue

        if report.action in {"buy", "strong_buy"} and held_qty == 0.0:
            pending_side = "buy"
        elif report.action in {"sell", "strong_sell"} and held_qty > 0.0:
            pending_side = "sell"

    # Close any open position at last bar's close
    if held_qty > 0.0 and bars:
        close_price = float(bars[-1].close)
        commission = held_qty * close_price * cfg.commission_pct
        pnl = (close_price - held_avg_cost) * held_qty - commission
        cash += held_qty * close_price - commission
        trades.append(_Trade(n - 1, "sell", close_price, held_qty, commission, pnl=pnl))
        held_qty = 0.0
        equity_curve[-1] = cash

    final_equity = cash
    initial = cfg.starting_cash
    total_return_pct = (final_equity / initial - 1.0) * 100.0 if initial > 0 else 0.0

    # Annualized return (assume 252 trading days)
    n_days = max(n - _WARMUP_BARS, 1)
    annual_factor = 252.0 / n_days
    annualized_return_pct = (
        ((final_equity / initial) ** annual_factor - 1.0) * 100.0
        if initial > 0 and final_equity > 0 else 0.0
    )

    # Max drawdown from equity curve
    peak = initial
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak * 100.0
            max_dd = max(max_dd, dd)

    # Sharpe + Sortino from daily equity returns
    sharpe = sortino = 0.0
    if len(equity_curve) >= 3:
        rets = [
            (equity_curve[k] - equity_curve[k - 1]) / equity_curve[k - 1]
            for k in range(1, len(equity_curve))
            if equity_curve[k - 1] > 0
        ]
        if len(rets) >= 2:
            mean_r = sum(rets) / len(rets)
            var_r = sum((r - mean_r) ** 2 for r in rets) / len(rets)
            std_r = math.sqrt(var_r) if var_r > 0 else 0.0
            sharpe = mean_r / std_r * math.sqrt(252) if std_r > 0 else 0.0

            neg = [r for r in rets if r < 0]
            if neg:
                down_var = sum(r ** 2 for r in neg) / len(neg)
                down_std = math.sqrt(down_var)
                sortino = mean_r / down_std * math.sqrt(252) if down_std > 0 else 0.0

    calmar = annualized_return_pct / max_dd if max_dd > 0 else 0.0

    # Round-trip stats (pair each buy with the next sell)
    sell_trades = [t for t in trades if t.side == "sell"]
    num_round_trips = len(sell_trades)
    wins = [t for t in sell_trades if t.pnl > 0]
    losses = [t for t in sell_trades if t.pnl <= 0]
    win_rate = len(wins) / num_round_trips if num_round_trips > 0 else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Consecutive wins/losses
    streak_w = streak_l = max_w = max_l = 0
    for t in sell_trades:
        if t.pnl > 0:
            streak_w += 1; streak_l = 0
        else:
            streak_l += 1; streak_w = 0
        max_w = max(max_w, streak_w)
        max_l = max(max_l, streak_l)

    return DEBacktestResult(
        symbol=symbol,
        config=cfg,
        bars_total=n,
        bars_simulated=bars_simulated,
        total_return_pct=round(total_return_pct, 2),
        annualized_return_pct=round(annualized_return_pct, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        num_trades=len(trades),
        num_round_trips=num_round_trips,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        max_consec_wins=max_w,
        max_consec_losses=max_l,
        expectancy=round(expectancy, 2),
        equity_curve=equity_curve,
        trades=trades,
    )
