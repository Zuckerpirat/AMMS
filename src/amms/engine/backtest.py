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
    mode: str = "swing",
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
            mode=mode,
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


# ── Multi-symbol batch backtest ───────────────────────────────────────────────

def run_batch_de_backtest(
    data_client,
    symbols: list[str],
    *,
    limit: int = 400,
    config: DEBacktestConfig | None = None,
) -> list[DEBacktestResult]:
    """Run `run_de_backtest` for each symbol using `data_client.get_bars()`.

    Symbols that fail to fetch or have too few bars are silently skipped.
    Returns results sorted by annualized return (best first).
    """
    results: list[DEBacktestResult] = []
    cfg = config or DEBacktestConfig()
    for sym in symbols:
        try:
            bars = data_client.get_bars(sym, limit=limit)
        except Exception:
            continue
        if not bars or len(bars) < 210:
            continue
        try:
            r = run_de_backtest(bars, symbol=sym, config=cfg)
        except Exception:
            continue
        results.append(r)
    results.sort(key=lambda r: r.annualized_return_pct, reverse=True)
    return results


def run_de_vs_buyhold(
    bars: list[Any],
    symbol: str = "",
    config: DEBacktestConfig | None = None,
) -> dict:
    """Compare DE strategy return against passive buy-and-hold on the same bars.

    Returns a dict with DE result, buy-and-hold equity curve, and comparison
    metrics. The buy-and-hold baseline invests 100% at the first bar's close
    and holds until the last bar.

    Keys returned:
      de_result        : DEBacktestResult
      bnh_return_pct   : buy-and-hold total return
      bnh_max_dd_pct   : buy-and-hold max drawdown
      alpha            : de_result.total_return_pct - bnh_return_pct
      outperformed     : bool — DE beat buy-and-hold
      summary          : str — formatted comparison text
    """
    cfg = config or DEBacktestConfig()
    de_result = run_de_backtest(bars, symbol=symbol, config=cfg)

    # Buy-and-hold: full position at first bar, held until last bar
    if len(bars) >= 2:
        entry = float(bars[0].close)
        equity = cfg.starting_cash
        if entry > 0:
            shares = equity / entry
        else:
            shares = 0.0

        bnh_curve = [shares * float(b.close) for b in bars]
        bnh_final = bnh_curve[-1]
        bnh_return_pct = (bnh_final / cfg.starting_cash - 1.0) * 100.0

        # Max drawdown for buy-and-hold
        peak = cfg.starting_cash
        bnh_max_dd = 0.0
        for eq in bnh_curve:
            peak = max(peak, eq)
            if peak > 0:
                dd = (peak - eq) / peak * 100.0
                bnh_max_dd = max(bnh_max_dd, dd)
    else:
        bnh_return_pct = 0.0
        bnh_max_dd = 0.0

    alpha = de_result.total_return_pct - bnh_return_pct
    outperformed = alpha > 0

    de_line = (
        f"DE Strategy:    {de_result.total_return_pct:>+7.2f}%  "
        f"(Sharpe {de_result.sharpe_ratio:.2f}  MaxDD {de_result.max_drawdown_pct:.2f}%  "
        f"{de_result.num_round_trips} trades)"
    )
    bnh_line = (
        f"Buy & Hold:     {bnh_return_pct:>+7.2f}%  "
        f"(Sharpe n/a        MaxDD {bnh_max_dd:.2f}%)"
    )
    verdict = "DE WINS" if outperformed else "BUY-AND-HOLD WINS"
    alpha_line = f"Alpha:          {alpha:>+7.2f}%  → {verdict}"

    summary = "\n".join([
        f"── DE vs Buy-and-Hold: {symbol} ({len(bars)} bars) ──",
        de_line,
        bnh_line,
        alpha_line,
    ])

    return {
        "de_result": de_result,
        "bnh_return_pct": round(bnh_return_pct, 2),
        "bnh_max_dd_pct": round(bnh_max_dd, 2),
        "alpha": round(alpha, 2),
        "outperformed": outperformed,
        "summary": summary,
    }


def run_regime_performance(
    bars: list[Any],
    symbol: str = "",
    config: DEBacktestConfig | None = None,
    *,
    regime_lookback: int = 20,
) -> dict:
    """Analyse DE backtest performance split by market regime.

    For each bar where the DE made a round-trip trade, classifies the regime
    at the entry bar and accumulates P&L by regime type.

    Regimes: "trending_up", "trending_down", "ranging_low_vol",
             "ranging_high_vol" (from amms.analysis.regime_classifier).

    Returns a dict:
      regime_stats : dict[regime_name, {trades, wins, pnl, win_rate}]
      de_result    : DEBacktestResult (full backtest)
      summary      : str (formatted table)
    """
    from amms.analysis.regime_classifier import classify as classify_regime

    cfg = config or DEBacktestConfig()
    de_result = run_de_backtest(bars, symbol=symbol, config=cfg)

    # Map each sell trade back to the regime at entry (buy) bar
    # Trade list: alternating buy/sell for the same position
    regime_pnl: dict[str, list[float]] = {}
    buy_idx: int | None = None
    buy_bar: int | None = None

    for trade in de_result.trades:
        if trade.side == "buy":
            buy_idx = trade.bar_index
            buy_bar = trade.bar_index
        elif trade.side == "sell" and buy_idx is not None:
            # Classify regime at the buy bar
            entry_bars = bars[: buy_idx + 1]
            regime = "unknown"
            if len(entry_bars) >= regime_lookback + 5:
                r = classify_regime(entry_bars, lookback=regime_lookback)
                if r is not None:
                    regime = r.regime
            regime_pnl.setdefault(regime, []).append(trade.pnl)
            buy_idx = None

    regime_stats: dict[str, dict] = {}
    for reg, pnls in regime_pnl.items():
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        regime_stats[reg] = {
            "trades": len(pnls),
            "wins": len(wins),
            "pnl": round(sum(pnls), 2),
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0.0,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
            "avg_loss": round(abs(sum(losses) / len(losses)), 2) if losses else 0.0,
        }

    # Format summary
    lines = [f"── DE Regime Performance: {symbol} ──", ""]
    if not regime_stats:
        lines.append("  No completed round-trips to analyse.")
    else:
        lines.append(f"  {'Regime':<22}  {'Trades':>6}  {'WinRate':>7}  {'Net P&L':>10}")
        for reg in sorted(regime_stats, key=lambda r: -regime_stats[r]["pnl"]):
            s = regime_stats[reg]
            lines.append(
                f"  {reg:<22}  {s['trades']:>6}  "
                f"{s['win_rate']:>6.0%}  ${s['pnl']:>+9,.2f}"
            )

    return {
        "regime_stats": regime_stats,
        "de_result": de_result,
        "summary": "\n".join(lines),
    }


def format_batch_summary(results: list[DEBacktestResult], *, top_n: int = 10) -> str:
    """Compact leaderboard table for a batch backtest run."""
    if not results:
        return "No backtest results to display."

    lines = [
        f"── DE Batch Backtest ({len(results)} symbols) ──",
        f"{'Symbol':<8}  {'Return':>8}  {'AnnRet':>8}  {'MaxDD':>7}  "
        f"{'Sharpe':>7}  {'WR':>5}  {'Trades':>6}",
    ]
    for r in results[:top_n]:
        lines.append(
            f"{r.symbol:<8}  {r.total_return_pct:>+7.2f}%  "
            f"{r.annualized_return_pct:>+7.2f}%  "
            f"{r.max_drawdown_pct:>6.2f}%  "
            f"{r.sharpe_ratio:>7.2f}  "
            f"{r.win_rate:>4.0%}  "
            f"{r.num_round_trips:>6}"
        )
    if len(results) > top_n:
        lines.append(f"  … and {len(results) - top_n} more")
    return "\n".join(lines)


_ALL_MODES = ("conservative", "swing", "meme", "event")


def run_mode_comparison(
    bars: list,
    symbol: str,
    config: DEBacktestConfig | None = None,
) -> dict:
    """Backtest the same symbol under all four trading modes.

    Runs run_de_backtest() four times — once per mode — and ranks them
    by total return. Also includes buy-and-hold for reference.

    Returns dict with:
      - "results": dict[mode_name, DEBacktestResult]
      - "ranking": list of mode names sorted by total_return_pct desc
      - "bnh_return_pct": buy-and-hold return over the same period
      - "best_mode": mode with highest return
      - "summary": formatted text table
    """
    cfg = config or DEBacktestConfig()

    results: dict[str, DEBacktestResult] = {}
    for mode in _ALL_MODES:
        results[mode] = run_de_backtest(bars, symbol=symbol, config=cfg, mode=mode)

    # Buy-and-hold
    bnh = run_de_vs_buyhold(bars, symbol=symbol, config=DEBacktestConfig(min_score=999_999.0))
    bnh_return = bnh.get("bnh_return_pct", 0.0)

    ranking = sorted(_ALL_MODES, key=lambda m: results[m].total_return_pct, reverse=True)
    best_mode = ranking[0]

    lines = [
        f"── DE Mode Comparison: {symbol} ──",
        f"{'Mode':<14}  {'Return':>8}  {'AnnRet':>8}  {'MaxDD':>7}  "
        f"{'Sharpe':>7}  {'WR':>5}  {'Trades':>6}",
    ]
    for mode in ranking:
        r = results[mode]
        best_marker = " ←" if mode == best_mode else ""
        lines.append(
            f"{mode:<14}  {r.total_return_pct:>+7.2f}%  "
            f"{r.annualized_return_pct:>+7.2f}%  "
            f"{r.max_drawdown_pct:>6.2f}%  "
            f"{r.sharpe_ratio:>7.2f}  "
            f"{r.win_rate:>4.0%}  "
            f"{r.num_round_trips:>6}"
            f"{best_marker}"
        )
    lines.append(
        f"{'Buy & Hold':<14}  {bnh_return:>+7.2f}%  "
        f"{'n/a':>8}  {'n/a':>7}  {'n/a':>7}  {'n/a':>5}  {'n/a':>6}"
    )

    return {
        "results": results,
        "ranking": ranking,
        "bnh_return_pct": bnh_return,
        "best_mode": best_mode,
        "summary": "\n".join(lines),
    }


def optimize_de_params(
    bars: list,
    symbol: str,
    *,
    min_score_range: tuple[float, float, float] = (20.0, 60.0, 10.0),   # start, stop, step
    min_confidence_range: tuple[float, float, float] = (0.50, 0.80, 0.10),
    config: DEBacktestConfig | None = None,
    mode: str = "swing",
) -> dict:
    """Grid-search the best min_score and min_confidence for the DE.

    Runs run_de_backtest() for each parameter combination and ranks by
    Sharpe ratio (ties broken by total return).

    Args:
        bars: bar history (needs 200+ bars for meaningful results)
        symbol: symbol name
        min_score_range: (start, stop, step) for min_score grid
        min_confidence_range: (start, stop, step) for min_confidence grid
        config: base DEBacktestConfig (min_score/min_confidence overridden by grid)
        mode: trading mode for DE weight selection

    Returns dict:
      - "best_params": {"min_score": float, "min_confidence": float}
      - "best_result": DEBacktestResult
      - "all_results": list of (min_score, min_confidence, DEBacktestResult)
      - "summary": formatted text table
    """
    cfg = config or DEBacktestConfig()

    ms_start, ms_stop, ms_step = min_score_range
    mc_start, mc_stop, mc_step = min_confidence_range

    grid: list[tuple[float, float]] = []
    ms = ms_start
    while ms <= ms_stop + 1e-9:
        mc = mc_start
        while mc <= mc_stop + 1e-9:
            grid.append((round(ms, 2), round(mc, 2)))
            mc += mc_step
        ms += ms_step

    all_results: list[tuple[float, float, DEBacktestResult]] = []
    for min_score, min_confidence in grid:
        trial_cfg = DEBacktestConfig(
            starting_cash=cfg.starting_cash,
            position_pct=cfg.position_pct,
            commission_pct=cfg.commission_pct,
            min_confidence=min_confidence,
            min_score=min_score,
            allow_strong_only=cfg.allow_strong_only,
            cooldown_bars=cfg.cooldown_bars,
        )
        result = run_de_backtest(bars, symbol=symbol, config=trial_cfg, mode=mode)
        all_results.append((min_score, min_confidence, result))

    if not all_results:
        return {"best_params": {}, "best_result": None, "all_results": [], "summary": "No results."}

    # Sort by Sharpe, then total return
    all_results.sort(key=lambda x: (x[2].sharpe_ratio, x[2].total_return_pct), reverse=True)
    best_ms, best_mc, best_result = all_results[0]

    lines = [
        f"── DE Parameter Optimization: {symbol} (mode={mode}) ──",
        f"  Grid: {len(grid)} combinations",
        f"  Best: min_score={best_ms:.0f}  min_confidence={best_mc:.0%}",
        f"  {'min_score':>9}  {'min_conf':>8}  {'Return':>8}  {'Sharpe':>7}  {'WR':>5}  {'Trades':>6}",
    ]
    for ms, mc, r in all_results[:10]:  # top 10
        lines.append(
            f"  {ms:>9.0f}  {mc:>8.0%}  {r.total_return_pct:>+7.2f}%  "
            f"{r.sharpe_ratio:>7.2f}  {r.win_rate:>4.0%}  {r.num_round_trips:>6}"
        )

    return {
        "best_params": {"min_score": best_ms, "min_confidence": best_mc},
        "best_result": best_result,
        "all_results": all_results,
        "summary": "\n".join(lines),
    }
