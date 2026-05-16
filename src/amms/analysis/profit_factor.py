"""Profit Factor Decomposition.

Analyses the profit factor (gross wins / gross losses) and decomposes it
into its two main drivers:
  - Win rate: how often you win
  - Win/loss size ratio: how much you win vs lose on average

Also computes:
  - Rolling profit factor (trend over time)
  - Profit factor by symbol and by bucket (scalp/intraday/swing)
  - Expectancy per trade (avg PnL per trade)
  - Payoff ratio (avg_win / abs(avg_loss))
  - Required win rate to break even at current payoff ratio
  - Kelly criterion from these metrics

These are the fundamental metrics that determine if a strategy is viable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingPF:
    window: int
    values: list[float]   # profit factor per rolling window
    trend: str            # "improving", "deteriorating", "stable"


@dataclass(frozen=True)
class SymbolPF:
    symbol: str
    n_trades: int
    profit_factor: float
    win_rate: float
    payoff_ratio: float


@dataclass(frozen=True)
class ProfitFactorReport:
    # Core metrics
    profit_factor: float          # gross_wins / gross_losses; inf if no losses
    win_rate: float               # 0-100
    payoff_ratio: float           # avg_win / abs(avg_loss)
    expectancy: float             # avg PnL per trade
    gross_wins: float
    gross_losses: float           # negative sum of losses

    # Breakeven analysis
    breakeven_win_rate: float     # win rate needed to break even at this payoff
    edge: float                   # win_rate - breakeven_win_rate (positive = edge)

    # Kelly
    kelly_pct: float              # full Kelly; use half in practice

    # Rolling
    rolling_20: RollingPF | None
    rolling_pf_trend: str         # from rolling_20

    # By symbol (top 5)
    by_symbol: list[SymbolPF]

    # Counts
    n_trades: int
    n_winners: int
    n_losers: int
    avg_winner_pct: float
    avg_loser_pct: float          # negative

    verdict: str


def _pf(wins: list[float], losses: list[float]) -> float:
    """Compute profit factor; returns 0 if no wins, large number if no losses."""
    gw = sum(wins) if wins else 0.0
    gl = abs(sum(losses)) if losses else 0.0
    if gl == 0:
        return 999.0 if gw > 0 else 0.0
    return gw / gl


def compute(conn, *, limit: int = 500) -> ProfitFactorReport | None:
    """Compute profit factor decomposition from closed trade history.

    conn: SQLite connection.
    limit: max trades to load.
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct, symbol, closed_at
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
            ORDER BY closed_at ASC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 10:
        return None

    pnls = [float(r[0]) for r in rows]
    symbols = [str(r[1]) if r[1] else "?" for r in rows]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    if not winners or not losers:
        return None

    n = len(pnls)
    n_w = len(winners)
    n_l = len(losers)
    wr = n_w / n * 100.0

    avg_w = sum(winners) / n_w
    avg_l = sum(losers) / n_l  # negative
    gross_w = sum(winners)
    gross_l = abs(sum(losers))

    pf = gross_w / gross_l if gross_l > 0 else 999.0
    payoff = avg_w / abs(avg_l) if avg_l != 0 else 999.0
    expectancy = sum(pnls) / n

    # Breakeven: wr_be × avg_w + (1-wr_be) × avg_l = 0
    # wr_be = |avg_l| / (avg_w + |avg_l|)
    be_wr = abs(avg_l) / (avg_w + abs(avg_l)) * 100.0 if (avg_w + abs(avg_l)) > 0 else 50.0
    edge = wr - be_wr

    # Kelly: f = wr - (1-wr)/payoff
    wr_dec = wr / 100.0
    kelly = wr_dec - (1.0 - wr_dec) / payoff if payoff > 0 else 0.0
    kelly = max(0.0, kelly) * 100.0  # as %

    # Rolling PF (window=20)
    rolling_vals: list[float] = []
    window = 20
    if n >= window + 5:
        for i in range(window, n + 1):
            chunk = pnls[i - window:i]
            cw = [p for p in chunk if p > 0]
            cl = [p for p in chunk if p <= 0]
            rolling_vals.append(_pf(cw, cl))

        # Trend: last 30% vs first 30%
        split = max(1, len(rolling_vals) // 3)
        early_pf = sum(rolling_vals[:split]) / split
        late_pf = sum(rolling_vals[-split:]) / split
        if late_pf > early_pf * 1.1:
            pf_trend = "improving"
        elif late_pf < early_pf * 0.9:
            pf_trend = "deteriorating"
        else:
            pf_trend = "stable"

        rolling_20 = RollingPF(window=window, values=[round(v, 3) for v in rolling_vals], trend=pf_trend)
    else:
        rolling_20 = None
        pf_trend = "stable"

    # By symbol
    sym_data: dict[str, list[float]] = {}
    for p, sym in zip(pnls, symbols):
        sym_data.setdefault(sym, []).append(p)

    sym_stats: list[SymbolPF] = []
    for sym, ps in sorted(sym_data.items(), key=lambda kv: -len(kv[1]))[:5]:
        sw = [p for p in ps if p > 0]
        sl = [p for p in ps if p <= 0]
        spf = _pf(sw, sl)
        swr = len(sw) / len(ps) * 100.0
        spo = (sum(sw) / len(sw)) / abs(sum(sl) / len(sl)) if sw and sl else 0.0
        sym_stats.append(SymbolPF(
            symbol=sym,
            n_trades=len(ps),
            profit_factor=round(spf, 3),
            win_rate=round(swr, 1),
            payoff_ratio=round(spo, 3),
        ))

    # Verdict
    parts = []
    pf_display = f"{pf:.2f}" if pf < 100 else ">100"
    parts.append(f"profit factor {pf_display}")
    parts.append(f"WR {wr:.1f}% vs breakeven {be_wr:.1f}% (edge {edge:+.1f}pp)")
    parts.append(f"payoff {payoff:.2f}× | Kelly {kelly:.1f}%")
    if rolling_20 and pf_trend != "stable":
        parts.append(f"PF is {pf_trend}")

    verdict = "Profit factor: " + "; ".join(parts) + "."

    return ProfitFactorReport(
        profit_factor=round(pf, 4),
        win_rate=round(wr, 2),
        payoff_ratio=round(payoff, 4),
        expectancy=round(expectancy, 4),
        gross_wins=round(gross_w, 3),
        gross_losses=round(gross_l, 3),
        breakeven_win_rate=round(be_wr, 2),
        edge=round(edge, 2),
        kelly_pct=round(kelly, 2),
        rolling_20=rolling_20,
        rolling_pf_trend=pf_trend,
        by_symbol=sym_stats,
        n_trades=n,
        n_winners=n_w,
        n_losers=n_l,
        avg_winner_pct=round(avg_w, 4),
        avg_loser_pct=round(avg_l, 4),
        verdict=verdict,
    )
