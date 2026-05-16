"""Win rate and PnL analysis broken down by sector.

Uses the SYMBOL_SECTOR lookup from sector_exposure to map each trade's
symbol to a sector, then aggregates performance metrics by sector.

Identifies which sectors the trader performs best and worst in,
helping focus trading activity on high-edge sectors.

Reads from trade_pairs (symbol, pnl, buy_price, qty).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from amms.analysis.sector_exposure import SYMBOL_SECTOR
except ImportError:
    SYMBOL_SECTOR: dict = {}


@dataclass(frozen=True)
class SectorWinRate:
    sector: str
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float          # 0-100
    avg_pnl_pct: float
    total_pnl: float
    profit_factor: float | None
    symbols: list[str]       # distinct symbols traded in this sector


@dataclass(frozen=True)
class SectorWinRateReport:
    sectors: list[SectorWinRate]  # sorted by total_pnl desc
    best_sector: str | None
    worst_sector: str | None
    unknown_trades: int           # trades where sector couldn't be mapped
    n_sectors: int
    n_trades: int
    verdict: str


def compute(conn, *, limit: int = 500, min_trades: int = 2) -> SectorWinRateReport | None:
    """Analyze win rate by sector from trade_pairs.

    conn: SQLite connection with trade_pairs table.
    Returns None if fewer than 5 usable trades.
    """
    try:
        rows = conn.execute(
            "SELECT symbol, pnl, buy_price, qty "
            "FROM trade_pairs "
            "WHERE pnl IS NOT NULL AND symbol IS NOT NULL "
            "ORDER BY sell_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return None

    if not rows or len(rows) < 5:
        return None

    # Group by sector
    sector_data: dict[str, dict] = {}
    unknown = 0

    for symbol, pnl, buy_price, qty in rows:
        try:
            sym = str(symbol).upper()
            pnl_f = float(pnl)
            bp = float(buy_price) if buy_price else 0.0
            qty_f = float(qty) if qty else 1.0
            entry_value = bp * qty_f
            pnl_pct = pnl_f / entry_value * 100 if entry_value > 0 else 0.0

            sector = SYMBOL_SECTOR.get(sym, "Unknown")
            if sector == "Unknown":
                unknown += 1

            if sector not in sector_data:
                sector_data[sector] = {"pnls": [], "pnl_pcts": [], "symbols": set()}
            sector_data[sector]["pnls"].append(pnl_f)
            sector_data[sector]["pnl_pcts"].append(pnl_pct)
            sector_data[sector]["symbols"].add(sym)
        except Exception:
            continue

    stats: list[SectorWinRate] = []
    for sector, data in sector_data.items():
        pnls = data["pnls"]
        pnl_pcts = data["pnl_pcts"]
        if len(pnls) < min_trades:
            continue
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else None
        stats.append(SectorWinRate(
            sector=sector,
            n_trades=n,
            n_wins=len(wins),
            n_losses=len(losses),
            win_rate=round(len(wins) / n * 100, 1),
            avg_pnl_pct=round(sum(pnl_pcts) / n, 3),
            total_pnl=round(sum(pnls), 2),
            profit_factor=round(pf, 2) if pf is not None else None,
            symbols=sorted(data["symbols"]),
        ))

    if not stats:
        return None

    stats.sort(key=lambda s: s.total_pnl, reverse=True)

    # Exclude Unknown from best/worst if possible
    known = [s for s in stats if s.sector != "Unknown"]
    best = known[0].sector if known else stats[0].sector
    worst = known[-1].sector if known else stats[-1].sector

    n_trades = sum(s.n_trades for s in stats)

    if best != worst:
        best_stats = next(s for s in stats if s.sector == best)
        worst_stats = next(s for s in stats if s.sector == worst)
        verdict = (
            f"Best sector: {best} ({best_stats.win_rate:.0f}% WR, "
            f"{best_stats.avg_pnl_pct:+.2f}% avg). "
            f"Worst: {worst} ({worst_stats.win_rate:.0f}% WR, "
            f"{worst_stats.avg_pnl_pct:+.2f}% avg). "
            f"{len(stats)} sectors analyzed."
        )
    else:
        verdict = f"{len(stats)} sector(s) analyzed."

    return SectorWinRateReport(
        sectors=stats,
        best_sector=best,
        worst_sector=worst,
        unknown_trades=unknown,
        n_sectors=len(stats),
        n_trades=n_trades,
        verdict=verdict,
    )
