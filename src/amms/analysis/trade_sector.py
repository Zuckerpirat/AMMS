"""Trade Sector Performance Analyser.

Analyses trade performance grouped by sector to detect which sectors have
been performing best recently versus historically, and whether there is
evidence of rotation (leadership changing over time).

The sector mapping is passed in as an optional dict; if not supplied the
analyser falls back to a built-in mapping of well-known tickers.

Metrics per sector:
  - n_trades, win_rate, avg_pnl_pct, total_pnl
  - Recent performance (last 30% of trades) vs historical (first 70%)
  - Rotation score = recent_avg_pnl - historical_avg_pnl

Report:
  - Ranked sector table
  - Current leader / laggard
  - Rotation signal: is the recent leader different from the historical leader?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeSectorStats:
    sector: str
    n_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl: float
    recent_avg_pnl: float       # last 30% of sector trades
    historical_avg_pnl: float   # first 70% of sector trades
    rotation_score: float       # recent - historical; positive = improving
    is_reliable: bool           # n_trades >= min_reliable


@dataclass(frozen=True)
class TradeSectorReport:
    sectors: list[TradeSectorStats]    # sorted by recent_avg_pnl desc
    leader: TradeSectorStats | None    # best recent performance
    laggard: TradeSectorStats | None   # worst recent performance
    historical_leader: TradeSectorStats | None
    rotation_detected: bool       # leader != historical_leader
    rotation_magnitude: float     # abs rank change for the current leader
    n_trades: int
    n_sectors: int
    verdict: str


# Built-in symbol → sector mapping
_KNOWN: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "TSM": "Technology",
    "AMZN": "Consumer", "TSLA": "Consumer", "NKE": "Consumer",
    "HD": "Consumer", "MCD": "Consumer", "SBUX": "Consumer",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance",
    "MS": "Finance", "WFC": "Finance", "C": "Finance",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "OXY": "Energy",
    "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "GLD": "Commodities", "SLV": "Commodities", "USO": "Commodities",
    "VIXY": "Volatility", "VXX": "Volatility",
}


def _assign_sector(symbol: str, sector_map: dict[str, str] | None) -> str:
    sym = str(symbol).upper().strip()
    if sector_map and sym in sector_map:
        return sector_map[sym]
    return _KNOWN.get(sym, "Other")


def compute(
    conn,
    *,
    limit: int = 1000,
    sector_map: dict[str, str] | None = None,
    min_reliable: int = 5,
) -> TradeSectorReport | None:
    """Analyse sector performance from closed trade history.

    conn: SQLite connection.
    limit: max trades to load.
    sector_map: optional symbol→sector override dict.
    min_reliable: minimum trades per sector to mark reliable.
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct, symbol, closed_at
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
              AND symbol IS NOT NULL
            ORDER BY closed_at ASC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 10:
        return None

    sector_pnls: dict[str, list[float]] = {}
    for pnl, symbol, _ in rows:
        sector = _assign_sector(str(symbol), sector_map)
        sector_pnls.setdefault(sector, []).append(float(pnl))

    if len(sector_pnls) < 2:
        return None

    sector_stats: list[TradeSectorStats] = []
    for sector, pnls in sector_pnls.items():
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100.0
        avg = sum(pnls) / n
        total = sum(pnls)

        split = max(1, int(n * 0.7))
        hist_pnls = pnls[:split]
        recent_pnls = pnls[split:] if split < n else pnls

        hist_avg = sum(hist_pnls) / len(hist_pnls) if hist_pnls else avg
        recent_avg = sum(recent_pnls) / len(recent_pnls) if recent_pnls else avg

        sector_stats.append(TradeSectorStats(
            sector=sector,
            n_trades=n,
            win_rate=round(wr, 1),
            avg_pnl_pct=round(avg, 3),
            total_pnl=round(total, 3),
            recent_avg_pnl=round(recent_avg, 3),
            historical_avg_pnl=round(hist_avg, 3),
            rotation_score=round(recent_avg - hist_avg, 3),
            is_reliable=n >= min_reliable,
        ))

    sector_stats.sort(key=lambda s: -s.recent_avg_pnl)

    reliable = [s for s in sector_stats if s.is_reliable]
    rank_pool = reliable if reliable else sector_stats

    leader = max(rank_pool, key=lambda s: s.recent_avg_pnl) if rank_pool else None
    laggard = min(rank_pool, key=lambda s: s.recent_avg_pnl) if rank_pool else None
    hist_leader = max(rank_pool, key=lambda s: s.historical_avg_pnl) if rank_pool else None

    rotation_detected = bool(leader and hist_leader and leader.sector != hist_leader.sector)

    if rotation_detected and leader and hist_leader:
        recent_ranks = {s.sector: i for i, s in enumerate(
            sorted(rank_pool, key=lambda s: -s.recent_avg_pnl))}
        hist_ranks = {s.sector: i for i, s in enumerate(
            sorted(rank_pool, key=lambda s: -s.historical_avg_pnl))}
        mag = float(abs(recent_ranks.get(leader.sector, 0) - hist_ranks.get(leader.sector, 0)))
    else:
        mag = 0.0

    parts = []
    if leader:
        parts.append(f"leader: {leader.sector} (recent {leader.recent_avg_pnl:+.2f}%)")
    if laggard and laggard.sector != (leader.sector if leader else ""):
        parts.append(f"laggard: {laggard.sector} ({laggard.recent_avg_pnl:+.2f}%)")
    if rotation_detected and hist_leader:
        parts.append(f"rotation detected — was {hist_leader.sector}")
    else:
        parts.append("no significant rotation")

    verdict = "Sector performance: " + "; ".join(parts) + "."

    return TradeSectorReport(
        sectors=sector_stats,
        leader=leader,
        laggard=laggard,
        historical_leader=hist_leader,
        rotation_detected=rotation_detected,
        rotation_magnitude=round(mag, 1),
        n_trades=len(rows),
        n_sectors=len(sector_stats),
        verdict=verdict,
    )
