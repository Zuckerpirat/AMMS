"""Trade Clustering Analysis.

Identifies whether trades cluster in time or price — patterns that
suggest systematic behavioural biases:

  Time clustering: many entries at the same time of day
    → could be chasing opening moves, FOMO at end-of-day, etc.

  Price clustering: many entries at round numbers or the same price level
    → could be anchoring bias (always buying at 100, 150, 200, etc.)

  Burst trading: many trades in a short time window
    → impulsive over-trading after a win/loss streak

  Symbol concentration: repeated trades in the same few stocks
    → lack of diversification, over-confidence in specific names

Outputs counts, ratios, and a verdict on each clustering dimension.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeCluster:
    hour: int                # 0–23
    n_trades: int
    avg_pnl_pct: float
    win_rate: float


@dataclass(frozen=True)
class BurstWindow:
    window_start: str        # ISO timestamp of first trade in burst
    n_trades: int
    duration_minutes: float
    avg_pnl_pct: float


@dataclass(frozen=True)
class TradingClusterReport:
    # Time clustering
    top_hours: list[TimeCluster]      # top 3 busiest hours
    hour_concentration: float         # Herfindahl-like: 0 = spread, 1 = all one hour
    # Burst trading
    burst_windows: list[BurstWindow]  # windows with >3 trades in <30 min
    n_burst_trades: int               # trades inside burst windows
    burst_pct: float                  # % of total trades in bursts
    burst_avg_pnl: float              # avg pnl_pct for burst trades
    non_burst_avg_pnl: float          # avg pnl_pct for non-burst trades
    # Symbol concentration
    top_symbols: list[tuple[str, int]]  # (symbol, n_trades) top 5
    symbol_concentration: float       # HHI-like score
    # Round number anchoring
    round_number_pct: float           # % of entries at round $5 levels
    # Metadata
    n_trades: int
    verdict: str


def _hhi(counts: list[int]) -> float:
    """Herfindahl-Hirschman Index normalized to [0, 1].
    1 = fully concentrated, 0 = perfectly spread.
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    return sum((c / total) ** 2 for c in counts)


def _parse_hour(ts: str) -> int | None:
    """Extract hour from ISO timestamp like '2024-03-15T10:30:00' or '2024-03-15 10:30:00'."""
    try:
        t = ts.replace("T", " ").strip()
        time_part = t.split(" ")[1] if " " in t else ""
        return int(time_part.split(":")[0]) if time_part else None
    except Exception:
        return None


def _parse_minutes_since_epoch(ts: str) -> float | None:
    """Convert ISO timestamp to minutes since 2000-01-01 for duration calculation."""
    try:
        t = ts.replace("T", " ").strip()
        parts = t.split(" ")
        date_parts = parts[0].split("-")
        time_parts = parts[1].split(":") if len(parts) > 1 else ["0", "0", "0"]
        y, mo, d = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
        h, mi = int(time_parts[0]), int(time_parts[1])
        # Simple days since 2000-01-01 (approximate, ignoring leap years)
        days = (y - 2000) * 365 + (mo - 1) * 30 + d
        return days * 1440.0 + h * 60.0 + mi
    except Exception:
        return None


def compute(conn, *, limit: int = 500, burst_window_min: int = 30, burst_min_trades: int = 3) -> TradingClusterReport | None:
    """Analyse trade clustering patterns from closed trade history.

    conn: SQLite connection.
    limit: max trades to analyse.
    burst_window_min: window size in minutes for burst detection (default 30).
    burst_min_trades: minimum trades in window to count as burst (default 3).
    """
    try:
        rows = conn.execute("""
            SELECT symbol, entered_at, pnl_pct, buy_price
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
              AND entered_at IS NOT NULL
            ORDER BY entered_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 5:
        return None

    n = len(rows)

    # --- Time clustering ---
    hour_stats: dict[int, list[float]] = {}
    for sym, ts, pnl, buy in rows:
        h = _parse_hour(str(ts))
        if h is None:
            continue
        if h not in hour_stats:
            hour_stats[h] = []
        hour_stats[h].append(float(pnl))

    top_hours: list[TimeCluster] = []
    for h in sorted(hour_stats, key=lambda x: -len(hour_stats[x]))[:3]:
        pnls = hour_stats[h]
        wins = [p for p in pnls if p > 0]
        top_hours.append(TimeCluster(
            hour=h,
            n_trades=len(pnls),
            avg_pnl_pct=round(sum(pnls) / len(pnls), 3),
            win_rate=round(len(wins) / len(pnls) * 100, 1),
        ))
    hour_counts = [len(v) for v in hour_stats.values()]
    hour_concentration = round(_hhi(hour_counts), 4)

    # --- Burst detection ---
    # Sort by time
    timed_trades = []
    for sym, ts, pnl, buy in rows:
        t = _parse_minutes_since_epoch(str(ts))
        if t is not None:
            timed_trades.append((t, float(pnl), str(ts)))
    timed_trades.sort(key=lambda x: x[0])

    burst_windows: list[BurstWindow] = []
    burst_trade_indices: set[int] = set()
    i = 0
    while i < len(timed_trades):
        j = i + 1
        while j < len(timed_trades) and timed_trades[j][0] - timed_trades[i][0] <= burst_window_min:
            j += 1
        count = j - i
        if count >= burst_min_trades:
            window_pnls = [timed_trades[k][1] for k in range(i, j)]
            burst_windows.append(BurstWindow(
                window_start=timed_trades[i][2],
                n_trades=count,
                duration_minutes=round(timed_trades[j - 1][0] - timed_trades[i][0], 1),
                avg_pnl_pct=round(sum(window_pnls) / len(window_pnls), 3),
            ))
            for k in range(i, j):
                burst_trade_indices.add(k)
            i = j  # skip past this burst
        else:
            i += 1

    n_burst = len(burst_trade_indices)
    burst_pnls = [timed_trades[k][1] for k in burst_trade_indices]
    non_burst_pnls = [timed_trades[k][1] for k in range(len(timed_trades)) if k not in burst_trade_indices]
    burst_avg = sum(burst_pnls) / len(burst_pnls) if burst_pnls else 0.0
    non_burst_avg = sum(non_burst_pnls) / len(non_burst_pnls) if non_burst_pnls else 0.0

    # --- Symbol concentration ---
    sym_counts: dict[str, int] = {}
    for sym, ts, pnl, buy in rows:
        sym_counts[sym] = sym_counts.get(sym, 0) + 1
    top_syms = sorted(sym_counts.items(), key=lambda x: -x[1])[:5]
    symbol_concentration = round(_hhi(list(sym_counts.values())), 4)

    # --- Round number anchoring ---
    round_count = 0
    for sym, ts, pnl, buy in rows:
        try:
            price = float(buy)
            # Round to nearest $5 — if within 1%, consider it a round number entry
            nearest_5 = round(price / 5.0) * 5.0
            if nearest_5 > 0 and abs(price - nearest_5) / nearest_5 < 0.01:
                round_count += 1
        except Exception:
            pass
    round_pct = round_count / n * 100.0

    # --- Verdict ---
    notes = []
    if hour_concentration > 0.5:
        h = top_hours[0].hour if top_hours else "?"
        notes.append(f"strong time clustering at {h}:00")
    if len(burst_windows) > 0:
        notes.append(f"{len(burst_windows)} burst windows ({n_burst} trades, avg {burst_avg:+.1f}% vs {non_burst_avg:+.1f}% outside)")
    if symbol_concentration > 0.3:
        top = top_syms[0][0] if top_syms else "?"
        notes.append(f"high symbol concentration ({top} dominates)")
    if round_pct > 30:
        notes.append(f"{round_pct:.0f}% of entries at round $5 levels (anchoring)")

    verdict = (
        f"Trade clustering ({n} trades): " +
        ("; ".join(notes) if notes else "no significant clustering detected") + "."
    )

    return TradingClusterReport(
        top_hours=top_hours,
        hour_concentration=hour_concentration,
        burst_windows=burst_windows[:5],
        n_burst_trades=n_burst,
        burst_pct=round(n_burst / n * 100.0, 1),
        burst_avg_pnl=round(burst_avg, 3),
        non_burst_avg_pnl=round(non_burst_avg, 3),
        top_symbols=top_syms,
        symbol_concentration=symbol_concentration,
        round_number_pct=round(round_pct, 1),
        n_trades=n,
        verdict=verdict,
    )
