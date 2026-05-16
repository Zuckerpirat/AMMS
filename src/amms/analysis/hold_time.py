"""Position Hold Time Analyser.

Analyses how long positions are held across the closed trade journal and
whether hold time correlates with outcome.

Metrics:
  - Median and mean hold time (minutes) overall
  - Hold time distribution buckets (scalp/intraday/swing/multi-day)
  - Hold time vs outcome: do longer holds win more?
  - Best hold-time bucket by avg PnL%
  - Per-symbol hold time (top/bottom symbols)
  - Hold time trend: are recent trades held longer or shorter?
"""

from __future__ import annotations

from dataclasses import dataclass

# Duration bucket boundaries (minutes)
_SCALP_MAX = 30       # < 30 min
_INTRADAY_MAX = 390   # < 6.5 h (one trading session)
_SWING_MAX = 2880     # < 2 days (2 × 24 h)
# >= 2880 = multi-day


@dataclass(frozen=True)
class HoldBucket:
    label: str          # "scalp", "intraday", "swing", "multi-day"
    min_minutes: float
    max_minutes: float  # exclusive; inf for last bucket
    n_trades: int
    win_rate: float
    avg_pnl_pct: float
    avg_hold_min: float


@dataclass(frozen=True)
class SymbolHoldStats:
    symbol: str
    n_trades: int
    avg_hold_min: float
    avg_pnl_pct: float


@dataclass(frozen=True)
class HoldTimeReport:
    median_hold_min: float
    mean_hold_min: float
    min_hold_min: float
    max_hold_min: float

    buckets: list[HoldBucket]        # scalp → multi-day
    best_bucket: HoldBucket | None   # highest avg PnL%
    worst_bucket: HoldBucket | None

    # Correlation: hold time vs pnl (Pearson, rough)
    hold_pnl_correlation: float      # -1 to +1

    # Per symbol (top 5 by trade count)
    by_symbol: list[SymbolHoldStats]

    # Recent trend: avg hold in last 20% of trades vs first 20%
    recent_hold_min: float | None
    early_hold_min: float | None
    hold_trend: str      # "increasing", "decreasing", "stable"

    n_trades: int
    n_with_duration: int
    verdict: str


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _parse_minutes(entered: str, exited: str) -> float | None:
    """Parse two ISO timestamps and return difference in minutes."""
    try:
        def _to_minutes(ts: str) -> float:
            t = str(ts).replace("T", " ").strip().split(".")[0]
            date_part, time_part = t.split(" ")
            y, mo, d = [int(x) for x in date_part.split("-")]
            h, mi, s = [int(x) for x in time_part.split(":")]
            # Days since epoch (rough, ignores DST)
            days = (y - 1970) * 365 + (y - 1970) // 4 + _days_to_month(mo, y) + d - 1
            return days * 1440 + h * 60 + mi + s / 60.0

        return _to_minutes(exited) - _to_minutes(entered)
    except Exception:
        return None


def _days_to_month(m: int, y: int) -> int:
    _MD = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    leap = 1 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) and m > 2 else 0
    return _MD[m - 1] + leap


def compute(conn, *, limit: int = 500) -> HoldTimeReport | None:
    """Analyse hold times from closed trade history.

    conn: SQLite connection.
    limit: max trades to load (ordered by closed_at ASC).
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct, entered_at, closed_at, symbol
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
              AND entered_at IS NOT NULL
              AND closed_at IS NOT NULL
            ORDER BY closed_at ASC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 5:
        return None

    durations: list[float] = []
    pnls_with_dur: list[float] = []
    symbol_data: dict[str, list[tuple[float, float]]] = {}  # symbol → [(hold_min, pnl)]

    for pnl, entered, exited, symbol in rows:
        pnl = float(pnl)
        dur = _parse_minutes(str(entered), str(exited))
        if dur is not None and dur >= 0:
            durations.append(dur)
            pnls_with_dur.append(pnl)
            sym = str(symbol) if symbol else "UNKNOWN"
            symbol_data.setdefault(sym, []).append((dur, pnl))

    if len(durations) < 3:
        return None

    n = len(rows)
    nd = len(durations)

    median_hold = _median(durations)
    mean_hold = sum(durations) / nd
    min_hold = min(durations)
    max_hold = max(durations)

    # Buckets
    bucket_defs = [
        ("scalp", 0.0, float(_SCALP_MAX)),
        ("intraday", float(_SCALP_MAX), float(_INTRADAY_MAX)),
        ("swing", float(_INTRADAY_MAX), float(_SWING_MAX)),
        ("multi-day", float(_SWING_MAX), float("inf")),
    ]
    buckets: list[HoldBucket] = []
    for label, lo, hi in bucket_defs:
        items = [(d, p) for d, p in zip(durations, pnls_with_dur) if lo <= d < hi]
        if not items:
            buckets.append(HoldBucket(label=label, min_minutes=lo, max_minutes=hi,
                                      n_trades=0, win_rate=0.0, avg_pnl_pct=0.0, avg_hold_min=0.0))
            continue
        ds, ps = zip(*items)
        wr = sum(1 for p in ps if p > 0) / len(ps) * 100.0
        buckets.append(HoldBucket(
            label=label,
            min_minutes=lo,
            max_minutes=hi,
            n_trades=len(items),
            win_rate=round(wr, 1),
            avg_pnl_pct=round(sum(ps) / len(ps), 3),
            avg_hold_min=round(sum(ds) / len(ds), 1),
        ))

    active_buckets = [b for b in buckets if b.n_trades > 0]
    best_bucket = max(active_buckets, key=lambda b: b.avg_pnl_pct) if active_buckets else None
    worst_bucket = min(active_buckets, key=lambda b: b.avg_pnl_pct) if active_buckets else None

    # Correlation hold time vs pnl
    corr = _pearson(durations, pnls_with_dur)

    # Per symbol stats (top 5 by n_trades)
    by_symbol_list: list[SymbolHoldStats] = []
    for sym, items in sorted(symbol_data.items(), key=lambda kv: -len(kv[1]))[:5]:
        ds, ps = zip(*items)
        by_symbol_list.append(SymbolHoldStats(
            symbol=sym,
            n_trades=len(items),
            avg_hold_min=round(sum(ds) / len(ds), 1),
            avg_pnl_pct=round(sum(ps) / len(ps), 3),
        ))

    # Hold trend: first 20% vs last 20%
    window = max(1, nd // 5)
    early_hold = sum(durations[:window]) / window if window > 0 else None
    recent_hold = sum(durations[-window:]) / window if window > 0 else None
    if early_hold and recent_hold:
        diff = recent_hold - early_hold
        hold_trend = "increasing" if diff > early_hold * 0.1 else "decreasing" if diff < -early_hold * 0.1 else "stable"
    else:
        hold_trend = "stable"

    # Verdict
    parts = []
    parts.append(f"median hold {median_hold:.0f} min ({median_hold / 60:.1f} h)")
    if best_bucket and best_bucket.n_trades >= 3:
        parts.append(f"best bucket '{best_bucket.label}' avg {best_bucket.avg_pnl_pct:+.2f}%")
    if abs(corr) > 0.15:
        direction = "longer holds = better" if corr > 0 else "longer holds = worse"
        parts.append(f"{direction} (r={corr:.2f})")
    if hold_trend != "stable":
        parts.append(f"hold time is {hold_trend} recently")

    verdict = "Hold time analysis: " + "; ".join(parts) + "."

    return HoldTimeReport(
        median_hold_min=round(median_hold, 2),
        mean_hold_min=round(mean_hold, 2),
        min_hold_min=round(min_hold, 2),
        max_hold_min=round(max_hold, 2),
        buckets=buckets,
        best_bucket=best_bucket,
        worst_bucket=worst_bucket,
        hold_pnl_correlation=round(corr, 4),
        by_symbol=by_symbol_list,
        recent_hold_min=round(recent_hold, 2) if recent_hold is not None else None,
        early_hold_min=round(early_hold, 2) if early_hold is not None else None,
        hold_trend=hold_trend,
        n_trades=n,
        n_with_duration=nd,
        verdict=verdict,
    )
