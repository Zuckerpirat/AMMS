"""Calendar Anomaly Detector.

Tests for systematic performance patterns in the closed trade history:

1. Day-of-week effect: Monday-Friday win rate and avg PnL%
2. Month-of-year effect: January through December patterns
3. Quarter-end effect: last 5 trading days of each quarter vs. otherwise
4. Turn-of-month effect: first 3 trading days of each month vs. otherwise

These are classic "market anomalies" — if you systematically perform
better on certain days or months, it can inform when to be more/less
aggressive.

A statistical significance note is included: small samples (< 20 per
group) are flagged as insufficient for reliable conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalendarSlice:
    label: str           # e.g. "Monday", "January", "Q-end", "Month-start"
    n_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl: float
    reliable: bool       # True if n_trades >= 20


@dataclass(frozen=True)
class CalendarAnomalyReport:
    by_weekday: list[CalendarSlice]   # Mon-Fri
    by_month: list[CalendarSlice]     # Jan-Dec
    qend_vs_other: tuple[CalendarSlice, CalendarSlice]   # (Q-end, other)
    month_start_vs_other: tuple[CalendarSlice, CalendarSlice]  # (start, other)
    best_weekday: CalendarSlice | None
    worst_weekday: CalendarSlice | None
    best_month: CalendarSlice | None
    worst_month: CalendarSlice | None
    n_trades: int
    verdict: str


_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_QEND_MONTHS = {3, 6, 9, 12}       # last month of each quarter
_QEND_DAYS = set(range(25, 32))     # last ~5 trading days


def _parse_date(ts: str) -> tuple[int, int, int] | None:
    """Return (year, month, day) from ISO timestamp."""
    try:
        t = str(ts).replace("T", " ").strip()
        date_part = t.split(" ")[0]
        y, m, d = date_part.split("-")
        return int(y), int(m), int(d)
    except Exception:
        return None


def _weekday(y: int, m: int, d: int) -> int:
    """0=Mon, 6=Sun using Zeller's formula."""
    if m < 3:
        m += 12
        y -= 1
    k = y % 100
    j = y // 100
    h = (d + (13 * (m + 1)) // 5 + k + k // 4 + j // 4 - 2 * j) % 7
    # h: 0=Sat, 1=Sun, 2=Mon, ..., 6=Fri → convert to 0=Mon
    return (h + 5) % 7


def _slice(label: str, pnls: list[float], min_reliable: int = 20) -> CalendarSlice:
    if not pnls:
        return CalendarSlice(label=label, n_trades=0, win_rate=0.0, avg_pnl_pct=0.0, total_pnl=0.0, reliable=False)
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return CalendarSlice(
        label=label,
        n_trades=n,
        win_rate=round(wins / n * 100, 1),
        avg_pnl_pct=round(sum(pnls) / n, 3),
        total_pnl=round(sum(pnls), 3),
        reliable=n >= min_reliable,
    )


def compute(conn, *, limit: int = 1000, min_reliable: int = 20) -> CalendarAnomalyReport | None:
    """Analyse calendar anomalies from closed trade history.

    conn: SQLite connection.
    limit: max trades to analyse.
    min_reliable: minimum trades per group to mark as reliable.
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct, entered_at
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
              AND entered_at IS NOT NULL
            ORDER BY entered_at DESC LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 10:
        return None

    weekday_pnls: dict[int, list[float]] = {i: [] for i in range(7)}
    month_pnls: dict[int, list[float]] = {i: [] for i in range(1, 13)}
    qend_pnls: list[float] = []
    other_qend_pnls: list[float] = []
    mstart_pnls: list[float] = []
    other_mstart_pnls: list[float] = []

    for pnl, ts in rows:
        pnl = float(pnl)
        date = _parse_date(str(ts))
        if date is None:
            continue
        y, mo, d = date

        wd = _weekday(y, mo, d)
        weekday_pnls[wd].append(pnl)
        month_pnls[mo].append(pnl)

        if mo in _QEND_MONTHS and d in _QEND_DAYS:
            qend_pnls.append(pnl)
        else:
            other_qend_pnls.append(pnl)

        if d <= 3:
            mstart_pnls.append(pnl)
        else:
            other_mstart_pnls.append(pnl)

    weekday_slices = [
        _slice(_WEEKDAY_NAMES[i], weekday_pnls[i], min_reliable)
        for i in range(5)  # Mon-Fri only
    ]

    month_slices = [
        _slice(_MONTH_NAMES[mo - 1], month_pnls[mo], min_reliable)
        for mo in range(1, 13)
    ]

    qend_slice = _slice("Q-end", qend_pnls, min_reliable)
    qother_slice = _slice("Non-Q-end", other_qend_pnls, min_reliable)
    mstart_slice = _slice("Month-start", mstart_pnls, min_reliable)
    mother_slice = _slice("Other days", other_mstart_pnls, min_reliable)

    # Best/worst weekday and month (reliable slices preferred)
    reliable_wd = [s for s in weekday_slices if s.reliable and s.n_trades > 0]
    reliable_mo = [s for s in month_slices if s.reliable and s.n_trades > 0]

    # Fall back to all if no reliable ones
    rank_wd = reliable_wd or [s for s in weekday_slices if s.n_trades > 0]
    rank_mo = reliable_mo or [s for s in month_slices if s.n_trades > 0]

    best_wd = max(rank_wd, key=lambda s: s.avg_pnl_pct) if rank_wd else None
    worst_wd = min(rank_wd, key=lambda s: s.avg_pnl_pct) if rank_wd else None
    best_mo = max(rank_mo, key=lambda s: s.avg_pnl_pct) if rank_mo else None
    worst_mo = min(rank_mo, key=lambda s: s.avg_pnl_pct) if rank_mo else None

    # Verdict
    notes = []
    if best_wd and worst_wd and abs(best_wd.avg_pnl_pct - worst_wd.avg_pnl_pct) > 0.5:
        notes.append(f"best day {best_wd.label} ({best_wd.avg_pnl_pct:+.2f}%), worst {worst_wd.label} ({worst_wd.avg_pnl_pct:+.2f}%)")
    if qend_slice.n_trades >= 5 and other_qend_pnls:
        diff = qend_slice.avg_pnl_pct - qother_slice.avg_pnl_pct
        if abs(diff) > 0.3:
            notes.append(f"Q-end effect {diff:+.2f}% vs normal")
    if mstart_slice.n_trades >= 5:
        diff2 = mstart_slice.avg_pnl_pct - mother_slice.avg_pnl_pct
        if abs(diff2) > 0.3:
            notes.append(f"month-start effect {diff2:+.2f}% vs other days")

    verdict = (
        "Calendar patterns: " + ("; ".join(notes) if notes else "no significant anomalies found") + "."
    )
    if not reliable_wd and not reliable_mo:
        verdict += " (note: most groups have < 20 trades — results are suggestive only)"

    return CalendarAnomalyReport(
        by_weekday=weekday_slices,
        by_month=month_slices,
        qend_vs_other=(qend_slice, qother_slice),
        month_start_vs_other=(mstart_slice, mother_slice),
        best_weekday=best_wd,
        worst_weekday=worst_wd,
        best_month=best_mo,
        worst_month=worst_mo,
        n_trades=len(rows),
        verdict=verdict,
    )
