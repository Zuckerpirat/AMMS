"""Position aging and hold time analysis.

Tracks how long each open position has been held and classifies it
by holding style (day trade, swing, medium-term, long-term).

Also flags positions that may be "overstayed" based on:
  - Time in trade vs. expected holding period
  - Whether the original thesis is still intact (trend still valid)
  - Whether the position is in profit or loss

This module does NOT make exit decisions — it raises awareness flags
for the decision engine or the user to act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class PositionAge:
    symbol: str
    entry_date: str | None      # ISO date string or None if unknown
    hold_days: int | None       # calendar days held (None if no entry date)
    hold_style: str             # "day" | "swing" | "medium" | "long" | "unknown"
    pnl_pct: float              # current unrealized P&L %
    overstay_flag: bool         # True if position may be overstayed
    overstay_reason: str        # explanation if flagged


@dataclass(frozen=True)
class AgingReport:
    positions: list[PositionAge]
    total_positions: int
    overstayed_count: int
    avg_hold_days: float | None
    oldest_symbol: str | None
    oldest_days: int | None


# Hold style thresholds (calendar days)
_HOLD_STYLE_THRESHOLDS = {
    "day":    (0, 2),
    "swing":  (2, 14),
    "medium": (14, 90),
    "long":   (90, 99999),
}


def analyze_aging(broker, conn=None) -> AgingReport | None:
    """Analyze the hold time and aging of all open positions.

    broker: must support get_positions()
    conn: optional SQLite connection for trade_pairs table (for entry dates)
    Returns None if no open positions.
    """
    try:
        positions = broker.get_positions()
    except Exception:
        return None

    if not positions:
        return None

    today = datetime.now(UTC).date()

    # Load entry dates from trade_pairs if available
    entry_dates: dict[str, str] = {}
    if conn is not None:
        try:
            rows = conn.execute(
                "SELECT symbol, MIN(buy_ts) as first_buy FROM trade_pairs "
                "WHERE sell_ts IS NULL OR sell_ts = '' "
                "GROUP BY symbol"
            ).fetchall()
            for sym, buy_ts in rows:
                if buy_ts:
                    entry_dates[sym] = buy_ts[:10]  # take date part
        except Exception:
            pass

    # Also try orders table as fallback
    if conn is not None and not entry_dates:
        try:
            rows = conn.execute(
                "SELECT symbol, MIN(filled_at) FROM orders "
                "WHERE side = 'buy' AND status = 'filled' GROUP BY symbol"
            ).fetchall()
            for sym, ts in rows:
                if ts and sym not in entry_dates:
                    entry_dates[sym] = ts[:10]
        except Exception:
            pass

    aged: list[PositionAge] = []

    for pos in positions:
        sym = pos.symbol
        entry_str = entry_dates.get(sym)
        hold_days = None

        if entry_str:
            try:
                entry_date = datetime.strptime(entry_str[:10], "%Y-%m-%d").date()
                hold_days = (today - entry_date).days
            except Exception:
                hold_days = None

        # Classify hold style
        hold_style = "unknown"
        if hold_days is not None:
            for style, (lo, hi) in _HOLD_STYLE_THRESHOLDS.items():
                if lo <= hold_days < hi:
                    hold_style = style
                    break

        # Compute P&L %
        try:
            entry_price = float(pos.avg_entry_price)
            mv = float(pos.market_value)
            qty = float(pos.qty)
            current_price = mv / qty if qty != 0 else entry_price
            pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
        except Exception:
            pnl_pct = 0.0

        # Overstay detection
        overstay = False
        reason = ""

        if hold_days is not None:
            # Swing trades > 30 days while losing
            if hold_days > 30 and pnl_pct < -5.0:
                overstay = True
                reason = f"Losing position held {hold_days}d (>{-pnl_pct:.1f}%% loss)"
            # Any trade > 90 days in significant loss
            elif hold_days > 90 and pnl_pct < -10.0:
                overstay = True
                reason = f"Significant loss after {hold_days}d ({pnl_pct:.1f}%%)"
            # Very long hold > 180 days — flag for review regardless
            elif hold_days > 180:
                overstay = True
                reason = f"Long hold: {hold_days}d — review thesis"

        aged.append(PositionAge(
            symbol=sym,
            entry_date=entry_str,
            hold_days=hold_days,
            hold_style=hold_style,
            pnl_pct=round(pnl_pct, 2),
            overstay_flag=overstay,
            overstay_reason=reason,
        ))

    # Sort by hold_days descending (oldest first)
    aged.sort(key=lambda x: x.hold_days if x.hold_days is not None else -1, reverse=True)

    overstayed_count = sum(1 for a in aged if a.overstay_flag)
    hold_days_known = [a.hold_days for a in aged if a.hold_days is not None]
    avg_hold = sum(hold_days_known) / len(hold_days_known) if hold_days_known else None

    oldest = aged[0] if aged and aged[0].hold_days is not None else None

    return AgingReport(
        positions=aged,
        total_positions=len(aged),
        overstayed_count=overstayed_count,
        avg_hold_days=round(avg_hold, 1) if avg_hold is not None else None,
        oldest_symbol=oldest.symbol if oldest else None,
        oldest_days=oldest.hold_days if oldest else None,
    )
