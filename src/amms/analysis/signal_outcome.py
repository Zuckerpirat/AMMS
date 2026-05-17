"""Signal outcome analysis: did DE signals predict price direction correctly?

Compares recorded DE signals (from de_signal_history) to actual price
movement N bars later, computing directional accuracy per mode and action.

Only signals with a recorded price are evaluated. Signals less than
`min_age_days` old are excluded (need time to play out).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_BUY_ACTIONS = {"buy", "strong_buy"}
_SELL_ACTIONS = {"sell", "strong_sell"}
_DIRECTIONAL_ACTIONS = _BUY_ACTIONS | _SELL_ACTIONS


@dataclass
class OutcomeStats:
    mode: str
    action: str
    total: int = 0
    correct: int = 0
    avg_score: float = 0.0
    avg_confidence: float = 0.0
    avg_return_pct: float = 0.0  # mean price change (+ = up, - = down)

    @property
    def accuracy_pct(self) -> float:
        return self.correct / self.total * 100.0 if self.total > 0 else 0.0


@dataclass
class SignalOutcomeReport:
    """Aggregated signal outcome results."""
    stats: list[OutcomeStats] = field(default_factory=list)
    total_evaluated: int = 0
    total_skipped: int = 0   # too new / no price / symbol data unavailable
    lookback_days: int = 5

    def summary_lines(self) -> list[str]:
        if not self.stats:
            return ["No signal outcomes to report."]
        lines = [
            f"── Signal Outcome Report (lookback={self.lookback_days}d) ──",
            f"  Evaluated: {self.total_evaluated}  |  Skipped (too new): {self.total_skipped}",
            "",
        ]
        for s in sorted(self.stats, key=lambda x: (-x.total, x.mode)):
            bar = "█" * int(s.accuracy_pct / 10)
            lines.append(
                f"  {s.mode:<12} {s.action:<12} "
                f"n={s.total:>3}  acc={s.accuracy_pct:>5.1f}%  {bar}"
            )
            lines.append(
                f"             avg score={s.avg_score:>+.0f}  "
                f"conf={s.avg_confidence:.0%}  "
                f"avg_ret={s.avg_return_pct:>+.2f}%"
            )
        return lines


def _find_outcome_price(bars, signal_ts: str, lookback_days: int) -> float | None:
    """Find the bar close approximately `lookback_days` after signal_ts.

    bars must be sorted oldest→newest. Returns None if no suitable bar found.
    """
    try:
        sig_dt = datetime.fromisoformat(signal_ts)
        if sig_dt.tzinfo is None:
            sig_dt = sig_dt.replace(tzinfo=timezone.utc)
        target_dt = sig_dt + timedelta(days=lookback_days)
    except (ValueError, TypeError):
        return None

    # Find first bar at or after target_dt
    for bar in bars:
        try:
            bar_dt = datetime.fromisoformat(str(bar.ts).rstrip("Z"))
            if bar_dt.tzinfo is None:
                bar_dt = bar_dt.replace(tzinfo=timezone.utc)
            if bar_dt >= target_dt:
                return float(bar.close)
        except Exception:
            continue
    # If target is in the future, return the most recent close
    if bars:
        return float(bars[-1].close)
    return None


def compute_signal_outcomes(
    conn,
    data_client,
    *,
    lookback_days: int = 5,
    min_age_days: int = 2,
    limit: int = 200,
) -> SignalOutcomeReport:
    """Evaluate historical DE signals against actual price outcomes.

    Args:
        conn: SQLite connection with de_signal_history table.
        data_client: market data client with get_bars(symbol, limit=N).
        lookback_days: how many days after the signal to measure price.
        min_age_days: exclude signals newer than this (haven't played out yet).
        limit: max number of signals to evaluate.

    Returns:
        SignalOutcomeReport with per-mode/action accuracy stats.
    """
    from amms.data.signal_history import fetch_recent

    report = SignalOutcomeReport(lookback_days=lookback_days)
    if conn is None or data_client is None:
        return report

    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
    signals = fetch_recent(conn, limit=limit)
    # Only signals with a price and old enough to have played out
    eligible = [
        s for s in signals
        if s.price and s.action in _DIRECTIONAL_ACTIONS and s.ts < cutoff
    ]

    # Cache bars per symbol to avoid redundant fetches
    bars_cache: dict[str, list] = {}
    stats_map: dict[tuple[str, str], OutcomeStats] = {}

    for sig in eligible:
        sym = sig.symbol
        if sym not in bars_cache:
            try:
                bars_cache[sym] = data_client.get_bars(sym, limit=400)
            except Exception:
                bars_cache[sym] = []

        bars = bars_cache[sym]
        if not bars:
            report.total_skipped += 1
            continue

        outcome_price = _find_outcome_price(bars, sig.ts, lookback_days)
        if outcome_price is None:
            report.total_skipped += 1
            continue

        entry_price = sig.price
        ret_pct = (outcome_price / entry_price - 1.0) * 100.0

        # Directional correctness
        if sig.action in _BUY_ACTIONS:
            correct = outcome_price > entry_price
        else:
            correct = outcome_price < entry_price

        key = (sig.mode, sig.action)
        if key not in stats_map:
            stats_map[key] = OutcomeStats(mode=sig.mode, action=sig.action)

        s = stats_map[key]
        # Running averages (incremental)
        n = s.total
        s.avg_score = (s.avg_score * n + sig.score) / (n + 1)
        s.avg_confidence = (s.avg_confidence * n + sig.confidence) / (n + 1)
        s.avg_return_pct = (s.avg_return_pct * n + ret_pct) / (n + 1)
        s.total += 1
        if correct:
            s.correct += 1
        report.total_evaluated += 1

    report.stats = sorted(stats_map.values(), key=lambda x: x.mode)
    return report


def format_outcome_report(report: SignalOutcomeReport) -> str:
    return "\n".join(report.summary_lines())
