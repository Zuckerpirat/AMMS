"""Walk-forward harness.

Splits a long backtest range into rolling windows (train+test pairs) and
reports per-window stats so a strategy that only looks good in-sample is
exposed before it gets to trade live money — even paper money.

For now, training is a no-op (the strategy has fixed params from config).
The harness still validates stability of returns across out-of-sample
windows, which is the most common failure mode for hand-tuned strategies.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

from amms.backtest.engine import BacktestConfig, run_backtest
from amms.backtest.stats import BacktestStats, compute_stats


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: Date
    train_end: Date
    test_start: Date
    test_end: Date
    stats: BacktestStats


def generate_windows(
    start: Date,
    end: Date,
    *,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[tuple[Date, Date, Date, Date]]:
    """Generate (train_start, train_end, test_start, test_end) tuples."""
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be > 0")
    step = step_days if step_days is not None else test_days
    if step <= 0:
        raise ValueError("step_days must be > 0")
    windows: list[tuple[Date, Date, Date, Date]] = []
    cursor = start
    while True:
        train_end = cursor + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > end:
            break
        windows.append((cursor, train_end, train_end + timedelta(days=1), test_end))
        cursor = cursor + timedelta(days=step)
    return windows


def run_walk_forward(
    base_config: BacktestConfig,
    conn: sqlite3.Connection,
    *,
    train_days: int = 180,
    test_days: int = 30,
    step_days: int | None = None,
) -> list[WalkForwardWindow]:
    """Run a backtest on each out-of-sample window and collect stats."""
    windows = generate_windows(
        base_config.start,
        base_config.end,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    results: list[WalkForwardWindow] = []
    for train_start, train_end, test_start, test_end in windows:
        window_config = BacktestConfig(
            start=test_start,
            end=test_end,
            symbols=base_config.symbols,
            initial_equity=base_config.initial_equity,
            risk=base_config.risk,
            strategy=base_config.strategy,
            timeframe=base_config.timeframe,
            universe=base_config.universe,
        )
        try:
            result = run_backtest(window_config, conn)
        except ValueError:
            # No bars in this window; skip silently.
            continue
        stats = compute_stats(result)
        results.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                stats=stats,
            )
        )
    return results
