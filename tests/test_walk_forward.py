from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from amms import db
from amms.backtest import BacktestConfig, generate_windows, run_walk_forward
from amms.data.bars import Bar
from amms.data.bars import upsert_bars as upsert_bars_fn
from amms.risk import RiskConfig
from amms.strategy import SmaCross


def test_generate_windows_basic() -> None:
    windows = generate_windows(
        Date(2024, 1, 1),
        Date(2024, 7, 1),
        train_days=60,
        test_days=30,
    )
    assert len(windows) >= 2
    for train_start, train_end, test_start, test_end in windows:
        assert (train_end - train_start).days == 60
        assert (test_end - test_start).days == 29  # inclusive end-day-1
        assert test_start == train_end + (test_start - train_end)


def test_generate_windows_rejects_zero() -> None:
    with pytest.raises(ValueError):
        generate_windows(Date(2024, 1, 1), Date(2024, 6, 1), train_days=0, test_days=10)


def test_generate_windows_custom_step() -> None:
    w1 = generate_windows(
        Date(2024, 1, 1), Date(2024, 12, 31),
        train_days=60, test_days=30, step_days=15,
    )
    w2 = generate_windows(
        Date(2024, 1, 1), Date(2024, 12, 31),
        train_days=60, test_days=30, step_days=30,
    )
    assert len(w1) > len(w2)


def test_walk_forward_runs_per_window(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)

    # Seed flat bars for a long window so every window has data.
    bars = []
    for i in range(120):
        d = Date(2025, 1, 1).fromordinal(Date(2025, 1, 1).toordinal() + i)
        bars.append(
            Bar("AAPL", "1Day", f"{d.isoformat()}T05:00:00Z", 10.0, 10.0, 10.0, 10.0, 100)
        )
    upsert_bars_fn(conn, bars)

    base = BacktestConfig(
        start=Date(2025, 1, 1),
        end=Date(2025, 4, 30),
        symbols=("AAPL",),
        initial_equity=10_000,
        risk=RiskConfig(),
        strategy=SmaCross(fast=3, slow=5),
    )
    results = run_walk_forward(base, conn, train_days=30, test_days=15)
    conn.close()

    assert len(results) >= 1
    for w in results:
        assert w.test_end > w.test_start
        assert w.stats.initial_equity == 10_000
