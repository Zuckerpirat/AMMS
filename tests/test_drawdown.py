from __future__ import annotations

import sqlite3

from amms.risk.drawdown import compute_drawdown, should_alert


def _conn_with_snapshots(snapshots: list[tuple[str, float]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE equity_snapshots ("
        "ts TEXT PRIMARY KEY, equity REAL, cash REAL, buying_power REAL)"
    )
    conn.executemany(
        "INSERT INTO equity_snapshots VALUES (?, ?, 0, 0)", snapshots
    )
    conn.commit()
    return conn


def test_compute_drawdown_uses_peak_in_window() -> None:
    conn = _conn_with_snapshots(
        [
            ("2026-05-01T14:00:00+00:00", 100_000.0),
            ("2026-05-05T14:00:00+00:00", 110_000.0),  # peak
            ("2026-05-10T14:00:00+00:00", 105_000.0),
        ]
    )
    dd = compute_drawdown(conn, current_equity=104_000.0, lookback_days=30)
    assert dd.peak_equity == 110_000.0
    assert dd.current_equity == 104_000.0
    # 104k vs 110k peak ≈ -5.45%
    assert -6.0 < dd.drawdown_pct < -5.0


def test_compute_drawdown_handles_no_snapshots() -> None:
    conn = _conn_with_snapshots([])
    dd = compute_drawdown(conn, current_equity=100_000.0)
    assert dd.peak_equity == 100_000.0
    assert dd.drawdown_pct == 0.0


def test_should_alert_triggers_above_threshold() -> None:
    conn = _conn_with_snapshots(
        [("2026-05-01T14:00:00+00:00", 100_000.0)]
    )
    dd = compute_drawdown(conn, current_equity=92_000.0)
    assert should_alert(dd, threshold_pct=5.0) is True
    assert should_alert(dd, threshold_pct=10.0) is False
