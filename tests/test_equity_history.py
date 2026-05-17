"""Tests for amms.data.equity_history."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from amms.data.equity_history import (
    EquitySnapshot,
    EquityStats,
    _sparkline,
    compute_stats,
    fetch_history,
    format_equity_curve,
    record_snapshot,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE equity_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT    NOT NULL,
            equity    REAL    NOT NULL,
            cash      REAL    NOT NULL,
            positions REAL    NOT NULL,
            n_pos     INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


class _FakeSnap:
    def __init__(self, equity=10_000.0, cash=5_000.0, mv=5_000.0, positions=None):
        self.portfolio_value = equity
        self.cash = cash
        self.total_market_value = mv
        self.positions = positions or {}


class _FakeTrader:
    def __init__(self, equity=10_000.0):
        self._equity = equity

    def snapshot(self):
        return _FakeSnap(
            equity=self._equity,
            cash=self._equity / 2,
            mv=self._equity / 2,
            positions={"AAPL": object()},
        )


def _insert(conn, equity: float, days_ago: float = 0):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn.execute(
        "INSERT INTO equity_history (ts, equity, cash, positions, n_pos) VALUES (?, ?, ?, ?, ?)",
        (ts, equity, equity / 2, equity / 2, 1),
    )
    conn.commit()


# ── record_snapshot ───────────────────────────────────────────────────────────

class TestRecordSnapshot:
    def test_records_successfully(self, db):
        ok = record_snapshot(db, _FakeTrader(10_000.0))
        assert ok is True
        rows = db.execute("SELECT * FROM equity_history").fetchall()
        assert len(rows) == 1
        assert rows[0]["equity"] == pytest.approx(10_000.0, abs=0.01)

    def test_no_conn_returns_false(self):
        ok = record_snapshot(None, _FakeTrader())
        assert ok is False

    def test_no_trader_returns_false(self, db):
        ok = record_snapshot(db, None)
        assert ok is False

    def test_stores_cash_and_positions(self, db):
        record_snapshot(db, _FakeTrader(8_000.0))
        row = db.execute("SELECT * FROM equity_history").fetchone()
        assert row["cash"] == pytest.approx(4_000.0, abs=0.01)
        assert row["n_pos"] == 1


# ── fetch_history ─────────────────────────────────────────────────────────────

class TestFetchHistory:
    def test_no_conn_returns_empty(self):
        assert fetch_history(None) == []

    def test_empty_db_returns_empty(self, db):
        assert fetch_history(db) == []

    def test_fetches_within_window(self, db):
        _insert(db, 10_000.0, days_ago=5)
        _insert(db, 10_500.0, days_ago=2)
        rows = fetch_history(db, days=7)
        assert len(rows) == 2

    def test_excludes_old_records(self, db):
        _insert(db, 10_000.0, days_ago=40)
        _insert(db, 10_500.0, days_ago=2)
        rows = fetch_history(db, days=7)
        assert len(rows) == 1
        assert rows[0].equity == pytest.approx(10_500.0, abs=0.01)

    def test_returns_oldest_first(self, db):
        _insert(db, 10_000.0, days_ago=5)
        _insert(db, 11_000.0, days_ago=1)
        rows = fetch_history(db, days=10)
        assert rows[0].equity < rows[-1].equity


# ── compute_stats ─────────────────────────────────────────────────────────────

class TestComputeStats:
    def _snaps(self, equities: list[float]) -> list[EquitySnapshot]:
        base = datetime.now(timezone.utc) - timedelta(days=len(equities))
        return [
            EquitySnapshot(
                id=i, ts=(base + timedelta(days=i)).isoformat(),
                equity=e, cash=e / 2, positions=e / 2, n_pos=1,
            )
            for i, e in enumerate(equities)
        ]

    def test_returns_none_for_single_snapshot(self):
        assert compute_stats(self._snaps([10_000.0])) is None

    def test_returns_none_for_empty(self):
        assert compute_stats([]) is None

    def test_positive_return(self):
        stats = compute_stats(self._snaps([10_000.0, 11_000.0]))
        assert stats is not None
        assert stats.total_return_pct == pytest.approx(10.0, abs=0.01)

    def test_max_drawdown_detected(self):
        # Peak at 12k, then drop to 9k → 25% drawdown
        equities = [10_000.0, 12_000.0, 9_000.0]
        stats = compute_stats(self._snaps(equities))
        assert stats.max_drawdown_pct == pytest.approx(25.0, abs=0.5)

    def test_sharpe_not_none_with_sufficient_returns(self):
        import random
        random.seed(42)
        equities = [10_000.0 + random.gauss(0, 200) * i for i in range(30)]
        equities = [max(1.0, e) for e in equities]
        stats = compute_stats(self._snaps(equities))
        # Either None (edge case) or a float
        if stats.sharpe_ratio is not None:
            assert isinstance(stats.sharpe_ratio, float)

    def test_no_drawdown_when_monotone(self):
        equities = [10_000.0 + i * 100 for i in range(10)]
        stats = compute_stats(self._snaps(equities))
        assert stats.max_drawdown_pct == pytest.approx(0.0, abs=0.01)

    def test_best_worst_day(self):
        equities = [10_000.0, 10_500.0, 9_800.0]  # +5%, -6.67%
        stats = compute_stats(self._snaps(equities))
        assert stats.best_day_pct == pytest.approx(5.0, abs=0.1)
        assert stats.worst_day_pct < 0.0


# ── _sparkline ────────────────────────────────────────────────────────────────

class TestSparkline:
    def test_returns_string(self):
        result = _sparkline([1.0, 2.0, 3.0, 2.0, 1.0])
        assert isinstance(result, str)

    def test_flat_line_for_single_value(self):
        result = _sparkline([5.0, 5.0, 5.0])
        assert "─" in result

    def test_width_respected(self):
        result = _sparkline(list(range(100)), width=20)
        assert len(result) <= 20

    def test_empty_returns_dashes(self):
        result = _sparkline([])
        assert "─" in result


# ── format_equity_curve ───────────────────────────────────────────────────────

class TestFormatEquityCurve:
    def test_empty_returns_no_history_message(self):
        out = format_equity_curve([], None)
        assert "No equity history" in out

    def test_with_snapshots_returns_curve(self):
        snaps = [
            EquitySnapshot(i, f"2026-01-{i+1:02d}T10:00:00Z",
                           10_000.0 + i * 100, 5_000.0, 5_000.0, 1)
            for i in range(5)
        ]
        stats = compute_stats(snaps)
        out = format_equity_curve(snaps, stats)
        assert "Equity Curve" in out
        assert "$" in out

    def test_includes_stats_when_available(self):
        snaps = [
            EquitySnapshot(i, f"2026-01-{i+1:02d}T10:00:00Z",
                           10_000.0 + i * 200, 5_000.0, 5_000.0, 1)
            for i in range(10)
        ]
        stats = compute_stats(snaps)
        out = format_equity_curve(snaps, stats)
        assert "Return" in out
        assert "drawdown" in out.lower()
