"""Tests for amms.analysis.trade_timing."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from amms.analysis.trade_timing import TimeBucket, TradeTimingReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (buy_ts, pnl, buy_price, qty)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (buy_ts, pnl, buy_price, qty) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'SYM', ?, '2026-01-10T12:00:00', ?, 105.0, ?, ?)",
            (i, buy_ts, buy_price, qty, pnl),
        )
    conn.commit()
    return conn


def _monday(hour: int = 10) -> str:
    # 2026-01-05 is a Monday
    return f"2026-01-05T{hour:02d}:00:00"


def _tuesday(hour: int = 10) -> str:
    return f"2026-01-06T{hour:02d}:00:00"


def _wednesday(hour: int = 10) -> str:
    return f"2026-01-07T{hour:02d}:00:00"


def _thursday(hour: int = 10) -> str:
    return f"2026-01-08T{hour:02d}:00:00"


def _friday(hour: int = 10) -> str:
    return f"2026-01-09T{hour:02d}:00:00"


def _make_rows(n: int = 20) -> list[tuple]:
    """Create n rows cycling through weekdays and hours."""
    days = [_monday, _tuesday, _wednesday, _thursday, _friday]
    rows = []
    for i in range(n):
        day_fn = days[i % 5]
        hour = 9 + (i % 7)
        pnl = 50.0 if i % 2 == 0 else -30.0
        rows.append((day_fn(hour), pnl, 100.0, 10.0))
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [(_monday(), 50.0, 100.0, 10.0) for _ in range(5)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, TradeTimingReport)


class TestByWeekday:
    def test_weekday_buckets_present(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.by_weekday) > 0

    def test_weekday_labels(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        for b in result.by_weekday:
            assert b.label in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

    def test_win_rate_in_range(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        for b in result.by_weekday:
            assert 0 <= b.win_rate <= 100

    def test_n_trades_sum_correct(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        total = sum(b.n_trades for b in result.by_weekday)
        assert total == result.n_trades

    def test_best_worst_weekday_present(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.best_weekday is not None
        assert result.worst_weekday is not None


class TestByHour:
    def test_hour_buckets_present(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.by_hour) > 0

    def test_hour_labels_format(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        for b in result.by_hour:
            assert ":" in b.label
            assert b.label.endswith(":00")

    def test_best_hour_present(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.best_hour is not None


class TestSignalQuality:
    def test_monday_winner(self):
        """All Monday trades win, all others lose → Monday is best weekday."""
        rows = []
        for i in range(5):
            rows.append((_monday(10), 100.0, 100.0, 10.0))
        for i in range(5):
            rows.append((_tuesday(10), -50.0, 100.0, 10.0))
        for i in range(5):
            rows.append((_wednesday(10), -50.0, 100.0, 10.0))
        # Pad to 10+ trades
        for i in range(5):
            rows.append((_thursday(10), -20.0, 100.0, 10.0))
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.best_weekday == "Monday"

    def test_limit_respected(self):
        rows = _make_rows(50)
        conn = _make_conn(rows)
        result = compute(conn, limit=20)
        assert result is not None
        assert result.n_trades == 20


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_make_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 5
