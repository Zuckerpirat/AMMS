"""Tests for amms.analysis.scorecard."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.scorecard import MetricScore, ScorecardReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (pnl, buy_price, qty)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (pnl, buy_price, qty) in enumerate(rows):
        month = (i // 5) + 1
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'AAPL', '2026-01-01T09:00:00', ?, ?, 105.0, ?, ?)",
            (i, f"2026-{month:02d}-{(i%5)+1:02d}T14:00:00", buy_price, qty, pnl),
        )
    conn.commit()
    return conn


def _winning_rows(n: int = 20) -> list[tuple]:
    """Strong winning system: 70% win rate, avg win 2%, avg loss 1%."""
    rows = []
    for i in range(n):
        if i % 10 < 7:
            rows.append((200.0, 100.0, 10.0))   # win: $200 on $1000 = 2%
        else:
            rows.append((-100.0, 100.0, 10.0))  # loss: -$100 on $1000 = -1%
    return rows


def _losing_rows(n: int = 20) -> list[tuple]:
    """Losing system: 30% win rate."""
    rows = []
    for i in range(n):
        if i % 10 < 3:
            rows.append((100.0, 100.0, 10.0))
        else:
            rows.append((-200.0, 100.0, 10.0))
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [(50.0, 100.0, 10.0) for _ in range(5)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result, ScorecardReport)


class TestScore:
    def test_score_in_range(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert 0 <= result.overall_score <= 100

    def test_grade_valid(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_winning_system_higher_score(self):
        win_conn = _make_conn(_winning_rows(20))
        lose_conn = _make_conn(_losing_rows(20))
        win_result = compute(win_conn)
        lose_result = compute(lose_conn)
        assert win_result is not None
        assert lose_result is not None
        assert win_result.overall_score > lose_result.overall_score

    def test_winning_gets_good_grade(self):
        conn = _make_conn(_winning_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.grade in ("A", "B", "C")

    def test_losing_gets_poor_grade(self):
        conn = _make_conn(_losing_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.grade in ("D", "F")


class TestMetrics:
    def test_has_metrics(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.metrics) > 0

    def test_metric_scores_in_range(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        for m in result.metrics:
            assert 0 <= m.score <= 100

    def test_metric_grades_valid(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        for m in result.metrics:
            assert m.grade in ("A", "B", "C", "D", "F")

    def test_metric_weights_sum(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        total_weight = sum(m.weight for m in result.metrics)
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_n_trades_correct(self):
        conn = _make_conn(_winning_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 20


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_grade_in_verdict(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert result.grade in result.verdict
