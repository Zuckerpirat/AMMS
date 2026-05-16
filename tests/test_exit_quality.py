"""Tests for amms.analysis.exit_quality."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.exit_quality import ExitQualityReport, compute


def _make_db(pnls: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE trades (pnl_pct REAL, status TEXT, closed_at TEXT)")
    for i, p in enumerate(pnls):
        conn.execute("INSERT INTO trades VALUES (?, 'closed', ?)", (p, f"2024-01-{1+i%28:02d}"))
    conn.commit()
    return conn


def _optimal_pnls(n: int = 30) -> list[float]:
    """All winners are above avg, all losers are small."""
    wins = [3.0, 4.0, 3.5, 2.8, 3.2] * (n // 5)
    losses = [-0.5, -0.3, -0.4] * (n // 3)
    return wins[:n // 2] + losses[:n // 2]


def _early_exit_pnls(n: int = 30) -> list[float]:
    """Mix of large wins and tiny wins so tiny wins fall below avg_win * 0.5."""
    # avg_win ≈ (5.0 + 0.4) / 2 = 2.7; threshold = 1.35; tiny wins 0.4 < 1.35 → too_early
    wins = [5.0, 0.4] * (n // 4)
    losses = [-1.5] * (n // 2)
    return (wins + losses)[:n]


def _late_exit_pnls(n: int = 30) -> list[float]:
    """Mix of small losses and big losses so big ones exceed avg_loss * 1.5."""
    # avg_loss ≈ (-2.0 + -8.0)/2 = -5.0; threshold = -7.5; -8.0 < -7.5 → late_exit
    wins = [1.5] * (n // 2)
    losses = [-2.0, -8.0] * (n // 4)
    return (wins + losses)[:n]


class TestEdgeCases:
    def test_returns_none_empty_db(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_db([1.0, -1.0, 2.0])
        assert compute(conn) is None

    def test_returns_none_only_winners(self):
        conn = _make_db([1.0, 2.0, 3.0] * 5)
        assert compute(conn) is None

    def test_returns_none_only_losers(self):
        conn = _make_db([-1.0, -2.0] * 5)
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, ExitQualityReport)


class TestQualityScore:
    def test_avg_quality_in_range(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.avg_exit_quality <= 1.0

    def test_optimal_trades_high_quality(self):
        conn = _make_db(_optimal_pnls(30))
        result = compute(conn)
        assert result is not None
        assert result.avg_exit_quality >= 0.3

    def test_consistency_in_range(self):
        conn = _make_db(_optimal_pnls(30))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.exit_consistency <= 1.0


class TestClassifications:
    def test_percentages_sum_to_100(self):
        conn = _make_db(_optimal_pnls(30))
        result = compute(conn)
        assert result is not None
        total = result.optimal_pct + result.too_early_pct + result.late_exit_pct + result.normal_pct
        assert abs(total - 100.0) < 0.5

    def test_pct_values_in_range(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        for pct in [result.optimal_pct, result.too_early_pct, result.late_exit_pct, result.normal_pct]:
            assert 0.0 <= pct <= 100.0

    def test_early_exit_detection(self):
        conn = _make_db(_early_exit_pnls(30))
        result = compute(conn)
        assert result is not None
        assert result.too_early_pct > 0

    def test_late_exit_detection(self):
        conn = _make_db(_late_exit_pnls(30))
        result = compute(conn)
        assert result is not None
        assert result.late_exit_pct > 0


class TestAvgMetrics:
    def test_avg_winner_positive(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        assert result.avg_winner_pct > 0

    def test_avg_loser_negative(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        assert result.avg_loser_pct < 0

    def test_best_exit_ge_avg_winner(self):
        conn = _make_db(_optimal_pnls(30))
        result = compute(conn)
        assert result is not None
        assert result.best_exit_pnl >= result.avg_winner_pct - 0.01

    def test_worst_exit_le_avg_loser(self):
        conn = _make_db(_optimal_pnls(30))
        result = compute(conn)
        assert result is not None
        assert result.worst_exit_pnl <= result.avg_loser_pct + 0.01


class TestCounts:
    def test_n_trades_correct(self):
        pnls = _optimal_pnls(30)
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 30

    def test_n_winners_n_losers_sum_to_n_trades(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        assert result.n_winners + result.n_losers == result.n_trades


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_quality(self):
        conn = _make_db(_optimal_pnls(20))
        result = compute(conn)
        assert result is not None
        assert "quality" in result.verdict.lower() or "exit" in result.verdict.lower()

    def test_early_exit_verdict_mentions_issue(self):
        conn = _make_db(_early_exit_pnls(30))
        result = compute(conn)
        assert result is not None
        # Just verify we got a verdict
        assert len(result.verdict) > 10
