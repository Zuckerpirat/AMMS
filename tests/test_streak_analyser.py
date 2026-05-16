"""Tests for amms.analysis.streak_analyser."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.streak_analyser import StreakAnalyserReport, StreakRun, compute


def _make_db(pnls: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE trades (pnl_pct REAL, status TEXT, closed_at TEXT)")
    for i, p in enumerate(pnls):
        conn.execute("INSERT INTO trades VALUES (?, 'closed', ?)", (p, f"2024-01-{1 + i % 28:02d} {i % 24:02d}:00:00"))
    conn.commit()
    return conn


def _alternating(n: int = 20) -> list[float]:
    """Win/loss/win/loss..."""
    return [1.0 if i % 2 == 0 else -1.0 for i in range(n)]


def _hot_hand(n: int = 30) -> list[float]:
    """Long win streaks separated by single losses (strong within-streak transitions)."""
    # 10 wins → 2 losses → 10 wins → 2 losses ...
    # P(win|prev win) = 9/10 per block >> baseline ≈ 10/12
    pattern = [1.5] * 10 + [-0.5] * 2
    return (pattern * (n // len(pattern) + 1))[:n]


def _cold_hand(n: int = 30) -> list[float]:
    """Lots of loss clusters."""
    pattern = [-1.5, -1.5, -1.5, 0.5, -1.5, -1.5, -1.5, 0.5]
    return (pattern * (n // len(pattern) + 1))[:n]


def _long_streak(n: int = 30) -> list[float]:
    """Long winning streak then losses."""
    return [2.0] * 10 + [-1.0] * 5 + [2.0] * (n - 15)


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_db([1.0, -1.0, 2.0])
        assert compute(conn) is None

    def test_returns_result_enough_trades(self):
        conn = _make_db(_alternating(20))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, StreakAnalyserReport)


class TestStreakIdentification:
    def test_alternating_all_streaks_length_one(self):
        conn = _make_db(_alternating(20))
        result = compute(conn)
        assert result is not None
        for s in result.all_streaks:
            assert s.length == 1

    def test_long_streak_detected(self):
        conn = _make_db(_long_streak(30))
        result = compute(conn)
        assert result is not None
        assert result.longest_win_streak is not None
        assert result.longest_win_streak.length >= 10

    def test_all_wins_no_loss_streak(self):
        conn = _make_db([1.0] * 15)
        result = compute(conn)
        assert result is not None
        assert result.longest_loss_streak is None
        assert result.n_loss_streaks == 0

    def test_all_losses_no_win_streak(self):
        conn = _make_db([-1.0] * 15)
        result = compute(conn)
        assert result is not None
        assert result.longest_win_streak is None
        assert result.n_win_streaks == 0


class TestStreakDataclass:
    def test_longest_win_streak_is_run(self):
        conn = _make_db(_long_streak(30))
        result = compute(conn)
        assert result is not None
        assert isinstance(result.longest_win_streak, StreakRun)

    def test_streak_kind_is_win_or_loss(self):
        conn = _make_db(_hot_hand(30))
        result = compute(conn)
        assert result is not None
        for s in result.all_streaks:
            assert s.kind in {"win", "loss"}

    def test_streak_length_positive(self):
        conn = _make_db(_alternating(20))
        result = compute(conn)
        assert result is not None
        for s in result.all_streaks:
            assert s.length >= 1


class TestCurrentStreak:
    def test_current_streak_present(self):
        conn = _make_db(_alternating(20))
        result = compute(conn)
        assert result is not None
        assert result.current_streak is not None

    def test_current_streak_matches_last_trades(self):
        pnls = [1.0] * 10 + [-1.0] * 3  # ends on 3-loss streak
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert result.current_streak is not None
        assert result.current_streak.kind == "loss"
        assert result.current_streak.length == 3


class TestConditionalProbabilities:
    def test_baseline_win_rate_in_range(self):
        conn = _make_db(_alternating(20))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.baseline_win_rate <= 1.0

    def test_p_win_after_win_in_range(self):
        conn = _make_db(_hot_hand(30))
        result = compute(conn)
        assert result is not None
        if result.p_win_after_win is not None:
            assert 0.0 <= result.p_win_after_win <= 1.0

    def test_p_loss_after_loss_in_range(self):
        conn = _make_db(_cold_hand(30))
        result = compute(conn)
        assert result is not None
        if result.p_loss_after_loss is not None:
            assert 0.0 <= result.p_loss_after_loss <= 1.0

    def test_hot_hand_positive_for_clustered_wins(self):
        conn = _make_db(_hot_hand(30))
        result = compute(conn)
        assert result is not None
        # With clustered wins, p_win_after_win > baseline
        if result.hot_hand_effect is not None:
            assert result.hot_hand_effect > 0

    def test_alternating_hot_hand_negative(self):
        conn = _make_db(_alternating(20))
        result = compute(conn)
        assert result is not None
        # With perfect alternation, P(win|prev win) = 0 < baseline ≈ 0.5
        if result.hot_hand_effect is not None:
            assert result.hot_hand_effect < 0


class TestAverages:
    def test_avg_win_streak_len_ge_one(self):
        conn = _make_db(_hot_hand(30))
        result = compute(conn)
        assert result is not None
        if result.n_win_streaks > 0:
            assert result.avg_win_streak_len >= 1.0

    def test_avg_loss_streak_len_ge_one(self):
        conn = _make_db(_cold_hand(30))
        result = compute(conn)
        assert result is not None
        if result.n_loss_streaks > 0:
            assert result.avg_loss_streak_len >= 1.0


class TestNTrades:
    def test_n_trades_correct(self):
        pnls = _alternating(20)
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 20


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_alternating(20))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_streak(self):
        conn = _make_db(_long_streak(30))
        result = compute(conn)
        assert result is not None
        assert "streak" in result.verdict.lower()
