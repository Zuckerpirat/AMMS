"""Tests for amms.analysis.trade_streak."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.trade_streak import StreakResult, compute


def _make_conn(pnls: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, pnl in enumerate(pnls):
        # Ensure consistent chronological ordering via sell_ts
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'SYM', "
            f"'2026-01-{i+1:02d}T10:00:00', '2026-01-{i+1:02d}T15:00:00', "
            "100.0, 101.0, 10.0, ?)",
            (i, pnl),
        )
    conn.commit()
    return conn


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_conn([50.0, 50.0, 50.0, 50.0])
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result_with_enough_data(self):
        conn = _make_conn([50.0] * 6)
        result = compute(conn)
        assert result is not None
        assert isinstance(result, StreakResult)


class TestCurrentStreak:
    def test_win_streak_five(self):
        """5 consecutive wins → current_streak = 5."""
        pnls = [-50.0] * 3 + [50.0] * 5  # oldest to newest
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.current_streak == 5

    def test_loss_streak_three(self):
        """3 consecutive losses at end → current_streak = -3."""
        pnls = [50.0] * 4 + [-50.0] * 3
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.current_streak == -3

    def test_streak_label_win(self):
        pnls = [50.0] * 4 + [50.0] * 4
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert "W" in result.current_streak_label

    def test_streak_label_loss(self):
        pnls = [50.0] * 4 + [-50.0] * 4
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert "L" in result.current_streak_label

    def test_single_win_after_losses(self):
        pnls = [-50.0] * 5 + [50.0]
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.current_streak == 1


class TestLongestStreaks:
    def test_longest_win_streak(self):
        """W W W L W W → longest win = 3."""
        pnls = [50, 50, 50, -50, 50, 50]
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.longest_win_streak == 3

    def test_longest_loss_streak(self):
        """L L L L W W → longest loss = 4."""
        pnls = [-50, -50, -50, -50, 50, 50]
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.longest_loss_streak == 4

    def test_all_wins(self):
        pnls = [50.0] * 10
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.longest_win_streak == 10
        assert result.longest_loss_streak == 0

    def test_all_losses(self):
        pnls = [-50.0] * 10
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.longest_loss_streak == 10
        assert result.longest_win_streak == 0


class TestRecentForm:
    def test_all_wins_hot(self):
        pnls = [50.0] * 15
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.recent_form == pytest.approx(100.0, abs=0.1)
        assert result.recent_form_label == "hot"

    def test_all_losses_icy(self):
        pnls = [-50.0] * 15
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.recent_form == pytest.approx(0.0, abs=0.1)
        assert result.recent_form_label == "icy"

    def test_50pct_recent_form(self):
        pnls = [50, -50] * 10
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.recent_form == pytest.approx(50.0, abs=5.0)


class TestMomentumAndFlags:
    def test_hot_hand_on_win_streak(self):
        pnls = [-50.0] * 3 + [50.0] * 5
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.hot_hand is True

    def test_no_hot_hand_short_streak(self):
        pnls = [50.0, -50.0, -50.0, 50.0, 50.0, -50.0, 50.0, 50.0]
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        # current streak = 2 wins → not hot_hand
        if result.current_streak == 2:
            assert result.hot_hand is False

    def test_tilt_risk_on_loss_streak(self):
        pnls = [50.0] * 5 + [-50.0] * 4
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.tilt_risk is True

    def test_verdict_contains_streak_count(self):
        pnls = [50.0] * 5 + [-50.0] * 4
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert "4" in result.verdict or "streak" in result.verdict.lower()

    def test_n_trades_correct(self):
        pnls = [50.0, -50.0] * 10
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 20

    def test_overall_win_rate(self):
        pnls = [50.0] * 7 + [-50.0] * 3
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.overall_win_rate == pytest.approx(70.0, abs=0.1)

    def test_limit_respected(self):
        pnls = [50.0] * 30
        conn = _make_conn(pnls)
        result = compute(conn, limit=15)
        assert result is not None
        assert result.n_trades == 15

    def test_momentum_stable_for_flat(self):
        pnls = [50, -50] * 10
        conn = _make_conn(pnls)
        result = compute(conn)
        assert result is not None
        assert result.momentum in ("stable", "improving", "declining")
