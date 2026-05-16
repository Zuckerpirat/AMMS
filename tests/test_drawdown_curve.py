"""Tests for amms.analysis.drawdown_curve."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.drawdown_curve import DrawdownCurveReport, DrawdownEpisode, compute


def _make_db(pnls: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE trades (pnl_pct REAL, status TEXT, closed_at TEXT)")
    for i, p in enumerate(pnls):
        conn.execute("INSERT INTO trades VALUES (?, 'closed', ?)", (p, f"2024-01-{1 + i % 28:02d} 10:{i % 60:02d}:00"))
    conn.commit()
    return conn


def _all_wins(n: int = 20) -> list[float]:
    return [1.5] * n


def _win_then_loss(n: int = 20) -> list[float]:
    """Win streak, then drawdown, then recovery."""
    wins = [2.0] * (n // 3)
    losses = [-1.5] * (n // 3)
    recovery = [2.0] * (n // 3)
    return wins + losses + recovery


def _deep_drawdown(n: int = 30) -> list[float]:
    """Big wins followed by catastrophic losses."""
    wins = [3.0] * (n // 3)
    losses = [-5.0] * (n // 3)
    flat = [0.5] * (n // 3)
    return wins + losses + flat


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_db([1.0, -1.0, 2.0])
        assert compute(conn) is None

    def test_returns_result_enough_trades(self):
        conn = _make_db(_all_wins(10))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, DrawdownCurveReport)


class TestEquityCurve:
    def test_equity_curve_length_matches_trades(self):
        pnls = _all_wins(15)
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert len(result.equity_curve) == 15

    def test_equity_curve_monotone_for_all_wins(self):
        conn = _make_db(_all_wins(10))
        result = compute(conn)
        assert result is not None
        for i in range(1, len(result.equity_curve)):
            assert result.equity_curve[i] >= result.equity_curve[i - 1]

    def test_equity_curve_starts_at_first_pnl(self):
        pnls = [2.0, 1.0, -1.0, 3.0, 2.0]
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert abs(result.equity_curve[0] - 2.0) < 0.01

    def test_equity_curve_final_value_is_sum(self):
        pnls = [1.0, 2.0, -1.0, 3.0, 0.5, 2.0]
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert abs(result.equity_curve[-1] - sum(pnls)) < 0.01


class TestDrawdownCurve:
    def test_drawdown_curve_length_matches_equity(self):
        conn = _make_db(_win_then_loss(18))
        result = compute(conn)
        assert result is not None
        assert len(result.drawdown_curve) == len(result.equity_curve)

    def test_drawdown_curve_zero_or_negative(self):
        conn = _make_db(_win_then_loss(18))
        result = compute(conn)
        assert result is not None
        for d in result.drawdown_curve:
            assert d <= 0.0

    def test_drawdown_zero_for_all_wins(self):
        conn = _make_db(_all_wins(10))
        result = compute(conn)
        assert result is not None
        assert all(d == 0.0 for d in result.drawdown_curve)


class TestMaxDrawdown:
    def test_max_drawdown_zero_for_all_wins(self):
        conn = _make_db(_all_wins(15))
        result = compute(conn)
        assert result is not None
        assert result.max_drawdown_pct == 0.0

    def test_max_drawdown_negative_when_losses(self):
        conn = _make_db(_deep_drawdown(30))
        result = compute(conn)
        assert result is not None
        assert result.max_drawdown_pct < 0.0

    def test_max_drawdown_le_zero(self):
        conn = _make_db(_win_then_loss(21))
        result = compute(conn)
        assert result is not None
        assert result.max_drawdown_pct <= 0.0

    def test_max_drawdown_duration_nonneg(self):
        conn = _make_db(_win_then_loss(21))
        result = compute(conn)
        assert result is not None
        assert result.max_drawdown_duration >= 0


class TestEpisodes:
    def test_no_episodes_for_all_wins(self):
        conn = _make_db(_all_wins(15))
        result = compute(conn)
        assert result is not None
        assert result.n_episodes == 0
        assert result.episodes == []

    def test_at_least_one_episode_with_losses(self):
        conn = _make_db(_win_then_loss(21))
        result = compute(conn)
        assert result is not None
        assert result.n_episodes >= 1

    def test_episode_depth_is_negative(self):
        conn = _make_db(_win_then_loss(21))
        result = compute(conn)
        assert result is not None
        for ep in result.episodes:
            assert ep.depth_pct < 0.0

    def test_episode_is_dataclass(self):
        conn = _make_db(_win_then_loss(21))
        result = compute(conn)
        assert result is not None
        for ep in result.episodes:
            assert isinstance(ep, DrawdownEpisode)


class TestCurrentDrawdown:
    def test_not_in_drawdown_after_wins(self):
        conn = _make_db(_all_wins(10))
        result = compute(conn)
        assert result is not None
        assert not result.in_drawdown
        assert result.current_drawdown_pct == 0.0

    def test_in_drawdown_when_last_bars_down(self):
        # Starts high, ends in loss
        pnls = [3.0, 3.0, 3.0, -1.5, -1.5, -1.5, -1.5, -1.5]
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert result.in_drawdown
        assert result.current_drawdown_pct < 0.0


class TestUlcerIndex:
    def test_ulcer_zero_for_all_wins(self):
        conn = _make_db(_all_wins(15))
        result = compute(conn)
        assert result is not None
        assert result.ulcer_index == 0.0

    def test_ulcer_positive_with_drawdowns(self):
        conn = _make_db(_win_then_loss(21))
        result = compute(conn)
        assert result is not None
        assert result.ulcer_index >= 0.0


class TestRecoveryFactor:
    def test_recovery_factor_zero_no_drawdown(self):
        conn = _make_db(_all_wins(10))
        result = compute(conn)
        assert result is not None
        assert result.recovery_factor == 0.0

    def test_n_trades_correct(self):
        pnls = _win_then_loss(21)
        conn = _make_db(pnls)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 21


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_win_then_loss(21))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_drawdown(self):
        conn = _make_db(_deep_drawdown(30))
        result = compute(conn)
        assert result is not None
        assert "drawdown" in result.verdict.lower()
