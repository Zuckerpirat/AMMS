"""Tests for amms.analysis.kelly_sizer."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.kelly_sizer import KellyResult, compute


def _make_conn(pnls: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, buy_price REAL, "
        "sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, pnl in enumerate(pnls):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'SYM', "
            "'2026-01-01T10:00:00', '2026-01-04T10:00:00', "
            "100.0, 101.0, 10.0, ?)",
            (i, pnl),
        )
    conn.commit()
    return conn


class TestComputeEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_conn([50.0, 50.0, 50.0, 50.0])
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_none_all_wins(self):
        """No losses → cannot compute Kelly."""
        conn = _make_conn([50.0] * 10)
        assert compute(conn) is None

    def test_returns_none_all_losses(self):
        """No wins → cannot compute Kelly."""
        conn = _make_conn([-50.0] * 10)
        assert compute(conn) is None


class TestKellyFormula:
    def _make_50pct_2r(self):
        """50% win rate, 2:1 payoff → Kelly = (0.5*2 - 0.5)/2 = 0.25 = 25%."""
        wins = [200.0] * 5
        losses = [-100.0] * 5
        conn = _make_conn(wins + losses)
        return compute(conn)

    def test_returns_result(self):
        assert self._make_50pct_2r() is not None

    def test_win_rate_correct(self):
        r = self._make_50pct_2r()
        assert r.win_rate == pytest.approx(50.0, abs=0.1)

    def test_payoff_ratio_correct(self):
        r = self._make_50pct_2r()
        assert r.payoff_ratio == pytest.approx(2.0, abs=0.01)

    def test_kelly_pct_capped(self):
        """25% full Kelly → capped at 25%."""
        r = self._make_50pct_2r()
        assert r.kelly_pct == pytest.approx(25.0, abs=0.1)

    def test_half_kelly_is_half(self):
        r = self._make_50pct_2r()
        assert r.half_kelly_pct == pytest.approx(r.kelly_pct * 0.5, abs=0.01)

    def test_quarter_kelly_is_quarter(self):
        r = self._make_50pct_2r()
        assert r.quarter_kelly_pct == pytest.approx(r.kelly_pct * 0.25, abs=0.01)

    def test_negative_edge_gives_zero_kelly(self):
        """30% win rate, 1:1 payoff → negative Kelly → clamped to 0."""
        wins = [100.0] * 3
        losses = [-100.0] * 7
        conn = _make_conn(wins + losses)
        r = compute(conn)
        assert r is not None
        assert r.kelly_pct == 0.0

    def test_edge_positive_for_good_system(self):
        """50% win, 2:1 payoff → edge = 0.5*200 - 0.5*100 = 50."""
        r = self._make_50pct_2r()
        assert r.edge > 0

    def test_edge_negative_for_bad_system(self):
        """30% win, 1:1 → edge < 0."""
        wins = [100.0] * 3
        losses = [-100.0] * 7
        conn = _make_conn(wins + losses)
        r = compute(conn)
        assert r is not None
        assert r.edge < 0

    def test_n_trades_correct(self):
        r = self._make_50pct_2r()
        assert r.n_trades == 10

    def test_n_wins_losses_correct(self):
        r = self._make_50pct_2r()
        assert r.n_wins == 5
        assert r.n_losses == 5


class TestSuggestions:
    def _result_with_portfolio(self, portfolio=10_000, price=100.0):
        wins = [200.0] * 6
        losses = [-100.0] * 4
        conn = _make_conn(wins + losses)
        return compute(conn, portfolio_value=portfolio, current_price=price)

    def test_suggested_value_present(self):
        r = self._result_with_portfolio()
        assert r.suggested_value is not None
        assert r.suggested_value > 0

    def test_suggested_shares_present(self):
        r = self._result_with_portfolio()
        assert r.suggested_shares is not None
        assert r.suggested_shares >= 0

    def test_suggested_value_within_portfolio(self):
        r = self._result_with_portfolio(portfolio=10_000)
        assert r.suggested_value <= 10_000

    def test_suggested_shares_consistent_with_value(self):
        r = self._result_with_portfolio(portfolio=10_000, price=100.0)
        assert r.suggested_shares == int(r.suggested_value / 100.0)

    def test_no_portfolio_no_suggestion(self):
        wins = [200.0] * 6
        losses = [-100.0] * 4
        conn = _make_conn(wins + losses)
        r = compute(conn)
        assert r.suggested_shares is None
        assert r.suggested_value is None

    def test_zero_kelly_zero_value(self):
        """Negative edge → kelly=0 → no suggested_value."""
        wins = [100.0] * 3
        losses = [-100.0] * 7
        conn = _make_conn(wins + losses)
        r = compute(conn, portfolio_value=10_000, current_price=100.0)
        assert r is not None
        assert r.suggested_value is None or r.suggested_value == 0.0


class TestGradeAndNote:
    def test_grade_a_high_edge(self):
        """Large wins relative to losses → A grade."""
        wins = [300.0] * 7
        losses = [-50.0] * 3
        conn = _make_conn(wins + losses)
        r = compute(conn)
        assert r is not None
        assert r.grade in ("A", "B")

    def test_grade_f_negative_edge(self):
        wins = [100.0] * 3
        losses = [-100.0] * 7
        conn = _make_conn(wins + losses)
        r = compute(conn)
        assert r is not None
        assert r.grade == "F"

    def test_note_present(self):
        wins = [200.0] * 5
        losses = [-100.0] * 5
        conn = _make_conn(wins + losses)
        r = compute(conn)
        assert r is not None
        assert len(r.note) > 0

    def test_limit_respected(self):
        pnls = [50.0 if i % 2 == 0 else -30.0 for i in range(50)]
        conn = _make_conn(pnls)
        r = compute(conn, limit=10)
        assert r is not None
        assert r.n_trades == 10
