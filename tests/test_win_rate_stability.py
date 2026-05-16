"""Tests for amms.analysis.win_rate_stability."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.win_rate_stability import WinRateWindow, WinRateStabilityReport, compute


def _make_conn(pnls: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, pnl in enumerate(pnls):
        day = (i % 28) + 1
        month = (i // 28) + 1
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'AAPL', '2026-01-01T09:00:00', ?, 100.0, 105.0, 10.0, ?)",
            (i, f"2026-{month:02d}-{day:02d}T14:00:00", pnl),
        )
    conn.commit()
    return conn


def _consistent_pnls(n: int = 60, win_rate: float = 0.6) -> list[float]:
    """Perfectly consistent win rate."""
    pnls = []
    for i in range(n):
        pnls.append(100.0 if (i / n) < win_rate else -50.0)
    return pnls


def _alternating_pnls(n: int = 60) -> list[float]:
    """Alternating win/loss = 50% WR, very stable."""
    return [100.0 if i % 2 == 0 else -50.0 for i in range(n)]


def _volatile_pnls(n: int = 80) -> list[float]:
    """First half all wins, second half all losses — unstable."""
    return [100.0] * (n // 2) + [-50.0] * (n // 2)


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        pnls = [50.0 if i % 2 == 0 else -30.0 for i in range(30)]
        conn = _make_conn(pnls)
        assert compute(conn) is None  # need 2 full windows of 20 = 40 trades

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, WinRateStabilityReport)


class TestWindows:
    def test_windows_present(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert len(result.windows) >= 2

    def test_window_size_stored(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn, window_size=20)
        assert result is not None
        assert result.window_size == 20

    def test_each_window_size_correct(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn, window_size=20)
        assert result is not None
        for w in result.windows:
            assert w.n_trades == 20

    def test_win_rate_in_range(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        for w in result.windows:
            assert 0 <= w.win_rate <= 100

    def test_n_trades_correct(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn, window_size=20)
        assert result is not None
        assert result.n_trades == 60


class TestStability:
    def test_stable_grade_for_alternating(self):
        """Perfectly alternating = very stable win rate."""
        conn = _make_conn(_alternating_pnls(80))
        result = compute(conn, window_size=20)
        assert result is not None
        assert result.stability_grade in ("Excellent", "Good")

    def test_unstable_grade_for_volatile(self):
        """All wins then all losses = maximum instability."""
        conn = _make_conn(_volatile_pnls(80))
        result = compute(conn, window_size=20)
        assert result is not None
        assert result.stability_grade in ("Moderate", "Unstable")
        assert result.win_rate_range > 30

    def test_grade_valid(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert result.stability_grade in ("Excellent", "Good", "Moderate", "Unstable")

    def test_cv_non_negative(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert result.win_rate_cv >= 0

    def test_min_le_max(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert result.win_rate_min <= result.win_rate_max


class TestConfidenceInterval:
    def test_ci_bounds_valid(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert 0 <= result.ci_lower <= result.ci_upper <= 100

    def test_overall_wr_in_ci(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert result.ci_lower <= result.overall_win_rate <= result.ci_upper


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_consistent_pnls(60))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10
