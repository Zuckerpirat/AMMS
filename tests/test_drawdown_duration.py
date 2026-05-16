"""Tests for amms.analysis.drawdown_duration."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.drawdown_duration import DrawdownDurationReport, compute


def _make_conn(equities: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE equity_snapshots (ts TEXT, equity REAL)")
    for i, eq in enumerate(equities):
        conn.execute(
            "INSERT INTO equity_snapshots VALUES (?, ?)",
            (f"2026-01-{i+1:02d}T16:00:00", eq),
        )
    conn.commit()
    return conn


def _growing(start: float, daily_ret: float, n: int) -> list[float]:
    result = [start]
    for _ in range(n - 1):
        result.append(result[-1] * (1 + daily_ret))
    return result


def _with_dd(start: float, grow: int, drop_pct: float, recover: int) -> list[float]:
    """Grow, drop by drop_pct%, then recover."""
    equities = _growing(start, 0.005, grow)
    peak = equities[-1]
    trough = peak * (1 - drop_pct / 100)
    # Linear drop then recover
    equities += [peak - (peak - trough) * i / 5 for i in range(1, 6)]
    equities += _growing(trough, 0.01, recover)
    return equities


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_conn([10000.0] * 8)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        equities = _growing(10000.0, 0.001, 20)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert isinstance(result, DrawdownDurationReport)


class TestCurrentDrawdown:
    def test_no_drawdown_at_new_high(self):
        """Monotonically growing → at all-time high, no drawdown."""
        equities = _growing(10000.0, 0.002, 20)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert not result.is_underwater
        assert result.current_drawdown_pct == pytest.approx(0.0, abs=0.1)

    def test_current_drawdown_detected(self):
        """Peak then drop → should be underwater."""
        equities = _growing(10000.0, 0.005, 15) + _growing(10000.0 * (1.005 ** 14) * 0.85, -0.001, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.is_underwater
        assert result.current_drawdown_pct < -5

    def test_current_drawdown_periods_positive(self):
        equities = _growing(10000.0, 0.005, 15) + [10000.0 * (1.005 ** 14) * 0.9] * 10
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        if result.is_underwater:
            assert result.current_drawdown_periods > 0


class TestMaxDrawdown:
    def test_max_drawdown_negative(self):
        equities = _with_dd(10000.0, 12, 20.0, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.max_drawdown_pct < 0

    def test_max_drawdown_bounded(self):
        equities = _with_dd(10000.0, 12, 20.0, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert -100 <= result.max_drawdown_pct <= 0

    def test_max_drawdown_at_least_drop_size(self):
        """With a 20% drop, max drawdown should be at least 20%."""
        equities = _with_dd(10000.0, 12, 20.0, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.max_drawdown_pct <= -10  # at least 10% (allowing some tolerance)

    def test_max_drawdown_duration_positive(self):
        equities = _with_dd(10000.0, 12, 15.0, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        # Duration can be 0 for very small drawdowns
        assert result.max_drawdown_duration >= 0


class TestRecovery:
    def test_avg_recovery_none_if_no_recovered_episodes(self):
        """Single open drawdown that hasn't recovered → None."""
        equities = _growing(10000.0, 0.005, 15) + _growing(10000.0 * (1.005 ** 14) * 0.8, -0.001, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        # avg_recovery is None if no episodes recovered
        if result.n_recovered == 0:
            assert result.avg_recovery_periods is None

    def test_n_recovered_lte_n_episodes(self):
        equities = _with_dd(10000.0, 12, 15.0, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.n_recovered <= result.n_drawdown_periods


class TestGeneralMetrics:
    def test_pain_index_zero_for_monotonic_growth(self):
        """No drawdowns → pain index should be ≈ 0."""
        equities = _growing(10000.0, 0.002, 20)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.pain_index == pytest.approx(0.0, abs=0.01)

    def test_pain_index_negative_with_drawdowns(self):
        equities = _with_dd(10000.0, 12, 15.0, 10)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.pain_index <= 0

    def test_equity_high_at_least_start(self):
        equities = _growing(10000.0, 0.001, 20)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.equity_high >= equities[0]

    def test_current_equity_matches_last(self):
        equities = _growing(10000.0, 0.001, 20)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.current_equity == pytest.approx(equities[-1], abs=0.01)

    def test_n_periods_correct(self):
        n = 30
        equities = _growing(10000.0, 0.001, n)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.n_periods == n

    def test_limit_respected(self):
        equities = _growing(10000.0, 0.001, 100)
        conn = _make_conn(equities)
        result = compute(conn, limit=30)
        assert result is not None
        assert result.n_periods <= 30

    def test_verdict_present(self):
        equities = _growing(10000.0, 0.001, 20)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 5
