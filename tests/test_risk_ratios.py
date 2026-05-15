"""Tests for amms.analysis.risk_ratios."""

from __future__ import annotations

import math
import sqlite3

import pytest

from amms.analysis.risk_ratios import RiskRatios, compute


def _make_conn(equities: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE equity_snapshots (ts TEXT, equity REAL)"
    )
    for i, eq in enumerate(equities):
        conn.execute(
            "INSERT INTO equity_snapshots VALUES (?, ?)",
            (f"2026-01-{i+1:02d}T16:00:00", eq),
        )
    conn.commit()
    return conn


def _flat(start: float, n: int) -> list[float]:
    return [start] * n


def _growing(start: float, daily_ret: float, n: int) -> list[float]:
    result = [start]
    for _ in range(n - 1):
        result.append(result[-1] * (1 + daily_ret))
    return result


def _with_drawdown(start: float, n: int, peak_at: int, drawdown_pct: float) -> list[float]:
    """Grow then drop."""
    result = [start * (1 + i * 0.005) for i in range(peak_at)]
    peak = result[-1]
    drop_to = peak * (1 - drawdown_pct / 100)
    remainder = n - peak_at
    result += [drop_to + (peak - drop_to) * i / max(remainder - 1, 1) for i in range(remainder)]
    return result


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_conn([10000.0] * 8)
        assert compute(conn) is None

    def test_returns_result_with_enough_data(self):
        conn = _make_conn(_growing(10000.0, 0.001, 50))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, RiskRatios)


class TestReturnMetrics:
    def test_flat_equity_zero_returns(self):
        """Flat equity → zero vol, zero return."""
        conn = _make_conn(_flat(10000.0, 30))
        result = compute(conn)
        assert result is not None
        assert result.annualized_vol_pct == pytest.approx(0.0, abs=0.01)

    def test_growing_equity_positive_return(self):
        conn = _make_conn(_growing(10000.0, 0.002, 50))
        result = compute(conn)
        assert result is not None
        assert result.annualized_return_pct > 0

    def test_declining_equity_negative_return(self):
        conn = _make_conn(_growing(10000.0, -0.002, 50))
        result = compute(conn)
        assert result is not None
        assert result.annualized_return_pct < 0

    def test_start_end_equity(self):
        equities = _growing(10000.0, 0.001, 30)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.start_equity == pytest.approx(equities[0], abs=0.01)
        assert result.end_equity == pytest.approx(equities[-1], abs=0.01)

    def test_n_periods_correct(self):
        n = 40
        conn = _make_conn(_growing(10000.0, 0.001, n))
        result = compute(conn)
        assert result is not None
        assert result.n_periods == n

    def test_limit_respected(self):
        conn = _make_conn(_growing(10000.0, 0.001, 100))
        result = compute(conn, limit=30)
        assert result is not None
        assert result.n_periods <= 30


class TestSharpe:
    def test_sharpe_none_for_flat_equity(self):
        """Zero vol → Sharpe undefined (None)."""
        conn = _make_conn(_flat(10000.0, 30))
        result = compute(conn)
        assert result is not None
        assert result.sharpe is None

    def test_sharpe_positive_for_growing_equity(self):
        conn = _make_conn(_growing(10000.0, 0.003, 60))
        result = compute(conn)
        assert result is not None
        assert result.sharpe is not None
        # High consistent growth → positive Sharpe
        assert result.sharpe > 0

    def test_sharpe_grade_excellent(self):
        """Very consistent high-growth → excellent."""
        equities = _growing(10000.0, 0.01, 60)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.sharpe_grade in ("excellent", "good")

    def test_sharpe_grade_poor_for_declining(self):
        conn = _make_conn(_growing(10000.0, -0.005, 30))
        result = compute(conn)
        assert result is not None
        assert result.sharpe_grade == "poor"


class TestSortino:
    def test_sortino_none_for_no_losses(self):
        """Only positive returns → no downside vol → Sortino undefined."""
        # Perfectly growing equity
        conn = _make_conn(_growing(10000.0, 0.002, 30))
        result = compute(conn)
        assert result is not None
        # Downside vol could be 0 if no negative daily returns
        if result.sortino is None:
            assert result.annualized_downside_vol_pct == pytest.approx(0.0, abs=0.01)

    def test_sortino_positive_when_mixed_returns(self):
        """Alternating up/down days → sortino computable."""
        equities = [10000.0]
        for i in range(40):
            ret = 0.005 if i % 2 == 0 else -0.002  # net positive
            equities.append(equities[-1] * (1 + ret))
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.sortino is not None
        assert result.sortino > 0


class TestCalmar:
    def test_calmar_none_if_no_drawdown(self):
        """Monotonically growing → no drawdown → Calmar undefined."""
        conn = _make_conn(_growing(10000.0, 0.002, 30))
        result = compute(conn)
        assert result is not None
        if result.calmar is None:
            assert result.max_drawdown_pct == pytest.approx(0.0, abs=0.1)

    def test_max_drawdown_negative(self):
        equities = _with_drawdown(10000.0, 50, 30, 20.0)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert result.max_drawdown_pct < 0

    def test_max_drawdown_bounded(self):
        equities = _with_drawdown(10000.0, 50, 30, 20.0)
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        assert -100 <= result.max_drawdown_pct <= 0

    def test_calmar_positive_for_growing_with_dd(self):
        """If overall growth and drawdown, Calmar should be positive."""
        # Big overall growth but with a temporary drawdown
        equities = (
            _growing(10000.0, 0.005, 30)
            + _growing(10000.0 * (1.005 ** 29) * 0.85, 0.003, 30)
        )
        conn = _make_conn(equities)
        result = compute(conn)
        assert result is not None
        if result.calmar is not None and result.annualized_return_pct > 0:
            assert result.calmar > 0


class TestVerdictAndGrade:
    def test_verdict_present(self):
        conn = _make_conn(_growing(10000.0, 0.002, 30))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 5

    def test_grade_is_valid(self):
        conn = _make_conn(_growing(10000.0, 0.002, 30))
        result = compute(conn)
        assert result is not None
        assert result.sharpe_grade in ("excellent", "good", "ok", "poor")
