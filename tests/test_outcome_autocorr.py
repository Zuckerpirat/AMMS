"""Tests for amms.analysis.outcome_autocorr."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.outcome_autocorr import OutcomeAutocorrReport, compute


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


def _alternating(n: int = 40) -> list[float]:
    """W, L, W, L, ... = strong negative autocorrelation."""
    return [100.0 if i % 2 == 0 else -50.0 for i in range(n)]


def _clustering(n: int = 40) -> list[float]:
    """WWWWLLLLWWWWLLLL... = positive autocorrelation."""
    return [100.0 if (i // 4) % 2 == 0 else -50.0 for i in range(n)]


def _random_ish(n: int = 40) -> list[float]:
    """Nearly balanced, no obvious pattern."""
    return [100.0 if i % 3 != 0 else -50.0 for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        pnls = [50.0 if i % 2 == 0 else -30.0 for i in range(15)]
        conn = _make_conn(pnls)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, OutcomeAutocorrReport)


class TestAutocorrelation:
    def test_alternating_negative_ac1(self):
        """Alternating W/L has strongly negative lag-1 autocorr."""
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert result.lag1_autocorr < 0

    def test_clustering_positive_ac1(self):
        """Clustered W/L has positive lag-1 autocorr."""
        conn = _make_conn(_clustering(40))
        result = compute(conn)
        assert result is not None
        assert result.lag1_autocorr > 0

    def test_ac1_bounded(self):
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert -1.0 <= result.lag1_autocorr <= 1.0

    def test_all_lags_bounded(self):
        conn = _make_conn(_random_ish(40))
        result = compute(conn)
        assert result is not None
        for ac in [result.lag1_autocorr, result.lag2_autocorr, result.lag3_autocorr]:
            assert -1.0 <= ac <= 1.0


class TestInterpretation:
    def test_alternating_is_mean_reversion(self):
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert result.interpretation == "mean_reversion"

    def test_clustering_is_hot_hand(self):
        conn = _make_conn(_clustering(40))
        result = compute(conn)
        assert result is not None
        assert result.interpretation == "hot_hand"

    def test_interpretation_valid(self):
        conn = _make_conn(_random_ish(40))
        result = compute(conn)
        assert result is not None
        assert result.interpretation in ("hot_hand", "mean_reversion", "random")


class TestRunsTest:
    def test_alternating_high_runs(self):
        """Alternating series has many runs."""
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert result.runs_count > result.expected_runs

    def test_clustering_few_runs(self):
        """Clustered series has few runs."""
        conn = _make_conn(_clustering(40))
        result = compute(conn)
        assert result is not None
        assert result.runs_count < result.expected_runs

    def test_runs_z_score_stored(self):
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert isinstance(result.runs_z_score, float)

    def test_runs_significant_bool(self):
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert isinstance(result.runs_significant, bool)


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_n_trades_correct(self):
        conn = _make_conn(_alternating(40))
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 40
