"""Tests for amms.analysis.cvar."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.cvar import CVaRLevel, CVaRReport, from_bars, from_trades


class _Bar:
    def __init__(self, close):
        self.close = close


def _make_db(pnls: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE trades (pnl_pct REAL, status TEXT, closed_at TEXT)")
    for p in pnls:
        conn.execute("INSERT INTO trades VALUES (?, 'closed', '2024-01-01')", (p,))
    conn.commit()
    return conn


def _mixed_pnls(n: int = 50) -> list[float]:
    """Mix of wins and losses."""
    pnls = []
    import random
    rng = random.Random(42)
    for _ in range(n):
        pnls.append(rng.uniform(-5.0, 3.0))
    return pnls


def _bar_series(n: int = 60, seed: int = 42) -> list[_Bar]:
    import random
    rng = random.Random(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] * (1 + rng.uniform(-0.02, 0.015))))
    return [_Bar(p) for p in prices]


class TestFromTradesEdgeCases:
    def test_returns_none_empty_db(self):
        conn = _make_db([])
        assert from_trades(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_db([1.0, -1.0, 2.0])
        assert from_trades(conn) is None

    def test_returns_none_no_losses(self):
        conn = _make_db([1.0, 2.0, 1.5] * 10)
        assert from_trades(conn) is None

    def test_returns_result(self):
        conn = _make_db(_mixed_pnls(30))
        result = from_trades(conn)
        assert result is not None
        assert isinstance(result, CVaRReport)


class TestFromBarsEdgeCases:
    def test_returns_none_empty(self):
        assert from_bars([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(20)]
        assert from_bars(bars) is None

    def test_returns_result(self):
        bars = _bar_series(60)
        result = from_bars(bars)
        assert result is not None
        assert isinstance(result, CVaRReport)


class TestCVaRLevels:
    def test_three_levels_returned(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        assert len(result.levels) == 3

    def test_levels_are_cvar_level(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        for lv in result.levels:
            assert isinstance(lv, CVaRLevel)

    def test_confidence_levels_correct(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        confs = [lv.confidence for lv in result.levels]
        assert 0.90 in confs
        assert 0.95 in confs
        assert 0.99 in confs

    def test_cvar_ge_var(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        for lv in result.levels:
            assert lv.cvar >= lv.var - 0.01  # CVaR >= VaR (avg of worse outcomes)

    def test_higher_confidence_higher_cvar(self):
        conn = _make_db(_mixed_pnls(100))
        result = from_trades(conn)
        assert result is not None
        cvar_90 = result.levels[0].cvar
        cvar_99 = result.levels[2].cvar
        assert cvar_99 >= cvar_90 - 0.01


class TestHeadlineMetrics:
    def test_cvar_95_positive(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        assert result.cvar_95 > 0

    def test_var_95_positive(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        assert result.var_95 > 0

    def test_max_loss_ge_cvar_95(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        assert result.max_loss >= result.cvar_95 - 0.01

    def test_avg_loss_positive(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        assert result.avg_loss > 0

    def test_tail_risk_label_valid(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        assert result.tail_risk_label in ("low", "moderate", "high", "extreme")

    def test_tail_risk_score_ge_one(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        # CVaR >= avg_loss by definition (tail is worse than average)
        assert result.tail_risk_score >= 1.0 - 0.01


class TestMetadata:
    def test_n_observations_correct(self):
        pnls = _mixed_pnls(40)
        conn = _make_db(pnls)
        result = from_trades(conn)
        assert result is not None
        assert result.n_observations == 40

    def test_n_losses_le_n_observations(self):
        conn = _make_db(_mixed_pnls(50))
        result = from_trades(conn)
        assert result is not None
        assert result.n_losses <= result.n_observations

    def test_source_set(self):
        conn = _make_db(_mixed_pnls(30))
        result = from_trades(conn)
        assert result is not None
        assert "historical" in result.source


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_mixed_pnls(30))
        result = from_trades(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_cvar(self):
        conn = _make_db(_mixed_pnls(30))
        result = from_trades(conn)
        assert result is not None
        assert "CVaR" in result.verdict
