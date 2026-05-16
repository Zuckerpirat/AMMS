"""Tests for amms.analysis.trade_sector."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.trade_sector import TradeSectorReport, TradeSectorStats, compute


def _make_db(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (pnl_pct, symbol)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trades (pnl_pct REAL, symbol TEXT, closed_at TEXT, status TEXT)"
    )
    for i, (pnl, sym) in enumerate(rows):
        ts = f"2024-{1 + i // 28:02d}-{1 + i % 28:02d} 10:00:00"
        conn.execute("INSERT INTO trades VALUES (?, ?, ?, 'closed')", (pnl, sym, ts))
    conn.commit()
    return conn


def _two_sector_rows(n: int = 20) -> list[tuple]:
    """Half AAPL (Technology), half JPM (Finance), alternating."""
    rows = []
    for i in range(n):
        sym = "AAPL" if i % 2 == 0 else "JPM"
        pnl = 2.0 if i % 2 == 0 else -1.0
        rows.append((pnl, sym))
    return rows


def _three_sector_rows(n: int = 30) -> list[tuple]:
    syms = ["AAPL", "JPM", "XOM"]  # Tech, Finance, Energy
    pnls = [2.0, 0.5, -1.0]
    return [(pnls[i % 3], syms[i % 3]) for i in range(n)]


def _rotation_rows() -> list[tuple]:
    """AAPL wins early, JPM wins late → rotation."""
    early = [(2.0, "AAPL"), (-0.5, "JPM")] * 8   # 16 rows
    late = [(-1.0, "AAPL"), (3.0, "JPM")] * 5    # 10 rows
    return early + late


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_db([(1.0, "AAPL")] * 5)
        assert compute(conn) is None

    def test_returns_none_single_sector(self):
        conn = _make_db([(1.0, "AAPL")] * 15)
        assert compute(conn) is None

    def test_returns_result_two_sectors(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, TradeSectorReport)


class TestSectorStats:
    def test_n_sectors_correct(self):
        conn = _make_db(_three_sector_rows(30))
        result = compute(conn)
        assert result is not None
        assert result.n_sectors == 3

    def test_sectors_are_stats(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        for s in result.sectors:
            assert isinstance(s, TradeSectorStats)

    def test_win_rate_in_range(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        for s in result.sectors:
            if s.n_trades > 0:
                assert 0.0 <= s.win_rate <= 100.0

    def test_sectors_sorted_by_recent_pnl(self):
        conn = _make_db(_three_sector_rows(30))
        result = compute(conn)
        assert result is not None
        recent_pnls = [s.recent_avg_pnl for s in result.sectors]
        assert recent_pnls == sorted(recent_pnls, reverse=True)


class TestLeaderLaggard:
    def test_leader_is_best_sector(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        if result.leader:
            for s in result.sectors:
                assert result.leader.recent_avg_pnl >= s.recent_avg_pnl - 0.001

    def test_laggard_is_worst_sector(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        if result.laggard:
            for s in result.sectors:
                assert result.laggard.recent_avg_pnl <= s.recent_avg_pnl + 0.001

    def test_leader_is_technology_for_aapl_wins(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        if result.leader:
            assert result.leader.sector == "Technology"


class TestRotation:
    def test_rotation_detected_when_leadership_changes(self):
        conn = _make_db(_rotation_rows())
        result = compute(conn)
        assert result is not None
        assert result.rotation_detected is True

    def test_no_rotation_when_same_leader(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.rotation_detected is False

    def test_rotation_magnitude_nonneg(self):
        conn = _make_db(_rotation_rows())
        result = compute(conn)
        assert result is not None
        assert result.rotation_magnitude >= 0.0


class TestCustomSectorMap:
    def test_custom_sector_map_used(self):
        rows = [(1.0, "MYMEME"), (-1.0, "OTHERTICK")] * 8
        conn = _make_db(rows)
        sector_map = {"MYMEME": "Meme", "OTHERTICK": "Boring"}
        result = compute(conn, sector_map=sector_map)
        assert result is not None
        labels = {s.sector for s in result.sectors}
        assert "Meme" in labels
        assert "Boring" in labels


class TestNTrades:
    def test_n_trades_correct(self):
        rows = _two_sector_rows(20)
        conn = _make_db(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 20


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_sector(self):
        conn = _make_db(_two_sector_rows(20))
        result = compute(conn)
        assert result is not None
        assert "sector" in result.verdict.lower() or "leader" in result.verdict.lower()
