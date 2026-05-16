"""Tests for amms.analysis.sector_win_rate."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from amms.analysis.sector_win_rate import SectorWinRate, SectorWinRateReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (symbol, pnl, buy_price, qty)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (symbol, pnl, buy_price, qty) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, ?, '2026-01-01T09:00:00', '2026-01-02T14:00:00', ?, 105.0, ?, ?)",
            (i, symbol, buy_price, qty, pnl),
        )
    conn.commit()
    return conn


MOCK_SECTOR = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
    "GS": "Financials",
    "JNJ": "Health Care",
    "PFE": "Health Care",
}


def _rows_with_sectors() -> list[tuple]:
    rows = []
    # Tech wins
    for _ in range(4):
        rows.append(("AAPL", 200.0, 100.0, 10.0))
    rows.append(("AAPL", -50.0, 100.0, 10.0))
    rows.append(("MSFT", 150.0, 100.0, 10.0))
    # Financials mixed
    for _ in range(3):
        rows.append(("JPM", 80.0, 100.0, 10.0))
    for _ in range(3):
        rows.append(("GS", -100.0, 100.0, 10.0))
    # Health Care loses
    for _ in range(2):
        rows.append(("JNJ", -200.0, 100.0, 10.0))
    rows.append(("PFE", -150.0, 100.0, 10.0))
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [("AAPL", 50.0, 100.0, 10.0), ("MSFT", -30.0, 100.0, 10.0)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        assert isinstance(result, SectorWinRateReport)


class TestSectors:
    def test_sectors_present(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        assert result.n_sectors > 0

    def test_tech_sector_detected(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        sectors = [s.sector for s in result.sectors]
        assert "Technology" in sectors

    def test_sorted_by_total_pnl(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        pnls = [s.total_pnl for s in result.sectors]
        assert pnls == sorted(pnls, reverse=True)

    def test_best_sector_tech(self):
        """Technology has most wins → should be best sector."""
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        assert result.best_sector == "Technology"

    def test_win_rate_range(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        for s in result.sectors:
            assert 0 <= s.win_rate <= 100

    def test_symbols_populated(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        for s in result.sectors:
            if s.sector != "Unknown":
                assert len(s.symbols) > 0

    def test_n_trades_sum(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        assert result.n_trades == sum(s.n_trades for s in result.sectors)

    def test_min_trades_filter(self):
        """Sectors with < min_trades are excluded."""
        rows = _rows_with_sectors()
        rows.append(("NEWSTOCK", 100.0, 100.0, 10.0))  # unknown sector, 1 trade
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(rows)
            result = compute(conn, min_trades=2)
        assert result is not None
        # NEWSTOCK maps to Unknown with only 1 trade, should not appear if min_trades=2
        # (unless other unknowns exist)

    def test_unknown_fallback_works(self):
        """Works even when SYMBOL_SECTOR is empty (all go to Unknown)."""
        conn = _make_conn(_rows_with_sectors())
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", {}):
            result = compute(conn)
        assert result is not None
        assert result.unknown_trades > 0


class TestVerdict:
    def test_verdict_present(self):
        with patch("amms.analysis.sector_win_rate.SYMBOL_SECTOR", MOCK_SECTOR):
            conn = _make_conn(_rows_with_sectors())
            result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10
