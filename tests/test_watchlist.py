"""Tests for amms.data.watchlist."""

from __future__ import annotations

import sqlite3

import pytest

from amms.data.watchlist import WatchEntry, add, remove, list_all, contains, ensure_table


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    return conn


class TestEnsureTable:
    def test_creates_table(self):
        conn = _conn()
        ensure_table(conn)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='watchlist'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        conn = _conn()
        ensure_table(conn)
        ensure_table(conn)  # should not raise


class TestAdd:
    def test_add_returns_true(self):
        conn = _conn()
        assert add(conn, "AAPL") is True

    def test_add_uppercase(self):
        conn = _conn()
        add(conn, "aapl")
        assert contains(conn, "AAPL")

    def test_add_duplicate_returns_false(self):
        conn = _conn()
        add(conn, "AAPL")
        assert add(conn, "AAPL") is False

    def test_add_with_note(self):
        conn = _conn()
        add(conn, "TSLA", note="AI play")
        entries = list_all(conn)
        assert entries[0].note == "AI play"


class TestRemove:
    def test_remove_existing(self):
        conn = _conn()
        add(conn, "AAPL")
        assert remove(conn, "AAPL") is True

    def test_remove_non_existing(self):
        conn = _conn()
        assert remove(conn, "ZZZZ") is False

    def test_remove_deletes_entry(self):
        conn = _conn()
        add(conn, "AAPL")
        remove(conn, "AAPL")
        assert not contains(conn, "AAPL")

    def test_remove_case_insensitive(self):
        conn = _conn()
        add(conn, "AAPL")
        assert remove(conn, "aapl") is True


class TestListAll:
    def test_empty_returns_empty(self):
        conn = _conn()
        assert list_all(conn) == []

    def test_returns_entries(self):
        conn = _conn()
        add(conn, "AAPL")
        add(conn, "TSLA")
        entries = list_all(conn)
        assert len(entries) == 2

    def test_entries_are_watchentry(self):
        conn = _conn()
        add(conn, "AAPL")
        entries = list_all(conn)
        assert isinstance(entries[0], WatchEntry)

    def test_symbol_uppercase(self):
        conn = _conn()
        add(conn, "nvda")
        entries = list_all(conn)
        assert entries[0].symbol == "NVDA"

    def test_added_ts_present(self):
        conn = _conn()
        add(conn, "AAPL")
        entries = list_all(conn)
        assert len(entries[0].added_ts) > 0


class TestContains:
    def test_contains_true(self):
        conn = _conn()
        add(conn, "AAPL")
        assert contains(conn, "AAPL") is True

    def test_contains_false(self):
        conn = _conn()
        assert contains(conn, "AAPL") is False

    def test_contains_case_insensitive(self):
        conn = _conn()
        add(conn, "AAPL")
        assert contains(conn, "aapl") is True
