"""Tests for DE signal history storage and retrieval."""

from __future__ import annotations

import sqlite3

import pytest


def _make_conn():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS de_signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            action TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            horizon TEXT,
            price REAL,
            macro_level TEXT
        )
    """)
    return conn


# ── record_signal ─────────────────────────────────────────────────────────────

def test_record_signal_returns_true_on_success():
    from amms.data.signal_history import record_signal
    conn = _make_conn()
    result = record_signal(conn, symbol="AAPL", mode="swing", action="buy",
                           score=55.0, confidence=0.70)
    assert result is True


def test_record_signal_inserts_row():
    from amms.data.signal_history import record_signal
    conn = _make_conn()
    record_signal(conn, symbol="TSLA", mode="meme", action="strong_buy",
                  score=80.0, confidence=0.85, price=200.0)
    row = conn.execute("SELECT * FROM de_signal_history").fetchone()
    assert row["symbol"] == "TSLA"
    assert row["mode"] == "meme"
    assert row["action"] == "strong_buy"
    assert row["score"] == pytest.approx(80.0)
    assert row["price"] == pytest.approx(200.0)


def test_record_signal_normalizes_symbol_to_upper():
    from amms.data.signal_history import record_signal
    conn = _make_conn()
    record_signal(conn, symbol="aapl", mode="swing", action="buy",
                  score=40.0, confidence=0.65)
    row = conn.execute("SELECT symbol FROM de_signal_history").fetchone()
    assert row["symbol"] == "AAPL"


def test_record_signal_returns_false_on_none_conn():
    from amms.data.signal_history import record_signal
    result = record_signal(None, symbol="GME", mode="meme", action="buy",
                           score=60.0, confidence=0.70)
    assert result is False


def test_record_signal_stores_macro_level():
    from amms.data.signal_history import record_signal
    conn = _make_conn()
    record_signal(conn, symbol="SPY", mode="swing", action="hold",
                  score=5.0, confidence=0.50, macro_level="stressed")
    row = conn.execute("SELECT macro_level FROM de_signal_history").fetchone()
    assert row["macro_level"] == "stressed"


def test_record_signal_stores_horizon():
    from amms.data.signal_history import record_signal
    conn = _make_conn()
    record_signal(conn, symbol="AAPL", mode="swing", action="buy",
                  score=45.0, confidence=0.65, horizon="1-3 weeks")
    row = conn.execute("SELECT horizon FROM de_signal_history").fetchone()
    assert row["horizon"] == "1-3 weeks"


# ── fetch_recent ──────────────────────────────────────────────────────────────

def test_fetch_recent_returns_empty_list_for_empty_table():
    from amms.data.signal_history import fetch_recent
    conn = _make_conn()
    records = fetch_recent(conn)
    assert records == []


def test_fetch_recent_returns_none_conn_as_empty():
    from amms.data.signal_history import fetch_recent
    records = fetch_recent(None)
    assert records == []


def test_fetch_recent_returns_inserted_records():
    from amms.data.signal_history import fetch_recent, record_signal
    conn = _make_conn()
    record_signal(conn, symbol="AAPL", mode="swing", action="buy",
                  score=50.0, confidence=0.70)
    record_signal(conn, symbol="MSFT", mode="conservative", action="hold",
                  score=10.0, confidence=0.55)
    records = fetch_recent(conn)
    assert len(records) == 2


def test_fetch_recent_filter_by_symbol():
    from amms.data.signal_history import fetch_recent, record_signal
    conn = _make_conn()
    record_signal(conn, symbol="AAPL", mode="swing", action="buy",
                  score=50.0, confidence=0.70)
    record_signal(conn, symbol="MSFT", mode="swing", action="sell",
                  score=-40.0, confidence=0.65)
    records = fetch_recent(conn, symbol="AAPL")
    assert all(r.symbol == "AAPL" for r in records)
    assert len(records) == 1


def test_fetch_recent_filter_by_mode():
    from amms.data.signal_history import fetch_recent, record_signal
    conn = _make_conn()
    record_signal(conn, symbol="GME", mode="meme", action="buy",
                  score=70.0, confidence=0.75)
    record_signal(conn, symbol="SPY", mode="swing", action="hold",
                  score=5.0, confidence=0.50)
    records = fetch_recent(conn, mode="meme")
    assert all(r.mode == "meme" for r in records)
    assert len(records) == 1


def test_fetch_recent_filter_by_action():
    from amms.data.signal_history import fetch_recent, record_signal
    conn = _make_conn()
    for action in ("buy", "sell", "hold", "buy"):
        record_signal(conn, symbol="X", mode="swing", action=action,
                      score=30.0, confidence=0.60)
    records = fetch_recent(conn, action="buy")
    assert all(r.action == "buy" for r in records)
    assert len(records) == 2


def test_fetch_recent_respects_limit():
    from amms.data.signal_history import fetch_recent, record_signal
    conn = _make_conn()
    for i in range(10):
        record_signal(conn, symbol="X", mode="swing", action="hold",
                      score=float(i), confidence=0.50)
    records = fetch_recent(conn, limit=3)
    assert len(records) == 3


# ── signal_accuracy_by_mode ───────────────────────────────────────────────────

def test_signal_accuracy_by_mode_empty():
    from amms.data.signal_history import signal_accuracy_by_mode
    conn = _make_conn()
    result = signal_accuracy_by_mode(conn)
    assert result == {}


def test_signal_accuracy_by_mode_aggregates_correctly():
    from amms.data.signal_history import record_signal, signal_accuracy_by_mode
    conn = _make_conn()
    record_signal(conn, symbol="A", mode="swing", action="buy",
                  score=50.0, confidence=0.70)
    record_signal(conn, symbol="B", mode="swing", action="buy",
                  score=55.0, confidence=0.72)
    record_signal(conn, symbol="C", mode="swing", action="sell",
                  score=-45.0, confidence=0.68)
    record_signal(conn, symbol="D", mode="meme", action="strong_buy",
                  score=80.0, confidence=0.85)
    result = signal_accuracy_by_mode(conn)
    assert "swing" in result
    assert result["swing"]["buy"] == 2
    assert result["swing"]["sell"] == 1
    assert "meme" in result
    assert result["meme"]["strong_buy"] == 1


def test_signal_accuracy_by_mode_none_conn():
    from amms.data.signal_history import signal_accuracy_by_mode
    result = signal_accuracy_by_mode(None)
    assert result == {}
