"""Tests for price alert management."""

from __future__ import annotations

import sqlite3

import pytest

from amms.data.alerts import (
    PriceAlert,
    add_alert,
    check_alerts,
    delete_alert,
    list_alerts,
    mark_triggered,
)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def test_add_and_list(conn):
    a = add_alert(conn, "AAPL", 200.0, "above")
    assert a.symbol == "AAPL"
    assert a.price == 200.0
    assert a.direction == "above"
    assert not a.triggered

    alerts = list_alerts(conn)
    assert len(alerts) == 1
    assert alerts[0].symbol == "AAPL"


def test_list_excludes_triggered(conn):
    a = add_alert(conn, "AAPL", 200.0, "above")
    mark_triggered(conn, a.id)
    assert list_alerts(conn) == []
    assert len(list_alerts(conn, include_triggered=True)) == 1


def test_delete(conn):
    a = add_alert(conn, "TSLA", 300.0, "below")
    deleted = delete_alert(conn, a.id)
    assert deleted is True
    assert list_alerts(conn) == []


def test_delete_nonexistent(conn):
    assert delete_alert(conn, 999) is False


def test_invalid_direction(conn):
    with pytest.raises(ValueError, match="direction"):
        add_alert(conn, "AAPL", 200.0, "sideways")


def test_invalid_price(conn):
    with pytest.raises(ValueError, match="positive"):
        add_alert(conn, "AAPL", -10.0, "above")


def test_check_alerts_fires_above(conn):
    add_alert(conn, "AAPL", 200.0, "above")
    fired = check_alerts(conn, {"AAPL": 205.0})
    assert len(fired) == 1
    assert fired[0].symbol == "AAPL"
    # Should be marked triggered — won't fire again.
    fired2 = check_alerts(conn, {"AAPL": 210.0})
    assert fired2 == []


def test_check_alerts_fires_below(conn):
    add_alert(conn, "NVDA", 100.0, "below")
    fired = check_alerts(conn, {"NVDA": 95.0})
    assert len(fired) == 1


def test_check_alerts_does_not_fire_when_not_crossed(conn):
    add_alert(conn, "AAPL", 200.0, "above")
    fired = check_alerts(conn, {"AAPL": 195.0})
    assert fired == []


def test_check_alerts_exact_boundary(conn):
    add_alert(conn, "SPY", 500.0, "above")
    fired = check_alerts(conn, {"SPY": 500.0})
    assert len(fired) == 1


def test_check_alerts_missing_symbol(conn):
    add_alert(conn, "AAPL", 200.0, "above")
    fired = check_alerts(conn, {"TSLA": 300.0})
    assert fired == []
