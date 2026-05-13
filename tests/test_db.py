from __future__ import annotations

import json
from pathlib import Path

import pytest

from amms import db
from amms.broker.alpaca import Account, Order


@pytest.fixture
def conn(tmp_path: Path):
    connection = db.connect(tmp_path / "test.sqlite")
    db.migrate(connection)
    yield connection
    connection.close()


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    applied_first = db.migrate(conn)
    applied_second = db.migrate(conn)
    assert applied_first >= 1
    assert applied_second == 0
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {r[0] for r in rows}
    assert {"bars", "orders", "equity_snapshots", "signals", "features"} <= table_names
    signal_cols = {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
    assert "score" in signal_cols
    conn.close()


def test_upsert_features_inserts_and_updates(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    db.upsert_features(conn, "2026-05-13T14:00:00Z", "AAPL", {"momentum_20d": 0.05})
    db.upsert_features(conn, "2026-05-13T14:00:00Z", "AAPL", {"momentum_20d": 0.07})
    rows = conn.execute(
        "SELECT name, value FROM features WHERE symbol=?", ("AAPL",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["name"] == "momentum_20d"
    assert rows[0]["value"] == pytest.approx(0.07)
    conn.close()


def test_upsert_features_handles_empty(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    assert db.upsert_features(conn, "ts", "AAPL", {}) == 0
    conn.close()


def test_upsert_order_inserts_then_updates(conn) -> None:
    order = Order(
        id="order-1",
        client_order_id="cid-1",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        type="market",
        status="accepted",
        submitted_at="2026-05-13T14:00:00Z",
        filled_at=None,
        filled_avg_price=None,
        raw={"id": "order-1"},
    )
    db.upsert_order(conn, order)

    filled = Order(**{**order.__dict__, "status": "filled", "filled_avg_price": 200.5})
    db.upsert_order(conn, filled)

    row = conn.execute(
        "SELECT status, filled_avg_price, raw_json FROM orders WHERE id=?",
        ("order-1",),
    ).fetchone()
    assert row["status"] == "filled"
    assert row["filled_avg_price"] == pytest.approx(200.5)
    assert json.loads(row["raw_json"]) == {"id": "order-1"}


def test_orders_side_check_rejects_invalid_side(conn) -> None:
    with pytest.raises(Exception, match="CHECK"):
        conn.execute(
            "INSERT INTO orders("
            "id, client_order_id, symbol, side, qty, type, status, submitted_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("x", "cx", "AAPL", "short", 1.0, "market", "new", "2026-05-13T14:00:00Z"),
        )


def test_insert_equity_snapshot(conn) -> None:
    account = Account(equity=100000.0, cash=50000.0, buying_power=50000.0, status="ACTIVE", raw={})
    ts = db.insert_equity_snapshot(conn, account)
    row = conn.execute("SELECT * FROM equity_snapshots WHERE ts=?", (ts,)).fetchone()
    assert row["equity"] == pytest.approx(100000.0)
    assert row["cash"] == pytest.approx(50000.0)
