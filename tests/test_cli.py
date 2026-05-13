from __future__ import annotations

import sqlite3

import httpx
import respx
from typer.testing import CliRunner

from amms import __version__
from amms.cli import app

runner = CliRunner()
PAPER_URL = "https://paper-api.alpaca.markets"


def test_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "paper trading" in result.stdout.lower()


def test_run_prints_ready_banner() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
    assert f"amms {__version__} ready" in result.stdout


def test_status_refuses_without_paper_url(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 2
    assert "paper" in result.stdout.lower()


def test_backtest_not_implemented_yet(paper_env) -> None:
    result = runner.invoke(app, ["backtest"])
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)


def test_init_db_creates_schema(paper_env) -> None:
    result = runner.invoke(app, ["init-db"])
    assert result.exit_code == 0, result.stdout
    conn = sqlite3.connect(paper_env["AMMS_DB_PATH"])
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"bars", "orders", "equity_snapshots", "signals"} <= tables


@respx.mock
def test_status_calls_alpaca_and_records_snapshot(paper_env) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(
            200,
            json={
                "equity": "100000",
                "cash": "100000",
                "buying_power": "100000",
                "status": "ACTIVE",
            },
        )
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "Account" in result.stdout
    assert "$100,000" in result.stdout

    conn = sqlite3.connect(paper_env["AMMS_DB_PATH"])
    row = conn.execute("SELECT equity FROM equity_snapshots").fetchone()
    conn.close()
    assert row[0] == 100000.0


@respx.mock
def test_buy_submits_order_and_records(paper_env) -> None:
    respx.post(f"{PAPER_URL}/v2/orders").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "o-1",
                "client_order_id": "cid-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "1",
                "type": "market",
                "status": "accepted",
                "submitted_at": "2026-05-13T14:00:00Z",
            },
        )
    )

    result = runner.invoke(app, ["buy", "AAPL", "1"])
    assert result.exit_code == 0, result.stdout
    assert "BUY" in result.stdout
    assert "o-1" in result.stdout

    conn = sqlite3.connect(paper_env["AMMS_DB_PATH"])
    row = conn.execute(
        "SELECT side, symbol, qty, status FROM orders WHERE id=?", ("o-1",)
    ).fetchone()
    conn.close()
    assert row == ("buy", "AAPL", 1.0, "accepted")


@respx.mock
def test_sell_refuses_when_position_too_small(paper_env) -> None:
    respx.get(f"{PAPER_URL}/v2/positions").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "symbol": "AAPL",
                    "qty": "0",
                    "avg_entry_price": "0",
                    "market_value": "0",
                    "unrealized_pl": "0",
                }
            ],
        )
    )

    result = runner.invoke(app, ["sell", "AAPL", "1"])
    assert result.exit_code != 0
    assert "short" in result.stdout.lower()


@respx.mock
def test_sell_submits_when_position_held(paper_env) -> None:
    respx.get(f"{PAPER_URL}/v2/positions").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "symbol": "AAPL",
                    "qty": "5",
                    "avg_entry_price": "180.0",
                    "market_value": "1000.0",
                    "unrealized_pl": "0",
                }
            ],
        )
    )
    respx.post(f"{PAPER_URL}/v2/orders").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "o-2",
                "client_order_id": "cid-2",
                "symbol": "AAPL",
                "side": "sell",
                "qty": "1",
                "type": "market",
                "status": "accepted",
                "submitted_at": "2026-05-13T14:00:00Z",
            },
        )
    )

    result = runner.invoke(app, ["sell", "AAPL", "1"])
    assert result.exit_code == 0, result.stdout
    assert "SELL" in result.stdout


def _bars_response_for(symbol: str, closes: list[float]) -> dict:
    return {
        "bars": {
            symbol.upper(): [
                {
                    "t": f"2025-01-{i + 1:02d}T05:00:00Z",
                    "o": c,
                    "h": c,
                    "l": c,
                    "c": c,
                    "v": 100,
                }
                for i, c in enumerate(closes)
            ]
        },
        "next_page_token": None,
    }


@respx.mock
def test_tick_dry_run_prints_signals_without_orders(paper_env_with_config) -> None:
    # Closes engineered so SMA(3) crosses above SMA(5) on the last bar.
    closes = [1, 1, 1, 1, 1, 1, 10]
    respx.get("https://data.alpaca.markets/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_response_for("AAPL", closes))
    )
    account_route = respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(
            200,
            json={
                "equity": "100000",
                "cash": "100000",
                "buying_power": "100000",
                "status": "ACTIVE",
            },
        )
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    orders_route = respx.post(f"{PAPER_URL}/v2/orders")

    result = runner.invoke(app, ["tick"])
    assert result.exit_code == 0, result.stdout
    assert "buy" in result.stdout.lower()
    assert "Dry run" in result.stdout
    assert account_route.called
    assert not orders_route.called

    conn = sqlite3.connect(paper_env_with_config["AMMS_DB_PATH"])
    rows = conn.execute("SELECT signal FROM signals WHERE symbol=?", ("AAPL",)).fetchall()
    conn.close()
    assert rows and rows[-1][0] == "buy"


@respx.mock
def test_tick_execute_places_order_when_allowed(paper_env_with_config) -> None:
    closes = [1, 1, 1, 1, 1, 1, 10]
    respx.get("https://data.alpaca.markets/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_response_for("AAPL", closes))
    )
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(
            200,
            json={
                "equity": "100000",
                "cash": "100000",
                "buying_power": "100000",
                "status": "ACTIVE",
            },
        )
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    orders_route = respx.post(f"{PAPER_URL}/v2/orders").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "ord-1",
                "client_order_id": "cid-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "400",
                "type": "market",
                "status": "accepted",
                "submitted_at": "2026-05-13T14:00:00Z",
            },
        )
    )

    result = runner.invoke(app, ["tick", "--execute"])
    assert result.exit_code == 0, result.stdout
    assert orders_route.called
    assert "Placed BUY" in result.stdout

    conn = sqlite3.connect(paper_env_with_config["AMMS_DB_PATH"])
    row = conn.execute("SELECT symbol, side, status FROM orders").fetchone()
    conn.close()
    assert row == ("AAPL", "buy", "accepted")


@respx.mock
def test_tick_does_not_buy_when_already_holding(paper_env_with_config) -> None:
    closes = [1, 1, 1, 1, 1, 1, 10]
    respx.get("https://data.alpaca.markets/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_response_for("AAPL", closes))
    )
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(
            200,
            json={
                "equity": "100000",
                "cash": "100000",
                "buying_power": "100000",
                "status": "ACTIVE",
            },
        )
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "symbol": "AAPL",
                    "qty": "10",
                    "avg_entry_price": "5",
                    "market_value": "50",
                    "unrealized_pl": "0",
                }
            ],
        )
    )
    orders_route = respx.post(f"{PAPER_URL}/v2/orders")

    result = runner.invoke(app, ["tick", "--execute"])
    assert result.exit_code == 0, result.stdout
    assert "BLOCKED" in result.stdout
    assert "already long" in result.stdout
    assert not orders_route.called


@respx.mock
def test_fetch_bars_stores_in_db(paper_env) -> None:
    respx.get("https://data.alpaca.markets/v2/stocks/bars").mock(
        return_value=httpx.Response(
            200,
            json={
                "bars": {
                    "AAPL": [
                        {
                            "t": "2025-01-02T05:00:00Z",
                            "o": 1.0,
                            "h": 2.0,
                            "l": 0.5,
                            "c": 1.5,
                            "v": 100,
                        }
                    ]
                },
                "next_page_token": None,
            },
        )
    )

    result = runner.invoke(
        app,
        ["fetch-bars", "AAPL", "--start", "2025-01-01", "--end", "2025-01-04"],
    )
    assert result.exit_code == 0, result.stdout
    assert "1 1Day bars" in result.stdout

    conn = sqlite3.connect(paper_env["AMMS_DB_PATH"])
    rows = conn.execute("SELECT symbol, ts, close FROM bars").fetchall()
    conn.close()
    assert rows == [("AAPL", "2025-01-02T05:00:00Z", 1.5)]
