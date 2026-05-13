from __future__ import annotations

import json

import httpx
import pytest
import respx

from amms.broker.alpaca import AlpacaClient, AlpacaError

PAPER_URL = "https://paper-api.alpaca.markets"


@pytest.fixture
def client() -> AlpacaClient:
    return AlpacaClient("k", "s", PAPER_URL)


def test_constructor_refuses_live_endpoint() -> None:
    with pytest.raises(RuntimeError, match="paper-api"):
        AlpacaClient("k", "s", "https://api.alpaca.markets")


@respx.mock
def test_get_account_parses_response(client: AlpacaClient) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(
            200,
            json={
                "equity": "100000.50",
                "cash": "50000.25",
                "buying_power": "200000.00",
                "status": "ACTIVE",
                "extra": "ignored",
            },
        )
    )
    account = client.get_account()
    assert account.equity == pytest.approx(100000.50)
    assert account.cash == pytest.approx(50000.25)
    assert account.buying_power == pytest.approx(200000.0)
    assert account.status == "ACTIVE"


@respx.mock
def test_get_positions_returns_list(client: AlpacaClient) -> None:
    respx.get(f"{PAPER_URL}/v2/positions").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "symbol": "AAPL",
                    "qty": "10",
                    "avg_entry_price": "180.0",
                    "market_value": "1900.0",
                    "unrealized_pl": "100.0",
                }
            ],
        )
    )
    positions = client.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == pytest.approx(10.0)


@respx.mock
def test_submit_order_sends_expected_body(client: AlpacaClient) -> None:
    route = respx.post(f"{PAPER_URL}/v2/orders").mock(
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
    order = client.submit_order("aapl", 1, "buy", client_order_id="cid-1")
    assert route.called
    body = json.loads(route.calls.last.request.read())
    assert body["symbol"] == "AAPL"
    assert body["side"] == "buy"
    assert body["client_order_id"] == "cid-1"
    assert body["type"] == "market"
    assert body["time_in_force"] == "day"
    assert order.id == "o-1"
    assert order.status == "accepted"


def test_submit_order_rejects_invalid_side(client: AlpacaClient) -> None:
    with pytest.raises(ValueError, match="side"):
        client.submit_order("AAPL", 1, "short")  # type: ignore[arg-type]


def test_submit_order_rejects_non_positive_qty(client: AlpacaClient) -> None:
    with pytest.raises(ValueError, match="qty"):
        client.submit_order("AAPL", 0, "buy")


@respx.mock
def test_request_raises_on_error_status(client: AlpacaClient) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(403, json={"message": "forbidden"})
    )
    with pytest.raises(AlpacaError, match="403"):
        client.get_account()


@respx.mock
def test_cancel_order_handles_204(client: AlpacaClient) -> None:
    respx.delete(f"{PAPER_URL}/v2/orders/o-1").mock(return_value=httpx.Response(204))
    client.cancel_order("o-1")  # should not raise
